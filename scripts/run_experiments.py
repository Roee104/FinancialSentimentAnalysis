"""scripts/run_experiments.py

A *single‑purpose* runner that fine‑tunes a transformer for sentiment
classification using **LoRA** (or full fine‑tune) and logs experiment metadata.
It replaces the old "batch pipeline" script so that we can train FinBERT and
DeBERTa‑Fin on the **3 000‑article gold standard** and save compact LoRA
adapters under `models/lora/<model>/<timestamp>/`.

Run examples
------------
```powershell
python -m scripts.run_experiments --model finbert --lora --epochs 2 --lr 2e-5 \
       --rank 8 --alpha 32 --gold data/3000_gold_standard.jsonl
```

Key behaviour
-------------
* Creates `experiments/<timestamp>_<model>_lora.json` with args + metrics.
* Saves adapter to `models/lora/<model>/<timestamp>/`.
* Prints the adapter path to **STDERR** so calling wrappers can capture it.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          Trainer, TrainingArguments)

# ------------------------------------------------  logging setup
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
    stream=sys.stdout,
)
LOG = logging.getLogger("run_experiments")

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models" / "lora"
EXPT_DIR = ROOT / "experiments"
EXPT_DIR.mkdir(exist_ok=True, parents=True)

# ----------------------------- helper ----------------------------------------


def timestamp() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")


# ----------------------------- CLI ------------------------------------------

def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine‑tune a model with LoRA")
    p.add_argument("--model", required=True,
                   choices=["finbert", "deberta-fin"],
                   help="Base model ID (shortcut names)")
    p.add_argument("--lora", action="store_true",
                   help="Enable LoRA fine‑tuning (recommended)")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--rank", type=int, default=8,
                   help="LoRA rank (only if --lora)")
    p.add_argument("--alpha", type=int, default=32,
                   help="LoRA alpha (only if --lora)")
    p.add_argument("--gold", type=Path, required=True,
                   help="Path to gold‑standard JSONL")
    return p.parse_args(argv)


# ----------------------- dataset preparation ---------------------------------

def load_gold_dataset(gold_path: Path):
    """Return HF Dataset with `text` and `label` columns."""
    raw = []
    label2id = {"Positive": 0, "Neutral": 1, "Negative": 2}
    with gold_path.open(encoding="utf-8") as fh:
        for line in fh:
            obj = json.loads(line)
            raw.append({
                "text": obj["article_text"],  # assume field present
                "label": label2id[obj["true_overall"]],
            })
    return load_dataset("json", data_files={"train": raw, "validation": raw[:200]})


# ----------------------------- main logic ------------------------------------

def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)
    LOG.info("🏷  Model: %s", args.model)
    LOG.info("📜 Gold:  %s", args.gold)

    # --- map shortcut → HF model id
    HF_ID = {
        "finbert": "ProsusAI/finbert",
        "deberta-fin": "deepset/deberta-v3-base-finetuned-financial-sentiment",
    }[args.model]

    tokenizer = AutoTokenizer.from_pretrained(HF_ID)
    model = AutoModelForSequenceClassification.from_pretrained(
        HF_ID, num_labels=3)

    if args.lora:
        lora_cfg = LoraConfig(r=args.rank, lora_alpha=args.alpha,
                              target_modules=["query", "value"],
                              lora_dropout=0.05, bias="none")
        model = get_peft_model(model, lora_cfg)
        LOG.info("🔧 Enabled LoRA ‑ rank=%d alpha=%d", args.rank, args.alpha)

    # dataset
    ds = load_gold_dataset(args.gold)

    def tokenize(batch: Dict[str, str]):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)

    ds = ds.map(tokenize, batched=True)
    ds.set_format(type="torch", columns=[
                  "input_ids", "attention_mask", "label"])

    out_dir = MODELS_DIR / args.model / timestamp()
    out_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="no",
        load_best_model_at_end=False,
        fp16=torch.cuda.is_available(),
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        tokenizer=tokenizer,
    )

    trainer.train()

    # save adapter or full model
    if args.lora:
        model.save_pretrained(out_dir)
    else:
        model.save_pretrained(out_dir / "full_model")
    tokenizer.save_pretrained(out_dir)

    # ---------------------- metrics + metadata ------------------------------
    metrics = trainer.evaluate()
    meta = {
        "timestamp": timestamp(),
        "base_model": HF_ID,
        "lora": bool(args.lora),
        "epochs": args.epochs,
        "lr": args.lr,
        "rank": args.rank if args.lora else None,
        "alpha": args.alpha if args.lora else None,
        "train_samples": len(ds["train"]),
        "val_samples": len(ds["validation"]),
        "metrics": metrics,
    }
    expt_path = EXPT_DIR / \
        f"{timestamp()}_{args.model}_{'lora' if args.lora else 'full'}.json"
    expt_path.write_text(json.dumps(meta, indent=2))

    print(f"[✓] Saved adapter to {out_dir}", file=sys.stderr)
    LOG.info("Done. Adapter dir: %s", out_dir)


if __name__ == "__main__":
    main()
