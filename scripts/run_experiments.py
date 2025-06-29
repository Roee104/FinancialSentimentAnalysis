"""scripts/run_experiments.py

A *single‑purpose* runner that fine‑tunes a transformer for sentiment
classification using **LoRA** (or full fine‑tune) and logs experiment metadata.
It replaces the old "batch pipeline" script so that we can train FinBERT and
DeBERTa‑Fin on the **3 000‑article gold standard** and save compact LoRA
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
from datasets import Dataset
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
    return datetime.now().strftime("%Y-%m-%dT%H-%M-%S")


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
    raw_train = []
    raw_val = []
    label2id = {"Positive": 0, "Neutral": 1, "Negative": 2}

    # Count labels for logging
    label_counts = {"Positive": 0, "Neutral": 0,
                    "Negative": 0, "Mixed": 0, "Other": 0}

    with gold_path.open(encoding="utf-8") as fh:
        articles = []
        for line in fh:
            article = json.loads(line)
            true_label = article.get("true_overall", "")

            # Handle Mixed labels by mapping to Neutral
            if true_label == "Mixed":
                label_counts["Mixed"] += 1
                article["true_overall"] = "Neutral"
                true_label = "Neutral"

            # Only include articles with valid labels
            if true_label in label2id:
                articles.append(article)
                label_counts[true_label] += 1
            else:
                label_counts["Other"] += 1
                LOG.warning(
                    f"Skipping article with unknown label: {true_label}")

    LOG.info(f"Label distribution: {label_counts}")
    LOG.info(f"Mapped {label_counts['Mixed']} 'Mixed' labels to 'Neutral'")
    LOG.info(f"Total valid articles: {len(articles)}")

    # Use 80/20 split for train/validation
    split_idx = int(len(articles) * 0.8)
    train_articles = articles[:split_idx]
    val_articles = articles[split_idx:]

    for article in train_articles:
        raw_train.append({
            "text": article["content"],
            "label": label2id[article["true_overall"]],
        })

    for article in val_articles:
        raw_val.append({
            "text": article["content"],
            "label": label2id[article["true_overall"]],
        })

    return {
        "train": Dataset.from_list(raw_train),
        "validation": Dataset.from_list(raw_val)
    }


# ----------------------------- main logic ------------------------------------

def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)
    LOG.info("🏷  Model: %s", args.model)
    LOG.info("📜 Gold:  %s", args.gold)

    # --- map shortcut → HF model id
    # Using alternative DeBERTa models that actually exist
    HF_ID = {
        "finbert": "ProsusAI/finbert",
        # Alternative that exists
        "deberta-fin": "mrm8488/deberta-v3-ft-financial-news-sentiment-analysis",
    }[args.model]

    LOG.info("Loading tokenizer and model: %s", HF_ID)
    tokenizer = AutoTokenizer.from_pretrained(HF_ID)
    model = AutoModelForSequenceClassification.from_pretrained(
        HF_ID, num_labels=3)

    if args.lora:
        lora_cfg = LoraConfig(
            r=args.rank,
            lora_alpha=args.alpha,
            target_modules=["query", "value"],
            lora_dropout=0.05,
            bias="none"
        )
        model = get_peft_model(model, lora_cfg)
        LOG.info("🔧 Enabled LoRA ‑ rank=%d alpha=%d", args.rank, args.alpha)

    # dataset
    ds = load_gold_dataset(args.gold)
    LOG.info(
        f"Train samples: {len(ds['train'])}, Val samples: {len(ds['validation'])}")

    def tokenize(batch: Dict[str, str]):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)

    ds["train"] = ds["train"].map(tokenize, batched=True)
    ds["validation"] = ds["validation"].map(tokenize, batched=True)

    ds["train"].set_format(type="torch", columns=[
                           "input_ids", "attention_mask", "label"])
    ds["validation"].set_format(type="torch", columns=[
                                "input_ids", "attention_mask", "label"])

    out_dir = MODELS_DIR / args.model / timestamp()
    out_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(out_dir),
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
