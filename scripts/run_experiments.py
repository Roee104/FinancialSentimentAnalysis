"""scripts/run_experiments.py

A *single‑purpose* runner that fine‑tunes a transformer for sentiment
classification using **LoRA** (or full fine‑tune) and logs experiment metadata.
"""

from __future__ import annotations
import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List
from collections import Counter
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import TrainerCallback
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          Trainer, TrainingArguments)

# Fix NumPy 2.0 compatibility issue
import numpy as np
np.set_printoptions(legacy='1.25')

# ---------------------------------  logging setup ----------------------------
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
    # p.add_argument("--gold", type=Path, required=True,
    # help="Path to gold‑standard JSONL")
    p.add_argument("--train", type=Path, required=True,
                   help="Path to train set JSONL")
    p.add_argument("--val", type=Path, required=True,
                   help="Path to validation set JSONL")
    p.add_argument("--patience", type=int, default=2,
                   help="Scheduler patience (ReduceLROnPlateau)")
    p.add_argument("--factor", type=float, default=0.5,
                   help="Scheduler factor (ReduceLROnPlateau)")
    p.add_argument("--lr_scheduler", choices=["reduce", "none"], default="none",
                   help="Use ReduceLROnPlateau if set to 'reduce'")
    p.add_argument("--metric_for_best_model", type=str, default="eval_loss",
                   help="Metric to monitor for LR scheduler or early stopping")

    return p.parse_args(argv)


# ----------------------- dataset preparation ---------------------------------

# ----------------------- gold_path ---------------------------------
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


# ----------------------- train_path|val_path ---------------------------------

def load_dataset_from_two_files(train_path: Path, val_path: Path):
    """Load HF Datasets from separate train and val JSONL files."""
    def load_file(path):
        data = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                a = json.loads(line)
                label = a.get("true_overall", "")
                if label == "Mixed":
                    label = "Neutral"
                if label in ["Positive", "Neutral", "Negative"]:
                    data.append({
                        "text": a["content"],
                        "label": {"Positive": 0, "Neutral": 1, "Negative": 2}[label]
                    })
        return Dataset.from_list(data)

    return {
        "train": load_file(train_path),
        "validation": load_file(val_path)
    }

# ----------------------------- evaluate ------------------------------------


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted')
    acc = accuracy_score(labels, predictions)
    return {
        'accuracy': round(acc, 3),
        'precision': round(precision, 3),
        'recall': round(recall, 3),
        'f1': round(f1, 3),
    }


# ----------------------------- Custom Trainer with weighted loss ------------------------------------

class CustomTrainer(Trainer):
    def __init__(self, *args, class_weights_tensor=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights_tensor = class_weights_tensor

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        loss_fn = CrossEntropyLoss(weight=self.class_weights_tensor)
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss

# ----------------------------- optimizer: LRScheduler ------------------------------------


class LRSchedulerCallback(TrainerCallback):
    def __init__(self, scheduler):
        self.scheduler = scheduler

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is not None and "eval_loss" in metrics:
            self.scheduler.step(metrics["eval_loss"])


# ----------------------------- main logic ------------------------------------

def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)
    LOG.info("🏷  Model: %s", args.model)
    # LOG.info(" Gold:  %s", args.gold)
    LOG.info("Train file: %s", args.train)
    LOG.info("Validation file: %s", args.val)

    # --- map shortcut → HF model id
    HF_ID = {
        "finbert": "ProsusAI/finbert",
        "deberta-fin": "mrm8488/deberta-v3-ft-financial-news-sentiment-analysis",
    }[args.model]

    LOG.info("Loading tokenizer and model: %s", HF_ID)
    tokenizer = AutoTokenizer.from_pretrained(HF_ID)
    model = AutoModelForSequenceClassification.from_pretrained(
        HF_ID, num_labels=3)

    if args.lora:
        # Different target modules for different models
        if args.model == "finbert":
            target_modules = ["query", "value"]  # BERT-based models
        else:  # deberta
            # Based on environment check output
            target_modules = ["query_proj", "value_proj"]  # DeBERTa models

        lora_cfg = LoraConfig(
            r=args.rank,
            lora_alpha=args.alpha,
            target_modules=target_modules,
            lora_dropout=0.01,
            bias="none"
        )
        model = get_peft_model(model, lora_cfg)
        LOG.info("🔧 Enabled LoRA ‑ rank=%d alpha=%d, modules=%s",
                 args.rank, args.alpha, target_modules)

    # dataset
    # ds = load_gold_dataset(args.gold)
    ds = load_dataset_from_two_files(args.train, args.val)

    #
    label_counts = Counter(ds["train"]["label"])
    total = sum(label_counts.values())
    num_classes = 3

    # class_weight[c] = total_samples / (num_classes * count[c])
    class_weights = [total / (num_classes * label_counts[i])
                     for i in range(num_classes)]
    class_weights_tensor = torch.tensor(class_weights).to(
        torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    print("Class weights:", class_weights_tensor)

    LOG.info(
        f"Train samples: {len(ds['train'])}, Val samples: {len(ds['validation'])}")

    def tokenize(batch: Dict[str, str]):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)

    ds["train"] = ds["train"].map(tokenize, batched=True)
    ds["validation"] = ds["validation"].map(tokenize, batched=True)

    # Set format without copy issues (NumPy 2.0 fix)
    ds["train"] = ds["train"].with_format(
        "torch", columns=["input_ids", "attention_mask", "label"])
    ds["validation"] = ds["validation"].with_format(
        "torch", columns=["input_ids", "attention_mask", "label"])

    out_dir = MODELS_DIR / args.model / timestamp()
    out_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        eval_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=5,
        save_steps=None,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        fp16=torch.cuda.is_available(),
        report_to=[],
        label_names=["labels"],  # Explicitly set label names
    )

    # Create optimizer manually
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Scheduler: Reduce LR when validation loss plateaus
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=args.factor,
        patience=args.patience,
        verbose=True
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        optimizers=(optimizer, scheduler),
        callbacks=[LRSchedulerCallback(scheduler)]
    )

    trainer.train()
    trainer.save_state()

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
