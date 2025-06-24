"""comprehensive_evaluation.py
---------------------------------
Evaluate one or more processed‑pipeline JSONL files against the gold‑standard
annotations and write a summary report (`data/evaluation_results.json`).

* Uses the default `GOLD_FILE` from `config.settings`, but you can override
  with `--gold path/to/gold.jsonl`.
* Supports `--split dev|all` where *dev* = deterministic 1 000‑row slice for
  quick checks.

Run examples
~~~~~~~~~~~~
$ python scripts/comprehensive_evaluation.py                 # full eval on all
$ python scripts/comprehensive_evaluation.py --split dev     # fast sanity run
$ python scripts/comprehensive_evaluation.py --gold data/...jsonl --split all
"""
from __future__ import annotations

import argparse, json, random, sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from config.settings import GOLD_FILE

# ---------------------------------------------------------------------------
# Helper --------------------------------------------------------------------
# ---------------------------------------------------------------------------

def load_gold(file: Path, split: str = "all", seed: int = 42):
    """Return gold_standard dict & gold_ticker_sentiments dict.

    *split*:
      **all** – full file.
      **dev** – deterministic first 1 000 (or full file if <1 000).
    """
    gold: Dict[str, str] = {}
    gold_tickers: Dict[str, Dict] = {}

    with file.open("r", encoding="utf-8") as fh:
        lines = list(fh)

    if split == "dev":
        random.seed(seed)
        random.shuffle(lines)
        lines = lines[:1000]

    for ln in lines:
        ann = json.loads(ln)
        h = ann["article_hash"]
        gold[h] = ann["true_overall"]
        gold_tickers[h] = ann.get("ticker_sentiments", {})

    return gold, gold_tickers


def evaluate_pipeline(result_path: Path, gold: Dict, gold_tickers: Dict):
    """Return metrics dict or None if file missing."""
    if not result_path.exists():
        return None

    y_true: List[str] = []
    y_pred: List[str] = []
    ticker_ok = total_ticker = 0

    with result_path.open("r", encoding="utf-8") as fh:
        for ln in fh:
            try:
                pred = json.loads(ln)
                h = pred.get("article_hash")
                if h not in gold:
                    continue

                y_true.append(gold[h])
                y_pred.append(pred["overall_sentiment"])

                if h in gold_tickers:
                    g_map = gold_tickers[h]
                    p_map = {t["symbol"]: t for t in pred.get("tickers", [])}
                    for sym, g_info in g_map.items():
                        if sym in p_map:
                            total_ticker += 1
                            if p_map[sym]["label"] == g_info["sentiment"]:
                                ticker_ok += 1
            except Exception:
                continue

    if not y_true:
        return None

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=["Positive", "Neutral", "Negative"], average="macro"
    )
    cm = confusion_matrix(y_true, y_pred, labels=["Positive", "Neutral", "Negative"])

    prec_c, rec_c, f1_c, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=["Positive", "Neutral", "Negative"], average=None
    )

    ticker_acc = ticker_ok / total_ticker if total_ticker else None

    return {
        "accuracy": acc,
        "macro_f1": f1,
        "precision": prec,
        "recall": rec,
        "confusion_matrix": cm.tolist(),
        "n_samples": len(y_true),
        "ticker_accuracy": ticker_acc,
        "per_class": {
            lab: {"precision": float(prec_c[i]), "recall": float(rec_c[i]), "f1": float(f1_c[i])}
            for i, lab in enumerate(["Positive", "Neutral", "Negative"])
        },
    }


# ---------------------------------------------------------------------------
# Main ----------------------------------------------------------------------
# ---------------------------------------------------------------------------

def main(argv: List[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Evaluate processed pipeline outputs against gold standard",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--gold", type=Path, default=Path(GOLD_FILE), help="Gold‑standard JSONL path")
    parser.add_argument("--split", choices=["all", "dev"], default="all", help="Subset to evaluate")
    args = parser.parse_args(argv)

    gold_path: Path = args.gold
    split: str = args.split

    print(f"Loading gold standard from {gold_path} …")
    gold_standard, gold_tickers = load_gold(gold_path, split=split)
    print(f"Loaded {len(gold_standard)} gold annotations (split={split})")

    pipelines = {
        "Standard": "data/processed_articles_standard.jsonl",
        "Optimized": "data/processed_articles_optimized.jsonl",
        "VADER": "data/vader_baseline_results.jsonl",
        "Calibrated": "data/processed_articles_calibrated.jsonl",
    }

    results = {}

    for name, path_str in pipelines.items():
        path = Path(path_str)
        print("\n" + "=" * 50)
        print(f"Evaluating {name} (file = {path.name})")
        print("=" * 50)
        metrics = evaluate_pipeline(path, gold_standard, gold_tickers)
        if metrics is None:
            print(f"Skipping {name} — file missing or no matching hashes")
            continue

        results[name] = metrics

        # pretty print summary for this pipeline
        print("Overall Accuracy:", f"{metrics['accuracy']:.3f}")
        print("Macro F1:        ", f"{metrics['macro_f1']:.3f}")
        if metrics["ticker_accuracy"] is not None:
            print("Ticker Acc.:     ", f"{metrics['ticker_accuracy']:.3f} (n={len(gold_tickers)})")

    # save json
    out = Path("data/evaluation_results.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print("\n[✓] Saved", out)

    # tiny summary table
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Pipeline':<12} | {'Accuracy':<8} | {'Macro‑F1':<8} | Samples")
    print("-" * 50)
    for n, m in results.items():
        print(f"{n:<12} | {m['accuracy']:.3f}    | {m['macro_f1']:.3f}   | {m['n_samples']}")


if __name__ == "__main__":
    main()
