"""comprehensive_evaluation_extended.py
---------------------------------
Extended version that includes fine-tuned models in the evaluation.
Based on the original comprehensive_evaluation.py but adds support for LoRA models.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
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
# Helper functions (same as original)
# ---------------------------------------------------------------------------


def load_gold(file: Path, split: str = "all", seed: int = 42):
    """Return gold_standard dict & gold_ticker_sentiments dict."""
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
    cm = confusion_matrix(y_true, y_pred, labels=[
                          "Positive", "Neutral", "Negative"])

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
        "ticker_stats": {
            "correct": ticker_ok,
            "total": total_ticker
        },
        "per_class": {
            lab: {"precision": float(prec_c[i]), "recall": float(
                rec_c[i]), "f1": float(f1_c[i])}
            for i, lab in enumerate(["Positive", "Neutral", "Negative"])
        },
    }


# ---------------------------------------------------------------------------
# Main (extended to include fine-tuned models)
# ---------------------------------------------------------------------------

def main(argv: List[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Evaluate ALL pipeline outputs including fine-tuned models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--gold", type=Path, default=Path(GOLD_FILE), help="Gold‑standard JSONL path")
    parser.add_argument(
        "--split", choices=["all", "dev"], default="all", help="Subset to evaluate")
    parser.add_argument("--include-finetuned", action="store_true", default=True,
                        help="Include fine-tuned models in evaluation")
    args = parser.parse_args(argv)

    gold_path: Path = args.gold
    split: str = args.split

    print(f"Loading gold standard from {gold_path} …")
    gold_standard, gold_tickers = load_gold(gold_path, split=split)
    print(f"Loaded {len(gold_standard)} gold annotations (split={split})")

    # Extended pipeline list including fine-tuned models
    pipelines = {
        "Standard": "data/processed_articles_standard.jsonl",
        "Optimized": "data/processed_articles_optimized.jsonl",
        "VADER": "data/vader_baseline_results.jsonl",
        "Calibrated": "data/processed_articles_calibrated.jsonl",
    }

    # Add fine-tuned models if requested
    if args.include_finetuned:
        pipelines.update({
            "FinBERT-LoRA": "data/processed_articles_finetuned_finbert.jsonl",
            "DeBERTa-LoRA": "data/processed_articles_finetuned_deberta.jsonl",
            "Distance-Weighted": "data/processed_articles_distance_weighted.jsonl",
        })

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
        print(f"Overall Accuracy: {metrics['accuracy']:.3f}")
        print(f"Macro F1:         {metrics['macro_f1']:.3f}")
        if metrics["ticker_accuracy"] is not None:
            ticker_stats = metrics.get("ticker_stats", {})
            print(f"Ticker Acc:       {metrics['ticker_accuracy']:.3f} " +
                  f"({ticker_stats.get('correct', 0)}/{ticker_stats.get('total', 0)})")

    # save json
    out = Path("data/evaluation_results_extended.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\n[✓] Saved {out}")

    # Enhanced summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Pipeline':<15} | {'Accuracy':<8} | {'Macro‑F1':<8} | {'Ticker Acc':<10} | Samples")
    print("-" * 70)

    # Sort by Macro F1 score
    sorted_results = sorted(
        results.items(), key=lambda x: x[1]['macro_f1'], reverse=True)

    for n, m in sorted_results:
        ticker_acc_str = f"{m['ticker_accuracy']:.3f}" if m['ticker_accuracy'] is not None else "N/A"
        print(
            f"{n:<15} | {m['accuracy']:.3f}    | {m['macro_f1']:.3f}    | {ticker_acc_str:<10} | {m['n_samples']}")

    # Find best model
    if sorted_results:
        best_name, best_metrics = sorted_results[0]
        print(
            f"\n🏆 Best Model: {best_name} (Macro F1: {best_metrics['macro_f1']:.3f})")


if __name__ == "__main__":
    main()
