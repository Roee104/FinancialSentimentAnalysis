
import json
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np

# Load gold standard
print("Loading gold standard...")
gold_standard = {}
gold_ticker_sentiments = {}

with open('data/gold_standard_annotations.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        ann = json.loads(line)
        article_hash = ann['article_hash']
        gold_standard[article_hash] = ann['true_overall']
        gold_ticker_sentiments[article_hash] = ann.get('ticker_sentiments', {})

print(f"Loaded {len(gold_standard)} gold standard annotations")

# Evaluate each pipeline
pipelines = {
    'Standard': 'data/processed_articles_standard.jsonl',
    'Optimized': 'data/processed_articles_optimized.jsonl',
    'VADER': 'data/vader_baseline_results.jsonl',
    'Calibrated': 'data/processed_articles_calibrated.jsonl'  # if you ran it
}

results = {}

for name, file in pipelines.items():
    if not Path(file).exists():
        print(f"\nSkipping {name} - file not found")
        continue

    print(f"\n{'='*50}")
    print(f"Evaluating {name} Pipeline")
    print('='*50)

    y_true = []
    y_pred = []
    ticker_level_correct = 0
    ticker_level_total = 0

    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                pred = json.loads(line)
                article_hash = pred.get('article_hash')

                if article_hash in gold_standard:
                    # Overall sentiment
                    y_true.append(gold_standard[article_hash])
                    y_pred.append(pred['overall_sentiment'])

                    # Ticker-level evaluation (if available)
                    if article_hash in gold_ticker_sentiments:
                        gold_tickers = gold_ticker_sentiments[article_hash]
                        pred_tickers = {t['symbol']: t for t in pred.get('tickers', [])}

                        for ticker, gold_info in gold_tickers.items():
                            if ticker in pred_tickers:
                                ticker_level_total += 1
                                if pred_tickers[ticker]['label'] == gold_info['sentiment']:
                                    ticker_level_correct += 1
            except Exception as e:
                # Skip any problematic lines
                continue

    # Calculate metrics
    if y_true:
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=['Positive', 'Neutral', 'Negative'], average='macro'
        )
        cm = confusion_matrix(y_true, y_pred, labels=['Positive', 'Neutral', 'Negative'])

        print(f"\nOverall Sentiment Metrics:")
        print(f"  Accuracy: {accuracy:.3f} ({len(y_true)} samples)")
        print(f"  Macro Precision: {precision:.3f}")
        print(f"  Macro Recall: {recall:.3f}")
        print(f"  Macro F1: {f1:.3f}")

        # Per-class metrics
        precision_c, recall_c, f1_c, support_c = precision_recall_fscore_support(
            y_true, y_pred, labels=['Positive', 'Neutral', 'Negative'], average=None
        )

        print(f"\nPer-Class Metrics:")
        for i, label in enumerate(['Positive', 'Neutral', 'Negative']):
            print(f"  {label}: Precision={precision_c[i]:.3f}, Recall={recall_c[i]:.3f}, F1={f1_c[i]:.3f}")

        if ticker_level_total > 0:
            ticker_accuracy = ticker_level_correct / ticker_level_total
            print(f"\nTicker-Level Accuracy: {ticker_accuracy:.3f} ({ticker_level_total} ticker evaluations)")

        # Store results
        results[name] = {
            'accuracy': accuracy,
            'macro_f1': f1,
            'precision': precision,
            'recall': recall,
            'confusion_matrix': cm.tolist(),
            'n_samples': len(y_true),
            'ticker_accuracy': ticker_accuracy if ticker_level_total > 0 else None
        }

# Save results
with open('data/evaluation_results.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2)

print("\n[DONE] Evaluation complete! Results saved to data/evaluation_results.json")

# Print summary comparison
print("\n" + "="*60)
print("SUMMARY COMPARISON")
print("="*60)
print(f"{'Pipeline':<15} {'Accuracy':<10} {'Macro F1':<10} {'Samples':<10}")
print("-"*45)
for name, metrics in results.items():
    print(f"{name:<15} {metrics['accuracy']:.3f}      {metrics['macro_f1']:.3f}      {metrics['n_samples']}")
