import json
from collections import Counter
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from pathlib import Path

INPUT_PATH = "data/final_gold_standard_6000.jsonl"
OUTPUT_PATH = "data/class_weights.json"
VALID_CLASSES = ["Positive", "Neutral", "Negative"]

overall_labels = []
ticker_labels = []

with open(INPUT_PATH, "r", encoding="utf-8") as f:
    for line in f:
        try:
            item = json.loads(line)

            # Overall sentiment
            overall = item.get("overall_sentiment")
            if overall in VALID_CLASSES:
                overall_labels.append(overall)

            # Ticker-level sentiments
            ticker_dict = item.get("ticker_sentiments", {})
            for sent in ticker_dict.values():
                if isinstance(sent, dict):
                    label = sent.get("sentiment")
                    if label in VALID_CLASSES:
                        ticker_labels.append(label)
        except json.JSONDecodeError:
            continue

# --- Compute weights ---


def compute_weights(y, task_name):
    counts = Counter(y)
    total = sum(counts.values())

    print(f"\n📊 {task_name} label distribution:")
    for label in VALID_CLASSES:
        count = counts.get(label, 0)
        pct = (count / total) * 100 if total > 0 else 0
        print(f"  {label:<8}: {count} ({pct:.2f}%)")

    weights = compute_class_weight(
        class_weight="balanced", classes=np.array(VALID_CLASSES), y=y)
    weight_dict = dict(zip(VALID_CLASSES, [float(w) for w in weights]))

    print(f"\n⚖️ {task_name} class weights:")
    for cls, w in weight_dict.items():
        print(f"  {cls:<8}: {w:.4f}")

    return weight_dict


# Compute both heads
overall_weights = compute_weights(overall_labels, "Overall sentiment")
ticker_weights = compute_weights(ticker_labels, "Ticker sentiment")

# Save to JSON
Path("data").mkdir(parents=True, exist_ok=True)
with open(OUTPUT_PATH, "w", encoding="utf-8") as fout:
    json.dump({
        "overall": overall_weights,
        "ticker": ticker_weights
    }, fout, indent=2)

print(f"\n✅ Saved class weights to: {OUTPUT_PATH}")
