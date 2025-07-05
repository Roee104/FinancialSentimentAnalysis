import json
from collections import defaultdict

# Load gold standard
gold_data = {}
with open('data/gold_standard_annotations.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        try:
            ann = json.loads(line)
            gold_data[ann['article_hash']] = ann
        except:
            continue

# Load predictions (using optimized as example)
predictions = {}
with open('data/processed_articles_optimized.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        try:
            pred = json.loads(line)
            if pred['article_hash'] in gold_data:
                predictions[pred['article_hash']] = pred
        except:
            continue

# Find misclassifications
errors = defaultdict(list)

for article_hash, gold in gold_data.items():
    if article_hash in predictions:
        pred = predictions[article_hash]
        if pred['overall_sentiment'] != gold['true_overall']:
            errors[f"{gold['true_overall']} -> {pred['overall_sentiment']}"].append({
                'title': gold['title'][:100],
                'gold': gold['true_overall'],
                'predicted': pred['overall_sentiment'],
                'gold_confidence': gold.get('overall_confidence', 0),
                'pred_confidence': pred.get('overall_confidence', 0)
            })

print("="*60)
print("ERROR ANALYSIS")
print("="*60)

for error_type, cases in errors.items():
    print(f"\n{error_type}: {len(cases)} cases")
    print("-"*40)
    # Show top 3 examples
    for case in cases[:3]:
        print(f"Title: {case['title']}...")
        print(f"  Gold confidence: {case['gold_confidence']:.2f}")
        print(f"  Pred confidence: {case['pred_confidence']:.2f}")
        print()
