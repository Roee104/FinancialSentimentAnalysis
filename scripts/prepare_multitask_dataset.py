import json
from pathlib import Path

INPUT_PATH = "data/final_gold_standard_6000.jsonl"
OUTPUT_PATH = "data/train_ready_multitask.jsonl"
MIN_CONFIDENCE = 0.5

kept, dropped_conf, dropped_empty = 0, 0, 0

with open(INPUT_PATH, "r", encoding="utf-8") as fin, open(OUTPUT_PATH, "w", encoding="utf-8") as fout:
    for line in fin:
        try:
            article = json.loads(line)

            overall = article.get("overall_sentiment")
            confidence = article.get("overall_confidence", 0.0)
            tickers = article.get("ticker_sentiments", {})
            title = article.get("title", "").strip()
            content = article.get("content", "").strip()

            # Filtering
            if not tickers or not isinstance(tickers, dict):
                dropped_empty += 1
                continue
            if confidence < MIN_CONFIDENCE:
                dropped_conf += 1
                continue

            # Format
            text = f"{title}\n{content}".strip()
            ticker_labels = {
                symbol: info["sentiment"]
                for symbol, info in tickers.items()
                if isinstance(info, dict) and "sentiment" in info
            }

            if not ticker_labels:
                dropped_empty += 1
                continue

            record = {
                "text": text,
                "overall_label": overall,
                "ticker_sentiments": ticker_labels
            }

            fout.write(json.dumps(record) + "\n")
            kept += 1

        except json.JSONDecodeError:
            continue

print("✅ Dataset generated:", OUTPUT_PATH)
print(f"🧮 Articles kept: {kept}")
print(f"⚠️  Dropped (low confidence): {dropped_conf}")
print(f"⚠️  Dropped (empty or invalid tickers): {dropped_empty}")
