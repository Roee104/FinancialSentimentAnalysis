"""
pipeline.py

End-to-end orchestration for multi-ticker financial news sentiment analysis:
1. Load sampled articles from Parquet.
2. Sentence/Clause Splitting (splitter).
3. Ticker Extraction (ner).
4. Chunk-level sentiment via FinBERT (sentiment).
5. Aggregate to per-ticker, per-sector, and overall article sentiment (aggregator).
6. Export results as JSON Lines for downstream use.
"""
import json
import pandas as pd
from tqdm import tqdm
from src.splitter import split_to_chunks
from src.ner import extract_symbols_from_text
from src.sentiment import FinBERTSentimentAnalyzer
from src.aggregator import load_sector_lookup, compute_ticker_sentiment, compute_sector_sentiment, compute_article_sentiment


def main(
    input_path: str = "data/financial_news_2020_2025.parquet",
    output_path: str = "data/processed_articles.jsonl"
):
    # Load sector lookup and build ticker dictionary
    sector_lookup = load_sector_lookup()
    ticker_dict = set(sector_lookup.keys())

    # Initialize FinBERT analyzer
    analyzer = FinBERTSentimentAnalyzer()

    # Read articles
    df = pd.read_parquet(input_path, engine="pyarrow")

    # Prepare output
    with open(output_path, "w", encoding="utf-8") as fout:
        # Iterate with progress bar
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing articles"):
            date = row["date"]
            title = row.get("title", "")
            content = row.get("content", "")
            full_text = f"{title}\n\n{content}"

            # 1. Split into chunks
            chunks = split_to_chunks(full_text)
            if not chunks:
                continue

            # 2. Sentiment prediction for all chunks at once
            preds = analyzer.predict(chunks)

            # 3. Assign chunk sentiments to tickers
            ticker_chunks = []
            for chunk, pred in zip(chunks, preds):
                symbols = extract_symbols_from_text(chunk, ticker_dict)
                for sym in symbols:
                    ticker_chunks.append((sym, pred["label"], pred["confidence"]))

            # 4. Aggregate
            ticker_sents = compute_ticker_sentiment(ticker_chunks)
            sector_sents = compute_sector_sentiment(ticker_sents)
            article_label, article_conf = compute_article_sentiment(ticker_sents)

            # 5. Build output record
            record = {
                "date": date.isoformat() if hasattr(date, 'isoformat') else str(date),
                "title": title,
                "overall_sentiment": article_label,
                "overall_confidence": article_conf,
                "tickers": [ {"symbol": t, **info} for t, info in ticker_sents.items() ],
                "sectors": [ {"sector": s, **info} for s, info in sector_sents.items() ]
            }

            # 6. Write JSON line
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"✅ Done! Processed {len(df)} articles → {output_path}")


if __name__ == "__main__":
    main()
