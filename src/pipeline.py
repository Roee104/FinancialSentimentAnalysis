# src/pipeline.py

"""
Fixed original pipeline to work with Enhanced NER.
This maintains the original structure but fixes imports.
"""

import json
import pandas as pd
from tqdm import tqdm

from splitter import split_to_chunks
from sentiment import FinBERTSentimentAnalyzer
from aggregator import compute_ticker_sentiment, compute_sector_sentiment, compute_article_sentiment
from ner import EnhancedNER, get_enhanced_symbols


def main(
    input_path: str = "data/financial_news_2020_2025_100k.parquet",
    output_path: str = "data/processed_articles.jsonl",
    ticker_csv: str = "data/master_ticker_list.csv"
):
    """
    Original pipeline with fixed imports for Enhanced NER.
    """
    
    print("🚀 Starting Original Pipeline (Fixed)")
    print("=" * 50)
    
    # 1. Initialize Enhanced NER
    print("Loading Enhanced NER...")
    ner = EnhancedNER(ticker_csv)

    # 2. Initialize FinBERT analyzer
    print("Loading FinBERT analyzer...")
    analyzer = FinBERTSentimentAnalyzer()

    # 3. Read articles
    print(f"Loading articles from {input_path}...")
    df = pd.read_parquet(input_path, engine="pyarrow")
    print(f"Loaded {len(df)} articles")

    # 4. Prepare output
    processed_count = 0
    with open(output_path, "w", encoding="utf-8") as fout:
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing articles"):
            try:
                date = row.get("date")
                title = str(row.get("title", ""))
                content = str(row.get("content", ""))
                full_text = f"{title}\n\n{content}"

                # Split into chunks
                chunks = split_to_chunks(full_text)
                if not chunks:
                    continue

                # Sentiment prediction for all chunks
                preds = analyzer.predict(chunks)

                # Extract symbols using Enhanced NER
                article_dict = {
                    "title": title,
                    "content": content,
                    "symbols": row.get("symbols", [])
                }
                
                symbols = get_enhanced_symbols(
                    article=article_dict,
                    ner_extractor=ner,
                    min_confidence=0.6,
                    use_metadata=True
                )
                
                # Assign chunk sentiments to tickers
                ticker_chunks = []
                for chunk, pred in zip(chunks, preds):
                    for sym in symbols:
                        ticker_chunks.append(
                            (sym, pred["label"], pred["confidence"])
                        )

                # Aggregate
                ticker_sents = compute_ticker_sentiment(ticker_chunks)
                sector_sents = compute_sector_sentiment(ticker_sents)
                article_label, article_conf = compute_article_sentiment(ticker_sents)

                # Build output record
                record = {
                    "date": date.isoformat() if hasattr(date, "isoformat") else str(date),
                    "title": title,
                    "overall_sentiment": article_label,
                    "overall_confidence": article_conf,
                    "tickers": [{"symbol": t, **info} for t, info in ticker_sents.items()],
                    "sectors": [{"sector": s, **info} for s, info in sector_sents.items()]
                }

                # Write JSON line
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                processed_count += 1
                
            except Exception as e:
                print(f"Error processing article: {e}")
                continue

    print(f"\n✅ Done! Processed {processed_count} articles → {output_path}")


if __name__ == "__main__":
    main()