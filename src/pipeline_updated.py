# src/pipeline_updated.py

"""
Updated pipeline with Enhanced NER and improved aggregation.
Uses frozen FinBERT but with better ticker extraction and aggregation.
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
    ticker_csv: str = "data/master_ticker_list.csv",
    method: str = "conf_weighted",
    threshold: float = 0.1
):
    """
    Updated pipeline with enhanced NER and configurable aggregation.

    Args:
        input_path: Input parquet file
        output_path: Output JSONL file
        ticker_csv: Master ticker list CSV
        method: Aggregation method (default, majority, conf_weighted)
        threshold: Threshold for sentiment classification
    """

    print(f"🚀 Starting Enhanced Pipeline")
    print(f"   Method: {method}")
    print(f"   Threshold: {threshold}")
    print("=" * 50)

    # 1. Initialize Enhanced NER
    print("Loading Enhanced NER...")
    ner = EnhancedNER(ticker_csv)

    # 2. Initialize FinBERT analyzer (frozen)
    print("Loading FinBERT sentiment analyzer...")
    analyzer = FinBERTSentimentAnalyzer()

    # 3. Read articles
    print(f"Loading articles from {input_path}...")
    df = pd.read_parquet(input_path, engine="pyarrow")
    print(f"Loaded {len(df)} articles")

    # 4. Statistics tracking
    stats = {
        'total_articles': 0,
        'articles_with_tickers': 0,
        'articles_without_tickers': 0,
        'total_tickers_found': 0,
        'sentiment_distribution': {'Positive': 0, 'Neutral': 0, 'Negative': 0}
    }

    # 5. Process articles
    with open(output_path, "w", encoding="utf-8") as fout:
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing articles"):
            stats['total_articles'] += 1

            date = row.get("date")
            title = row.get("title", "")
            content = row.get("content", "")
            full_text = f"{title}\n\n{content}"

            # 5.1 Split into chunks
            chunks = split_to_chunks(full_text)
            if not chunks:
                continue

            # 5.2 Extract tickers using Enhanced NER
            article_dict = {
                "title": title,
                "content": content,
                "symbols": row.get("symbols", [])
            }

            # Get enhanced symbols (handles exchange suffixes)
            enhanced_symbols = get_enhanced_symbols(
                article=article_dict,
                ner_extractor=ner,
                min_confidence=0.6,
                use_metadata=True
            )

            if not enhanced_symbols:
                stats['articles_without_tickers'] += 1
                # Still process even without tickers
                enhanced_symbols = []
            else:
                stats['articles_with_tickers'] += 1
                stats['total_tickers_found'] += len(enhanced_symbols)

            # 5.3 Sentiment prediction for all chunks
            preds = analyzer.predict(chunks, batch_size=32)

            # 5.4 Assign chunk sentiments to tickers
            ticker_chunks = []
            for chunk, pred in zip(chunks, preds):
                for sym in enhanced_symbols:
                    ticker_chunks.append(
                        (sym, pred["label"], pred["confidence"])
                    )

            # 5.5 Aggregate sentiments
            ticker_sents = compute_ticker_sentiment(
                ticker_chunks,
                method=method,
                threshold=threshold
            )
            sector_sents = compute_sector_sentiment(
                ticker_sents,
                threshold=threshold
            )
            article_label, article_conf = compute_article_sentiment(
                ticker_sents,
                threshold=threshold
            )

            # Track sentiment distribution
            stats['sentiment_distribution'][article_label] += 1

            # 5.6 Build output record
            record = {
                "date": date.isoformat() if hasattr(date, "isoformat") else str(date),
                "title": title,
                "overall_sentiment": article_label,
                "overall_confidence": article_conf,
                "tickers": [{"symbol": t, **info} for t, info in ticker_sents.items()],
                "sectors": [{"sector": s, **info} for s, info in sector_sents.items()],
                "enhanced_symbols": enhanced_symbols,  # Track what NER found
                "chunk_count": len(chunks)
            }

            # 5.7 Write JSON line
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    # 6. Print statistics
    print("\n" + "="*50)
    print("📊 PIPELINE STATISTICS")
    print("="*50)
    print(f"Total articles processed: {stats['total_articles']}")
    print(
        f"Articles with tickers: {stats['articles_with_tickers']} ({stats['articles_with_tickers']/stats['total_articles']*100:.1f}%)")
    print(
        f"Articles without tickers: {stats['articles_without_tickers']} ({stats['articles_without_tickers']/stats['total_articles']*100:.1f}%)")
    print(
        f"Average tickers per article: {stats['total_tickers_found']/stats['total_articles']:.2f}")

    print("\nSentiment Distribution:")
    total = stats['total_articles']
    for sentiment, count in stats['sentiment_distribution'].items():
        pct = count / total * 100 if total > 0 else 0
        print(f"  {sentiment}: {count} ({pct:.1f}%)")

    # Show NER extraction stats
    ner_stats = ner.get_extraction_stats()
    if ner_stats:
        print("\nNER Extraction Methods:")
        for method, count in sorted(ner_stats.items(), key=lambda x: x[1], reverse=True):
            print(f"  {method}: {count}")

    print(f"\n✅ Done! Processed {len(df)} articles → {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run enhanced pipeline with different settings")
    parser.add_argument("--method", type=str, default="conf_weighted",
                        choices=["default", "majority", "conf_weighted"],
                        help="Aggregation method")
    parser.add_argument("--threshold", type=float, default=0.1,
                        help="Threshold for sentiment classification")
    parser.add_argument("--input", type=str, default="data/financial_news_2020_2025_100k.parquet",
                        help="Input parquet file")
    parser.add_argument("--output", type=str, default="data/processed_articles.jsonl",
                        help="Output JSONL file")

    args = parser.parse_args()

    main(
        input_path=args.input,
        output_path=args.output,
        method=args.method,
        threshold=args.threshold
    )
