# src/pipeline.py

"""
Optimized version of the original pipeline that won't freeze in Colab.
Key improvements:
- Batch processing with memory management
- Progress saving (can resume if interrupted)
- Smaller batch sizes for Colab
- Memory cleanup between batches
"""

import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import gc
import os
from datetime import datetime

from splitter import split_to_chunks
from sentiment import FinBERTSentimentAnalyzer
from aggregator import compute_ticker_sentiment, compute_sector_sentiment, compute_article_sentiment
from ner import load_symbol_list, get_combined_symbols


def process_batch(batch_df, analyzer, ticker_dict, method="default", threshold=0.1):
    """Process a batch of articles"""
    results = []
    
    for _, row in batch_df.iterrows():
        try:
            date = row.get("date")
            title = str(row.get("title", ""))
            content = str(row.get("content", ""))
            
            # Skip if empty
            if not title and not content:
                continue
                
            full_text = f"{title}\n\n{content}"
            
            # Split into chunks
            chunks = split_to_chunks(full_text)
            if not chunks:
                continue
            
            # Limit chunks to prevent memory issues
            if len(chunks) > 50:
                chunks = chunks[:50]
            
            # Sentiment prediction
            preds = analyzer.predict(chunks, batch_size=8)  # Smaller batch for Colab
            
            # Extract tickers
            article_meta = {
                "symbols": row.get("symbols", []),
                "title": title,
                "content": content
            }
            symbols = get_combined_symbols(
                article_meta,
                dictionary=ticker_dict,
                use_text_extraction=True
            )
            
            # Assign sentiments to tickers
            ticker_chunks = []
            for chunk, pred in zip(chunks, preds):
                for sym in symbols:
                    ticker_chunks.append(
                        (sym, pred["label"], pred["confidence"])
                    )
            
            # Aggregate
            ticker_sents = compute_ticker_sentiment(ticker_chunks, method=method, threshold=threshold)
            sector_sents = compute_sector_sentiment(ticker_sents, threshold=threshold)
            article_label, article_conf = compute_article_sentiment(ticker_sents, threshold=threshold)
            
            # Build record
            record = {
                "date": date.isoformat() if hasattr(date, "isoformat") else str(date),
                "title": title[:500],  # Limit title length
                "overall_sentiment": article_label,
                "overall_confidence": float(article_conf),
                "tickers": [{"symbol": t, **info} for t, info in ticker_sents.items()],
                "sectors": [{"sector": s, **info} for s, info in sector_sents.items()]
            }
            
            results.append(record)
            
        except Exception as e:
            print(f"Error processing article: {e}")
            continue
    
    return results


def main(
    input_path: str = "data/financial_news_2020_2025_100k.parquet",
    output_path: str = "data/processed_articles.jsonl",
    ticker_csv: str = "data/master_ticker_list.csv",
    batch_size: int = 100,
    max_articles: int = None,
    resume: bool = True
):
    """
    Optimized pipeline for Colab with batch processing and memory management.
    
    Args:
        input_path: Input parquet file
        output_path: Output JSONL file
        ticker_csv: Master ticker list
        batch_size: Number of articles per batch (lower for Colab)
        max_articles: Maximum articles to process (None for all)
        resume: Whether to resume from previous run
    """
    
    print(f"🚀 Starting Optimized Pipeline")
    print(f"   Batch size: {batch_size}")
    print(f"   Max articles: {max_articles or 'All'}")
    print("=" * 50)
    
    # Check if we're resuming
    start_idx = 0
    processed_ids = set()
    
    if resume and os.path.exists(output_path):
        print("📂 Found existing output, resuming...")
        with open(output_path, 'r') as f:
            for line in f:
                try:
                    record = json.loads(line)
                    processed_ids.add(record.get('title', ''))
                except:
                    continue
        print(f"   Already processed: {len(processed_ids)} articles")
    
    # Load ticker dictionary
    print("Loading ticker dictionary...")
    ticker_dict = load_symbol_list(ticker_csv)
    
    # Initialize sentiment analyzer
    print("Loading FinBERT analyzer...")
    analyzer = FinBERTSentimentAnalyzer()
    
    # Load data
    print(f"Loading articles from {input_path}...")
    df = pd.read_parquet(input_path, engine="pyarrow")
    
    # Limit if requested
    if max_articles:
        df = df.head(max_articles)
    
    print(f"Total articles to process: {len(df)}")
    
    # Open output file in append mode if resuming
    mode = 'a' if (resume and os.path.exists(output_path)) else 'w'
    
    # Statistics
    stats = {
        'processed': 0,
        'errors': 0,
        'sentiment_dist': {'Positive': 0, 'Neutral': 0, 'Negative': 0}
    }
    
    # Process in batches
    total_batches = (len(df) + batch_size - 1) // batch_size
    
    with open(output_path, mode, encoding='utf-8') as fout:
        for batch_idx in tqdm(range(0, len(df), batch_size), total=total_batches, desc="Processing batches"):
            # Get batch
            batch_df = df.iloc[batch_idx:batch_idx + batch_size]
            
            # Skip if already processed (when resuming)
            if resume:
                batch_df = batch_df[~batch_df['title'].isin(processed_ids)]
                if len(batch_df) == 0:
                    continue
            
            # Process batch
            try:
                results = process_batch(batch_df, analyzer, ticker_dict)
                
                # Write results
                for record in results:
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                    stats['processed'] += 1
                    stats['sentiment_dist'][record['overall_sentiment']] += 1
                
                # Flush to disk
                fout.flush()
                
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                stats['errors'] += 1
            
            # Memory cleanup every 10 batches
            if (batch_idx // batch_size) % 10 == 0:
                gc.collect()
                
            # Print progress every 20 batches
            if (batch_idx // batch_size) % 20 == 0 and batch_idx > 0:
                print(f"\nProgress: {stats['processed']} articles processed")
                print(f"Sentiment distribution so far: {stats['sentiment_dist']}")
    
    # Final statistics
    print("\n" + "="*50)
    print("📊 PIPELINE STATISTICS")
    print("="*50)
    print(f"Total articles processed: {stats['processed']}")
    print(f"Errors: {stats['errors']}")
    
    print("\nSentiment Distribution:")
    total = stats['processed']
    for sentiment, count in stats['sentiment_dist'].items():
        pct = count / total * 100 if total > 0 else 0
        print(f"  {sentiment}: {count} ({pct:.1f}%)")
    
    print(f"\n✅ Pipeline completed! Results saved to {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run optimized pipeline")
    parser.add_argument("--batch-size", type=int, default=100,
                       help="Batch size (lower for Colab, e.g., 50-100)")
    parser.add_argument("--max-articles", type=int, default=None,
                       help="Maximum articles to process")
    parser.add_argument("--no-resume", action="store_true",
                       help="Start fresh (don't resume)")
    parser.add_argument("--output", type=str, default="data/processed_articles.jsonl",
                       help="Output file")
    
    args = parser.parse_args()
    
    main(
        batch_size=args.batch_size,
        max_articles=args.max_articles,
        resume=not args.no_resume,
        output_path=args.output
    )