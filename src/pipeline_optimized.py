# src/pipeline_optimized_fixed.py

"""
Fixed optimized pipeline that handles numpy arrays properly and has better error handling.
"""

import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import gc
import os
from datetime import datetime
import traceback

from splitter import split_to_chunks
from sentiment import FinBERTSentimentAnalyzer
from aggregator import compute_ticker_sentiment, compute_sector_sentiment, compute_article_sentiment
from ner import EnhancedNER, get_enhanced_symbols


def handle_symbols(symbols_raw):
    """Convert symbols to list format, handling numpy arrays"""
    if symbols_raw is None:
        return []
    elif isinstance(symbols_raw, np.ndarray):
        return symbols_raw.tolist()
    elif isinstance(symbols_raw, list):
        return symbols_raw
    elif isinstance(symbols_raw, str):
        # Handle string that looks like a list
        if symbols_raw.startswith('[') and symbols_raw.endswith(']'):
            try:
                return eval(symbols_raw)
            except:
                return []
        return [symbols_raw] if symbols_raw else []
    elif pd.isna(symbols_raw):
        return []
    else:
        return []


def process_batch(batch_df, analyzer, ner_instance, method="default", threshold=0.1):
    """Process a batch of articles with better error handling"""
    results = []
    
    for idx, row in batch_df.iterrows():
        try:
            date = row.get("date")
            title = str(row.get("title", ""))
            content = str(row.get("content", ""))
            
            # Skip if empty
            if not title and not content:
                continue
                
            # Limit content length to prevent memory issues
            if len(content) > 10000:
                content = content[:10000]
                
            full_text = f"{title}\n\n{content}"
            
            # Split into chunks
            chunks = split_to_chunks(full_text)
            if not chunks:
                continue
            
            # Limit chunks to prevent memory issues
            if len(chunks) > 30:
                chunks = chunks[:30]
            
            # Sentiment prediction with smaller batch
            try:
                preds = analyzer.predict(chunks, batch_size=4)  # Even smaller batch
            except Exception as e:
                print(f"Sentiment prediction error: {e}")
                continue
            
            # Handle symbols properly
            raw_symbols = row.get("symbols", [])
            symbols_list = handle_symbols(raw_symbols)
            
            # Extract tickers using enhanced NER
            article_dict = {
                "title": title,
                "content": content,
                "symbols": symbols_list
            }
            
            try:
                symbols = get_enhanced_symbols(
                    article=article_dict,
                    ner_extractor=ner_instance,
                    min_confidence=0.6,
                    use_metadata=True
                )
            except Exception as e:
                print(f"NER error: {e}")
                symbols = symbols_list  # Fallback to metadata symbols
            
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
                "sectors": [{"sector": s, **info} for s, info in sector_sents.items()],
                "ticker_count": len(symbols),
                "chunk_count": len(chunks)
            }
            
            results.append(record)
            
        except Exception as e:
            print(f"Error processing article idx {idx}: {str(e)[:100]}")
            # traceback.print_exc()  # Uncomment for full traceback
            continue
    
    return results


def main(
    input_path: str = "data/financial_news_2020_2025_100k.parquet",
    output_path: str = "data/processed_articles.jsonl",
    ticker_csv: str = "data/master_ticker_list.csv",
    batch_size: int = 50,
    max_articles: int = None,
    resume: bool = True,
    start_from: int = 0
):
    """
    Fixed optimized pipeline for Colab with better error handling.
    """
    
    print(f"🚀 Starting Fixed Optimized Pipeline")
    print(f"   Batch size: {batch_size}")
    print(f"   Max articles: {max_articles or 'All'}")
    print(f"   Start from: {start_from}")
    print("=" * 50)
    
    # Check if we're resuming
    processed_count = 0
    processed_titles = set()
    
    if resume and os.path.exists(output_path):
        print("📂 Found existing output, counting processed articles...")
        with open(output_path, 'r') as f:
            for line in f:
                try:
                    record = json.loads(line)
                    processed_titles.add(record.get('title', ''))
                    processed_count += 1
                except:
                    continue
        print(f"   Already processed: {processed_count} articles")
    
    # Initialize Enhanced NER
    print("Loading Enhanced NER...")
    ner = EnhancedNER(ticker_csv)
    
    # Initialize sentiment analyzer
    print("Loading FinBERT analyzer...")
    analyzer = FinBERTSentimentAnalyzer()
    
    # Load data
    print(f"Loading articles from {input_path}...")
    df = pd.read_parquet(input_path, engine="pyarrow")
    
    # Apply start_from if specified
    if start_from > 0:
        df = df.iloc[start_from:]
        print(f"Starting from article {start_from}")
    
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
        'skipped': 0,
        'sentiment_dist': {'Positive': 0, 'Neutral': 0, 'Negative': 0}
    }
    
    # Process in batches
    total_batches = (len(df) + batch_size - 1) // batch_size
    
    with open(output_path, mode, encoding='utf-8') as fout:
        for batch_idx in tqdm(range(0, len(df), batch_size), total=total_batches, desc="Processing batches"):
            # Get batch
            batch_df = df.iloc[batch_idx:batch_idx + batch_size]
            
            # Skip already processed articles when resuming
            if resume and processed_titles:
                # Filter out already processed
                mask = ~batch_df['title'].isin(processed_titles)
                batch_df = batch_df[mask]
                
                if len(batch_df) == 0:
                    stats['skipped'] += batch_size
                    continue
            
            # Process batch
            try:
                results = process_batch(batch_df, analyzer, ner)
                
                # Write results
                for record in results:
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                    stats['processed'] += 1
                    stats['sentiment_dist'][record['overall_sentiment']] += 1
                    
                    # Add to processed titles
                    if resume:
                        processed_titles.add(record['title'])
                
                # Flush to disk
                fout.flush()
                
            except Exception as e:
                print(f"\nError in batch starting at {batch_idx}: {e}")
                stats['errors'] += 1
                # Try to process articles individually
                for _, row in batch_df.iterrows():
                    try:
                        single_results = process_batch(pd.DataFrame([row]), analyzer, ner)
                        for record in single_results:
                            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                            stats['processed'] += 1
                            stats['sentiment_dist'][record['overall_sentiment']] += 1
                    except:
                        stats['errors'] += 1
                        continue
            
            # Memory cleanup every 5 batches
            if (batch_idx // batch_size) % 5 == 0:
                gc.collect()
                
            # Print progress every 10 batches
            if (batch_idx // batch_size) % 10 == 0 and batch_idx > 0:
                print(f"\nProgress: {stats['processed'] + processed_count} total articles processed")
                if stats['processed'] > 0:
                    print(f"Recent sentiment distribution: {stats['sentiment_dist']}")
    
    # Final statistics
    print("\n" + "="*50)
    print("📊 PIPELINE STATISTICS")
    print("="*50)
    print(f"New articles processed: {stats['processed']}")
    print(f"Previously processed: {processed_count}")
    print(f"Total processed: {stats['processed'] + processed_count}")
    print(f"Skipped (already processed): {stats['skipped']}")
    print(f"Errors: {stats['errors']}")
    
    if stats['processed'] > 0:
        print("\nNew Articles Sentiment Distribution:")
        total = stats['processed']
        for sentiment, count in stats['sentiment_dist'].items():
            pct = count / total * 100 if total > 0 else 0
            print(f"  {sentiment}: {count} ({pct:.1f}%)")
    
    print(f"\n✅ Pipeline completed! Results saved to {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run fixed optimized pipeline")
    parser.add_argument("--batch-size", type=int, default=50,
                       help="Batch size (lower for Colab, e.g., 25-50)")
    parser.add_argument("--max-articles", type=int, default=None,
                       help="Maximum articles to process")
    parser.add_argument("--no-resume", action="store_true",
                       help="Start fresh (don't resume)")
    parser.add_argument("--output", type=str, default="data/processed_articles.jsonl",
                       help="Output file")
    parser.add_argument("--start-from", type=int, default=0,
                       help="Start from article index (for debugging)")
    
    args = parser.parse_args()
    
    main(
        batch_size=args.batch_size,
        max_articles=args.max_articles,
        resume=not args.no_resume,
        output_path=args.output,
        start_from=args.start_from
    )