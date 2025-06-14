# src/vader_baseline.py

"""
VADER baseline for sentiment analysis comparison.
Processes the same articles as the main pipeline for fair comparison.
"""

import json
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from tqdm import tqdm
from collections import defaultdict

from ner import EnhancedNER, get_enhanced_symbols


def vader_sentiment_to_label(compound_score: float, threshold: float = 0.05) -> str:
    """
    Convert VADER compound score to sentiment label.
    
    Args:
        compound_score: VADER compound score (-1 to 1)
        threshold: Threshold for positive/negative classification
        
    Returns:
        Sentiment label (Positive/Neutral/Negative)
    """
    if compound_score >= threshold:
        return "Positive"
    elif compound_score <= -threshold:
        return "Negative"
    else:
        return "Neutral"


def main(
    input_path: str = "data/financial_news_2020_2025_100k.parquet",
    output_path: str = "data/vader_baseline_results.jsonl",
    ticker_csv: str = "data/master_ticker_list.csv",
    threshold: float = 0.05
):
    """
    Run VADER baseline on the same data as the main pipeline.
    """
    
    print("🎯 Running VADER Baseline")
    print(f"   Threshold: {threshold}")
    print("=" * 50)
    
    # Initialize VADER
    print("Initializing VADER...")
    vader = SentimentIntensityAnalyzer()
    
    # Initialize Enhanced NER (same as main pipeline)
    print("Loading Enhanced NER...")
    ner = EnhancedNER(ticker_csv)
    
    # Load articles
    print(f"Loading articles from {input_path}...")
    df = pd.read_parquet(input_path, engine="pyarrow")
    print(f"Loaded {len(df)} articles")
    
    # Statistics
    stats = {
        'total_articles': 0,
        'sentiment_distribution': defaultdict(int),
        'articles_with_tickers': 0,
        'articles_without_tickers': 0
    }
    
    # Process articles
    results = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing with VADER"):
        stats['total_articles'] += 1
        
        title = str(row.get("title", ""))
        content = str(row.get("content", ""))
        full_text = f"{title}. {content}"
        
        # Get VADER scores
        scores = vader.polarity_scores(full_text)
        
        # Convert to label
        sentiment_label = vader_sentiment_to_label(scores['compound'], threshold)
        stats['sentiment_distribution'][sentiment_label] += 1
        
        # Extract tickers (same as main pipeline)
        article_dict = {
            "title": title,
            "content": content,
            "symbols": row.get("symbols", [])
        }
        
        enhanced_symbols = get_enhanced_symbols(
            article=article_dict,
            ner_extractor=ner,
            min_confidence=0.6,
            use_metadata=True
        )
        
        if enhanced_symbols:
            stats['articles_with_tickers'] += 1
        else:
            stats['articles_without_tickers'] += 1
        
        # Create result record
        record = {
            "date": str(row.get("date", "")),
            "title": title,
            "overall_sentiment": sentiment_label,
            "vader_scores": scores,
            "compound_score": scores['compound'],
            "tickers": enhanced_symbols,
            "ticker_count": len(enhanced_symbols)
        }
        
        results.append(record)
    
    # Save results
    with open(output_path, 'w', encoding='utf-8') as f:
        for record in results:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    # Print statistics
    print("\n" + "="*50)
    print("📊 VADER BASELINE STATISTICS")
    print("="*50)
    print(f"Total articles processed: {stats['total_articles']}")
    
    print("\nSentiment Distribution:")
    total = stats['total_articles']
    for sentiment, count in sorted(stats['sentiment_distribution'].items()):
        pct = count / total * 100 if total > 0 else 0
        print(f"  {sentiment}: {count} ({pct:.1f}%)")
    
    print(f"\nArticles with tickers: {stats['articles_with_tickers']} ({stats['articles_with_tickers']/total*100:.1f}%)")
    print(f"Articles without tickers: {stats['articles_without_tickers']} ({stats['articles_without_tickers']/total*100:.1f}%)")
    
    print(f"\n✅ VADER baseline saved to {output_path}")
    
    return stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run VADER baseline")
    parser.add_argument("--threshold", type=float, default=0.05,
                       help="VADER threshold for pos/neg classification")
    parser.add_argument("--input", type=str, default="data/financial_news_2020_2025_100k.parquet",
                       help="Input parquet file")
    parser.add_argument("--output", type=str, default="data/vader_baseline_results.jsonl",
                       help="Output JSONL file")
    
    args = parser.parse_args()
    
    main(
        input_path=args.input,
        output_path=args.output,
        threshold=args.threshold
    )