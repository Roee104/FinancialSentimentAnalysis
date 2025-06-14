# debug_sentiment.py

"""
Debug script to understand the sentiment distribution discrepancy
"""

import pandas as pd
from collections import defaultdict
from sentiment import FinBERTSentimentAnalyzer
from sentiment_optimized import OptimizedSentimentAnalyzer
import json
import os


def analyze_pipeline_output():
    """Check the sentiment distribution from your pipeline output"""
    print("🔍 Analyzing Pipeline Output Distribution")
    print("=" * 50)

    # Check if processed_articles.jsonl exists
    pipeline_file = "processed_articles.jsonl"
    if os.path.exists(pipeline_file):
        # Load and analyze
        sentiments = defaultdict(int)
        total = 0

        with open(pipeline_file, 'r') as f:
            for line in f:
                article = json.loads(line)
                sentiment = article.get('overall_sentiment', 'Unknown')
                sentiments[sentiment] += 1
                total += 1

        print(f"Total articles: {total}")
        print("\nSentiment Distribution:")
        for sentiment, count in sentiments.items():
            pct = count / total * 100 if total > 0 else 0
            print(f"  {sentiment}: {count} ({pct:.1f}%)")
    else:
        print(f"❌ {pipeline_file} not found")

    print("\n" + "="*50)


def test_specific_texts():
    """Test on specific text examples to understand behavior"""
    print("\n🧪 Testing Specific Text Examples")
    print("=" * 50)

    # Test texts that should have clear sentiment
    test_texts = [
        # Clearly positive
        "Apple reported record-breaking revenue with profits exceeding expectations by 20%. The company's strong performance drove stock prices to all-time highs.",

        # Clearly negative
        "The company filed for bankruptcy after massive losses. Revenue plummeted 40% amid declining sales and mounting debt.",

        # Actually neutral
        "The quarterly earnings report will be released next Tuesday. The company will hold a conference call to discuss results.",

        # Mixed but leaning positive
        "Despite supply chain challenges, the company managed to maintain profitability and even increased market share slightly.",

        # Mixed but leaning negative
        "While revenue remained stable, rising costs and regulatory concerns cast doubt on future growth prospects."
    ]

    # Test with both analyzers
    original = FinBERTSentimentAnalyzer()
    optimized = OptimizedSentimentAnalyzer()

    print("Original FinBERT:")
    orig_results = original.predict(test_texts)
    for i, result in enumerate(orig_results):
        print(f"\nText {i+1}: {test_texts[i][:60]}...")
        print(
            f"  Prediction: {result['label']} (confidence: {result['confidence']:.3f})")
        print(f"  Scores: Pos={result['scores']['Positive']:.3f}, "
              f"Neu={result['scores']['Neutral']:.3f}, "
              f"Neg={result['scores']['Negative']:.3f}")

    print("\n" + "-"*50)
    print("Optimized FinBERT:")
    opt_results = optimized.predict_optimized(test_texts)
    for i, result in enumerate(opt_results):
        print(f"\nText {i+1}: {test_texts[i][:60]}...")
        print(
            f"  Prediction: {result['label']} (confidence: {result['confidence']:.3f})")
        print(f"  Scores: Pos={result['scores']['Positive']:.3f}, "
              f"Neu={result['scores']['Neutral']:.3f}, "
              f"Neg={result['scores']['Negative']:.3f}")
        if result['bias_corrected']:
            print(f"  ⚡ Changed from: {result['original_label']}")


def check_data_characteristics():
    """Check characteristics of the actual data being tested"""
    print("\n📊 Checking Data Characteristics")
    print("=" * 50)

    try:
        df = pd.read_parquet("data/financial_news_2020_2025_100k.parquet")

        # Sample for analysis
        sample = df.sample(n=100, random_state=42)

        # Check text lengths
        sample['text_length'] = sample.apply(lambda row:
                                             len(str(row['title']) + " " + str(row['content']).split()[:75]), axis=1)

        print(
            f"Average text length (words): {sample['text_length'].mean():.1f}")
        print(
            f"Min/Max: {sample['text_length'].min()} / {sample['text_length'].max()}")

        # Check original sentiment labels if available
        if 'sentiment_label' in df.columns:
            print("\nOriginal sentiment labels in data:")
            print(df['sentiment_label'].value_counts(normalize=True).round(3))

        # Sample some titles
        print("\nSample article titles:")
        for i, title in enumerate(sample['title'].head(5), 1):
            print(f"{i}. {title[:80]}...")

    except Exception as e:
        print(f"Error loading data: {e}")


def test_bias_correction_logic():
    """Test the bias correction logic directly"""
    print("\n🔧 Testing Bias Correction Logic")
    print("=" * 50)

    import torch

    # Simulate probability distributions
    test_probs = [
        # Strong neutral (should be reduced)
        torch.tensor([[0.1, 0.8, 0.1]]),  # [neg, neu, pos]
        # Weak neutral (should be changed)
        torch.tensor([[0.3, 0.4, 0.3]]),
        # Strong positive (should stay)
        torch.tensor([[0.1, 0.2, 0.7]]),
        # Strong negative (should stay)
        torch.tensor([[0.7, 0.2, 0.1]]),
    ]

    optimizer = OptimizedSentimentAnalyzer()

    for i, probs in enumerate(test_probs):
        print(f"\nTest case {i+1}:")
        print(
            f"  Original: Neg={probs[0, 0]:.2f}, Neu={probs[0, 1]:.2f}, Pos={probs[0, 2]:.2f}")

        # Apply correction
        corrected = optimizer._apply_bias_correction(probs)
        print(
            f"  Corrected: Neg={corrected[0, 0]:.2f}, Neu={corrected[0, 1]:.2f}, Pos={corrected[0, 2]:.2f}")

        # Determine classification
        orig_class = ['Negative', 'Neutral',
                      'Positive'][torch.argmax(probs).item()]
        corr_class = ['Negative', 'Neutral',
                      'Positive'][torch.argmax(corrected).item()]

        print(f"  Classification: {orig_class} → {corr_class}")


if __name__ == "__main__":
    print("🐛 Debugging Sentiment Analysis Distribution")
    print("=" * 50)

    # 1. Check pipeline output distribution
    analyze_pipeline_output()

    # 2. Test specific examples
    test_specific_texts()

    # 3. Check data characteristics
    check_data_characteristics()

    # 4. Test bias correction logic
    test_bias_correction_logic()

    print("\n✅ Debugging completed!")
