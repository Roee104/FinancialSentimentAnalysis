# test_calibration.py

"""
Quick test of the fixed calibrated sentiment analysis
"""

import pandas as pd
from collections import defaultdict
from sentiment import FinBERTSentimentAnalyzer
from sentiment_calibrated import FixedCalibratedFinBERTAnalyzer, find_optimal_calibration

def quick_comparison_test():
    """Quick comparison between original and fixed calibrated sentiment"""
    
    print("🔍 Quick Calibration Fix Test")
    print("=" * 40)
    
    # Load sample data
    try:
        df = pd.read_parquet("data/financial_news_2020_2025_100k.parquet")
        print(f"✅ Loaded {len(df)} articles")
        
        # Sample 50 articles for quick test
        sample_df = df.sample(n=50, random_state=42)
        
        # Prepare texts
        texts = []
        for _, row in sample_df.iterrows():
            title = str(row['title']) if pd.notna(row['title']) else ""
            content = str(row['content']) if pd.notna(row['content']) else ""
            content_short = " ".join(content.split()[:50])  # First 50 words
            full_text = f"{title}. {content_short}"
            texts.append(full_text)
            
    except FileNotFoundError:
        print("Using predefined test texts...")
        texts = [
            "Apple reported quarterly earnings that slightly exceeded analyst expectations.",
            "The company faced some challenges but maintained steady revenue growth.",
            "Stock prices remained relatively stable during the trading session.",
            "Investors showed cautious optimism about the company's future prospects.",
            "Management provided updated guidance that was in line with estimates.",
            "The quarterly dividend payment was maintained at previous levels.",
            "Revenue growth was modest compared to the previous quarter.",
            "Market reaction to the announcement was largely as expected.",
            "The company continues to navigate current industry conditions.",
            "Several business segments showed incremental improvement."
        ]
    
    # Test original FinBERT
    print("\n📊 Original FinBERT:")
    original_analyzer = FinBERTSentimentAnalyzer()
    original_results = original_analyzer.predict(texts)
    original_dist = get_distribution(original_results)
    
    for label, pct in original_dist.items():
        print(f"  {label}: {pct:.1f}%")
    
    # Test fixed calibrated FinBERT
    print("\n📊 Fixed Calibrated FinBERT:")
    calibrated_analyzer = FixedCalibratedFinBERTAnalyzer(
        neutral_penalty=0.7,
        pos_boost=1.3,
        neg_boost=1.3,
        min_confidence_diff=0.05
    )
    calibrated_results = calibrated_analyzer.predict_calibrated(texts)
    calibrated_dist = get_distribution(calibrated_results)
    
    for label, pct in calibrated_dist.items():
        print(f"  {label}: {pct:.1f}%")
    
    # Calculate improvement
    original_neutral = original_dist.get('Neutral', 0)
    calibrated_neutral = calibrated_dist.get('Neutral', 0)
    improvement = original_neutral - calibrated_neutral
    
    print(f"\n🎯 Neutral Bias Change: {improvement:.1f} percentage points")
    
    if improvement > 0:
        print("✅ SUCCESS: Neutral bias reduced!")
    else:
        print("❌ ISSUE: Neutral bias increased or unchanged")
    
    # Show some example changes
    print(f"\n📝 Example Classifications:")
    for i, (orig, calib) in enumerate(zip(original_results[:5], calibrated_results[:5])):
        text_preview = orig['text'][:60] + "..." if len(orig['text']) > 60 else orig['text']
        print(f"\n{i+1}. {text_preview}")
        print(f"   Original: {orig['label']} ({orig['confidence']:.3f})")
        print(f"   Calibrated: {calib['label']} ({calib['confidence']:.3f})")
        if orig['label'] != calib['label']:
            print(f"   >>> CHANGED: {orig['label']} → {calib['label']}")
    
    # Show calibration stats
    stats = calibrated_analyzer.get_calibration_stats()
    if stats:
        print(f"\n📈 Calibration Statistics:")
        for stat, count in stats.items():
            print(f"  {stat}: {count}")
    
    return {
        "original_distribution": original_dist,
        "calibrated_distribution": calibrated_dist,
        "improvement": improvement
    }

def get_distribution(results):
    """Get percentage distribution of sentiment labels"""
    if not results:
        return {}
        
    counts = defaultdict(int)
    for result in results:
        counts[result["label"]] += 1
    
    total = len(results)
    return {label: count/total*100 for label, count in counts.items()}

def test_parameter_optimization():
    """Test the parameter optimization function"""
    
    print("\n🎯 Testing Parameter Optimization")
    print("=" * 40)
    
    # Simple test texts that are typically classified as neutral
    neutral_heavy_texts = [
        "The company reported quarterly results that met expectations.",
        "Stock price movements were minimal following the announcement.",
        "The CEO discussed various initiatives during the earnings call.",
        "Revenue performance was in line with the previous quarter.",
        "The company provided standard guidance for the fiscal year.",
        "Market conditions remained stable throughout the period.",
        "Management outlined operational priorities for efficiency.",
        "The board approved the regular quarterly dividend payment.",
        "Business segments showed varied performance this quarter.",
        "Industry trends continue to impact the company's position."
    ]
    
    try:
        optimization_results = find_optimal_calibration(neutral_heavy_texts)
        
        if optimization_results["best_params"]:
            print("✅ Optimization completed successfully!")
            return optimization_results
        else:
            print("⚠️  Optimization did not find better parameters")
            return None
            
    except Exception as e:
        print(f"❌ Optimization failed: {e}")
        return None

if __name__ == "__main__":
    print("🧪 Testing Fixed Calibrated Sentiment Analysis")
    print("=" * 50)
    
    # Quick comparison test
    comparison_results = quick_comparison_test()
    
    # Parameter optimization test
    optimization_results = test_parameter_optimization()
    
    print("\n✅ Testing completed!")
    
    # Summary
    if comparison_results["improvement"] > 0:
        print(f"\n🎉 SUCCESS: Calibration is working!")
        print(f"   Neutral bias reduced by {comparison_results['improvement']:.1f} percentage points")
    else:
        print(f"\n⚠️  Need further tuning: bias change = {comparison_results['improvement']:.1f}")