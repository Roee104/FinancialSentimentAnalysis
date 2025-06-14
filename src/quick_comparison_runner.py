# quick_comparison_runner.py

"""
Streamlined runner that focuses on what you need for the interim report.
Assumes you already have processed_articles.jsonl with good results.
"""

import os
import time
import subprocess
import json
from collections import defaultdict

def analyze_file(filepath):
    """Quick analysis of a results file"""
    if not os.path.exists(filepath):
        return None
    
    stats = {
        'total': 0,
        'sentiments': defaultdict(int),
        'has_tickers': 0
    }
    
    with open(filepath, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                stats['total'] += 1
                stats['sentiments'][data.get('overall_sentiment', 'Unknown')] += 1
                if data.get('tickers') or data.get('ticker_count', 0) > 0:
                    stats['has_tickers'] += 1
            except:
                continue
    
    return stats

def main():
    print("🚀 QUICK COMPARISON FOR INTERIM REPORT")
    print("="*60)
    
    # Check what we already have
    existing_files = {
        "Enhanced Pipeline": "data/processed_articles.jsonl",
        "VADER Baseline": "data/vader_baseline_results.jsonl",
    }
    
    print("📁 Checking existing results...")
    for name, filepath in existing_files.items():
        if os.path.exists(filepath):
            stats = analyze_file(filepath)
            if stats and stats['total'] > 0:
                print(f"\n✅ {name}: {stats['total']} articles")
                neutral_pct = stats['sentiments']['Neutral'] / stats['total'] * 100
                print(f"   Neutral: {neutral_pct:.1f}%")
                ticker_pct = stats['has_tickers'] / stats['total'] * 100
                print(f"   Has tickers: {ticker_pct:.1f}%")
            else:
                print(f"\n❌ {name}: No data")
        else:
            print(f"\n❌ {name}: File not found")
    
    # Run only what's missing
    print("\n📊 Running missing analyses...")
    
    # 1. VADER baseline (if missing)
    if not os.path.exists("data/vader_baseline_results.jsonl"):
        print("\nRunning VADER baseline...")
        subprocess.run([
            "python", "src/vader_baseline.py",
            "--threshold", "0.05",
            "--input", "data/financial_news_2020_2025_100k.parquet"
        ])
    
    # 2. One alternative threshold (for comparison)
    alt_file = "data/processed_articles_t15.jsonl"
    if not os.path.exists(alt_file):
        print("\nTesting alternative threshold (0.15)...")
        subprocess.run([
            "python", "src/pipeline_updated.py",
            "--threshold", "0.15",
            "--output", alt_file,
            "--input", "data/financial_news_2020_2025_100k.parquet"
        ])
    
    # 3. Generate comparison plots
    print("\n📊 Generating comparison plots...")
    
    # Create simple comparison data
    results_data = []
    
    # Analyze each file
    files_to_compare = [
        ("Enhanced Pipeline (t=0.1)", "data/processed_articles.jsonl"),
        ("VADER Baseline", "data/vader_baseline_results.jsonl"),
        ("Enhanced Pipeline (t=0.15)", alt_file),
    ]
    
    for name, filepath in files_to_compare:
        if os.path.exists(filepath):
            stats = analyze_file(filepath)
            if stats:
                results_data.append({
                    'name': name,
                    'total': stats['total'],
                    'neutral_pct': stats['sentiments']['Neutral'] / stats['total'] * 100,
                    'positive_pct': stats['sentiments']['Positive'] / stats['total'] * 100,
                    'negative_pct': stats['sentiments']['Negative'] / stats['total'] * 100,
                    'ticker_coverage': stats['has_tickers'] / stats['total'] * 100
                })
    
    # Generate simple comparison plot
    if len(results_data) >= 2:
        import matplotlib.pyplot as plt
        import numpy as np
        
        os.makedirs("data/plots", exist_ok=True)
        
        # Sentiment distribution comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        
        names = [r['name'] for r in results_data]
        x = np.arange(len(names))
        width = 0.25
        
        positives = [r['positive_pct'] for r in results_data]
        neutrals = [r['neutral_pct'] for r in results_data]
        negatives = [r['negative_pct'] for r in results_data]
        
        ax.bar(x - width, positives, width, label='Positive', color='green', alpha=0.8)
        ax.bar(x, neutrals, width, label='Neutral', color='gray', alpha=0.8)
        ax.bar(x + width, negatives, width, label='Negative', color='red', alpha=0.8)
        
        ax.set_ylabel('Percentage (%)')
        ax.set_title('Sentiment Distribution Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=15, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for i, (pos, neu, neg) in enumerate(zip(positives, neutrals, negatives)):
            ax.text(i - width, pos + 1, f'{pos:.1f}%', ha='center', va='bottom', fontsize=9)
            ax.text(i, neu + 1, f'{neu:.1f}%', ha='center', va='bottom', fontsize=9)
            ax.text(i + width, neg + 1, f'{neg:.1f}%', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig('data/plots/sentiment_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Saved sentiment comparison plot")
        
        # Neutral reduction chart
        fig, ax = plt.subplots(figsize=(8, 6))
        
        baseline_neutral = results_data[0]['neutral_pct']  # First one is baseline
        
        for i, r in enumerate(results_data):
            reduction = baseline_neutral - r['neutral_pct']
            color = 'green' if reduction > 0 else 'red'
            bar = ax.bar(i, r['neutral_pct'], color=color, alpha=0.7)
            
            # Add reduction annotation
            if i > 0 and reduction != 0:
                ax.annotate(f'{reduction:+.1f}pp',
                           xy=(i, r['neutral_pct']/2),
                           ha='center', va='center',
                           fontsize=12, fontweight='bold',
                           color='white')
        
        ax.set_ylabel('Neutral Sentiment (%)')
        ax.set_title('Neutral Bias Reduction Analysis')
        ax.set_xticks(range(len(results_data)))
        ax.set_xticklabels([r['name'] for r in results_data], rotation=15, ha='right')
        ax.axhline(y=40, color='red', linestyle='--', alpha=0.5, label='Target (40%)')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('data/plots/neutral_reduction.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Saved neutral reduction plot")
    
    # Final summary
    print("\n📋 SUMMARY FOR INTERIM REPORT")
    print("="*60)
    
    if len(results_data) >= 2:
        baseline = results_data[0]
        print(f"Baseline ({baseline['name']}):")
        print(f"  Neutral: {baseline['neutral_pct']:.1f}%")
        
        best_reduction = 0
        best_method = None
        
        for r in results_data[1:]:
            reduction = baseline['neutral_pct'] - r['neutral_pct']
            print(f"\n{r['name']}:")
            print(f"  Neutral: {r['neutral_pct']:.1f}% (reduction: {reduction:+.1f}pp)")
            
            if reduction > best_reduction:
                best_reduction = reduction
                best_method = r['name']
        
        if best_method:
            print(f"\n🏆 Best result: {best_method}")
            print(f"   Achieved {best_reduction:.1f} percentage points reduction in neutral bias!")
    
    print("\n✅ Ready for interim presentation!")
    print("\n📁 Use these files:")
    print("  - data/plots/sentiment_comparison.png")
    print("  - data/plots/neutral_reduction.png")

if __name__ == "__main__":
    main()