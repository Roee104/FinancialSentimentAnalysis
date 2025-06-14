# src/compare_results.py

"""
Compare results from different pipeline runs and baselines.
Generates statistics and visualizations for the interim report.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from typing import Dict, List
import os


def load_jsonl(filepath: str) -> List[Dict]:
    """Load JSONL file into list of dictionaries"""
    results = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            results.append(json.loads(line))
    return results


def analyze_results(results: List[Dict], name: str) -> Dict:
    """Analyze results and compute statistics"""

    stats = {
        'name': name,
        'total_articles': len(results),
        'sentiment_dist': defaultdict(int),
        'articles_with_tickers': 0,
        'articles_without_tickers': 0,
        'avg_tickers_per_article': 0,
        'sector_coverage': defaultdict(int),
        'unknown_sectors': 0
    }

    total_tickers = 0

    for article in results:
        # Sentiment distribution
        sentiment = article.get('overall_sentiment', 'Unknown')
        stats['sentiment_dist'][sentiment] += 1

        # Ticker analysis
        tickers = article.get('tickers', [])
        if tickers:
            stats['articles_with_tickers'] += 1
            total_tickers += len(tickers)
        else:
            stats['articles_without_tickers'] += 1

        # Sector analysis (if available)
        sectors = article.get('sectors', [])
        for sector_info in sectors:
            sector_name = sector_info.get('sector', 'Unknown')
            stats['sector_coverage'][sector_name] += 1
            if sector_name == 'Unknown':
                stats['unknown_sectors'] += 1

    # Calculate averages
    if stats['total_articles'] > 0:
        stats['avg_tickers_per_article'] = total_tickers / \
            stats['total_articles']

    # Convert to percentages
    total = stats['total_articles']
    stats['sentiment_pct'] = {
        sentiment: (count / total * 100) for sentiment, count in stats['sentiment_dist'].items()
    }
    stats['ticker_coverage_pct'] = (
        stats['articles_with_tickers'] / total * 100) if total > 0 else 0

    return stats


def create_comparison_plots(stats_list: List[Dict], output_dir: str = "data/plots"):
    """Create comparison plots for the interim report"""

    os.makedirs(output_dir, exist_ok=True)

    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')

    # 1. Sentiment Distribution Comparison
    fig, ax = plt.subplots(figsize=(10, 6))

    sentiments = ['Positive', 'Neutral', 'Negative']
    x_pos = range(len(stats_list))
    width = 0.25

    for i, sentiment in enumerate(sentiments):
        values = [stats['sentiment_pct'].get(
            sentiment, 0) for stats in stats_list]
        offset = (i - 1) * width
        bars = ax.bar([x + offset for x in x_pos],
                      values, width, label=sentiment)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Method', fontsize=12)
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_title('Sentiment Distribution Comparison',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([stats['name'] for stats in stats_list])
    ax.legend()
    ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/sentiment_distribution_comparison.png",
                dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Ticker Coverage Comparison
    fig, ax = plt.subplots(figsize=(8, 6))

    names = [stats['name'] for stats in stats_list]
    ticker_coverage = [stats['ticker_coverage_pct'] for stats in stats_list]

    bars = ax.bar(names, ticker_coverage, color=[
                  '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=11)

    ax.set_ylabel('Articles with Tickers (%)', fontsize=12)
    ax.set_title('Ticker Extraction Coverage', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/ticker_coverage_comparison.png",
                dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Neutral Bias Reduction Chart
    fig, ax = plt.subplots(figsize=(8, 6))

    neutral_percentages = [stats['sentiment_pct'].get(
        'Neutral', 0) for stats in stats_list]
    colors = ['red' if pct > 60 else 'orange' if pct >
              40 else 'green' for pct in neutral_percentages]

    bars = ax.bar(names, neutral_percentages, color=colors, alpha=0.7)

    # Add value labels and improvement indicators
    for i, (bar, pct) in enumerate(zip(bars, neutral_percentages)):
        ax.annotate(f'{pct:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

        # Show improvement from baseline
        if i > 0:
            improvement = neutral_percentages[0] - pct
            if improvement > 0:
                ax.annotate(f'↓ {improvement:.1f}pp',
                            xy=(bar.get_x() + bar.get_width() /
                                2, bar.get_height() / 2),
                            ha='center', va='center', fontsize=10,
                            color='darkgreen', fontweight='bold')

    ax.set_ylabel('Neutral Sentiment (%)', fontsize=12)
    ax.set_title('Neutral Bias Analysis', fontsize=14, fontweight='bold')
    ax.axhline(y=50, color='red', linestyle='--',
               alpha=0.5, label='50% threshold')
    ax.set_ylim(0, 100)
    ax.legend()

    plt.tight_layout()
    plt.savefig(f"{output_dir}/neutral_bias_reduction.png",
                dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Plots saved to {output_dir}/")


def generate_comparison_report(stats_list: List[Dict]) -> str:
    """Generate text report for comparison"""

    report = []
    report.append("=" * 60)
    report.append("PIPELINE COMPARISON REPORT")
    report.append("=" * 60)

    # Find baseline (first in list)
    baseline = stats_list[0]

    for i, stats in enumerate(stats_list):
        report.append(f"\n{i+1}. {stats['name']}")
        report.append("-" * 40)
        report.append(f"Total articles: {stats['total_articles']}")
        report.append(
            f"Articles with tickers: {stats['articles_with_tickers']} ({stats['ticker_coverage_pct']:.1f}%)")

        report.append("\nSentiment Distribution:")
        for sentiment in ['Positive', 'Neutral', 'Negative']:
            pct = stats['sentiment_pct'].get(sentiment, 0)
            report.append(f"  {sentiment}: {pct:.1f}%")

            # Show improvement from baseline
            if i > 0 and sentiment == 'Neutral':
                baseline_pct = baseline['sentiment_pct'].get(sentiment, 0)
                improvement = baseline_pct - pct
                if improvement > 0:
                    report.append(
                        f"    → Improvement: {improvement:.1f} percentage points reduction")

        report.append(
            f"\nAverage tickers per article: {stats['avg_tickers_per_article']:.2f}")

    # Summary
    report.append("\n" + "=" * 60)
    report.append("SUMMARY")
    report.append("=" * 60)

    # Best neutral bias reduction
    neutral_scores = [(stats['name'], stats['sentiment_pct'].get(
        'Neutral', 0)) for stats in stats_list]
    best_method = min(neutral_scores, key=lambda x: x[1])

    report.append(
        f"Best neutral bias reduction: {best_method[0]} ({best_method[1]:.1f}% neutral)")

    # Ticker coverage improvement
    baseline_ticker_coverage = baseline['ticker_coverage_pct']
    for stats in stats_list[1:]:
        improvement = stats['ticker_coverage_pct'] - baseline_ticker_coverage
        if improvement > 0:
            report.append(
                f"Ticker coverage improvement ({stats['name']}): +{improvement:.1f} percentage points")

    return "\n".join(report)


def main():
    """Run comparison analysis"""

    print("🔍 Running Pipeline Comparison Analysis")
    print("=" * 50)

    # Define files to compare
    result_files = [
        ("data/processed_articles.jsonl", "Original Pipeline"),
        ("data/vader_baseline_results.jsonl", "VADER Baseline"),
        ("data/processed_articles_enhanced.jsonl",
         "Enhanced Pipeline (conf_weighted)"),
        # Add more files as you generate them with different settings
    ]

    stats_list = []

    # Analyze each result file
    for filepath, name in result_files:
        if os.path.exists(filepath):
            print(f"\nAnalyzing {name}...")
            results = load_jsonl(filepath)
            stats = analyze_results(results, name)
            stats_list.append(stats)

            # Print basic stats
            print(f"  Total articles: {stats['total_articles']}")
            print(
                f"  Neutral percentage: {stats['sentiment_pct'].get('Neutral', 0):.1f}%")
            print(f"  Ticker coverage: {stats['ticker_coverage_pct']:.1f}%")
        else:
            print(f"⚠️  File not found: {filepath}")

    if len(stats_list) >= 2:
        # Generate comparison plots
        print("\n📊 Generating comparison plots...")
        create_comparison_plots(stats_list)

        # Generate text report
        report = generate_comparison_report(stats_list)

        # Save report
        with open("data/comparison_report.txt", 'w') as f:
            f.write(report)

        print("\n" + report)

        print("\n✅ Comparison analysis complete!")
        print("   - Plots saved to data/plots/")
        print("   - Report saved to data/comparison_report.txt")
    else:
        print("\n❌ Need at least 2 result files to compare!")


if __name__ == "__main__":
    main()
