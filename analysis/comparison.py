# analysis/comparison.py
"""
Module for comparing results from different pipeline runs
"""

import json
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Optional
from pathlib import Path
import logging

from data.loader import DataLoader
from analysis.visualization import create_comparison_plots
from config.settings import DATA_DIR, OUTPUT_DIR

logger = logging.getLogger(__name__)


class ResultsComparator:
    """Handles comparison of results from different pipelines"""

    def __init__(self):
        """Initialize comparator"""
        self.results = {}
        logger.info("Initialized ResultsComparator")

    def load_results(self, name: str, filepath: Path) -> Optional[Dict]:
        """
        Load and analyze results from a file

        Args:
            name: Name for this result set
            filepath: Path to results file

        Returns:
            Analysis dict or None if error
        """
        if not filepath.exists():
            logger.warning(f"File not found: {filepath}")
            return None

        try:
            results = DataLoader.load_jsonl(filepath)
            analysis = self.analyze_results(results, name)
            self.results[name] = analysis
            return analysis
        except Exception as e:
            logger.error(f"Error loading {filepath}: {e}")
            return None

    def analyze_results(self, results: List[Dict], name: str) -> Dict:
        """
        Analyze results and compute statistics

        Args:
            results: List of result dictionaries
            name: Name for this result set

        Returns:
            Statistics dictionary
        """
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
            ticker_count = article.get('ticker_count', len(tickers))

            if ticker_count > 0:
                stats['articles_with_tickers'] += 1
                total_tickers += ticker_count
            else:
                stats['articles_without_tickers'] += 1

            # Sector analysis
            sectors = article.get('sectors', [])
            for sector_info in sectors:
                if isinstance(sector_info, dict):
                    sector_name = sector_info.get('sector', 'Unknown')
                else:
                    sector_name = str(sector_info)
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
            sentiment: (count / total * 100) if total > 0 else 0
            for sentiment, count in stats['sentiment_dist'].items()
        }
        stats['ticker_coverage_pct'] = (
            (stats['articles_with_tickers'] / total * 100)
            if total > 0 else 0
        )

        return stats

    def compare_all(self, result_files: List[tuple]) -> List[Dict]:
        """
        Compare multiple result files

        Args:
            result_files: List of (filepath, name) tuples

        Returns:
            List of analysis dictionaries
        """
        stats_list = []

        for filepath, name in result_files:
            filepath = Path(filepath)
            logger.info(f"\nAnalyzing {name}...")

            analysis = self.load_results(name, filepath)
            if analysis:
                stats_list.append(analysis)

                # Print basic stats
                logger.info(f"  Total articles: {analysis['total_articles']}")
                logger.info(
                    f"  Neutral percentage: {analysis['sentiment_pct'].get('Neutral', 0):.1f}%")
                logger.info(
                    f"  Ticker coverage: {analysis['ticker_coverage_pct']:.1f}%")

        return stats_list

    def generate_report(self, stats_list: List[Dict]) -> str:
        """
        Generate text comparison report

        Args:
            stats_list: List of analysis dictionaries

        Returns:
            Report string
        """
        report = []
        report.append("=" * 60)
        report.append("PIPELINE COMPARISON REPORT")
        report.append("=" * 60)

        # Find baseline (first in list)
        baseline = stats_list[0] if stats_list else None

        for i, stats in enumerate(stats_list):
            report.append(f"\n{i+1}. {stats['name']}")
            report.append("-" * 40)
            report.append(f"Total articles: {stats['total_articles']}")
            report.append(f"Articles with tickers: {stats['articles_with_tickers']} "
                          f"({stats['ticker_coverage_pct']:.1f}%)")

            report.append("\nSentiment Distribution:")
            for sentiment in ['Positive', 'Neutral', 'Negative']:
                pct = stats['sentiment_pct'].get(sentiment, 0)
                report.append(f"  {sentiment}: {pct:.1f}%")

                # Show improvement from baseline
                if i > 0 and baseline and sentiment == 'Neutral':
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

        if len(stats_list) > 1:
            # Best neutral bias reduction
            neutral_scores = [(stats['name'], stats['sentiment_pct'].get('Neutral', 0))
                              for stats in stats_list]
            best_method = min(neutral_scores, key=lambda x: x[1])

            report.append(
                f"Best neutral bias reduction: {best_method[0]} ({best_method[1]:.1f}% neutral)")

            # Ticker coverage improvement
            if baseline:
                baseline_ticker_coverage = baseline['ticker_coverage_pct']
                for stats in stats_list[1:]:
                    improvement = stats['ticker_coverage_pct'] - \
                        baseline_ticker_coverage
                    if improvement > 0:
                        report.append(f"Ticker coverage improvement ({stats['name']}): "
                                      f"+{improvement:.1f} percentage points")

        return "\n".join(report)

    def analyze_recent_results(self, filepath: Path, last_n: int = 5000) -> Optional[Dict]:
        """
        Analyze the most recent N results from a file

        Args:
            filepath: Path to results file
            last_n: Number of recent results to analyze

        Returns:
            Analysis dict or None
        """
        if not filepath.exists():
            return None

        sentiments = defaultdict(int)
        ticker_counts = []
        sector_counts = []
        total = 0

        # Read file and get last N entries
        with open(filepath, 'r') as f:
            lines = f.readlines()

        # Analyze last N lines
        for line in lines[-last_n:]:
            try:
                data = json.loads(line)
                total += 1

                # Sentiment
                sentiment = data.get('overall_sentiment', 'Unknown')
                sentiments[sentiment] += 1

                # Tickers
                tickers = data.get('tickers', [])
                ticker_count = data.get('ticker_count', len(tickers))
                ticker_counts.append(ticker_count)

                # Sectors
                sectors = data.get('sectors', [])
                sector_counts.append(len(sectors))

            except:
                continue

        if total == 0:
            return None

        return {
            'total': total,
            'sentiments': dict(sentiments),
            'avg_tickers': sum(ticker_counts) / len(ticker_counts) if ticker_counts else 0,
            'articles_with_tickers': sum(1 for c in ticker_counts if c > 0),
            'avg_sectors': sum(sector_counts) / len(sector_counts) if sector_counts else 0
        }


def run_comparison(result_files: List[tuple], output_dir: Path = None) -> Dict:
    """
    Run full comparison analysis

    Args:
        result_files: List of (filepath, name) tuples
        output_dir: Directory for output files

    Returns:
        Comparison results dictionary
    """
    output_dir = output_dir or OUTPUT_DIR

    logger.info("🔍 Running Pipeline Comparison Analysis")
    logger.info("=" * 50)

    comparator = ResultsComparator()

    # Analyze all files
    stats_list = comparator.compare_all(result_files)

    if len(stats_list) >= 2:
        # Generate plots
        logger.info("\n📊 Generating comparison plots...")
        create_comparison_plots(stats_list, output_dir=output_dir)

        # Generate report
        report = comparator.generate_report(stats_list)

        # Save report
        report_path = output_dir / "comparison_report.txt"
        with open(report_path, 'w') as f:
            f.write(report)

        logger.info("\n" + report)
        logger.info(f"\n✅ Comparison analysis complete!")
        logger.info(f"   - Plots saved to {output_dir}")
        logger.info(f"   - Report saved to {report_path}")
    else:
        logger.warning("\n❌ Need at least 2 result files to compare!")

    return {
        'stats_list': stats_list,
        'comparator': comparator
    }


# Main function for backward compatibility
def main():
    """Run comparison with default files"""
    result_files = [
        (DATA_DIR / "processed_articles.jsonl", "Original Pipeline"),
        (DATA_DIR / "vader_baseline_results.jsonl", "VADER Baseline"),
        (DATA_DIR / "processed_articles_enhanced.jsonl", "Enhanced Pipeline"),
    ]

    run_comparison(result_files)


if __name__ == "__main__":
    main()
