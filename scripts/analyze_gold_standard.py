# scripts/analyze_gold_standard.py
"""
Analyze and validate gold standard annotations
"""

import json
import pandas as pd
from pathlib import Path
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class GoldStandardAnalyzer:
    def __init__(self, gold_standard_file: Path):
        """Load and analyze gold standard data"""
        self.annotations = []
        with open(gold_standard_file, 'r') as f:
            for line in f:
                try:
                    self.annotations.append(json.loads(line))
                except:
                    continue
                    
        print(f"Loaded {len(self.annotations)} annotations")
        
    def analyze_distribution(self):
        """Analyze sentiment distribution"""
        # Overall sentiment distribution
        overall_sentiments = [a['overall_sentiment'] for a in self.annotations]
        overall_dist = Counter(overall_sentiments)
        
        print("\n=== Overall Sentiment Distribution ===")
        for sentiment, count in overall_dist.items():
            print(f"{sentiment}: {count} ({count/len(self.annotations)*100:.1f}%)")
            
        # Ticker-level sentiment distribution
        ticker_sentiments = []
        for ann in self.annotations:
            for ticker, info in ann.get('ticker_sentiments', {}).items():
                ticker_sentiments.append(info['sentiment'])
                
        ticker_dist = Counter(ticker_sentiments)
        
        print("\n=== Ticker-level Sentiment Distribution ===")
        for sentiment, count in ticker_dist.items():
            print(f"{sentiment}: {count} ({count/len(ticker_sentiments)*100:.1f}%)")
            
        # Articles by ticker count
        ticker_counts = [len(a.get('ticker_sentiments', {})) for a in self.annotations]
        
        print("\n=== Articles by Ticker Count ===")
        for i in range(6):
            count = ticker_counts.count(i)
            if count > 0:
                print(f"{i} tickers: {count} articles")
                
        # Mixed sentiment articles
        mixed_sentiment = 0
        for ann in self.annotations:
            sentiments = set()
            for ticker_info in ann.get('ticker_sentiments', {}).values():
                sentiments.add(ticker_info['sentiment'])
            if len(sentiments) > 1:
                mixed_sentiment += 1
                
        print(f"\n=== Mixed Sentiment Articles ===")
        print(f"Articles with mixed ticker sentiments: {mixed_sentiment} ({mixed_sentiment/len(self.annotations)*100:.1f}%)")
        
    def analyze_confidence(self):
        """Analyze confidence scores"""
        overall_confidences = [a.get('overall_confidence', 0) for a in self.annotations]
        
        print("\n=== Confidence Analysis ===")
        print(f"Overall confidence - Mean: {np.mean(overall_confidences):.3f}, Std: {np.std(overall_confidences):.3f}")
        
        # Low confidence articles
        low_conf = [a for a in self.annotations if a.get('overall_confidence', 0) < 0.6]
        print(f"Low confidence articles (<0.6): {len(low_conf)}")
        
        # Ticker-level confidence
        ticker_confidences = []
        for ann in self.annotations:
            for ticker_info in ann.get('ticker_sentiments', {}).values():
                ticker_confidences.append(ticker_info.get('confidence', 0))
                
        if ticker_confidences:
            print(f"Ticker confidence - Mean: {np.mean(ticker_confidences):.3f}, Std: {np.std(ticker_confidences):.3f}")
            
    def find_interesting_cases(self):
        """Find interesting cases for manual review"""
        print("\n=== Interesting Cases for Review ===")
        
        # 1. Mixed sentiment articles
        mixed = []
        for ann in self.annotations:
            sentiments = set()
            for ticker_info in ann.get('ticker_sentiments', {}).values():
                sentiments.add(ticker_info['sentiment'])
            if len(sentiments) > 1:
                mixed.append(ann)
                
        print(f"\n1. Mixed sentiment articles: {len(mixed)}")
        if mixed:
            example = mixed[0]
            print(f"   Example: {example['title'][:80]}...")
            for ticker, info in example['ticker_sentiments'].items():
                print(f"   - {ticker}: {info['sentiment']}")
                
        # 2. Low confidence
        low_conf = sorted(self.annotations, key=lambda x: x.get('overall_confidence', 1))[:5]
        print(f"\n2. Lowest confidence articles:")
        for ann in low_conf:
            print(f"   - Confidence {ann['overall_confidence']:.2f}: {ann['title'][:60]}...")
            
        # 3. High ticker count
        high_ticker = sorted(self.annotations, 
                           key=lambda x: len(x.get('ticker_sentiments', {})), 
                           reverse=True)[:5]
        print(f"\n3. Most tickers in single article:")
        for ann in high_ticker:
            n_tickers = len(ann['ticker_sentiments'])
            print(f"   - {n_tickers} tickers: {ann['title'][:60]}...")
            
    def validate_consistency(self):
        """Check annotation consistency"""
        print("\n=== Validation Checks ===")
        
        issues = defaultdict(int)
        
        for ann in self.annotations:
            # Check if overall sentiment matches ticker sentiments
            ticker_sentiments = [info['sentiment'] for info in ann.get('ticker_sentiments', {}).values()]
            if ticker_sentiments:
                sentiment_counts = Counter(ticker_sentiments)
                dominant = sentiment_counts.most_common(1)[0][0]
                
                # If all tickers have same sentiment, overall should match
                if len(set(ticker_sentiments)) == 1 and ann['overall_sentiment'] != ticker_sentiments[0]:
                    issues['overall_mismatch'] += 1
                    
            # Check for missing rationales
            if not ann.get('overall_rationale'):
                issues['missing_rationale'] += 1
                
            # Check confidence ranges
            if not 0 <= ann.get('overall_confidence', 0.5) <= 1:
                issues['invalid_confidence'] += 1
                
        print("Issues found:")
        for issue, count in issues.items():
            print(f"  - {issue}: {count}")
            
        if not issues:
            print("  No issues found! ✅")
            
    def create_visualizations(self, output_dir: Path):
        """Create visualization plots"""
        output_dir.mkdir(exist_ok=True)
        
        # 1. Sentiment distribution comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Overall sentiment
        overall = pd.Series([a['overall_sentiment'] for a in self.annotations])
        overall.value_counts().plot(kind='pie', ax=ax1, autopct='%1.1f%%',
                                   colors=['#2ecc71', '#95a5a6', '#e74c3c'])
        ax1.set_title('Overall Sentiment Distribution')
        ax1.set_ylabel('')
        
        # Ticker sentiment
        ticker_sents = []
        for ann in self.annotations:
            for info in ann.get('ticker_sentiments', {}).values():
                ticker_sents.append(info['sentiment'])
                
        if ticker_sents:
            pd.Series(ticker_sents).value_counts().plot(
                kind='pie', ax=ax2, autopct='%1.1f%%',
                colors=['#2ecc71', '#95a5a6', '#e74c3c']
            )
            ax2.set_title('Ticker-level Sentiment Distribution')
            ax2.set_ylabel('')
            
        plt.tight_layout()
        plt.savefig(output_dir / 'gold_standard_sentiment_dist.png', dpi=300)
        plt.close()
        
        # 2. Confidence distribution
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        overall_conf = [a.get('overall_confidence', 0) for a in self.annotations]
        ax1.hist(overall_conf, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(np.mean(overall_conf), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(overall_conf):.2f}')
        ax1.set_xlabel('Confidence Score')
        ax1.set_ylabel('Count')
        ax1.set_title('Overall Sentiment Confidence Distribution')
        ax1.legend()
        
        # Ticker confidence
        ticker_conf = []
        for ann in self.annotations:
            for info in ann.get('ticker_sentiments', {}).values():
                ticker_conf.append(info.get('confidence', 0))
                
        if ticker_conf:
            ax2.hist(ticker_conf, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
            ax2.axvline(np.mean(ticker_conf), color='red', linestyle='--',
                       label=f'Mean: {np.mean(ticker_conf):.2f}')
            ax2.set_xlabel('Confidence Score')
            ax2.set_ylabel('Count')
            ax2.set_title('Ticker Sentiment Confidence Distribution')
            ax2.legend()
            
        plt.tight_layout()
        plt.savefig(output_dir / 'gold_standard_confidence_dist.png', dpi=300)
        plt.close()
        
        print(f"\n✅ Visualizations saved to {output_dir}")
        
    def export_for_evaluation(self, output_file: Path):
        """Export in format expected by evaluation code"""
        # The evaluation code expects: article_hash/title and true_overall
        with open(output_file, 'w') as f:
            for ann in self.annotations:
                # Minimal format for current evaluation
                eval_format = {
                    'article_hash': ann.get('article_hash'),
                    'title': ann.get('title'),
                    'true_overall': ann.get('overall_sentiment')
                }
                f.write(json.dumps(eval_format) + '\n')
                
        print(f"\n✅ Exported evaluation format to {output_file}")
        
    def generate_report(self):
        """Generate comprehensive report"""
        report = []
        report.append("="*60)
        report.append("GOLD STANDARD DATASET REPORT")
        report.append("="*60)
        report.append(f"\nTotal annotations: {len(self.annotations)}")
        
        # Add all analyses
        self.analyze_distribution()
        self.analyze_confidence() 
        self.validate_consistency()
        self.find_interesting_cases()
        
        return "\n".join(report)


def main():
    """Run gold standard analysis"""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='data/gold_standard_annotations.jsonl')
    parser.add_argument('--output-dir', type=str, default='data/plots')
    parser.add_argument('--export', type=str, default='data/gold_standard_eval.jsonl')
    
    args = parser.parse_args()
    
    if not Path(args.input).exists():
        print(f"Gold standard file not found: {args.input}")
        return
        
    analyzer = GoldStandardAnalyzer(Path(args.input))
    
    # Generate report
    analyzer.generate_report()
    
    # Create visualizations
    analyzer.create_visualizations(Path(args.output_dir))
    
    # Export for evaluation
    analyzer.export_for_evaluation(Path(args.export))


if __name__ == "__main__":
    main()