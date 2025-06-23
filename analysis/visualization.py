# analysis/visualization.py
"""
Unified visualization module for creating plots and charts
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from pathlib import Path
import logging
import json

from config.settings import PLOT_CONFIG, PLOTS_DIR

logger = logging.getLogger(__name__)


def setup_plot_style():
    """Set up matplotlib style"""
    plt.style.use(PLOT_CONFIG["style"])
    plt.rcParams['figure.dpi'] = PLOT_CONFIG["dpi"]


def create_comparison_plots(stats_list: List[Dict], output_dir: Path = None):
    """
    Create comparison plots for analysis results

    Args:
        stats_list: List of statistics dictionaries
        output_dir: Output directory for plots
    """
    output_dir = output_dir or PLOTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_plot_style()

    # 1. Sentiment Distribution Comparison
    create_sentiment_distribution_plot(stats_list, output_dir)

    # 2. Ticker Coverage Comparison
    create_ticker_coverage_plot(stats_list, output_dir)

    # 3. Neutral Bias Reduction Chart
    create_neutral_bias_plot(stats_list, output_dir)

    logger.info(f"✅ Plots saved to {output_dir}")


def create_sentiment_distribution_plot(stats_list: List[Dict], output_dir: Path):
    """Create sentiment distribution comparison plot"""
    fig, ax = plt.subplots(figsize=PLOT_CONFIG["figsize"])

    sentiments = ['Positive', 'Neutral', 'Negative']
    x_pos = range(len(stats_list))
    width = 0.25

    for i, sentiment in enumerate(sentiments):
        values = [stats['sentiment_pct'].get(
            sentiment, 0) for stats in stats_list]
        offset = (i - 1) * width
        color = PLOT_CONFIG["colors"][sentiment.lower()]
        bars = ax.bar([x + offset for x in x_pos], values, width,
                      label=sentiment, color=color, alpha=0.8)

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
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
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "sentiment_distribution_comparison.png",
                dpi=PLOT_CONFIG["dpi"], bbox_inches='tight')
    plt.close()


def create_ticker_coverage_plot(stats_list: List[Dict], output_dir: Path):
    """Create ticker coverage comparison plot"""
    fig, ax = plt.subplots(figsize=(8, 6))

    names = [stats['name'] for stats in stats_list]
    ticker_coverage = [stats['ticker_coverage_pct'] for stats in stats_list]

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c',
              '#d62728', '#9467bd'][:len(names)]
    bars = ax.bar(names, ticker_coverage, color=colors, alpha=0.7)

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
    ax.grid(axis='y', alpha=0.3)

    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / "ticker_coverage_comparison.png",
                dpi=PLOT_CONFIG["dpi"], bbox_inches='tight')
    plt.close()


def create_neutral_bias_plot(stats_list: List[Dict], output_dir: Path):
    """Create neutral bias reduction plot"""
    fig, ax = plt.subplots(figsize=(8, 6))

    names = [stats['name'] for stats in stats_list]
    neutral_percentages = [stats['sentiment_pct'].get(
        'Neutral', 0) for stats in stats_list]

    # Color based on neutral percentage
    colors = ['red' if pct > 60 else 'orange' if pct > 40 else 'green'
              for pct in neutral_percentages]

    bars = ax.bar(names, neutral_percentages, color=colors, alpha=0.7)

    # Add value labels and improvement indicators
    baseline_neutral = neutral_percentages[0] if neutral_percentages else 0

    for i, (bar, pct) in enumerate(zip(bars, neutral_percentages)):
        # Value label
        ax.annotate(f'{pct:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

        # Improvement indicator
        if i > 0:
            improvement = baseline_neutral - pct
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
    ax.axhline(y=33.3, color='green', linestyle='--',
               alpha=0.5, label='Balanced (33.3%)')
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / "neutral_bias_reduction.png",
                dpi=PLOT_CONFIG["dpi"], bbox_inches='tight')
    plt.close()


def create_gold_standard_evaluation_plots(results: Dict[str, Dict], output_dir: Path = None):
    """Create evaluation plots for gold standard comparison"""
    output_dir = output_dir or PLOTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_plot_style()

    # 1. Accuracy Comparison Bar Chart
    fig, ax = plt.subplots(figsize=(10, 6))

    pipelines = list(results.keys())
    accuracies = [results[p]['accuracy'] * 100 for p in pipelines]
    f1_scores = [results[p]['macro_f1'] * 100 for p in pipelines]

    x = np.arange(len(pipelines))
    width = 0.35

    bars1 = ax.bar(x - width/2, accuracies, width, label='Accuracy', alpha=0.8)
    bars2 = ax.bar(x + width/2, f1_scores, width, label='Macro F1', alpha=0.8)

    ax.set_xlabel('Pipeline', fontsize=12)
    ax.set_ylabel('Score (%)', fontsize=12)
    ax.set_title('Pipeline Performance Comparison (Gold Standard Evaluation)',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(pipelines)
    ax.legend()
    ax.set_ylim(0, 100)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(output_dir / 'gold_standard_accuracy_comparison.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Confusion Matrices
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()

    for idx, (name, metrics) in enumerate(results.items()):
        if idx < 4:  # Maximum 4 pipelines
            cm = np.array(metrics['confusion_matrix'])

            # Normalize confusion matrix
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

            sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                        xticklabels=['Positive', 'Neutral', 'Negative'],
                        yticklabels=['Positive', 'Neutral', 'Negative'],
                        ax=axes[idx])
            axes[idx].set_title(f'{name} Pipeline')
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('Actual')

    plt.suptitle('Confusion Matrices (Normalized)',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrices.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    logger.info("✅ Evaluation plots created")


def create_performance_comparison_plot(metrics_dict: Dict[str, Dict], output_dir: Path):
    """Create performance metrics comparison plot"""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Sentiment accuracy plot
    methods = list(metrics_dict.keys())
    accuracies = [metrics_dict[m].get('accuracy', 0) * 100 for m in methods]

    bars1 = ax1.bar(methods, accuracies, color='skyblue', alpha=0.7)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('Sentiment Classification Accuracy', fontsize=14)
    ax1.set_ylim(0, 100)

    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.1f}%', ha='center', va='bottom')

    # F1 scores plot
    sentiments = ['Positive', 'Neutral', 'Negative']
    x = np.arange(len(methods))
    width = 0.25

    for i, sentiment in enumerate(sentiments):
        f1_scores = [
            metrics_dict[m]['per_class'][sentiment]['f1'] * 100
            for m in methods
        ]
        ax2.bar(x + (i-1)*width, f1_scores, width, label=sentiment)

    ax2.set_ylabel('F1 Score (%)', fontsize=12)
    ax2.set_xlabel('Method', fontsize=12)
    ax2.set_title('F1 Scores by Sentiment Class', fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods)
    ax2.legend()
    ax2.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(output_dir / "performance_comparison.png",
                dpi=PLOT_CONFIG["dpi"], bbox_inches='tight')
    plt.close()


def create_article_length_distribution(df: pd.DataFrame, output_dir: Path):
    """Create article length distribution plot"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Word count distribution
    word_counts = df['content'].str.split().str.len()
    ax1.hist(word_counts, bins=50, color='steelblue',
             alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Number of Words', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Article Word Count Distribution', fontsize=14)
    ax1.axvline(word_counts.mean(), color='red', linestyle='--',
                label=f'Mean: {word_counts.mean():.0f}')
    ax1.legend()

    # Token count distribution
    if 'token_count' in df.columns:
        token_counts = df['token_count']
        ax2.hist(token_counts, bins=50, color='forestgreen',
                 alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Number of Tokens', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title('Article Token Count Distribution', fontsize=14)
        ax2.axvline(token_counts.mean(), color='red', linestyle='--',
                    label=f'Mean: {token_counts.mean():.0f}')
        ax2.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "article_length_distribution.png",
                dpi=PLOT_CONFIG["dpi"], bbox_inches='tight')
    plt.close()


def create_sentiment_confidence_plot(results: List[Dict], output_dir: Path):
    """Create sentiment confidence distribution plot"""
    fig, ax = plt.subplots(figsize=PLOT_CONFIG["figsize"])

    # Extract confidence scores by sentiment
    confidence_by_sentiment = {
        'Positive': [],
        'Neutral': [],
        'Negative': []
    }

    for result in results:
        sentiment = result.get('overall_sentiment', 'Unknown')
        confidence = result.get('overall_confidence', 0)
        if sentiment in confidence_by_sentiment:
            confidence_by_sentiment[sentiment].append(confidence)

    # Create violin plot
    data = []
    labels = []
    for sentiment, confidences in confidence_by_sentiment.items():
        if confidences:
            data.append(confidences)
            labels.append(sentiment)

    parts = ax.violinplot(data, positions=range(len(labels)),
                          showmeans=True, showmedians=True)

    # Customize colors
    colors = [PLOT_CONFIG["colors"][label.lower()] for label in labels]
    for pc, color in zip(parts['bodies'], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel('Confidence Score', fontsize=12)
    ax.set_title('Confidence Distribution by Sentiment',
                 fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "sentiment_confidence_distribution.png",
                dpi=PLOT_CONFIG["dpi"], bbox_inches='tight')
    plt.close()


def create_final_summary_plot(stats_dict: Dict, output_dir: Path):
    """Create final summary visualization"""
    fig = plt.figure(figsize=(15, 10))

    # Create grid
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. Sentiment pie chart (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    sentiments = ['Positive', 'Neutral', 'Negative']
    sizes = [stats_dict['sentiment_dist'].get(s, 0) for s in sentiments]
    colors = [PLOT_CONFIG["colors"][s.lower()] for s in sentiments]
    ax1.pie(sizes, labels=sentiments, colors=colors,
            autopct='%1.1f%%', startangle=90)
    ax1.set_title('Overall Sentiment Distribution')

    # 2. Neutral reduction bar (top middle)
    ax2 = fig.add_subplot(gs[0, 1])
    reduction = stats_dict.get('neutral_reduction', 0)
    ax2.bar(['Before', 'After'], [80, 80-reduction],
            color=['red', 'green'], alpha=0.7)
    ax2.set_ylabel('Neutral %')
    ax2.set_title(f'Neutral Bias Reduction\n({reduction:.1f}pp improvement)')
    ax2.set_ylim(0, 100)

    # 3. Ticker coverage (top right)
    ax3 = fig.add_subplot(gs[0, 2])
    coverage = stats_dict.get('ticker_coverage_pct', 0)
    ax3.bar(['Coverage'], [coverage], color='blue', alpha=0.7)
    ax3.set_ylabel('Percentage (%)')
    ax3.set_title(f'Ticker Coverage\n({coverage:.1f}% of articles)')
    ax3.set_ylim(0, 100)

    # 4. Key metrics table (bottom)
    ax4 = fig.add_subplot(gs[1:, :])
    ax4.axis('tight')
    ax4.axis('off')

    # Create metrics table
    metrics_data = [
        ['Metric', 'Value', 'Target', 'Status'],
        ['Total Articles',
            f"{stats_dict.get('total_articles', 0):,}", '100,000+', '✓'],
        ['Neutral Bias', f"{80-reduction:.1f}%", '<40%',
            '✓' if (80-reduction) < 40 else '✗'],
        ['Ticker Coverage', f"{coverage:.1f}%",
            '>80%', '✓' if coverage > 80 else '✗'],
        ['Avg Tickers/Article', f"{stats_dict.get('avg_tickers', 0):.2f}", '>2.5',
         '✓' if stats_dict.get('avg_tickers', 0) > 2.5 else '✗'],
        ['Processing Time',
            f"{stats_dict.get('processing_time', 'N/A')}", '<2 hours', '-']
    ]

    table = ax4.table(cellText=metrics_data, cellLoc='center', loc='center',
                      colWidths=[0.3, 0.3, 0.2, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)

    # Style header row
    for i in range(4):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    fig.suptitle('Financial Sentiment Analysis Pipeline - Final Results',
                 fontsize=16, fontweight='bold')

    plt.savefig(output_dir / "final_summary.png",
                dpi=PLOT_CONFIG["dpi"], bbox_inches='tight')
    plt.close()

    logger.info("✅ Created final summary visualization")
