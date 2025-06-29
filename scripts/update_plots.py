"""Update plots after LoRA fine-tuning"""

from analysis.visualization import (
    create_final_summary_plot,
    create_sentiment_confidence_plot,
    create_comparison_plots,
    create_gold_standard_evaluation_plots
)
import sys
import json
from pathlib import Path

# Add project root to path
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))


def load_latest_results():
    """Load the most recent processed results for visualization"""
    data_dir = root / "data"

    # Find all processed files
    processed_files = list(data_dir.glob("processed_articles_*.jsonl"))

    results = []
    for file in processed_files:
        try:
            with open(file, 'r') as f:
                # Read first 1000 lines for visualization
                for i, line in enumerate(f):
                    if i >= 1000:
                        break
                    results.append(json.loads(line))
        except Exception as e:
            print(f"Error loading {file}: {e}")

    return results


def calculate_stats(results):
    """Calculate statistics for visualization"""
    total = len(results)
    sentiment_counts = {'Positive': 0, 'Neutral': 0, 'Negative': 0}
    ticker_count = 0
    total_tickers = 0

    for result in results:
        sentiment = result.get('overall_sentiment', 'Unknown')
        if sentiment in sentiment_counts:
            sentiment_counts[sentiment] += 1

        tickers = result.get('tickers', [])
        if tickers:
            ticker_count += 1
            total_tickers += len(tickers)

    sentiment_dist = {k: v for k, v in sentiment_counts.items()}
    sentiment_pct = {k: (v/total)*100 if total >
                     0 else 0 for k, v in sentiment_counts.items()}

    return {
        'total_articles': total,
        'sentiment_dist': sentiment_dist,
        'sentiment_pct': sentiment_pct,
        'ticker_coverage_pct': (ticker_count/total)*100 if total > 0 else 0,
        'avg_tickers': total_tickers/ticker_count if ticker_count > 0 else 0,
        # Assuming 80% baseline
        'neutral_reduction': 80 - sentiment_pct.get('Neutral', 0),
    }


def main():
    """Generate updated plots after LoRA fine-tuning"""

    print("📊 Generating updated visualizations...")

    # Create output directory
    plots_dir = root / "data" / "plots"
    plots_dir.mkdir(exist_ok=True, parents=True)

    # Load results
    print("\nLoading results...")
    results = load_latest_results()
    if not results:
        print("❌ No results found to visualize!")
        return

    print(f"✅ Loaded {len(results)} articles for visualization")

    # Calculate statistics
    stats = calculate_stats(results)

    # 1. Create final summary plot
    print("\n1. Creating final summary plot...")
    try:
        create_final_summary_plot(stats, plots_dir)
        print("✅ Final summary plot created")
    except Exception as e:
        print(f"❌ Error creating summary plot: {e}")

    # 2. Create confidence distribution plot
    print("\n2. Creating confidence distribution plot...")
    try:
        create_sentiment_confidence_plot(results, plots_dir)
        print("✅ Confidence distribution plot created")
    except Exception as e:
        print(f"❌ Error creating confidence plot: {e}")

    # 3. Create comparison plots if we have multiple pipeline results
    print("\n3. Creating comparison plots...")
    try:
        # Create stats list for different pipelines
        stats_list = []

        # Check for different pipeline outputs
        pipeline_files = {
            'Standard': root / "data" / "processed_articles.jsonl",
            'Optimized': root / "data" / "processed_articles_optimized.jsonl",
            'FinBERT LoRA': root / "data" / "processed_articles_finetuned_finbert.jsonl",
            'DeBERTa LoRA': root / "data" / "processed_articles_finetuned_deberta.jsonl",
        }

        for name, file in pipeline_files.items():
            if file.exists():
                # Load first 1000 articles
                pipeline_results = []
                with open(file, 'r') as f:
                    for i, line in enumerate(f):
                        if i >= 1000:
                            break
                        pipeline_results.append(json.loads(line))

                if pipeline_results:
                    pipeline_stats = calculate_stats(pipeline_results)
                    pipeline_stats['name'] = name
                    stats_list.append(pipeline_stats)

        if len(stats_list) > 1:
            create_comparison_plots(stats_list, plots_dir)
            print("✅ Comparison plots created")
        else:
            print("⚠️  Not enough pipeline results for comparison")

    except Exception as e:
        print(f"❌ Error creating comparison plots: {e}")

    # 4. Create evaluation plots if gold standard evaluation exists
    print("\n4. Checking for evaluation results...")
    eval_results_file = root / "experiments" / "evaluation_results.json"
    if eval_results_file.exists():
        try:
            with open(eval_results_file, 'r') as f:
                eval_results = json.load(f)
            create_gold_standard_evaluation_plots(eval_results, plots_dir)
            print("✅ Evaluation plots created")
        except Exception as e:
            print(f"⚠️  No evaluation results found or error: {e}")
    else:
        print("⚠️  No evaluation results file found")

    # List generated plots
    print("\n📁 Generated plots:")
    for plot_file in plots_dir.glob("*.png"):
        print(f"  - {plot_file.name}")

    print("\n✅ All visualizations updated!")


if __name__ == "__main__":
    main()
