# scripts/run_experiments.py
"""
Run all experiments for comparison and analysis
"""

from config.settings import DATA_DIR, OUTPUT_DIR
from analysis.comparison import run_comparison
from scripts.run_pipeline import main as run_pipeline
import sys
import logging
import time
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_experiment(name: str, pipeline_args: list) -> bool:
    """
    Run a single experiment

    Args:
        name: Experiment name
        pipeline_args: Arguments for pipeline

    Returns:
        Success status
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"🚀 {name}")
    logger.info(f"{'='*60}")

    start_time = time.time()

    try:
        # Modify sys.argv to pass arguments
        original_argv = sys.argv
        sys.argv = ['run_pipeline.py'] + pipeline_args

        # Run pipeline
        run_pipeline()

        # Restore original argv
        sys.argv = original_argv

        elapsed = time.time() - start_time
        logger.info(f"✅ Completed in {elapsed:.1f} seconds")
        return True

    except Exception as e:
        logger.error(f"❌ Failed: {e}")
        return False


def main():
    """Run all experiments"""

    logger.info("🔬 RUNNING ALL EXPERIMENTS")
    logger.info("="*60)
    logger.info(f"Start time: {datetime.now()}")

    # Check if data exists
    if not (DATA_DIR / "financial_news_2020_2025_100k.parquet").exists():
        logger.error("❌ Input data not found!")
        logger.error("Please run: python data/loader.py")
        return

    # Define experiments
    experiments = [
        # 1. Standard pipeline (baseline)
        {
            "name": "Standard Pipeline (Baseline)",
            "args": [
                "--pipeline", "standard",
                "--output", str(DATA_DIR / "processed_articles_standard.jsonl")
            ]
        },

        # 2. VADER baseline
        {
            "name": "VADER Baseline",
            "args": [
                "--pipeline", "vader",
                "--output", str(DATA_DIR / "vader_baseline_results.jsonl"),
                "--vader-threshold", "0.05"
            ]
        },

        # 3. Optimized pipeline (default)
        {
            "name": "Optimized Pipeline (Default Settings)",
            "args": [
                "--pipeline", "optimized",
                "--output", str(DATA_DIR /
                                "processed_articles_optimized.jsonl")
            ]
        },

        # 4. Optimized with different threshold
        {
            "name": "Optimized Pipeline (Threshold=0.15)",
            "args": [
                "--pipeline", "optimized",
                "--output", str(DATA_DIR /
                                "processed_articles_optimized_t15.jsonl"),
                "--threshold", "0.15"
            ]
        },

        # 5. Calibrated pipeline
        {
            "name": "Calibrated Pipeline (Advanced)",
            "args": [
                "--pipeline", "calibrated",
                "--output", str(DATA_DIR /
                                "processed_articles_calibrated.jsonl")
            ]
        },

        # 6. Majority voting method
        {
            "name": "Optimized Pipeline (Majority Method)",
            "args": [
                "--pipeline", "optimized",
                "--output", str(DATA_DIR /
                                "processed_articles_majority.jsonl"),
                "--method", "majority"
            ]
        }
    ]

    # Run experiments
    results = []
    total_start = time.time()

    for exp in experiments:
        success = run_experiment(exp["name"], exp["args"])
        results.append((exp["name"], success))

        if not success:
            logger.warning("⚠️ Continuing with next experiment...")

    # Run comparison analysis
    logger.info(f"\n{'='*60}")
    logger.info("📊 Running comparison analysis")
    logger.info(f"{'='*60}")

    # Prepare files for comparison
    comparison_files = [
        (DATA_DIR / "processed_articles_standard.jsonl", "Standard Pipeline"),
        (DATA_DIR / "vader_baseline_results.jsonl", "VADER Baseline"),
        (DATA_DIR / "processed_articles_optimized.jsonl", "Optimized Pipeline"),
        (DATA_DIR / "processed_articles_optimized_t15.jsonl", "Optimized (t=0.15)"),
        (DATA_DIR / "processed_articles_calibrated.jsonl", "Calibrated Pipeline"),
        (DATA_DIR / "processed_articles_majority.jsonl", "Optimized (Majority)"),
    ]

    # Filter to existing files
    existing_files = [(f, n) for f, n in comparison_files if f.exists()]

    if len(existing_files) >= 2:
        run_comparison(existing_files, output_dir=OUTPUT_DIR)
    else:
        logger.warning("Not enough result files for comparison")

    # Summary
    total_time = time.time() - total_start

    logger.info(f"\n{'='*60}")
    logger.info("📋 EXPERIMENT SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Total experiments: {len(experiments)}")

    successful = sum(1 for _, success in results if success)
    failed = len(results) - successful

    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Total time: {total_time/60:.1f} minutes")

    logger.info("\n📁 Output files:")
    for exp in experiments:
        output_file = None
        for arg_idx, arg in enumerate(exp["args"]):
            if arg == "--output" and arg_idx + 1 < len(exp["args"]):
                output_file = Path(exp["args"][arg_idx + 1])
                break

        if output_file and output_file.exists():
            logger.info(f"  ✅ {output_file.name}")
        else:
            logger.info(
                f"  ❌ {output_file.name if output_file else 'Unknown'}")

    logger.info(f"\n🎯 Next steps:")
    logger.info("1. Review comparison report in outputs/")
    logger.info("2. Check visualization plots in plots/")
    logger.info("3. Use results for presentation")

    logger.info("\n✨ All experiments completed!")


if __name__ == "__main__":
    main()
