# scripts/run_pipeline.py
"""
Main entry point for running the financial sentiment analysis pipeline
"""

from config.settings import PROCESSED_OUTPUT, INPUT_PARQUET
from pipelines.baselines import VADERBaseline
from pipelines.main_pipeline import create_pipeline
import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Run Financial Sentiment Analysis Pipeline"
    )

    # Pipeline selection
    parser.add_argument(
        "--pipeline",
        type=str,
        default="optimized",
        choices=["standard", "optimized", "calibrated", "vader"],
        help="Pipeline type to run"
    )

    # Data arguments
    parser.add_argument(
        "--input",
        type=str,
        default=str(INPUT_PARQUET),
        help="Input parquet file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(PROCESSED_OUTPUT),
        help="Output JSONL file"
    )

    # Processing arguments
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Batch size for processing"
    )
    parser.add_argument(
        "--max-articles",
        type=int,
        default=None,
        help="Maximum number of articles to process"
    )
    parser.add_argument(
        "--start-from",
        type=int,
        default=0,
        help="Starting article index"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Don't resume from existing output"
    )

    # Sentiment analysis arguments
    parser.add_argument(
        "--sentiment-mode",
        type=str,
        choices=["standard", "optimized", "calibrated"],
        help="Override sentiment analysis mode"
    )

    # Aggregation arguments
    parser.add_argument(
        "--method",
        type=str,
        choices=["default", "majority", "conf_weighted"],
        help="Aggregation method"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        help="Threshold for sentiment classification"
    )

    # VADER specific
    parser.add_argument(
        "--vader-threshold",
        type=float,
        default=0.05,
        help="VADER compound score threshold"
    )

    args = parser.parse_args()

    logger.info("🚀 Starting Financial Sentiment Analysis Pipeline")
    logger.info(f"Pipeline type: {args.pipeline}")

    try:
        if args.pipeline == "vader":
            # Run VADER baseline
            pipeline = VADERBaseline(
                input_path=args.input,
                output_path=args.output,
                batch_size=args.batch_size,
                resume=not args.no_resume,
                vader_threshold=args.vader_threshold
            )
        else:
            # Create main pipeline
            kwargs = {
                "input_path": args.input,
                "output_path": args.output,
                "batch_size": args.batch_size,
                "resume": not args.no_resume
            }

            # Add optional overrides
            if args.sentiment_mode:
                kwargs["sentiment_mode"] = args.sentiment_mode
            if args.method:
                kwargs["aggregation_method"] = args.method
            if args.threshold is not None:
                kwargs["aggregation_threshold"] = args.threshold

            pipeline = create_pipeline(args.pipeline, **kwargs)

        # Run pipeline
        pipeline.run(
            max_articles=args.max_articles,
            start_from=args.start_from
        )

        logger.info("✅ Pipeline completed successfully!")

    except KeyboardInterrupt:
        logger.info("⚠️ Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
