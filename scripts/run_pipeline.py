# scripts/run_pipeline.py
"""
Main entry point for running the financial sentiment analysis pipeline
"""

from utils.config_loader import load_config
from pipelines.main_pipeline import create_pipeline
from pipelines.baselines import VADERBaseline
from config.settings import PROCESSED_OUTPUT, INPUT_PARQUET, LOGGING_CONFIG
import argparse
import logging
import logging.config
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))


# Setup logging ONCE at the start
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Run Financial Sentiment Analysis Pipeline",
        allow_abbrev=False  # Disable abbreviation to avoid conflicts
    )

    # Config file
    parser.add_argument("--config", type=str,
                        help="Path to YAML configuration file")

    # Pipeline selection
    parser.add_argument(
        "--pipeline",
        type=str,
        default="optimized",
        choices=["standard", "optimized", "calibrated", "vader"],
        help="Pipeline type to run",
    )

    # Data arguments
    parser.add_argument("--input", type=str, help="Input parquet file")
    parser.add_argument("--output", type=str, help="Output JSONL file")

    # Processing arguments
    parser.add_argument("--batch-size", type=int,
                        help="Batch size for processing")
    parser.add_argument(
        "--max-articles",
        type=int,
        default=None,
        help="Maximum number of articles to process",
    )
    parser.add_argument(
        "--start-from", type=int, default=0, help="Starting article index"
    )
    parser.add_argument(
        "--no-resume", action="store_true", help="Don't resume from existing output"
    )

    # Sentiment analysis arguments
    parser.add_argument(
        "--sentiment-mode",
        type=str,
        choices=["standard", "optimized", "calibrated"],
        help="Override sentiment analysis mode",
    )
    parser.add_argument(
        "--sentiment-batch-size", type=int, help="Batch size for sentiment model"
    )

    # Aggregation arguments
    parser.add_argument(
        "--method",
        type=str,
        choices=["default", "majority", "conf_weighted"],
        help="Aggregation method",
    )
    parser.add_argument(
        "--agg-method",
        type=str,
        choices=["default", "majority", "conf_weighted"],
        help="Aggregation method (alias for --method)",
    )
    parser.add_argument(
        "--threshold", type=float, help="Threshold for sentiment classification"
    )
    parser.add_argument(
        "--no-distance-weighting",
        action="store_true",
        help="Disable distance-based weighting",
    )

    # VADER specific
    parser.add_argument(
        "--vader-threshold",
        type=float,
        default=0.05,
        help="VADER compound score threshold",
    )

    # Logging
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Override logging level",
    )

    args = parser.parse_args()

    # Override log level if requested
    if args.log_level:
        logging.getLogger().setLevel(getattr(logging, args.log_level))

    # Load configuration
    config_path = Path(args.config) if args.config else None
    config = load_config(config_path, parse_cli=False)

    # Apply command line overrides
    if args.input:
        config["input_parquet"] = args.input
    if args.output:
        config["processed_output"] = args.output
    if args.batch_size:
        config["pipeline_config"]["batch_size"] = args.batch_size
    if args.sentiment_batch_size:
        config["sentiment_config"]["batch_size"] = args.sentiment_batch_size

    logger.info("🚀 Starting Financial Sentiment Analysis Pipeline")
    logger.info(f"Pipeline type: {args.pipeline}")

    try:
        if args.pipeline == "vader":
            # Run VADER baseline
            pipeline = VADERBaseline(
                input_path=config.get("input_parquet", INPUT_PARQUET),
                output_path=config.get("processed_output", PROCESSED_OUTPUT),
                batch_size=config["pipeline_config"]["batch_size"],
                resume=not args.no_resume,
                vader_threshold=args.vader_threshold,
            )
        else:
            # Create main pipeline
            kwargs = {
                "input_path": config.get("input_parquet", INPUT_PARQUET),
                "output_path": config.get("processed_output", PROCESSED_OUTPUT),
                "batch_size": config["pipeline_config"]["batch_size"],
                "resume": not args.no_resume,
                "use_distance_weighting": not args.no_distance_weighting,
            }

            # Add optional overrides
            if args.sentiment_mode:
                kwargs["sentiment_mode"] = args.sentiment_mode
            if args.agg_method:
                kwargs["aggregation_method"] = args.agg_method
            if args.method:
                kwargs["aggregation_method"] = args.method
            if args.threshold is not None:
                kwargs["aggregation_threshold"] = args.threshold

            pipeline = create_pipeline(args.pipeline, **kwargs)

        # Run pipeline
        pipeline.run(
            max_articles=args.max_articles, start_from=args.start_from
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
