# ────────────────────────────────────────────────────────────────────────────
#  scripts/run_pipeline.py
#  Main entry-point for running any Financial-Sentiment-Analysis pipeline
# ────────────────────────────────────────────────────────────────────────────
from __future__ import annotations
from config.settings import PROCESSED_OUTPUT, INPUT_PARQUET
from pipelines.baselines import VADERBaseline
from pipelines.main_pipeline import create_pipeline
from utils.config_loader import load_config
import sys
from pathlib import Path
import argparse

# -------------------------------------------------------  logging FIRST  ---
import logging
import logging.config
from config.settings import LOGGING_CONFIG
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)

# ----------------------------------------------------------  stdlib / 3rd-party

# ----------------------------------------------------------  internal imports

# ────────────────────────────────────────────────────────────────────────────


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run Financial Sentiment Analysis Pipeline",
        allow_abbrev=False,
    )

    # Config overrides
    p.add_argument("--config", type=str, help="YAML configuration file")

    # Pipeline selection
    p.add_argument(
        "--pipeline",
        default="optimized",
        choices=["standard", "optimized", "calibrated", "vader"],
        help="Pipeline type to run",
    )

    # Data I/O
    p.add_argument("--input", type=str, help="Input Parquet file")
    p.add_argument("--output", type=str, help="Output JSONL file")

    # Processing
    p.add_argument("--batch-size", type=int, help="Article batch size")
    p.add_argument("--max-articles", type=int,
                   help="Process at most N articles")
    p.add_argument("--start-from", type=int, default=0, help="Start index")
    p.add_argument("--no-resume", action="store_true", help="Do not resume")

    # Sentiment specifics
    p.add_argument("--sentiment-mode",
                   choices=["standard", "optimized", "calibrated"])
    p.add_argument("--sentiment-batch-size", type=int)

    # Aggregation
    p.add_argument("--method", "--agg-method",
                   dest="agg_method",
                   choices=["default", "majority", "conf_weighted"])
    p.add_argument("--threshold", type=float)
    p.add_argument("--no-distance-weighting", action="store_true")

    # VADER
    p.add_argument("--vader-threshold", type=float, default=0.05)

    # Logging
    p.add_argument(
        "--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    # Dynamic log level
    if args.log_level:
        logging.getLogger().setLevel(args.log_level)

    # ------------------------------------------------------------------ config
    config_path = Path(args.config) if args.config else None
    cfg = load_config(config_path, parse_cli=False)

    # CLI overrides
    if args.input:
        cfg["input_parquet"] = args.input
    if args.output:
        cfg["processed_output"] = args.output
    if args.batch_size:
        cfg["pipeline_config"]["batch_size"] = args.batch_size
    if args.sentiment_batch_size:
        cfg["sentiment_config"]["batch_size"] = args.sentiment_batch_size

    logger.info(
        "🚀 Starting Financial Sentiment Analysis Pipeline — %s", args.pipeline)

    try:
        if args.pipeline == "vader":
            pipeline = VADERBaseline(
                input_path=cfg.get("input_parquet", INPUT_PARQUET),
                output_path=cfg.get("processed_output", PROCESSED_OUTPUT),
                batch_size=cfg["pipeline_config"]["batch_size"],
                resume=not args.no_resume,
                vader_threshold=args.vader_threshold,
            )
        else:
            kwargs = dict(
                input_path=cfg.get("input_parquet", INPUT_PARQUET),
                output_path=cfg.get("processed_output", PROCESSED_OUTPUT),
                batch_size=cfg["pipeline_config"]["batch_size"],
                resume=not args.no_resume,
                use_distance_weighting=not args.no_distance_weighting,
            )
            if args.sentiment_mode:
                kwargs["sentiment_mode"] = args.sentiment_mode
            if args.agg_method:
                kwargs["aggregation_method"] = args.agg_method
            if args.threshold is not None:
                kwargs["aggregation_threshold"] = args.threshold

            pipeline = create_pipeline(args.pipeline, **kwargs)

        pipeline.run(max_articles=args.max_articles,
                     start_from=args.start_from)
        logger.info("✅ Pipeline completed successfully")

    except KeyboardInterrupt:
        logger.warning("⚠️ Pipeline interrupted by user")
        sys.exit(1)
    except Exception as ex:  # noqa: BLE001
        logger.exception("❌ Pipeline failed: %s", ex)
        sys.exit(1)


if __name__ == "__main__":  # pragma: no cover
    main()
