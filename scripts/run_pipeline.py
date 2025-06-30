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
try:
    import core.ner
    if not hasattr(core.ner, 'UnifiedNER') and hasattr(core.ner, 'ImprovedNER'):
        core.ner.UnifiedNER = core.ner.ImprovedNER

    # Use pretrained NER
    from core.pretrained_financial_ner import FinancialNERWrapper
    core.ner.UnifiedNER = FinancialNERWrapper
    logger.info("Using pre-trained Financial NER")
except Exception as e:
    logger.warning(f"NER setup issue: {e}")
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

    # PEFT adapter support
    p.add_argument("--adapter", type=str,
                   help="Path to PEFT adapter directory")

    # Aggregation - updated to support new methods
    p.add_argument("--aggregation",
                   choices=["default", "majority", "conf_weighted",
                            "length_weighted", "distance_weighted", "confidence_weighted"],
                   help="Aggregation method")

    # Legacy support for --method/--agg-method
    p.add_argument("--method", "--agg-method",
                   dest="agg_method",
                   choices=["default", "majority", "conf_weighted"],
                   help="(Deprecated) Use --aggregation instead")

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
    if args.max_articles:
        cfg["pipeline_config"]["max_articles"] = args.max_articles
    if args.start_from:
        cfg["pipeline_config"]["start_from"] = args.start_from
    if args.no_resume:
        cfg["pipeline_config"]["resume"] = False
    if args.sentiment_mode:
        cfg["sentiment_config"]["mode"] = args.sentiment_mode
    if args.threshold:
        cfg["aggregation_config"]["aggregation_threshold"] = args.threshold

    # Handle aggregation method
    aggregation_method = args.aggregation or args.agg_method
    if aggregation_method:
        # Map new method names to existing ones if needed
        if aggregation_method in ["length_weighted", "confidence_weighted"]:
            cfg["aggregation_config"]["aggregation_method"] = "conf_weighted"
        elif aggregation_method == "distance_weighted":
            cfg["aggregation_config"]["aggregation_method"] = "conf_weighted"
            cfg["aggregation_config"]["use_distance_weighting"] = True
        else:
            cfg["aggregation_config"]["aggregation_method"] = aggregation_method

    if args.no_distance_weighting:
        cfg["aggregation_config"]["use_distance_weighting"] = False

    # Adapter support
    if args.adapter:
        cfg["sentiment_config"]["adapter_path"] = args.adapter

    # ---------------------------------------------------------- build pipeline
    if args.pipeline == "vader":
        logger.info("🏃 Running VADER baseline")
        pipeline = VADERBaseline(threshold=args.vader_threshold, **cfg)
    else:
        logger.info(f"🏃 Running {args.pipeline} pipeline")
        pipeline = create_pipeline(
            # mode=args.pipeline,
            pipeline_type=args.pipeline,
            input_parquet=cfg.get("input_parquet", INPUT_PARQUET),
            output_jsonl=cfg.get("processed_output", PROCESSED_OUTPUT),
            **cfg
        )

    # ---------------------------------------------------------------- process
    result = pipeline.process()

    if result.success:
        logger.info(f"✅ Success! Processed {result.total_processed} articles")
        logger.info(f"📄 Output: {result.output_path}")
    else:
        logger.error(f"❌ Pipeline failed: {result.error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
