# ────────────────────────────────────────────────────────────────────────────
#  scripts/run_pipeline.py
#  Main entry-point for running any Financial-Sentiment-Analysis pipeline
# ────────────────────────────────────────────────────────────────────────────
from __future__ import annotations
from utils.config_loader import load_config
from pipelines.main_pipeline import create_pipeline
from pipelines.baselines import VADERBaseline
from config.settings import PROCESSED_OUTPUT, INPUT_PARQUET
import sys
from pathlib import Path
import argparse

# -------------------------------------------------------  logging FIRST  ---
import logging
import logging.config
from config.settings import LOGGING_CONFIG

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)

# -------------------------------------------------------  NER monkey-patch  ---
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
                   help="Aggregation method (legacy)")

    # Additional controls
    p.add_argument("--threshold", type=float,
                   help="Sentiment threshold (0.0-1.0)")
    p.add_argument("--no-distance-weighting", action="store_true",
                   help="Disable distance-based weighting")

    # VADER-specific
    p.add_argument("--vader-threshold", type=float, default=0.05,
                   help="VADER sentiment threshold")

    # Dev options
    p.add_argument("--device", choices=["cpu", "cuda", "mps"],
                   help="Compute device")
    p.add_argument("--checkpoint-interval", type=int,
                   help="Save checkpoint every N batches")

    return p


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    # --------------------------------------------------------- config loading
    # Load base configuration from YAML if provided
    if args.config:
        logger.info(f"Loading config from {args.config}")
        cfg = load_config(args.config)
    else:
        cfg = {}

    # Ensure nested config safety
    cfg.setdefault("pipeline_config", {})
    cfg.setdefault("sentiment_config", {})
    cfg.setdefault("aggregation_config", {})

    # --------------------------------------------------------- override w/ CLI
    # Input/output paths - CRITICAL FIX: Use args directly
    input_path = args.input or cfg.get("input_parquet", INPUT_PARQUET)
    output_path = args.output or cfg.get("processed_output", PROCESSED_OUTPUT)

    # Store paths in config for pipeline
    cfg["input_parquet"] = input_path
    cfg["processed_output"] = output_path

    # Pipeline config
    if args.batch_size is not None:
        cfg["pipeline_config"]["batch_size"] = args.batch_size
    if args.max_articles is not None:
        cfg["pipeline_config"]["max_articles"] = args.max_articles
    if args.start_from:
        cfg["pipeline_config"]["start_from"] = args.start_from
    if args.no_resume:
        cfg["pipeline_config"]["resume"] = False
    if args.checkpoint_interval:
        cfg["pipeline_config"]["checkpoint_interval"] = args.checkpoint_interval

    # Sentiment config
    if args.sentiment_mode:
        cfg["sentiment_config"]["mode"] = args.sentiment_mode
    if args.sentiment_batch_size:
        cfg["sentiment_config"]["batch_size"] = args.sentiment_batch_size
    if args.device:
        cfg["device"] = args.device

    # Aggregation config
    if args.threshold is not None:
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
        pipeline = VADERBaseline(
            input_path=input_path,
            output_path=output_path,
            vader_threshold=args.vader_threshold,
            **cfg
        )
    else:
        logger.info(f"🏃 Running {args.pipeline} pipeline")
        logger.info(f"📄 Input: {input_path}")
        logger.info(f"📄 Output: {output_path}")

        # CRITICAL FIX: Pass input_path and output_path explicitly
        pipeline = create_pipeline(
            pipeline_type=args.pipeline,
            input_path=input_path,
            output_path=output_path,
            **cfg
        )

    # ---------------------------------------------------------------- process
    # Extract pipeline-specific parameters
    pipeline_params = cfg.get("pipeline_config", {})

    # Run the pipeline using the correct method
    pipeline.run(
        max_articles=pipeline_params.get("max_articles"),
        start_from=pipeline_params.get("start_from", 0),
        checkpoint_interval=pipeline_params.get("checkpoint_interval")
    )

    # Print final statistics
    logger.info(f"✅ Pipeline completed successfully!")
    logger.info(f"📄 Output: {pipeline.output_path}")
    logger.info(f"📊 Processed: {pipeline.stats['processed']} articles")
    logger.info(f"⚠️  Errors: {pipeline.stats['errors']}")


if __name__ == "__main__":
    main()
