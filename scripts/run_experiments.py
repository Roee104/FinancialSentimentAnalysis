# ────────────────────────────────────────────────────────────────────────────
#  scripts/run_experiments.py
#  Batch-runner that executes multiple pipeline flavours then plots results
# ────────────────────────────────────────────────────────────────────────────
from __future__ import annotations
from scripts.run_pipeline import main as run_pipeline
from analysis.visualization import create_comparison_plots
from config.settings import DATA_DIR, OUTPUT_DIR
from typing import List, Tuple
from pathlib import Path
from datetime import datetime
import time
import sys
import json

# -------------------------------------------------------  logging FIRST  ---
import logging
import logging.config
from config.settings import LOGGING_CONFIG
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)

# ----------------------------------------------------------  stdlib / 3rd-party

# ----------------------------------------------------------  internal imports

# ────────────────────────────────────────────────────────────────────────────


def run_experiment(name: str, pipeline_args: list[str]) -> bool:
    """Run a single experiment by forwarding `pipeline_args` to run_pipeline()."""
    logger.info("\n%s\n🚀 %s\n%s", "=" * 60, name, "=" * 60)
    start = time.time()

    try:
        original_argv = sys.argv
        sys.argv = ["run_pipeline.py", *pipeline_args]
        run_pipeline()
        elapsed = time.time() - start
        logger.info("✅ Completed in %.1f s", elapsed)
        return True
    except Exception as ex:   # noqa: BLE001
        logger.exception("❌ Failed: %s", ex)
        return False
    finally:
        sys.argv = original_argv


def run_comparison(file_label_pairs: List[Tuple[Path, str]],
                   output_dir: Path = OUTPUT_DIR) -> None:
    """Transform raw JSONL → stats list → comparison plots."""
    stats = []

    for fpath, label in file_label_pairs:
        if not fpath.exists():
            continue

        pos = neu = neg = with_ticker = total = 0
        with fpath.open(encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                total += 1
                sent = rec.get("overall_sentiment", "Neutral")
                if sent == "Positive":
                    pos += 1
                elif sent == "Negative":
                    neg += 1
                else:
                    neu += 1
                if rec.get("tickers"):
                    with_ticker += 1

        if total:
            stats.append(
                dict(
                    name=label,
                    sentiment_pct={
                        "Positive": pos * 100 / total,
                        "Neutral": neu * 100 / total,
                        "Negative": neg * 100 / total,
                    },
                    ticker_coverage_pct=with_ticker * 100 / total,
                )
            )

    if len(stats) >= 2:
        create_comparison_plots(stats, output_dir)
        logger.info("✅ Comparison plots saved to %s", output_dir)
    else:
        logger.warning("Not enough result files for comparison")


def main() -> None:
    logger.info("🔬 RUNNING ALL EXPERIMENTS")
    logger.info("Start: %s", datetime.now().isoformat(timespec="seconds"))

    experiments = [
        # Standard pipeline
        dict(
            name="Standard Pipeline",
            args=["--pipeline", "standard",
                  "--output", str(DATA_DIR / "processed_articles_standard.jsonl")],
        ),
        # Optimized (default conf_weighted)
        dict(
            name="Optimized Pipeline",
            args=["--pipeline", "optimized",
                  "--output", str(DATA_DIR / "processed_articles_optimized.jsonl")],
        ),
        # Optimized - threshold tweak
        dict(
            name="Optimized Pipeline (thr 0.15)",
            args=["--pipeline", "optimized",
                  "--output", str(DATA_DIR /
                                  "processed_articles_optimized_t15.jsonl"),
                  "--threshold", "0.15"],
        ),
        # Calibrated
        dict(
            name="Calibrated Pipeline",
            args=["--pipeline", "calibrated",
                  "--output", str(DATA_DIR / "processed_articles_calibrated.jsonl")],
        ),
        # VADER baseline
        dict(
            name="VADER Baseline",
            args=["--pipeline", "vader",
                  "--output", str(DATA_DIR / "vader_baseline_results.jsonl")],
        ),
        # Majority voting variant
        dict(
            name="Optimized Pipeline (Majority Method)",
            args=["--pipeline", "optimized",
                  "--output", str(DATA_DIR /
                                  "processed_articles_majority.jsonl"),
                  "--method", "majority"],
        ),
    ]

    # Run each experiment
    summary = [(exp["name"], run_experiment(exp["name"], exp["args"]))
               for exp in experiments]

    # Comparison plots
    file_map = {
        "Standard Pipeline": DATA_DIR / "processed_articles_standard.jsonl",
        "VADER Baseline": DATA_DIR / "vader_baseline_results.jsonl",
        "Optimized Pipeline": DATA_DIR / "processed_articles_optimized.jsonl",
        "Optimized Pipeline (thr 0.15)": DATA_DIR / "processed_articles_optimized_t15.jsonl",
        "Calibrated Pipeline": DATA_DIR / "processed_articles_calibrated.jsonl",
        "Optimized Pipeline (Majority Method)": DATA_DIR / "processed_articles_majority.jsonl",
    }
    available = [(f, name) for name, f in file_map.items() if f.exists()]
    run_comparison(available, OUTPUT_DIR)

    # Summary log
    ok = sum(s for _, s in summary)
    logger.info("Experiments finished: %d / %d succeeded", ok, len(summary))


if __name__ == "__main__":  # pragma: no cover
    main()
