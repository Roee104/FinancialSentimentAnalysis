# config/settings.py
"""
Centralised configuration for the Financial-Sentiment-Analysis project
"""

import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# ────────────────  CORE PATHS (created on import)  ───────────────── #
# ------------------------------------------------------------------ #
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = DATA_DIR / "outputs"
PLOTS_DIR = DATA_DIR / "plots"
CHECKPOINTS_DIR = DATA_DIR / "checkpoints"
CACHE_DIR = PROJECT_ROOT / ".cache"

for d in (DATA_DIR, OUTPUT_DIR, PLOTS_DIR, CHECKPOINTS_DIR, CACHE_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------ #
# ───────────────────────  DATA FILES / URLS  ─────────────────────── #
# ------------------------------------------------------------------ #
MASTER_TICKER_LIST = DATA_DIR / "master_ticker_list.csv"
TICKER_SECTOR_FILE = DATA_DIR / "ticker_sector.csv"
SECTOR_CACHE_FILE = DATA_DIR / "sp500_sectors_cache.json"
INPUT_PARQUET = DATA_DIR / "financial_news_2020_2025_100k.parquet"
PROCESSED_OUTPUT = DATA_DIR / "processed_articles.jsonl"

# ------------------------------------------------------------------ #
# ───────────────────────  EXTERNAL SECRETS  ──────────────────────── #
# ------------------------------------------------------------------ #
EODHD_API_TOKEN = os.getenv("EODHD_API_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not EODHD_API_TOKEN:
    logger.warning(
        "⚠  EODHD_API_TOKEN not set – data-collection scripts will fail.")

# ------------------------------------------------------------------ #
# ────────────────────────  PIPELINE CONFIGS  ─────────────────────── #
# ------------------------------------------------------------------ #
DATA_COLLECTION = dict(
    base_url="https://eodhd.com/api/news",
    from_date="2020-01-01",
    to_date="2025-06-07",
    target_per_tag=2000,
    batch_size=500,
    sleep_sec=0.3,
    min_content_words=50,
    max_duplicate_threshold=0.85,
    max_pages_per_tag=50,
)

MODELS = dict(
    finbert="yiyanghkust/finbert-tone",
    bert_tokenizer="bert-base-uncased",
    gpt_model="gpt-4",
    cache_dir=str(CACHE_DIR / "models"),
)

SENTIMENT_CONFIG = dict(batch_size=16,
                        max_length=512,
                        device=None,
                        method="conf_weighted",
                        threshold=0.10)

NER_CONFIG = dict(min_confidence=0.6,
                  use_metadata=True,
                  cache_sectors=True,
                  sector_cache_ttl=86400)

TEXT_PROCESSING = dict(min_clause_words=3,
                       max_chunks=30)

PLOT_CONFIG = dict(style="seaborn-v0_8-darkgrid",
                   dpi=300,
                   figsize=(10, 6),
                   colors=dict(positive="#2ecc71",
                               neutral="#95a5a6",
                               negative="#e74c3c"))

# ------------------------------------------------------------------ #
# ──────────────────────────  LOGGING  ────────────────────────────── #
# ------------------------------------------------------------------ #
LOGGING_CONFIG = {
    "version": 1,
    "formatters": {
        "std": {"format": "%(asctime)s | %(levelname)-8s | %(name)s: %(message)s"}
    },
    "handlers": {
        "console": {"class": "logging.StreamHandler",
                    "formatter": "std",
                    "level": "INFO"},
    },
    "root": {"handlers": ["console"], "level": "INFO"},
}
