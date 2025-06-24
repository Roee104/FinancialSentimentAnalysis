# config/settings.py
"""
Centralised configuration for the Financial-Sentiment-Analysis project
"""

import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────────────
#  CORE PATHS (created on import)
# ────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = DATA_DIR / "outputs"
PLOTS_DIR = DATA_DIR / "plots"
CHECKPOINTS_DIR = DATA_DIR / "checkpoints"
CACHE_DIR = PROJECT_ROOT / ".cache"

for d in (DATA_DIR, OUTPUT_DIR, PLOTS_DIR, CHECKPOINTS_DIR, CACHE_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ────────────────────────────────────────────────────────────────────────
#  DATA FILES / URLS
# ────────────────────────────────────────────────────────────────────────
MASTER_TICKER_LIST = DATA_DIR / "master_ticker_list.csv"
TICKER_SECTOR_FILE = DATA_DIR / "ticker_sector.csv"
SECTOR_CACHE_FILE = DATA_DIR / "sp500_sectors_cache.json"
INPUT_PARQUET = DATA_DIR / "financial_news_2020_2025_100k.parquet"
PROCESSED_OUTPUT = DATA_DIR / "processed_articles.jsonl"
GOLD_FILE = DATA_DIR / "1500_gold_standard.jsonl"

# ────────────────────────────────────────────────────────────────────────
#  EXTERNAL SECRETS  (set in your environment or .env)
# ────────────────────────────────────────────────────────────────────────
EODHD_API_TOKEN = os.getenv("EODHD_API_TOKEN")  # 🡒 news API
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")   # 🡒 GPT-4 / GPT-4o

if not EODHD_API_TOKEN:
    logger.warning(
        "⚠  EODHD_API_TOKEN not set – data-collection scripts will fail.")

# ────────────────────────────────────────────────────────────────────────
#  PIPELINE-WIDE CONFIGS
# ────────────────────────────────────────────────────────────────────────
# ------------------------------------------------------------------ #
#  DATA-COLLECTION DEFAULTS  (used by data.loader.NewsDataCollector)
# ------------------------------------------------------------------ #
DATA_COLLECTION = dict(
    # API and date range
    base_url="https://eodhd.com/api/news",
    from_date="2020-01-01",
    to_date="2025-06-07",

    # volume targets & paging
    target_per_tag=2000,         # ≈ articles per search tag
    batch_size=500,              # EODHD page size
    max_pages_per_tag=50,        # hard stop safeguard

    # rate-limit / politeness
    sleep_sec=0.3,               # pause between calls (seconds)

    # quality / deduplication thresholds
    min_content_words=50,
    min_title_words=3,
    max_duplicate_threshold=0.85,
    duplicate_check_window=1000,  # recent-article window for near-dupes

    # fault-tolerance / retries
    max_retries=5,               # per-tag page fetch
    backoff_factor=2,            # exponential factor
    initial_retry_delay=1,
)

MODELS = dict(
    finbert="yiyanghkust/finbert-tone",
    bert_tokenizer="bert-base-uncased",
    gpt_model="gpt-4o-2024-05-13",
    cache_dir=str(CACHE_DIR / "models"),
)

SENTIMENT_CONFIG = dict(
    batch_size=16,
    max_length=512,
    device=None,            # auto-detect GPU / CPU
    method="conf_weighted",
    threshold=0.10,
)

# NER Settings
NER_CONFIG = {
    "min_confidence": 0.6,
    "use_metadata": True,
    "exchange_suffixes": {
        ".US",
        ".TO",
        ".L",
        ".PA",
        ".F",
        ".HM",
        ".MI",
        ".AS",
        ".MX",
        ".SA",
        ".BE",
        ".DU",
        ".MU",
        ".STU",
        ".XETRA",
    },
    "multi_word_ticker_pattern": r"\b[A-Z]{1,5}(?:\.[A-Z])?\b",
    "cache_sectors": True,
    "sector_cache_ttl": 86400,  # 24 hours in seconds
}

TEXT_PROCESSING = dict(
    min_clause_words=3,
    max_chunks=30,
    max_content_length=10000,
    min_length_for_commas=40,
)

PIPELINE_CONFIG = dict(
    batch_size=50,
    sentiment_batch_size=16,
    checkpoint_interval=10,
    buffer_size=100,
)

PLOT_CONFIG = dict(
    style="seaborn-v0_8-darkgrid",
    dpi=300,
    figsize=(10, 6),
    colors=dict(
        positive="#2ecc71",
        neutral="#95a5a6",
        negative="#e74c3c",
    ),
)

# ────────────────────────────────────────────────────────────────────────
#  TAGS & HEURISTICS  (used by data.loader, core.ner, etc.)
# ────────────────────────────────────────────────────────────────────────
COLLECTION_TAGS = [
    "balance sheet",
    "capital employed",
    "class action",
    "company announcement",
    "consensus eps estimate",
    "consensus estimate",
    "credit rating",
    "discounted cash flow",
    "dividend payments",
    "earnings estimate",
    "earnings growth",
    "earnings per share",
    "earnings release",
    "earnings report",
    "earnings results",
    "earnings surprise",
    "estimate revisions",
    "european regulatory news",
    "financial results",
    "fourth quarter",
    "free cash flow",
    "future cash flows",
    "growth rate",
    "initial public offering",
    "insider ownership",
    "insider transactions",
    "institutional investors",
    "institutional ownership",
    "intrinsic value",
    "market research reports",
    "net income",
    "operating income",
    "present value",
    "press releases",
    "price target",
    "quarterly earnings",
    "quarterly results",
    "ratings",
    "research analysis and reports",
    "return on equity",
    "revenue estimates",
    "revenue growth",
    "roce",
    "roe",
    "share price",
    "shareholder rights",
    "shareholder",
    "shares outstanding",
    "split",
    "strong buy",
    "total revenue",
    "zacks investment research",
    "zacks rank",
]

EXCLUDED_WORDS = {
    "CEO", "CFO", "CTO", "COO", "IPO", "SEC", "FDA", "EPA", "GDP", "USD", "EUR", "AI", "API",
    "ETF", "NYSE", "NASDAQ", "SPY", "QQQ", "DJIA", "FED", "FOMC",
}

FINANCIAL_CONTEXTS = [
    "earnings", "revenue", "profit", "loss", "guidance", "outlook", "forecast",
    "dividend", "split", "merger", "acquisition", "buyback", "upgrade", "downgrade",
]

LOW_QUALITY_PATTERNS = [
    "this article was generated by", "automatically generated",
    "ai generated", "for educational purposes only", "disclaimer:",
]

QUALITY_FILTER_STATS = {
    "min_content_words": 0,
    "min_title_words": 0,
    "empty_content": 0,
    "empty_title": 0,
    "low_quality_pattern": 0,
    "repetitive_content": 0,
    "short_sentences": 0,
    "no_symbols": 0,
}

# ────────────────────────────────────────────────────────────────────────
#  LOGGING CONFIG (imported via logging.config.dictConfig)
# ────────────────────────────────────────────────────────────────────────
LOGGING_CONFIG = {
    "version": 1,
    "formatters": {
        "std": {
            "format": "%(asctime)s | %(levelname)-8s | %(name)s: %(message)s"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "std",
            "level": "INFO",
        },
    },
    "root": {"handlers": ["console"], "level": "INFO"},
}

# ────────────────────────────────────────────────────────────────────────
#  DEFAULT_CONFIG  (needed by utils.config_loader.load_config)
# ────────────────────────────────────────────────────────────────────────
DEFAULT_CONFIG = {
    # ----- core paths -----
    "project_root": str(PROJECT_ROOT),
    "data_dir": str(DATA_DIR),
    "output_dir": str(OUTPUT_DIR),
    "plots_dir": str(PLOTS_DIR),
    "checkpoints_dir": str(CHECKPOINTS_DIR),
    "cache_dir": str(CACHE_DIR),

    # ----- data files -----
    "master_ticker_list": str(MASTER_TICKER_LIST),
    "ticker_sector_file": str(TICKER_SECTOR_FILE),
    "sector_cache_file": str(SECTOR_CACHE_FILE),
    "input_parquet": str(INPUT_PARQUET),
    "processed_output": str(PROCESSED_OUTPUT),

    # ----- secrets -----
    "eodhd_api_token": EODHD_API_TOKEN,
    "openai_api_key": OPENAI_API_KEY,

    # ----- nested configs -----
    "data_collection": DATA_COLLECTION,
    "models": MODELS,
    "sentiment_config": SENTIMENT_CONFIG,
    "ner_config": NER_CONFIG,
    "text_processing": TEXT_PROCESSING,
    "pipeline_config": PIPELINE_CONFIG,
    "plot_config": PLOT_CONFIG,

    # extra lists the project sometimes serialises
    "collection_tags": COLLECTION_TAGS,
    "excluded_words": list(EXCLUDED_WORDS),
    "financial_contexts": FINANCIAL_CONTEXTS,
    "low_quality_patterns": LOW_QUALITY_PATTERNS,
}

# ------------------------  VADER baseline  ------------------------- #
VADER_CONFIG = {
    "threshold": 0.07,       # compound > 0.07 → Positive, < −0.07 → Negative
}

# ------------------------------------------------------------------ #
#  Public export list – prevents star-imports from dragging in
#  gigantic dicts we don’t need at runtime.
# ------------------------------------------------------------------ #
__all__ = [
    # core paths
    "PROJECT_ROOT", "DATA_DIR", "OUTPUT_DIR", "PLOTS_DIR",
    "CHECKPOINTS_DIR", "CACHE_DIR",

    # secrets
    "EODHD_API_TOKEN", "OPENAI_API_KEY",

    # config dicts
    "DATA_COLLECTION", "MODELS", "SENTIMENT_CONFIG", "NER_CONFIG",
    "TEXT_PROCESSING", "PIPELINE_CONFIG", "PLOT_CONFIG",

    # misc lists / constants
    "COLLECTION_TAGS", "EXCLUDED_WORDS", "FINANCIAL_CONTEXTS",
    "LOW_QUALITY_PATTERNS", "QUALITY_FILTER_STATS",

    # master dict + extra constants needed by other modules
    "DEFAULT_CONFIG", "VADER_CONFIG",
]
