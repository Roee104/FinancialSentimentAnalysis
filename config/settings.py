# config/settings.py
"""
Centralized configuration for Financial Sentiment Analysis Pipeline
Default values - can be overridden by config.yaml and CLI args
"""

import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = DATA_DIR / "outputs"
PLOTS_DIR = DATA_DIR / "plots"
CHECKPOINTS_DIR = DATA_DIR / "checkpoints"

# Ensure directories exist
for dir_path in [DATA_DIR, OUTPUT_DIR, PLOTS_DIR, CHECKPOINTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Data files
MASTER_TICKER_LIST = DATA_DIR / "master_ticker_list.csv"
TICKER_SECTOR_FILE = DATA_DIR / "ticker_sector.csv"
SECTOR_CACHE_FILE = DATA_DIR / "sp500_sectors_cache.csv"
INPUT_PARQUET = DATA_DIR / "financial_news_2020_2025_100k.parquet"
PROCESSED_OUTPUT = DATA_DIR / "processed_articles.jsonl"

# API Configuration
EODHD_API_TOKEN = os.getenv("EODHD_API_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Data Collection Settings
DATA_COLLECTION = {
    "base_url": "https://eodhd.com/api/news",
    "from_date": "2020-01-01",
    "to_date": "2025-06-07",
    "target_per_tag": 2000,
    "batch_size": 500,
    "sleep_sec": 0.3,
    "min_content_words": 50,
    "min_title_words": 3,
    "max_duplicate_threshold": 0.85,
    "max_retries": 5,
    "backoff_factor": 2,
    "initial_retry_delay": 1,
    "max_pages_per_tag": 50,
    "duplicate_check_window": 1000,  # Check against last N articles
}

# Model Settings
MODELS = {
    "finbert": "yiyanghkust/finbert-tone",
    "bert_tokenizer": "bert-base-uncased",
    "gpt_model": "gpt-4",
}

# Sentiment Analysis Settings
SENTIMENT_CONFIG = {
    "batch_size": 16,
    "max_length": 512,
    "device": None,  # Auto-detect
    "neutral_penalty": 0.5,
    "pos_boost": 1.1,
    "neg_boost": 1.1,
    "min_confidence_diff": 0.03,
    "method": "conf_weighted",
    "threshold": 0.1,
}

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

# Text Processing Settings
TEXT_PROCESSING = {
    "min_clause_words": 3,
    "min_length_for_commas": 40,
    "max_content_length": 10000,
    "max_chunks": 30,
}

# Pipeline Settings
PIPELINE_CONFIG = {
    "batch_size": 50,
    "sentiment_batch_size": 16,
    "max_retries": 3,
    "checkpoint_interval": 10,
    "buffer_size": 100,
}

# Visualization Settings
PLOT_CONFIG = {
    "style": "seaborn-v0_8-darkgrid",
    "dpi": 300,
    "figsize": (10, 6),
    "colors": {
        "positive": "#2ecc71",
        "neutral": "#95a5a6",
        "negative": "#e74c3c",
    },
}

# VADER Settings
VADER_CONFIG = {
    "threshold": 0.05,
}

# Logging Configuration
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        },
        "detailed": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "standard",
            "stream": "ext://sys.stdout",
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "DEBUG",
            "formatter": "detailed",
            "filename": str(DATA_DIR / "pipeline.log"),
            "maxBytes": 10485760,  # 10MB
            "backupCount": 3,
        },
    },
    "loggers": {
        "": {  # root logger
            "handlers": ["console", "file"],
            "level": "INFO",
            "propagate": False,
        }
    },
}

# Tags for data collection
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

# Excluded words for NER
EXCLUDED_WORDS = {
    "CEO",
    "CFO",
    "CTO",
    "COO",
    "IPO",
    "SEC",
    "FDA",
    "EPA",
    "GDP",
    "USA",
    "USD",
    "EUR",
    "API",
    "AI",
    "IT",
    "HR",
    "PR",
    "IR",
    "QA",
    "RD",
    "GAAP",
    "EBITDA",
    "ROI",
    "ROE",
    "MA",
    "B2B",
    "B2C",
    "SAAS",
    "COVID",
    "Q1",
    "Q2",
    "Q3",
    "Q4",
    "YOY",
    "QOQ",
    "ETF",
    "NYSE",
    "NASDAQ",
    "SPY",
    "QQQ",
    "VIX",
    "DJIA",
    "SP",
    "DOW",
    "FED",
    "FOMC",
    "CPI",
    "PPI",
    "PMI",
    "ISM",
    "NFIB",
    "ADP",
    "BLS",
    "BEA",
    "FRED",
    "OECD",
    "IMF",
    "ECB",
    "BOJ",
    "PBOC",
    "SNB",
    "BOE",
    "RBA",
    "BOC",
    "RBNZ",
    "SARB",
    "CBR",
    "BCB",
    "MXN",
    "CAD",
    "GBP",
    "JPY",
    "CHF",
    "AUD",
    "NZD",
    "ZAR",
    "RUB",
    "BRL",
    "CNY",
    "INR",
    "KRW",
    "TWD",
    "HKD",
    "SGD",
    "THB",
    "MYR",
    "IDR",
    "PHP",
    "VND",
    "LAK",
    "MMK",
    "REIT",
    "ETN",
    "SPAC",
    "IPO",
    "ICO",
    "DPO",
    "PIPE",
    "ESOP",
    "DRIP",
    "ADR",
    "GDR",
}

# Financial context keywords
FINANCIAL_CONTEXTS = [
    "earnings",
    "revenue",
    "profit",
    "loss",
    "guidance",
    "outlook",
    "forecast",
    "quarterly",
    "annual",
    "results",
    "report",
    "announcement",
    "statement",
    "dividend",
    "split",
    "merger",
    "acquisition",
    "buyback",
    "repurchase",
    "upgrade",
    "downgrade",
    "rating",
    "target",
    "price",
    "valuation",
    "analyst",
    "estimate",
    "consensus",
    "beat",
    "miss",
    "surprise",
]

# Low quality patterns
LOW_QUALITY_PATTERNS = [
    "this article was generated by",
    "automatically generated",
    "computer generated",
    "ai generated",
    "algorithmic trading",
    "please see important disclosures",
    "this is not investment advice",
    "past performance does not guarantee",
    "disclaimer:",
    "for educational purposes only",
]

# Quality filter statistics tracking
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

# Create default config dictionary
DEFAULT_CONFIG = {
    "project_root": str(PROJECT_ROOT),
    "data_dir": str(DATA_DIR),
    "output_dir": str(OUTPUT_DIR),
    "plots_dir": str(PLOTS_DIR),
    "checkpoints_dir": str(CHECKPOINTS_DIR),
    "master_ticker_list": str(MASTER_TICKER_LIST),
    "ticker_sector_file": str(TICKER_SECTOR_FILE),
    "sector_cache_file": str(SECTOR_CACHE_FILE),
    "input_parquet": str(INPUT_PARQUET),
    "processed_output": str(PROCESSED_OUTPUT),
    "eodhd_api_token": EODHD_API_TOKEN,
    "openai_api_key": OPENAI_API_KEY,
    "data_collection": DATA_COLLECTION,
    "models": MODELS,
    "sentiment_config": SENTIMENT_CONFIG,
    "ner_config": NER_CONFIG,
    "text_processing": TEXT_PROCESSING,
    "pipeline_config": PIPELINE_CONFIG,
    "plot_config": PLOT_CONFIG,
    "vader_config": VADER_CONFIG,
    "logging_config": LOGGING_CONFIG,
    "collection_tags": COLLECTION_TAGS,
    "excluded_words": list(EXCLUDED_WORDS),
    "financial_contexts": FINANCIAL_CONTEXTS,
    "low_quality_patterns": LOW_QUALITY_PATTERNS,
    "quality_filter_stats": QUALITY_FILTER_STATS,
}

# ─────────────────────────────────────────────────────────────
# Central logging configuration
# Imported by scripts/run_pipeline.py → logging.config.dictConfig
# ─────────────────────────────────────────────────────────────

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,

    # FORMATTERS  ────────────────
    "formatters": {
        "console": {
            "format": "%(asctime)s | %(levelname)-8s | %(name)s: %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },

    # HANDLERS  ─────────────────
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",              # default; overridden by --log-level
            "formatter": "console",
        },
    },

    # ROOT LOGGER  ──────────────
    "root": {
        "handlers": ["console"],
        "level": "INFO",
    },
}
