# config/settings.py
"""
Centralized configuration for Financial Sentiment Analysis Pipeline
"""

import os
from pathlib import Path

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
INPUT_PARQUET = DATA_DIR / "financial_news_2020_2025_100k.parquet"
PROCESSED_OUTPUT = DATA_DIR / "processed_articles.jsonl"

# API Configuration
EODHD_API_TOKEN = os.getenv("EODHD_API_TOKEN", "68442677069401.89798760")
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

    # Optimization parameters
    "neutral_penalty": 0.5,
    "pos_boost": 1.1,
    "neg_boost": 1.1,
    "min_confidence_diff": 0.03,

    # Aggregation settings
    "method": "conf_weighted",  # default, majority, conf_weighted
    "threshold": 0.1,
}

# NER Settings
NER_CONFIG = {
    "min_confidence": 0.6,
    "use_metadata": True,
    "exchange_suffixes": {'.US', '.TO', '.L', '.PA', '.F', '.HM', '.MI',
                          '.AS', '.MX', '.SA', '.BE', '.DU', '.MU', '.STU', '.XETRA'},
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
    "batch_size": 50,  # For Colab
    "sentiment_batch_size": 8,
    "max_retries": 3,
    "checkpoint_interval": 10,  # Save checkpoint every N tags
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
    }
}

# VADER Settings
VADER_CONFIG = {
    "threshold": 0.05,
}

# Tags for data collection
COLLECTION_TAGS = [
    'balance sheet', 'capital employed', 'class action', 'company announcement',
    'consensus eps estimate', 'consensus estimate', 'credit rating',
    'discounted cash flow', 'dividend payments', 'earnings estimate',
    'earnings growth', 'earnings per share', 'earnings release', 'earnings report',
    'earnings results', 'earnings surprise', 'estimate revisions',
    'european regulatory news', 'financial results', 'fourth quarter',
    'free cash flow', 'future cash flows', 'growth rate', 'initial public offering',
    'insider ownership', 'insider transactions', 'institutional investors',
    'institutional ownership', 'intrinsic value', 'market research reports',
    'net income', 'operating income', 'present value', 'press releases',
    'price target', 'quarterly earnings', 'quarterly results', 'ratings',
    'research analysis and reports', 'return on equity', 'revenue estimates',
    'revenue growth', 'roce', 'roe', 'share price', 'shareholder rights',
    'shareholder', 'shares outstanding', 'split', 'strong buy', 'total revenue',
    'zacks investment research', 'zacks rank'
]

# Excluded words for NER
EXCLUDED_WORDS = {
    'CEO', 'CFO', 'CTO', 'COO', 'IPO', 'SEC', 'FDA', 'EPA', 'GDP', 'USA', 'USD', 'EUR',
    'API', 'AI', 'IT', 'HR', 'PR', 'IR', 'QA', 'RD', 'GAAP', 'EBITDA', 'ROI', 'ROE',
    'MA', 'B2B', 'B2C', 'SAAS', 'COVID', 'Q1', 'Q2', 'Q3', 'Q4', 'YOY', 'QOQ', 'ETF',
    'NYSE', 'NASDAQ', 'SPY', 'QQQ', 'VIX', 'DJIA', 'SP', 'DOW', 'FED', 'FOMC', 'CPI',
    'PPI', 'PMI', 'ISM', 'NFIB', 'ADP', 'BLS', 'BEA', 'FRED', 'OECD', 'IMF', 'ECB',
    'BOJ', 'PBOC', 'SNB', 'BOE', 'RBA', 'BOC', 'RBNZ', 'SARB', 'CBR', 'BCB', 'MXN',
    'CAD', 'GBP', 'JPY', 'CHF', 'AUD', 'NZD', 'ZAR', 'RUB', 'BRL', 'CNY', 'INR',
    'KRW', 'TWD', 'HKD', 'SGD', 'THB', 'MYR', 'IDR', 'PHP', 'VND', 'LAK', 'MMK',
    'REIT', 'ETN', 'SPAC', 'IPO', 'ICO', 'DPO', 'PIPE', 'ESOP', 'DRIP', 'ADR', 'GDR'
}

# Financial context keywords
FINANCIAL_CONTEXTS = [
    'earnings', 'revenue', 'profit', 'loss', 'guidance', 'outlook', 'forecast',
    'quarterly', 'annual', 'results', 'report', 'announcement', 'statement',
    'dividend', 'split', 'merger', 'acquisition', 'buyback', 'repurchase',
    'upgrade', 'downgrade', 'rating', 'target', 'price', 'valuation',
    'analyst', 'estimate', 'consensus', 'beat', 'miss', 'surprise'
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
    "for educational purposes only"
]
