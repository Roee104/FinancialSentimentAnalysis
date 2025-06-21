# ────────────────────────────────────────────────────────────────────────────
#  scripts/setup_data.py
#  Prepare directories, download NLTK/VADER assets, check data presence
# ────────────────────────────────────────────────────────────────────────────
from __future__ import annotations
from data.builder import TickerListBuilder, SectorBuilder
from utils.helpers import ensure_dir, print_header, print_success, print_error
from config.settings import DATA_DIR, MASTER_TICKER_LIST
from pathlib import Path

# -------------------------------------------------------  logging FIRST  ---
import logging
import logging.config
from config.settings import LOGGING_CONFIG
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)

# ----------------------------------------------------------  stdlib / 3rd-party

# ----------------------------------------------------------  internal helpers

# ────────────────────────────────────────────────────────────────────────────


def setup_directories() -> None:
    print_header("Setting up directories")
    for d in [DATA_DIR, DATA_DIR / "outputs", DATA_DIR / "plots", DATA_DIR / "checkpoints"]:
        ensure_dir(d)
        print_success(f"Created {d}")


def setup_ticker_list() -> None:
    print_header("Setting up ticker list")
    if MASTER_TICKER_LIST.exists():
        print_success(f"Ticker list already exists at {MASTER_TICKER_LIST}")
        return
    try:
        df = TickerListBuilder().build_from_datahub()
        if df is not None:
            print_success(f"Downloaded {len(df)} tickers")
        else:
            print_error("Failed to download ticker list")
    except Exception as ex:  # noqa: BLE001
        print_error(f"Error building ticker list: {ex}")


def setup_sectors(skip_if_exists: bool = True) -> None:
    print_header("Setting up sector mappings")
    sector_file = DATA_DIR / "ticker_sector.csv"
    if skip_if_exists and sector_file.exists():
        print_success(f"Sector mappings already exist at {sector_file}")
        return
    try:
        df = SectorBuilder().build_sector_mapping()
        if df is not None:
            print_success(f"Built sector mappings for {len(df)} tickers")
        else:
            print_error("Failed to build sector mappings")
    except Exception as ex:  # noqa: BLE001
        print_error(f"Error building sectors: {ex}")


def check_data_file() -> None:
    print_header("Checking data file")
    data_file = DATA_DIR / "financial_news_2020_2025_100k.parquet"
    if data_file.exists():
        size_mb = data_file.stat().st_size / (1024 * 1024)
        print_success(f"Data file found ({size_mb:.1f} MB) → {data_file}")
    else:
        print_error(f"Data file not found: {data_file}")
        print("   1. Run data collection:  python data/loader.py")
        print("   2. Or copy an existing file to that path.")


def download_nltk_data() -> None:
    print_header("Setting up NLTK data")
    try:
        import nltk
        for name in ["punkt", "vader_lexicon"]:
            try:
                nltk.data.find(f"tokenizers/{name}")
                print_success(f"NLTK {name} already present")
            except LookupError:
                nltk.download(name, quiet=True)
                print_success(f"Downloaded NLTK {name}")
    except Exception as ex:  # noqa: BLE001
        print_error(f"Error downloading NLTK data: {ex}")


def main() -> None:
    print_header("FINANCIAL SENTIMENT ANALYSIS – DATA SETUP")
    setup_directories()
    download_nltk_data()
    setup_ticker_list()
    # Optional: build sector mappings (comment-in if you need them)
    # setup_sectors()
    check_data_file()
    print_header("Setup complete – you can now run `scripts/run_pipeline.py`")


if __name__ == "__main__":  # pragma: no cover
    main()
