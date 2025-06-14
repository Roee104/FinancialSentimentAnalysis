# scripts/setup_data.py
"""
Setup script for preparing data files and directories
"""

from config.settings import DATA_DIR, MASTER_TICKER_LIST
from utils.helpers import ensure_dir, print_header, print_success, print_error
from data.builder import TickerListBuilder, SectorBuilder
import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_directories():
    """Create all required directories"""
    print_header("Setting up directories")

    directories = [
        DATA_DIR,
        DATA_DIR / "outputs",
        DATA_DIR / "plots",
        DATA_DIR / "checkpoints"
    ]

    for directory in directories:
        ensure_dir(directory)
        print_success(f"Created {directory}")


def setup_ticker_list():
    """Download and setup master ticker list"""
    print_header("Setting up ticker list")

    if MASTER_TICKER_LIST.exists():
        print_success(f"Ticker list already exists at {MASTER_TICKER_LIST}")
        return

    try:
        builder = TickerListBuilder()
        df = builder.build_from_datahub()

        if df is not None:
            print_success(f"Downloaded {len(df)} tickers")
        else:
            print_error("Failed to download ticker list")
    except Exception as e:
        print_error(f"Error building ticker list: {e}")


def setup_sectors(skip_if_exists: bool = True):
    """Build sector mappings"""
    print_header("Setting up sector mappings")

    sector_file = DATA_DIR / "ticker_sector.csv"

    if skip_if_exists and sector_file.exists():
        print_success(f"Sector mappings already exist at {sector_file}")
        return

    try:
        builder = SectorBuilder()
        df = builder.build_sector_mapping()

        if df is not None:
            print_success(f"Built sector mappings for {len(df)} tickers")
        else:
            print_error("Failed to build sector mappings")
    except Exception as e:
        print_error(f"Error building sectors: {e}")


def check_data_file():
    """Check if main data file exists"""
    print_header("Checking data file")

    data_file = DATA_DIR / "financial_news_2020_2025_100k.parquet"

    if data_file.exists():
        print_success(f"Data file found: {data_file}")

        # Get file size
        size_mb = data_file.stat().st_size / (1024 * 1024)
        print_success(f"File size: {size_mb:.1f} MB")
    else:
        print_error(f"Data file not found: {data_file}")
        print("Please either:")
        print("  1. Run data collection: python data/loader.py")
        print("  2. Copy your existing file to: data/financial_news_2020_2025_100k.parquet")


def download_nltk_data():
    """Download required NLTK data"""
    print_header("Setting up NLTK data")

    try:
        import nltk

        # Download required data
        for data_name in ['punkt', 'punkt_tab']:
            try:
                nltk.data.find(f'tokenizers/{data_name}')
                print_success(f"NLTK {data_name} already downloaded")
            except LookupError:
                nltk.download(data_name, quiet=True)
                print_success(f"Downloaded NLTK {data_name}")

    except Exception as e:
        print_error(f"Error downloading NLTK data: {e}")


def main():
    """Run all setup steps"""
    print_header("FINANCIAL SENTIMENT ANALYSIS - DATA SETUP")

    # 1. Setup directories
    setup_directories()

    # 2. Download NLTK data
    download_nltk_data()

    # 3. Setup ticker list
    setup_ticker_list()

    # 4. Setup sectors (optional - takes time)
    # Uncomment to build sector mappings
    # setup_sectors()

    # 5. Check data file
    check_data_file()

    print_header("Setup Complete!")
    print("\nNext steps:")
    print("1. If data file is missing, copy it to data/ directory")
    print("2. Run pipeline: python scripts/run_pipeline.py")


if __name__ == "__main__":
    main()
