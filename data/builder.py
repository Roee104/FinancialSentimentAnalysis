# data/builder.py
"""
Builders for ticker lists and sector mappings
"""

import pandas as pd
import yfinance as yf
import time
import os
import logging
from pathlib import Path
from typing import Dict, Optional

from config.settings import DATA_DIR, MASTER_TICKER_LIST

logger = logging.getLogger(__name__)


class TickerListBuilder:
    """Builds master ticker list from various sources"""

    @staticmethod
    def build_from_datahub():
        """Build ticker list from DataHub NYSE/Other listings"""
        logger.info("Building ticker list from DataHub...")

        # DataHub URLs
        URL_NYSE = "https://datahub.io/core/nyse-other-listings/r/nyse-listed.csv"
        URL_OTHER = "https://datahub.io/core/nyse-other-listings/r/other-listed.csv"

        try:
            # Load both CSV files
            df_nyse = pd.read_csv(URL_NYSE, dtype=str)
            df_other = pd.read_csv(URL_OTHER, dtype=str)

            # Combine datasets
            df = pd.concat([df_nyse, df_other], ignore_index=True)

            # Drop ETFs if column exists
            if 'ETF' in df.columns:
                df = df[df['ETF'] != 'Y']

            # Select and rename columns
            df = df[['ACT Symbol', 'Company Name']].rename(
                columns={'ACT Symbol': 'symbol',
                         'Company Name': 'company_name'}
            )

            # Remove symbols with special characters
            df = df[~df['symbol'].str.contains(r"[\.$]", na=False)]

            # Deduplicate
            df = df.drop_duplicates(subset='symbol').reset_index(drop=True)

            # Save
            output_path = MASTER_TICKER_LIST
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False)

            logger.info(f"✅ Saved {len(df)} unique tickers to {output_path}")
            return df

        except Exception as e:
            logger.error(f"Error building ticker list: {e}")
            return None


class SectorBuilder:
    """Builds ticker to sector mappings"""

    def __init__(self, master_csv: Path = None):
        """
        Initialize sector builder

        Args:
            master_csv: Path to master ticker list
        """
        self.master_csv = master_csv or MASTER_TICKER_LIST
        self.output_csv = DATA_DIR / "ticker_sector.csv"

    def build_sector_mapping(self, pause: float = 0.1) -> Optional[pd.DataFrame]:
        """
        Build sector mapping using yfinance

        Args:
            pause: Pause between API calls

        Returns:
            DataFrame with ticker->sector mapping
        """
        if not self.master_csv.exists():
            logger.error(f"Master ticker list not found at {self.master_csv}")
            return None

        logger.info(f"Building sector mapping from {self.master_csv}")

        # Load tickers
        df = pd.read_csv(self.master_csv, dtype=str)
        records = []

        for i, row in df.iterrows():
            sym = row["symbol"].strip()

            if i % 100 == 0:
                logger.info(f"Processing ticker {i+1}/{len(df)}: {sym}")

            try:
                # Get info from yfinance
                ticker = yf.Ticker(sym)
                info = ticker.info

                # Try to get sector
                sector = (info.get("sector") or
                          info.get("industry") or
                          "Unknown")

            except Exception:
                sector = "Unknown"

            records.append({
                "symbol": sym,
                "company_name": row["company_name"],
                "sector": sector
            })

            # Rate limit
            if (i + 1) % 100 == 0:
                time.sleep(pause)

        # Create DataFrame
        out_df = pd.DataFrame(records)

        # Save
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(self.output_csv, index=False)

        logger.info(f"✅ Written {len(out_df)} rows to {self.output_csv}")

        # Summary statistics
        sector_counts = out_df['sector'].value_counts()
        logger.info("\nSector distribution:")
        for sector, count in sector_counts.head(10).items():
            logger.info(f"  {sector}: {count}")

        return out_df

    def load_sp500_sectors(self) -> Dict[str, str]:
        """Load S&P 500 sectors from Wikipedia"""
        try:
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            df = pd.read_html(url)[0]
            df["Symbol"] = df["Symbol"].str.replace(
                ".", "-", regex=False).str.strip()
            return dict(zip(df["Symbol"], df["GICS Sector"]))
        except Exception as e:
            logger.error(f"Error loading S&P 500 sectors: {e}")
            return {}

    def merge_sector_sources(self) -> pd.DataFrame:
        """Merge sector data from multiple sources"""
        sectors = {}

        # Load existing ticker_sector.csv if exists
        if self.output_csv.exists():
            df = pd.read_csv(self.output_csv, dtype=str)
            for _, row in df.iterrows():
                sectors[row['symbol']] = row['sector']

        # Add S&P 500 sectors
        sp500_sectors = self.load_sp500_sectors()
        sectors.update(sp500_sectors)

        # Convert to DataFrame
        df_merged = pd.DataFrame([
            {"symbol": sym, "sector": sect}
            for sym, sect in sectors.items()
        ])

        # Save merged version
        merged_path = DATA_DIR / "ticker_sector_merged.csv"
        df_merged.to_csv(merged_path, index=False)

        logger.info(f"✅ Merged sector data: {len(df_merged)} entries")
        return df_merged


# Main functions for command line usage
def build_ticker_list():
    """Build master ticker list"""
    builder = TickerListBuilder()
    return builder.build_from_datahub()


def build_sectors():
    """Build sector mappings"""
    builder = SectorBuilder()
    return builder.build_sector_mapping()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build ticker data")
    parser.add_argument("--tickers", action="store_true",
                        help="Build master ticker list")
    parser.add_argument("--sectors", action="store_true",
                        help="Build sector mappings")
    parser.add_argument("--all", action="store_true",
                        help="Build everything")

    args = parser.parse_args()

    if args.all or args.tickers:
        build_ticker_list()

    if args.all or args.sectors:
        build_sectors()
