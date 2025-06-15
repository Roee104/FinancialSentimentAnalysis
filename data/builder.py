# data/builder.py
"""
Builders for ticker lists and sector mappings
(2025-06-15 revision)
"""

from __future__ import annotations

import logging
import time
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import yfinance as yf

from config.settings import DATA_DIR, MASTER_TICKER_LIST

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Master list builder (unchanged besides doc-string tweaks)
# ──────────────────────────────────────────────────────────────────────────────
class TickerListBuilder:
    """Builds master ticker list from DataHub NYSE/Other listings."""

    @staticmethod
    def build_from_datahub() -> Optional[pd.DataFrame]:
        logger.info("Downloading ticker lists from DataHub …")
        URL_NYSE = "https://datahub.io/core/nyse-other-listings/r/nyse-listed.csv"
        URL_OTHER = "https://datahub.io/core/nyse-other-listings/r/other-listed.csv"

        try:
            df_nyse = pd.read_csv(URL_NYSE, dtype=str)
            df_other = pd.read_csv(URL_OTHER, dtype=str)
            df = pd.concat([df_nyse, df_other], ignore_index=True)
            if "ETF" in df.columns:
                df = df[df["ETF"] != "Y"]  # drop ETFs

            df = (
                df[["ACT Symbol", "Company Name"]]
                .rename(columns={"ACT Symbol": "symbol", "Company Name": "company_name"})
                .query("symbol.str.contains('[\.$]', regex=True) == False")  # noqa: E501
                .drop_duplicates(subset="symbol")
                .reset_index(drop=True)
            )

            MASTER_TICKER_LIST.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(MASTER_TICKER_LIST, index=False)
            logger.info("Saved %s unique tickers → %s",
                        len(df), MASTER_TICKER_LIST)
            return df
        except Exception as e:  # noqa: BLE001
            logger.error("Ticker-list build failed: %s", e)
            return None


# ──────────────────────────────────────────────────────────────────────────────
# Sector builder (biggest changes)
# ──────────────────────────────────────────────────────────────────────────────
class SectorBuilder:
    """Builds ticker→sector mapping with resumable progress."""

    def __init__(self, master_csv: Path | None = None) -> None:
        self.master_csv = master_csv or MASTER_TICKER_LIST
        self.output_csv = DATA_DIR / "ticker_sector.csv"
        self.progress_file = DATA_DIR / "ticker_sector_progress.parquet"

    # -------------- public --------------
    def build_sector_mapping(self, pause: float = 0.1, batch_save: int = 500) -> Optional[pd.DataFrame]:
        if not self.master_csv.exists():
            logger.error("Master ticker list missing: %s", self.master_csv)
            return None

        master_df = pd.read_csv(self.master_csv, dtype=str)
        done_symbols = set()

        if self.progress_file.exists():
            logger.info("Resuming from progress file %s", self.progress_file)
            existing_df = pd.read_parquet(self.progress_file)
            done_symbols = set(existing_df["symbol"])
        else:
            existing_df = pd.DataFrame(
                columns=["symbol", "company_name", "sector"])

        records = []
        start_time = time.time()

        for i, row in master_df.iterrows():
            sym = row["symbol"].strip()
            if sym in done_symbols:
                continue

            if (len(records) + len(existing_df)) % 100 == 0:
                logger.info("Processed %s/%s tickers",
                            len(records) + len(existing_df), len(master_df))

            sector = self._fetch_sector(sym)
            records.append(
                {"symbol": sym, "company_name": row["company_name"], "sector": sector})

            # periodic flush
            if len(records) and len(records) % batch_save == 0:
                self._flush(records, existing_df)
                records.clear()
                logger.info("Checkpoint flushed (%s rows).", len(existing_df))

            if (i + 1) % 100 == 0:
                time.sleep(pause)

        # final flush
        if records:
            self._flush(records, existing_df)

        # promote progress_file to final csv
        df_final = pd.read_parquet(self.progress_file)
        df_final.to_csv(self.output_csv, index=False)
        elapsed = time.time() - start_time
        logger.info("✅ Sector mapping built (%s rows, %.1f s)",
                    len(df_final), elapsed)
        self._log_sector_stats(df_final)
        return df_final

    # -------------- private --------------
    def _flush(self, new_records: list[dict], existing_df: pd.DataFrame) -> None:
        df_chunk = pd.DataFrame(new_records)
        df_all = pd.concat([existing_df, df_chunk], ignore_index=True)
        df_all.to_parquet(self.progress_file, index=False)

    @staticmethod
    @lru_cache(maxsize=50_000)
    def _fetch_sector(symbol: str, retries: int = 3) -> str:
        for attempt in range(1, retries + 1):
            try:
                info = yf.Ticker(symbol).info
                return info.get("sector") or info.get("industry") or "Unknown"
            except Exception:  # noqa: BLE001
                if attempt == retries:
                    return "Unknown"
                sleep = 2 ** attempt
                time.sleep(sleep)

    @staticmethod
    def _log_sector_stats(df: pd.DataFrame) -> None:
        cnts = df["sector"].value_counts()
        logger.info("Top sector counts:")
        for sector, n in cnts.head(10).items():
            logger.info("  %-20s %5d", sector, n)

    # ---------- helpers for Wikipedia merge (unchanged) ----------
    def load_sp500_sectors(self) -> Dict[str, str]:
        try:
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            df = pd.read_html(url)[0]
            df["Symbol"] = df["Symbol"].str.replace(
                ".", "-", regex=False).str.strip()
            return dict(zip(df["Symbol"], df["GICS Sector"]))
        except Exception as e:  # noqa: BLE001
            logger.error("Wikipedia scrape failed: %s", e)
            return {}

    def merge_sector_sources(self) -> pd.DataFrame:
        sectors = {}
        if self.output_csv.exists():
            df = pd.read_csv(self.output_csv, dtype=str)
            sectors.update(dict(zip(df["symbol"], df["sector"])))

        sectors.update(self.load_sp500_sectors())
        out_df = pd.DataFrame([{"symbol": s, "sector": sec}
                              for s, sec in sectors.items()])
        merged_path = DATA_DIR / "ticker_sector_merged.csv"
        out_df.to_csv(merged_path, index=False)
        logger.info("Merged sector data → %s rows", len(out_df))
        return out_df


# CLI helper (same flags)
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build ticker data")
    parser.add_argument("--tickers", action="store_true")
    parser.add_argument("--sectors", action="store_true")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if args.all or args.tickers:
        TickerListBuilder().build_from_datahub()
    if args.all or args.sectors:
        SectorBuilder().build_sector_mapping()
