# ticker_sector_builder.py

import pandas as pd
import yfinance as yf
import time
import os


def build_ticker_sector(
    master_csv: str = "data/master_ticker_list.csv",
    output_csv: str = "data/ticker_sector.csv",
    pause: float = 0.1
):
    """
    Reads master ticker list, fetches sector via yfinance for each symbol,
    and writes a full ticker->sector CSV.
    """
    df = pd.read_csv(master_csv, dtype=str)
    print("Script started")
    print(f"Reading master ticker list from {master_csv}...")
    records = []
    for i, row in df.iterrows():
        sym = row["symbol"].strip()
        print(f"Working on ticker: {sym}")
        try:
            info = yf.Ticker(sym).info
            sector = info.get("sector") or info.get("industry") or "Unknown"
        except Exception:
            sector = "Unknown"

        records.append({
            "symbol":       sym,
            "company_name": row["company_name"],
            "sector":       sector
        })

        # rate-limit pause
        if (i + 1) % 100 == 0:
            time.sleep(pause)

    out_df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    out_df.to_csv(output_csv, index=False)
    print(f"✅ Written {len(out_df)} rows to {output_csv}")


if __name__ == "__main__":
    build_ticker_sector()
