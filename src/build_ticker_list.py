import pandas as pd
import os

# 1. DataHub URLs for NYSE and other exchange listings
URL_NYSE  = "https://datahub.io/core/nyse-other-listings/r/nyse-listed.csv"
URL_OTHER = "https://datahub.io/core/nyse-other-listings/r/other-listed.csv"

# 2. Load both CSV files into DataFrames
#    dtype=str ensures symbols remain strings (including leading zeros, etc.)
df_nyse  = pd.read_csv(URL_NYSE, dtype=str)
df_other = pd.read_csv(URL_OTHER, dtype=str)

# 3. Combine the two datasets
df = pd.concat([df_nyse, df_other], ignore_index=True)

# 4. Drop ETFs (rows where 'ETF' column == 'Y')
if 'ETF' in df.columns:
    df = df[df['ETF'] != 'Y']

# 5. Select only the ticker ('ACT Symbol') and company name columns, rename to 'symbol' and 'company_name'
df = df[['ACT Symbol', 'Company Name']].rename(
    columns={'ACT Symbol': 'symbol', 'Company Name': 'company_name'}
)

# 6. Remove any symbols containing '.' or '$' (e.g. 'AACT.U', 'ABR$D')
df = df[~df['symbol'].str.contains(r"[\.$]")]

# 7. Deduplicate by symbol, keeping the first occurrence
df = df.drop_duplicates(subset='symbol').reset_index(drop=True)

# 8. Ensure the output directory exists
os.makedirs('data', exist_ok=True)

# 9. Save the cleaned master ticker list to CSV
out_path = 'data/master_ticker_list.csv'
df.to_csv(out_path, index=False)
print(f"✅ Saved {len(df)} unique tickers to {out_path}")
