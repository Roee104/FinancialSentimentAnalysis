#!/usr/bin/env python
"""
utils_fetch_wikidata_aliases.py  –  Wikidata symbol-alias fetcher  (paged)

Run:
  python scripts/utils_fetch_wikidata_aliases.py --out data/ticker_alias_table.csv
"""

from __future__ import annotations
import argparse
import csv
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd
from SPARQLWrapper import JSON, SPARQLWrapper, POST
from tqdm import tqdm

logging.basicConfig(level=logging.INFO,
                    format="[%(levelname)s] %(message)s", stream=sys.stdout)
LOG = logging.getLogger("alias-fetch")

BASE_QUERY = """
SELECT DISTINCT ?ticker ?label WHERE {{
  ?item p:P414 ?listing .
  ?listing pq:P249 ?ticker .
  FILTER(REGEX(?ticker, "^[A-Z]{{1,5}}$"))      # ← doubled braces
  {{
    ?item rdfs:label ?label .
    FILTER (LANG(?label) = "en")
  }} UNION {{
    ?item skos:altLabel ?label .
    FILTER (LANG(?label) = "en")
  }}
}}
LIMIT {limit} OFFSET {offset}
"""


def fetch_page(limit: int, offset: int) -> list[tuple[str, str]]:
    q = BASE_QUERY.format(limit=limit, offset=offset)
    sparql = SPARQLWrapper(
        "https://query.wikidata.org/sparql", agent="FinSent-Pipeline/1.0 (https://github.com/Roee104/FinancialSentimentAnalysis; mailto:Roee104@gmail.com)"
    )
    sparql.setMethod(POST)
    sparql.setQuery(q)
    sparql.setReturnFormat(JSON)
    data = sparql.query().convert()
    rows = [(b["ticker"]["value"].upper(), b["label"]["value"].lower())
            for b in data["results"]["bindings"]]
    return rows


def fetch_all(limit: int = 5000, sleep_s: float = 1.0) -> list[tuple[str, str]]:
    LOG.info("Querying Wikidata in pages of %d …", limit)
    offset = 0
    rows: list[tuple[str, str]] = []
    while True:
        page = fetch_page(limit, offset)
        if not page:
            break
        rows.extend(page)
        LOG.info("… got %d rows (total %d)", len(page), len(rows))
        offset += limit
        time.sleep(sleep_s)          # good-citizen pause
    return rows


def consolidate(rows: list[tuple[str, str]]) -> pd.DataFrame:
    bucket: defaultdict[str, set[str]] = defaultdict(set)
    for sym, alias in rows:
        clean = alias.replace(",", "").replace("'", "").strip()
        if len(clean) < 3:
            continue
        bucket[sym].add(clean)
    data = [(s, "|".join(sorted(a))) for s, a in bucket.items()]
    LOG.info("Consolidated → %d unique symbols", len(data))
    return pd.DataFrame(data, columns=["symbol", "aliases"])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Fetch ticker-alias table from Wikidata and save CSV")
    parser.add_argument("--out", type=Path, default=Path("data/ticker_alias_table.csv"),
                        help="CSV path (default: data/ticker_alias_table.csv)")
    args = parser.parse_args(argv)

    raw_rows = fetch_all()
    LOG.info("Fetched %d (symbol, alias) pairs", len(raw_rows))
    df = consolidate(raw_rows)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False, quoting=csv.QUOTE_MINIMAL)
    LOG.info("✅ Saved %s (%.1f KB)", args.out, args.out.stat().st_size / 1024)


if __name__ == "__main__":
    main()
