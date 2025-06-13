# src/ner.py

"""
NER for financial news:
 - Loads symbol→company_name mapping from CSV.
 - Regex extracts tickers (1–5 uppercase letters).
 - Single-letter only if parenthesized.
 - ALSO: any multi-letter company_name mention in text yields its ticker.
"""

import re
import pandas as pd
from typing import Dict, List, Set

# Regex to catch tokens of 1–5 uppercase letters
TICKER_PATTERN = re.compile(r"\b[A-Z]{1,5}\b")

def load_symbol_list(
    csv_path: str = "data/master_ticker_list.csv"
) -> Dict[str, str]:
    """
    Returns dict mapping SYMBOL (upper) -> company_name (lowercase).
    """
    df = pd.read_csv(csv_path, dtype=str)
    df["symbol"]       = df["symbol"].str.upper()
    df["company_name"] = df["company_name"].str.lower()
    return dict(zip(df["symbol"], df["company_name"]))

def extract_symbols_from_metadata(article: dict) -> Set[str]:
    """
    Trust the API's 'symbols' list first (already in metadata).
    """
    return {s.upper() for s in article.get("symbols", []) if isinstance(s, str)}

def extract_symbols_from_text(
    text: str,
    dictionary: Dict[str, str]
) -> Set[str]:
    """
    Extract tickers from free text:
      1. Multi-letter tokens if in dictionary.
      2. Single-letter only if seen in "(X)".
      3. Any company_name substring match → add its symbol.
    """
    candidates = set(TICKER_PATTERN.findall(text))
    valid = set()
    lower_text = text.lower()

    # 1) Multi-letter by regex
    for tok in candidates:
        if len(tok) > 1 and tok in dictionary:
            valid.add(tok)

    # 2) Single-letter in parentheses
    for tok in candidates:
        if len(tok) == 1 and tok in dictionary:
            if f"({tok})" in text:
                valid.add(tok)

    # 3) Company-name substring match
    for sym, cname in dictionary.items():
        # only for multi-letter symbols (single-letter would be noise here)
        if len(sym) > 1 and cname in lower_text:
            valid.add(sym)

    return valid

def get_combined_symbols(
    article: dict,
    dictionary: Dict[str, str],
    use_text_extraction: bool = True
) -> List[str]:
    """
    Merge metadata + text-extracted symbols into a sorted list.
    """
    syms = extract_symbols_from_metadata(article)
    if use_text_extraction:
        full = article.get("title","") + "\n" + article.get("content","")
        syms |= extract_symbols_from_text(full, dictionary)
    return sorted(syms)
