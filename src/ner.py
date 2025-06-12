"""
src/ner.py

Ticker extraction utilities using:
 - metadata from the API
 - a master ticker list (non-ETF US symbols)
 - regex-based fallback that ignores one-letter tokens unless parenthesized.
"""

import re
import pandas as pd
from typing import Dict, Set, List

# Regex to capture 1–5 uppercase letters
TICKER_PATTERN = re.compile(r"\b[A-Z]{1,5}\b")

def load_symbol_list(csv_path: str = "data/master_ticker_list.csv",
                     symbol_col: str = 'symbol') -> Set[str]:
    """
    Load the master ticker list CSV into a set for fast lookup.
    """
    df = pd.read_csv(csv_path, dtype=str)
    return set(df[symbol_col].str.strip().str.upper())

def extract_symbols_from_metadata(article: Dict) -> Set[str]:
    """
    Extract tickers directly from the article's API metadata.
    """
    raw = article.get('symbols', [])
    return {s.upper() for s in raw if isinstance(s, str)}

def extract_symbols_from_text(text: str,
                              dictionary: Set[str]) -> Set[str]:
    """
    Run regex on text to find uppercase tokens, then filter by dictionary.
    Only multi-letter tokens pass; single-letter symbols only if in "(X)" form.
    """
    candidates = set(TICKER_PATTERN.findall(text))
    valid = set()

    # 1) multi-letter tokens
    for tok in candidates:
        if len(tok) > 1 and tok in dictionary:
            valid.add(tok)

    # 2) single-letter tickers only if matched in parentheses
    for tok in candidates:
        if len(tok) == 1 and tok in dictionary and f"({tok})" in text:
            valid.add(tok)

    return valid

def get_combined_symbols(article: Dict,
                         dictionary: Set[str],
                         use_text_extraction: bool = True) -> List[str]:
    """
    Merge metadata tickers + optional text-based extraction into a sorted list.
    """
    symbols = extract_symbols_from_metadata(article)
    if use_text_extraction and dictionary:
        text = article.get('title','') + "\n" + article.get('content','')
        symbols |= extract_symbols_from_text(text, dictionary)
    return sorted(symbols)

if __name__ == '__main__':
    # quick sanity check
    dummy = {
        'symbols': ['AAPL'],
        'title': 'Agilent (A) declares dividend',
        'content': 'Investors love A.'
    }
    dict_set = load_symbol_list()
    print(get_combined_symbols(dummy, dict_set))
