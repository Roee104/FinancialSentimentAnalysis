"""
ner.py

Loads a master ticker→company mapping and extracts tickers from text:
 - multi-letter tokens always
 - single-letter tokens only if in parentheses or co-occur with their company name
"""

import re
import pandas as pd
from typing import Dict, List, Set

# Match 1–5 uppercase letters
TICKER_PATTERN = re.compile(r"\b[A-Z]{1,5}\b")


def load_symbol_list(
    csv_path: str = "data/master_ticker_list.csv"
) -> Dict[str, str]:
    """
    Returns a dict mapping ticker symbol (upper) → company_name (lowercase).
    """
    df = pd.read_csv(csv_path, dtype=str)
    df["symbol"] = df["symbol"].str.upper()
    df["company_name"] = df["company_name"].str.lower()
    return dict(zip(df["symbol"], df["company_name"]))


def extract_symbols_from_metadata(article: dict) -> Set[str]:
    """
    Always trust the API's 'symbols' list first.
    """
    raw = article.get("symbols", [])
    return {s.upper() for s in raw if isinstance(s, str)}


def extract_symbols_from_text(
    text: str,
    dictionary: Dict[str, str]
) -> Set[str]:
    """
    From free text, extract:
      • any multi-letter token that appears in dictionary
      • any single-letter token in dictionary if it appears in parentheses
        or if its company_name appears in the text.
    """
    candidates = set(TICKER_PATTERN.findall(text))
    valid = set()
    lower_text = text.lower()

    # Multi-letter tokens
    for tok in candidates:
        if len(tok) > 1 and tok in dictionary:
            valid.add(tok)

    # Single-letter with context
    for tok in candidates:
        if len(tok) == 1 and tok in dictionary:
            company = dictionary[tok]
            if f"({tok})" in text or company in lower_text:
                valid.add(tok)

    return valid


def get_combined_symbols(
    article: dict,
    dictionary: Dict[str, str],
    use_text_extraction: bool = True
) -> List[str]:
    """
    Merge metadata tickers with text-extracted tickers,
    returning a sorted list of unique symbols.
    """
    syms = extract_symbols_from_metadata(article)
    if use_text_extraction:
        full_text = article.get("title", "") + "\n" + \
            article.get("content", "")
        syms |= extract_symbols_from_text(full_text, dictionary)
    return sorted(syms)
