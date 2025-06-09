"""
ner.py

Provides ticker extraction utilities for financial news articles.
Supports loading a symbol dictionary and extracting known symbols from article metadata and text.
"""

import re
import pandas as pd
from typing import List, Set, Dict

# Regex pattern to capture potential tickers: 1-5 uppercase letters
TICKER_PATTERN = re.compile(r"\b[A-Z]{1,5}\b")


def load_symbol_list(csv_path: str, symbol_col: str = 'Symbol') -> Set[str]:
    """
    Load a CSV of tickers into a set for quick lookup.

    Args:
        csv_path (str): Path to CSV file containing a column of ticker symbols.
        symbol_col (str): Name of the column containing symbols. Defaults to 'Symbol'.

    Returns:
        Set[str]: Uppercased set of ticker strings.
    """
    df = pd.read_csv(csv_path)
    # Ensure symbols are strings, uppercase, stripped
    return set(df[symbol_col].astype(str).str.upper().str.strip())


def extract_symbols_from_metadata(article: Dict) -> Set[str]:
    """
    Extract tickers directly from the article's metadata 'symbols' field.

    Args:
        article (Dict): Article dict with optional 'symbols' list.

    Returns:
        Set[str]: Uppercased ticker symbols from metadata.
    """
    raw = article.get('symbols', [])
    return {s.upper() for s in raw if isinstance(s, str)}


def extract_symbols_from_text(text: str, dictionary: Set[str]) -> Set[str]:
    """
    Run a regex over the text to find uppercase 1-5 letter tokens,
    then filter by the provided dictionary.

    Args:
        text (str): Full article text (headline + body).
        dictionary (Set[str]): Set of valid tickers.

    Returns:
        Set[str]: Valid tickers found in text.
    """
    candidates = set(TICKER_PATTERN.findall(text))
    # Filter by dictionary membership
    return {tok for tok in candidates if tok in dictionary}


def get_combined_symbols(
    article: Dict,
    dictionary: Set[str] = None,
    use_text_extraction: bool = False
) -> List[str]:
    """
    Combine symbols from metadata and (optionally) text-based extraction.

    Args:
        article (Dict): Article dict containing 'symbols' and text fields.
        dictionary (Set[str], optional): Symbol dictionary for text extraction.
        use_text_extraction (bool): Whether to extract from text.

    Returns:
        List[str]: Sorted list of unique tickers for this article.
    """
    symbols = extract_symbols_from_metadata(article)
    if use_text_extraction and dictionary:
        text = article.get('title', '') + '\n' + article.get('content', '')
        symbols |= extract_symbols_from_text(text, dictionary)
    # Return a sorted list for consistency
    return sorted(symbols)


if __name__ == '__main__':
    # Quick self-test
    sample_article = {
        'symbols': ['AAPL'],
        'title': 'Apple launches new iPhone',
        'content': 'Meanwhile MSFT saw strong Azure growth.'
    }
    # Load a dummy dict
    dummy_dict = {'AAPL', 'MSFT', 'GOOG'}
    print(get_combined_symbols(sample_article, dictionary=dummy_dict, use_text_extraction=True))
