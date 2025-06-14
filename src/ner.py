# src/ner.py

"""
Enhanced NER for financial news:
- Context-aware ticker extraction
- Better handling of ambiguous cases
- Confidence scoring for extracted tickers
- Financial context detection
- Exchange suffix handling (.US, .TO, etc.)
"""

import re
import pandas as pd
import numpy as np  # Added for array handling
from typing import Dict, List, Set, Tuple
import logging
from collections import defaultdict

# Enhanced patterns for ticker detection
TICKER_PATTERN = re.compile(r"\b[A-Z]{1,5}\b")
PARENTHETICAL_PATTERN = re.compile(r"\(([A-Z]{1,5})\)")
EXCHANGE_PATTERN = re.compile(
    r"\b([A-Z]{2,5})(?:\s+(?:on|traded\s+on|listed\s+on)\s+(?:NYSE|NASDAQ|the\s+exchange))", re.IGNORECASE)
STOCK_MENTION_PATTERN = re.compile(
    r"\b([A-Z]{2,5})\s+(?:stock|shares|equity|securities)", re.IGNORECASE)
PRICE_PATTERN = re.compile(r"\$([A-Z]{2,5})\b")
FINANCIAL_ACTION_PATTERN = re.compile(
    r"(?:buy|sell|hold|upgrade|downgrade|target|rating)\s+([A-Z]{2,5})\b", re.IGNORECASE)

# Exchange suffixes to clean
EXCHANGE_SUFFIXES = {'.US', '.TO', '.L', '.PA', '.F', '.HM', '.MI',
                     '.AS', '.MX', '.SA', '.BE', '.DU', '.MU', '.STU', '.XETRA'}

# Common words that are not tickers but appear as uppercase
EXCLUDED_WORDS = {
    'CEO', 'CFO', 'CTO', 'COO', 'IPO', 'SEC', 'FDA', 'EPA', 'GDP', 'USA', 'USD', 'EUR',
    'API', 'AI', 'IT', 'HR', 'PR', 'IR', 'QA', 'RD', 'GAAP', 'EBITDA', 'ROI', 'ROE',
    'MA', 'B2B', 'B2C', 'SAAS', 'COVID', 'Q1', 'Q2', 'Q3', 'Q4', 'YOY', 'QOQ', 'ETF',
    'NYSE', 'NASDAQ', 'SPY', 'QQQ', 'VIX', 'DJIA', 'SP', 'DOW', 'FED', 'FOMC', 'CPI',
    'PPI', 'PMI', 'ISM', 'NFIB', 'ADP', 'BLS', 'BEA', 'FRED', 'OECD', 'IMF', 'ECB',
    'BOJ', 'PBOC', 'SNB', 'BOE', 'RBA', 'BOC', 'RBNZ', 'SARB', 'CBR', 'BCB', 'MXN',
    'CAD', 'GBP', 'JPY', 'CHF', 'AUD', 'NZD', 'ZAR', 'RUB', 'BRL', 'CNY', 'INR',
    'KRW', 'TWD', 'HKD', 'SGD', 'THB', 'MYR', 'IDR', 'PHP', 'VND', 'LAK', 'MMK',
    'REIT', 'ETN', 'SPAC', 'IPO', 'ICO', 'DPO', 'PIPE', 'ESOP', 'DRIP', 'ADR', 'GDR'
}

# Financial context keywords that suggest ticker relevance
FINANCIAL_CONTEXTS = [
    'earnings', 'revenue', 'profit', 'loss', 'guidance', 'outlook', 'forecast',
    'quarterly', 'annual', 'results', 'report', 'announcement', 'statement',
    'dividend', 'split', 'merger', 'acquisition', 'buyback', 'repurchase',
    'upgrade', 'downgrade', 'rating', 'target', 'price', 'valuation',
    'analyst', 'estimate', 'consensus', 'beat', 'miss', 'surprise'
]


def clean_symbol(symbol: str) -> str:
    """Remove exchange suffixes from symbols"""
    symbol = symbol.upper().strip()
    for suffix in EXCHANGE_SUFFIXES:
        if symbol.endswith(suffix):
            return symbol[:-len(suffix)]
    return symbol


def normalize_symbols_list(symbols_list: List[str]) -> List[str]:
    """Clean and normalize a list of symbols"""
    normalized = []
    for symbol in symbols_list:
        clean_sym = clean_symbol(symbol)
        if clean_sym and clean_sym not in normalized:
            normalized.append(clean_sym)
    return normalized


class EnhancedNER:
    """
    Enhanced Named Entity Recognition for financial tickers
    """

    def __init__(self, csv_path: str = "data/master_ticker_list.csv"):
        self.symbol_dict = self._load_symbol_list(csv_path)
        self.company_to_symbol = self._build_company_lookup()
        self.financial_context_pattern = self._build_context_pattern()

        # Statistics tracking
        self.extraction_stats = defaultdict(int)

    def _load_symbol_list(self, csv_path: str) -> Dict[str, str]:
        """Load symbol->company mapping"""
        try:
            df = pd.read_csv(csv_path, dtype=str)
            df["symbol"] = df["symbol"].str.upper().str.strip()
            df["company_name"] = df["company_name"].str.lower().str.strip()
            symbol_dict = dict(zip(df["symbol"], df["company_name"]))
            print(f"✅ Loaded {len(symbol_dict)} symbols from {csv_path}")
            return symbol_dict
        except FileNotFoundError:
            print(
                f"❌ Warning: {csv_path} not found. Creating empty symbol dictionary.")
            return {}
        except Exception as e:
            print(f"❌ Error loading symbol list: {e}")
            return {}

    def _build_company_lookup(self) -> Dict[str, str]:
        """Build company_name->symbol lookup for substring matching"""
        lookup = {}
        for symbol, company in self.symbol_dict.items():
            # Use company names with 2+ words to avoid false matches
            company_words = company.split()
            if len(company_words) >= 2:
                lookup[company] = symbol

                # Also add abbreviated versions for common cases
                if len(company_words) >= 3:
                    # Add first two words
                    short_name = " ".join(company_words[:2])
                    if len(short_name) > 8:  # Only if meaningful length
                        lookup[short_name] = symbol

        print(f"✅ Built company lookup with {len(lookup)} entries")
        return lookup

    def _build_context_pattern(self) -> re.Pattern:
        """Build regex pattern for financial contexts"""
        context_pattern = "|".join(FINANCIAL_CONTEXTS)
        return re.compile(f"({context_pattern})", re.IGNORECASE)

    def _has_financial_context(self, text: str, window_size: int = 50) -> bool:
        """Check if text contains financial context keywords"""
        return bool(self.financial_context_pattern.search(text))

    def _extract_context_window(self, text: str, ticker: str, window_size: int = 100) -> str:
        """Extract context window around ticker mention"""
        ticker_pos = text.upper().find(ticker.upper())
        if ticker_pos == -1:
            return ""

        start = max(0, ticker_pos - window_size)
        end = min(len(text), ticker_pos + len(ticker) + window_size)
        return text[start:end]

    def _is_valid_symbol(self, symbol: str) -> bool:
        """Check if a symbol is valid (exists in our dictionary)"""
        clean_sym = clean_symbol(symbol)
        return clean_sym in self.symbol_dict and clean_sym not in EXCLUDED_WORDS

    def extract_high_confidence_tickers(self, text: str) -> Set[str]:
        """Extract tickers with high confidence based on strong contextual signals"""
        high_conf = set()

        # 1. Parenthetical mentions: "Apple (AAPL)" or "Apple Inc. (AAPL)"
        for match in PARENTHETICAL_PATTERN.finditer(text):
            symbol = match.group(1).upper()
            if self._is_valid_symbol(symbol) and len(symbol) > 1:
                clean_sym = clean_symbol(symbol)
                high_conf.add(clean_sym)
                self.extraction_stats['parenthetical'] += 1

        # 2. Exchange mentions: "AAPL on NYSE", "traded on NASDAQ"
        for match in EXCHANGE_PATTERN.finditer(text):
            symbol = match.group(1).upper()
            if self._is_valid_symbol(symbol):
                clean_sym = clean_symbol(symbol)
                high_conf.add(clean_sym)
                self.extraction_stats['exchange_mention'] += 1

        # 3. Stock/shares mentions: "AAPL stock", "MSFT shares"
        for match in STOCK_MENTION_PATTERN.finditer(text):
            symbol = match.group(1).upper()
            if self._is_valid_symbol(symbol):
                clean_sym = clean_symbol(symbol)
                high_conf.add(clean_sym)
                self.extraction_stats['stock_mention'] += 1

        # 4. Price mentions: "$AAPL"
        for match in PRICE_PATTERN.finditer(text):
            symbol = match.group(1).upper()
            if self._is_valid_symbol(symbol):
                clean_sym = clean_symbol(symbol)
                high_conf.add(clean_sym)
                self.extraction_stats['price_mention'] += 1

        # 5. Financial action mentions: "buy AAPL", "upgrade MSFT"
        for match in FINANCIAL_ACTION_PATTERN.finditer(text):
            symbol = match.group(1).upper()
            if self._is_valid_symbol(symbol):
                clean_sym = clean_symbol(symbol)
                high_conf.add(clean_sym)
                self.extraction_stats['action_mention'] += 1

        return high_conf

    def extract_medium_confidence_tickers(self, text: str) -> Set[str]:
        """Extract tickers with medium confidence"""
        medium_conf = set()

        # Multi-letter uppercase tokens in financial context
        candidates = TICKER_PATTERN.findall(text)

        for candidate in candidates:
            candidate = candidate.upper()
            if self._is_valid_symbol(candidate) and len(candidate) > 1:
                clean_sym = clean_symbol(candidate)

                # Check if it appears in a financial context
                context_window = self._extract_context_window(text, candidate)
                if self._has_financial_context(context_window):
                    medium_conf.add(clean_sym)
                    self.extraction_stats['context_based'] += 1
                elif len(candidate) >= 3:  # 3+ letter tickers are more likely to be valid
                    medium_conf.add(clean_sym)
                    self.extraction_stats['multi_letter'] += 1

        return medium_conf

    def extract_company_mentions(self, text: str) -> Set[str]:
        """Extract symbols based on company name mentions"""
        company_mentions = set()
        text_lower = text.lower()

        # Sort by length (longest first) to avoid partial matches
        sorted_companies = sorted(self.company_to_symbol.items(),
                                  key=lambda x: len(x[0]), reverse=True)

        for company_name, symbol in sorted_companies:
            if company_name in text_lower:
                # Additional validation: word boundaries
                pattern = r'\b' + re.escape(company_name) + r'\b'
                if re.search(pattern, text_lower):
                    company_mentions.add(symbol)
                    self.extraction_stats['company_mention'] += 1

        return company_mentions

    def extract_single_letter_tickers(self, text: str) -> Set[str]:
        """Extract single-letter tickers with strict validation"""
        single_letter = set()

        # Only extract single letters if they appear in parentheses or specific contexts
        candidates = TICKER_PATTERN.findall(text)

        for candidate in candidates:
            candidate = candidate.upper()
            if self._is_valid_symbol(candidate) and len(candidate) == 1:
                clean_sym = clean_symbol(candidate)

                # Must be in parentheses for single letters
                if f"({candidate})" in text:
                    single_letter.add(clean_sym)
                    self.extraction_stats['single_letter_paren'] += 1
                # Or in a very specific financial context
                elif any(pattern in text.lower() for pattern in [
                    f"{candidate.lower()} stock",
                    f"{candidate.lower()} shares",
                    f"${candidate}",
                    f"buy {candidate.lower()}",
                    f"sell {candidate.lower()}"
                ]):
                    single_letter.add(clean_sym)
                    self.extraction_stats['single_letter_context'] += 1

        return single_letter

    def extract_symbols_with_confidence(self, text: str) -> List[Tuple[str, float]]:
        """
        Extract symbols with confidence scores
        Returns: List of (symbol, confidence) tuples
        """
        if not text or not isinstance(text, str):
            return []

        # Extract different categories
        high_conf = self.extract_high_confidence_tickers(text)
        medium_conf = self.extract_medium_confidence_tickers(text)
        company_mentions = self.extract_company_mentions(text)
        single_letter = self.extract_single_letter_tickers(text)

        results = []

        # High confidence (0.9)
        for symbol in high_conf:
            results.append((symbol, 0.9))

        # Medium confidence (0.7) - only if not already in high confidence
        for symbol in medium_conf - high_conf:
            results.append((symbol, 0.7))

        # Company mentions (0.8) - only if not already found
        existing_symbols = {s for s, _ in results}
        for symbol in company_mentions - existing_symbols:
            results.append((symbol, 0.8))

        # Single letter tickers (0.6) - only if not already found
        for symbol in single_letter - existing_symbols:
            results.append((symbol, 0.6))

        return results

    def get_extraction_stats(self) -> Dict[str, int]:
        """Get extraction statistics"""
        return dict(self.extraction_stats)

    def reset_stats(self):
        """Reset extraction statistics"""
        self.extraction_stats.clear()


def get_enhanced_symbols(
    article: dict,
    ner_extractor: EnhancedNER,
    min_confidence: float = 0.6,
    use_metadata: bool = True
) -> List[str]:
    """
    Get symbols using enhanced NER with confidence filtering

    Args:
        article: Article dictionary with title, content, symbols
        ner_extractor: Enhanced NER instance
        min_confidence: Minimum confidence threshold
        use_metadata: Whether to include metadata symbols

    Returns:
        List of extracted symbols (cleaned of exchange suffixes)
    """
    all_symbols = set()

    # Start with metadata symbols if available and requested
    if use_metadata:
        metadata_symbols = article.get("symbols", [])

        # FIX: Handle numpy arrays and other formats
        if metadata_symbols is not None:
            # Convert numpy array to list
            if isinstance(metadata_symbols, np.ndarray):
                metadata_symbols = metadata_symbols.tolist()
            # Handle single string
            elif isinstance(metadata_symbols, str):
                metadata_symbols = [
                    metadata_symbols] if metadata_symbols else []
            # Ensure it's a list
            elif not isinstance(metadata_symbols, list):
                try:
                    metadata_symbols = list(metadata_symbols)
                except:
                    metadata_symbols = []
        else:
            metadata_symbols = []

        if metadata_symbols:
            # Clean exchange suffixes from metadata symbols
            clean_metadata = normalize_symbols_list(metadata_symbols)
            all_symbols.update(clean_metadata)

    # Extract from text
    title = article.get("title", "")
    content = article.get("content", "")
    full_text = f"{title}\n\n{content}"

    if full_text.strip():
        text_symbols = ner_extractor.extract_symbols_with_confidence(full_text)

        # Filter by confidence threshold
        for symbol, confidence in text_symbols:
            if confidence >= min_confidence:
                all_symbols.add(symbol)

    return sorted(list(all_symbols))

# Compatibility function to replace the old NER


def get_combined_symbols(
    article: dict,
    dictionary: Dict[str, str],
    use_text_extraction: bool = True
) -> List[str]:
    """
    Compatibility wrapper for the old NER interface
    """
    # Create a temporary NER instance
    ner = EnhancedNER()

    # Use enhanced extraction
    return get_enhanced_symbols(
        article=article,
        ner_extractor=ner,
        min_confidence=0.6,
        use_metadata=use_text_extraction
    )


if __name__ == "__main__":
    # Test the enhanced NER
    print("Testing Enhanced NER System with Exchange Suffix Handling...")

    ner = EnhancedNER()

    test_articles = [
        {
            "title": "Apple (AAPL) Reports Strong Q4 Earnings",
            "content": "Apple Inc. (AAPL) reported quarterly earnings that beat analyst expectations. The AAPL stock price rose 5% in after-hours trading. Revenue from iPhone sales exceeded forecasts.",
            "symbols": ["AAPL.US"]
        },
        {
            "title": "Tech Stocks Rally",
            "content": "Microsoft shares gained 3% while investors bought MSFT on positive cloud revenue growth. Meanwhile, analysts upgraded GOOGL with a new price target of $150. Tesla performance was mixed with some concerns about delivery numbers.",
            "symbols": ["MSFT.US", "GOOGL.US"]
        }
    ]

    print("\nTesting Enhanced Symbol Extraction with Exchange Suffixes:")
    print("=" * 60)

    for i, article in enumerate(test_articles, 1):
        print(f"\nTest Article {i}:")
        print(f"Title: {article['title']}")
        print(f"Metadata symbols: {article['symbols']}")

        extracted = get_enhanced_symbols(article, ner, min_confidence=0.6)
        print(f"Enhanced extraction: {extracted}")

        # Show cleaned symbols
        cleaned_metadata = normalize_symbols_list(article['symbols'])
        print(f"Cleaned metadata: {cleaned_metadata}")

    print("\n✅ Enhanced NER with exchange suffix handling testing completed!")
