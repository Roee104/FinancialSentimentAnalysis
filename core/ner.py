# core/ner.py
"""
Unified Enhanced NER for financial news with robust ticker extraction
"""

import re
import pandas as pd
import numpy as np
from typing import Dict, List, Set, Tuple, Union, Optional
import logging
from collections import defaultdict
from pathlib import Path
import functools
import spacy
from spacy.matcher import PhraseMatcher

from config.settings import (
    MASTER_TICKER_LIST, NER_CONFIG, EXCLUDED_WORDS,
    FINANCIAL_CONTEXTS, DATA_DIR
)

logger = logging.getLogger(__name__)

# Singleton cache for symbol dictionary
_SYMBOL_CACHE = {}

# Updated regex patterns for ticker detection
# Matches single-letter and multi-letter tickers
TICKER_PATTERN = re.compile(r"\b[A-Z]{1,5}\b", re.IGNORECASE)
MULTI_WORD_TICKER_PATTERN = re.compile(
    r"\b[A-Z]{1,5}(?:\.[A-Z])?\b", re.IGNORECASE)  # Handles BRK.B
PARENTHETICAL_PATTERN = re.compile(
    r"\(([A-Z]{1,5}(?:\.[A-Z])?)\)", re.IGNORECASE)
EXCHANGE_PATTERN = re.compile(
    r"\b([A-Z]{2,5}(?:\.[A-Z])?)\s+(?:on|traded\s+on|listed\s+on)\s+(?:NYSE|NASDAQ|the\s+exchange)",
    re.IGNORECASE
)
STOCK_MENTION_PATTERN = re.compile(
    r"\b([A-Z]{2,5}(?:\.[A-Z])?)\s+(?:stock|shares|equity|securities)",
    re.IGNORECASE
)
PRICE_PATTERN = re.compile(r"\$([A-Z]{2,5}(?:\.[A-Z])?)\b")
FINANCIAL_ACTION_PATTERN = re.compile(
    r"(?:buy|sell|hold|upgrade|downgrade|target|rating)\s+([A-Z]{2,5}(?:\.[A-Z])?)\b",
    re.IGNORECASE
)


class UnifiedNER:
    """
    Unified Named Entity Recognition for financial tickers
    Combines enhanced extraction with numpy array handling
    """

    def __init__(self, ticker_csv_path: str = None):
        """
        Initialize NER with ticker dictionary

        Args:
            ticker_csv_path: Path to master ticker list CSV
        """
        self.ticker_csv_path = Path(ticker_csv_path or MASTER_TICKER_LIST)

        # Use cached symbol dict if available
        cache_key = str(self.ticker_csv_path)
        if cache_key in _SYMBOL_CACHE:
            self.symbol_dict = _SYMBOL_CACHE[cache_key]
            logger.debug(
                f"Using cached symbol dictionary ({len(self.symbol_dict)} symbols)")
        else:
            self.symbol_dict = self._load_symbol_list()
            _SYMBOL_CACHE[cache_key] = self.symbol_dict

        self.company_to_symbol = self._build_company_lookup()
        self.financial_context_pattern = self._build_context_pattern()

        # Initialize spaCy components
        self._init_spacy()

        # Configuration
        self.config = NER_CONFIG.copy()

        # Statistics tracking
        self.extraction_stats = defaultdict(int)

        logger.info(f"Initialized NER with {len(self.symbol_dict)} symbols")

    def _init_spacy(self):
        """Initialize spaCy NLP and PhraseMatcher"""
        try:
            self.nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
            self.phrase_matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")

            # Add company names to phrase matcher
            company_patterns = []
            # Limit for performance
            for company_name in list(self.company_to_symbol.keys())[:1000]:
                doc = self.nlp(company_name)
                company_patterns.append(doc)

            if company_patterns:
                self.phrase_matcher.add("COMPANY_NAMES", company_patterns)

            logger.debug("Initialized spaCy components")
        except Exception as e:
            logger.warning(f"Failed to initialize spaCy: {e}")
            self.nlp = None
            self.phrase_matcher = None

    def _load_symbol_list(self) -> Dict[str, str]:
        """Load symbol->company mapping"""
        try:
            df = pd.read_csv(self.ticker_csv_path, dtype=str)
            df["symbol"] = df["symbol"].str.upper().str.strip()
            df["company_name"] = df["company_name"].str.lower().str.strip()
            symbol_dict = dict(zip(df["symbol"], df["company_name"]))
            logger.info(
                f"Loaded {len(symbol_dict)} symbols from {self.ticker_csv_path}")
            return symbol_dict
        except FileNotFoundError:
            logger.warning(f"Ticker list not found at {self.ticker_csv_path}")
            return {}
        except Exception as e:
            logger.error(f"Error loading symbol list: {e}")
            return {}

    def _build_company_lookup(self) -> Dict[str, str]:
        """Build company_name->symbol lookup with better matching"""
        lookup = {}
        for symbol, company in self.symbol_dict.items():
            company_words = company.split()

            # Full company name
            lookup[company] = symbol

            # Company name without common suffixes
            cleaned_company = re.sub(
                r'\b(inc|corp|corporation|ltd|limited|plc|co|company)\b\.?', '', company, flags=re.IGNORECASE).strip()
            if cleaned_company and cleaned_company != company:
                lookup[cleaned_company] = symbol

            # First 2-3 significant words
            if len(company_words) >= 2:
                short_name = " ".join(company_words[:2])
                if len(short_name) > 5:  # Avoid too short names
                    lookup[short_name] = symbol

                if len(company_words) >= 3:
                    medium_name = " ".join(company_words[:3])
                    if len(medium_name) > 8:
                        lookup[medium_name] = symbol

        logger.info(f"Built company lookup with {len(lookup)} entries")
        return lookup

    def _build_context_pattern(self) -> re.Pattern:
        """Build regex pattern for financial contexts"""
        context_pattern = "|".join(re.escape(ctx)
                                   for ctx in FINANCIAL_CONTEXTS)
        return re.compile(f"\\b({context_pattern})\\b", re.IGNORECASE)

    def clean_symbol(self, symbol: str) -> str:
        """Remove exchange suffixes from symbols"""
        symbol = symbol.upper().strip()
        for suffix in self.config["exchange_suffixes"]:
            if symbol.endswith(suffix):
                return symbol[:-len(suffix)]
        return symbol

    def normalize_symbols_list(self, symbols_list: List[str]) -> List[str]:
        """Clean and normalize a list of symbols"""
        normalized = []
        seen = set()
        for symbol in symbols_list:
            clean_sym = self.clean_symbol(symbol)
            if clean_sym and clean_sym not in seen:
                normalized.append(clean_sym)
                seen.add(clean_sym)
        return normalized

    def handle_symbols_array(self, symbols_raw: Union[None, str, list, np.ndarray]) -> List[str]:
        """
        Convert various symbol formats to list, handling numpy arrays properly

        Args:
            symbols_raw: Raw symbols in various formats

        Returns:
            List of symbol strings
        """
        if symbols_raw is None:
            return []
        elif isinstance(symbols_raw, np.ndarray):
            return symbols_raw.tolist()
        elif isinstance(symbols_raw, list):
            return symbols_raw
        elif isinstance(symbols_raw, str):
            if symbols_raw.startswith('[') and symbols_raw.endswith(']'):
                try:
                    return eval(symbols_raw)
                except:
                    return []
            return [symbols_raw] if symbols_raw else []
        elif pd.isna(symbols_raw):
            return []
        else:
            try:
                return list(symbols_raw)
            except:
                return []

    def extract_symbols(self,
                        article: Dict,
                        min_confidence: float = None,
                        use_metadata: bool = None) -> List[Tuple[str, float]]:
        """
        Main extraction method for symbols from article

        Args:
            article: Article dict with title, content, symbols
            min_confidence: Minimum confidence threshold
            use_metadata: Whether to include metadata symbols

        Returns:
            List of (symbol, confidence) tuples
        """
        min_confidence = min_confidence or self.config["min_confidence"]
        use_metadata = use_metadata if use_metadata is not None else self.config[
            "use_metadata"]

        all_symbols = {}  # symbol -> max confidence

        # Handle metadata symbols
        if use_metadata:
            metadata_symbols = self.handle_symbols_array(
                article.get("symbols", []))
            if metadata_symbols:
                clean_metadata = self.normalize_symbols_list(metadata_symbols)
                for sym in clean_metadata:
                    # High confidence for metadata
                    all_symbols[sym] = max(all_symbols.get(sym, 0), 0.95)
                self.extraction_stats['metadata'] += len(clean_metadata)

        # Extract from text
        title = article.get("title", "")
        content = article.get("content", "")
        full_text = f"{title}\n\n{content}"

        if full_text.strip():
            text_symbols = self.extract_symbols_with_confidence(full_text)

            # Update with higher confidence
            for symbol, confidence in text_symbols:
                if confidence >= min_confidence:
                    all_symbols[symbol] = max(
                        all_symbols.get(symbol, 0), confidence)

        # Return as sorted list of tuples
        return sorted([(sym, conf) for sym, conf in all_symbols.items()],
                      key=lambda x: x[1], reverse=True)

    def extract_symbols_with_confidence(self, text: str) -> List[Tuple[str, float]]:
        """
        Extract symbols with confidence scores

        Returns:
            List of (symbol, confidence) tuples
        """
        if not text or not isinstance(text, str):
            return []

        results = {}  # symbol -> max confidence

        # Extract different categories
        high_conf = self._extract_high_confidence_tickers(text)
        medium_conf = self._extract_medium_confidence_tickers(text)
        company_mentions = self._extract_company_mentions(text)
        single_letter = self._extract_single_letter_tickers(text)

        # High confidence (0.9)
        for symbol in high_conf:
            results[symbol] = max(results.get(symbol, 0), 0.9)

        # Medium confidence (0.7)
        for symbol in medium_conf:
            results[symbol] = max(results.get(symbol, 0), 0.7)

        # Company mentions (0.8)
        for symbol in company_mentions:
            results[symbol] = max(results.get(symbol, 0), 0.8)

        # Single letter (0.6)
        for symbol in single_letter:
            results[symbol] = max(results.get(symbol, 0), 0.6)

        return list(results.items())

    def _extract_high_confidence_tickers(self, text: str) -> Set[str]:
        """Extract tickers with high confidence signals"""
        high_conf = set()

        # Parenthetical mentions
        for match in PARENTHETICAL_PATTERN.finditer(text):
            symbol = match.group(1).upper()
            if self._is_valid_symbol(symbol):
                clean_sym = self.clean_symbol(symbol)
                high_conf.add(clean_sym)
                self.extraction_stats['parenthetical'] += 1

        # Exchange mentions
        for match in EXCHANGE_PATTERN.finditer(text):
            symbol = match.group(1).upper()
            if self._is_valid_symbol(symbol):
                clean_sym = self.clean_symbol(symbol)
                high_conf.add(clean_sym)
                self.extraction_stats['exchange_mention'] += 1

        # Stock/shares mentions
        for match in STOCK_MENTION_PATTERN.finditer(text):
            symbol = match.group(1).upper()
            if self._is_valid_symbol(symbol):
                clean_sym = self.clean_symbol(symbol)
                high_conf.add(clean_sym)
                self.extraction_stats['stock_mention'] += 1

        # Price mentions
        for match in PRICE_PATTERN.finditer(text):
            symbol = match.group(1).upper()
            if self._is_valid_symbol(symbol):
                clean_sym = self.clean_symbol(symbol)
                high_conf.add(clean_sym)
                self.extraction_stats['price_mention'] += 1

        # Financial actions
        for match in FINANCIAL_ACTION_PATTERN.finditer(text):
            symbol = match.group(1).upper()
            if self._is_valid_symbol(symbol):
                clean_sym = self.clean_symbol(symbol)
                high_conf.add(clean_sym)
                self.extraction_stats['action_mention'] += 1

        return high_conf

    def _extract_medium_confidence_tickers(self, text: str) -> Set[str]:
        """Extract tickers with medium confidence"""
        medium_conf = set()

        # Use multi-word pattern for better coverage
        candidates = MULTI_WORD_TICKER_PATTERN.findall(text)

        for candidate in candidates:
            candidate = candidate.upper()
            if self._is_valid_symbol(candidate):
                clean_sym = self.clean_symbol(candidate)

                # Check financial context
                context_window = self._extract_context_window(text, candidate)
                if self._has_financial_context(context_window):
                    medium_conf.add(clean_sym)
                    self.extraction_stats['context_based'] += 1
                elif len(clean_sym) >= 3:  # Longer tickers without explicit context
                    medium_conf.add(clean_sym)
                    self.extraction_stats['multi_letter'] += 1

        return medium_conf

    def _extract_company_mentions(self, text: str) -> Set[str]:
        """Extract symbols based on company names using spaCy PhraseMatcher"""
        company_mentions = set()

        if self.phrase_matcher and self.nlp:
            try:
                doc = self.nlp(text.lower())
                matches = self.phrase_matcher(doc)

                for match_id, start, end in matches:
                    span = doc[start:end]
                    company_text = span.text

                    if company_text in self.company_to_symbol:
                        symbol = self.company_to_symbol[company_text]
                        company_mentions.add(symbol)
                        self.extraction_stats['company_mention_spacy'] += 1

            except Exception as e:
                logger.debug(f"SpaCy matching error: {e}")

        # Fallback to simple matching
        text_lower = text.lower()

        # Sort by length to avoid partial matches
        sorted_companies = sorted(
            self.company_to_symbol.items(),
            key=lambda x: len(x[0]),
            reverse=True
        )[:500]  # Limit for performance

        for company_name, symbol in sorted_companies:
            # Use word boundaries for better matching
            pattern = r'\b' + re.escape(company_name) + r'\b'
            if re.search(pattern, text_lower):
                company_mentions.add(symbol)
                self.extraction_stats['company_mention'] += 1

        return company_mentions

    def _extract_single_letter_tickers(self, text: str) -> Set[str]:
        """Extract single-letter tickers with strict validation"""
        single_letter = set()

        candidates = TICKER_PATTERN.findall(text)

        for candidate in candidates:
            candidate = candidate.upper()
            if self._is_valid_symbol(candidate) and len(candidate) == 1:
                clean_sym = self.clean_symbol(candidate)

                # Must be in parentheses
                if f"({candidate})" in text:
                    single_letter.add(clean_sym)
                    self.extraction_stats['single_letter_paren'] += 1
                # Or in specific context
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

    def _is_valid_symbol(self, symbol: str) -> bool:
        """Check if symbol is valid"""
        clean_sym = self.clean_symbol(symbol)
        return clean_sym in self.symbol_dict and clean_sym not in EXCLUDED_WORDS

    def _has_financial_context(self, text: str) -> bool:
        """Check if text contains financial context"""
        return bool(self.financial_context_pattern.search(text))

    def _extract_context_window(self, text: str, ticker: str, window_size: int = 100) -> str:
        """Extract context window around ticker"""
        ticker_pos = text.upper().find(ticker.upper())
        if ticker_pos == -1:
            return ""

        start = max(0, ticker_pos - window_size)
        end = min(len(text), ticker_pos + len(ticker) + window_size)
        return text[start:end]

    def get_extraction_stats(self) -> Dict[str, int]:
        """Get extraction statistics"""
        return dict(self.extraction_stats)

    def reset_stats(self):
        """Reset extraction statistics"""
        self.extraction_stats.clear()

    # Unit tests
    def test_ticker_extraction(self):
        """Test ticker extraction functionality"""
        test_cases = [
            ("Apple (AAPL) stock rose 5%", [("AAPL", 0.9)]),
            ("Buy MSFT shares on NYSE", [("MSFT", 0.9)]),
            ("BRK.B trading at $350", [("BRK.B", 0.9)]),
            ("Microsoft announced earnings", [
             ("MSFT", 0.8)]) if "microsoft" in self.company_to_symbol else (None, []),
        ]

        passed = 0
        for text, expected in test_cases:
            if expected is None:
                continue

            article = {"title": text, "content": ""}
            results = self.extract_symbols(article)

            # Check if expected symbols are found
            found_symbols = {sym for sym, _ in results}
            expected_symbols = {sym for sym, _ in expected}

            if expected_symbols.issubset(found_symbols):
                passed += 1
                logger.debug(f"✅ Test passed: '{text}' -> {results}")
            else:
                logger.warning(
                    f"❌ Test failed: '{text}' -> {results}, expected {expected}")

        logger.info(
            f"Ticker extraction tests: {passed}/{len([e for _, e in test_cases if e is not None])} passed")
        return passed == len([e for _, e in test_cases if e is not None])


# Convenience function for backward compatibility
def get_enhanced_symbols(article: dict,
                         ner_extractor: UnifiedNER = None,
                         min_confidence: float = 0.6,
                         use_metadata: bool = True) -> List[str]:
    """
    Extract symbols using enhanced NER (backward compatibility)

    Args:
        article: Article dictionary
        ner_extractor: NER instance (creates new if None)
        min_confidence: Minimum confidence threshold
        use_metadata: Whether to use metadata symbols

    Returns:
        List of extracted symbols
    """
    if ner_extractor is None:
        ner_extractor = UnifiedNER()

    symbol_confidence_pairs = ner_extractor.extract_symbols(
        article=article,
        min_confidence=min_confidence,
        use_metadata=use_metadata
    )

    # Return just symbols for backward compatibility
    return [sym for sym, _ in symbol_confidence_pairs]
