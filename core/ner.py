"""
Improved NER for financial ticker extraction with better accuracy
"""

import re
import pandas as pd
from typing import Dict, List, Set, Tuple
import logging
from collections import defaultdict
from pathlib import Path
import spacy
from spacy.matcher import PhraseMatcher

logger = logging.getLogger(__name__)

# Enhanced regex patterns for better ticker detection
PATTERNS = {
    # Parenthetical mentions with high confidence (e.g., "Apple (AAPL)")
    'parenthetical': re.compile(r'\(([A-Z]{1,5}(?:\.[A-Z])?)\)', re.IGNORECASE),

    # Dollar sign mentions (e.g., "$AAPL")
    'dollar_sign': re.compile(r'\$([A-Z]{1,5}(?:\.[A-Z])?)\b', re.IGNORECASE),

    # Exchange mentions (e.g., "AAPL on NYSE")
    'exchange': re.compile(
        r'\b([A-Z]{1,5}(?:\.[A-Z])?)\s+(?:on|at|traded\s+on|listed\s+on)\s+'
        r'(?:the\s+)?(?:NYSE|NASDAQ|AMEX|OTC|exchange)',
        re.IGNORECASE
    ),

    # Stock/shares mentions (e.g., "MSFT stock", "AAPL shares")
    'stock_mention': re.compile(
        r'\b([A-Z]{1,5}(?:\.[A-Z])?)\s+'
        r'(?:stock|shares?|equity|equities|securities|ticker|symbol)',
        re.IGNORECASE
    ),

    # Financial actions (e.g., "buy AAPL", "upgraded MSFT")
    'action': re.compile(
        r'(?:buy|sell|hold|upgrade[d]?|downgrade[d]?|rate[d]?|target|'
        r'initiate[d]?|reiterate[d]?|maintain[s]?|cut|raise[d]?)\s+'
        r'(?:on\s+)?([A-Z]{1,5}(?:\.[A-Z])?)\b',
        re.IGNORECASE
    ),

    # Analyst mentions (e.g., "AAPL PT", "MSFT price target")
    'analyst': re.compile(
        r'\b([A-Z]{1,5}(?:\.[A-Z])?)\s+'
        r'(?:PT|price\s+target|target\s+price|rating)',
        re.IGNORECASE
    ),

    # Trading mentions (e.g., "AAPL trading at", "MSFT closed")
    'trading': re.compile(
        r'\b([A-Z]{1,5}(?:\.[A-Z])?)\s+'
        r'(?:trading|traded|trade[s]?|close[d]?|open[ed]?|gain[ed]?|'
        r'fell|rose|jump[ed]?|surg[ed]?|tumbl[ed]?|plunge[d]?)\s+'
        r'(?:at|to|by|up|down)?',
        re.IGNORECASE
    ),

    # Possessive mentions (e.g., "Apple's earnings", "Microsoft's CEO")
    'possessive': re.compile(
        r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)'s\s+"
        r"(?:earnings?|revenue|sales|CEO|CFO|stock|shares?)",
        re.IGNORECASE
    ),
}

# Financial context words for validation
FINANCIAL_CONTEXTS = {
    'price', 'stock', 'share', 'shares', 'equity', 'market', 'trading',
    'earnings', 'revenue', 'profit', 'loss', 'gain', 'analyst', 'upgrade',
    'downgrade', 'buy', 'sell', 'hold', 'target', 'dividend', 'yield',
    'portfolio', 'investment', 'investor', 'traded', 'exchange', 'nasdaq',
    'nyse', 'ticker', 'symbol', 'ipo', 'offering', 'valuation', 'pe',
    'ratio', 'eps', 'guidance', 'forecast', 'quarter', 'fiscal', 'beat',
    'miss', 'consensus', 'estimate', 'analyst', 'rating', 'outperform',
    'underperform', 'neutral', 'bullish', 'bearish', 'long', 'short'
}

# Common names that should NOT be tickers
EXCLUDED_NAMES = {
    'COOK', 'TIM', 'ELON', 'JEFF', 'WARREN', 'BILL', 'STEVE', 'MARK',
    'PETER', 'PAUL', 'MARY', 'JOHN', 'JAMES', 'ROBERT', 'MICHAEL',
    'DAVID', 'RICHARD', 'CHARLES', 'JOSEPH', 'THOMAS', 'DANIEL',
    'CEO', 'CFO', 'CTO', 'COO', 'IPO', 'SEC', 'FDA', 'EPA', 'DOJ',
    'FBI', 'CIA', 'NSA', 'IRS', 'FTC', 'FCC', 'DOT', 'HHS', 'DHS',
    'AI', 'ML', 'API', 'IT', 'HR', 'PR', 'IR', 'VC', 'PE', 'LP'
}


class ImprovedNER:
    """
    Improved NER for financial ticker extraction with better pattern matching
    and context validation
    """

    def __init__(self, ticker_csv_path: str = None, debug: bool = False):
        """Initialize improved NER"""
        self.debug = debug
        self.extraction_stats = defaultdict(int)

        # Load valid tickers
        self.valid_tickers = self._load_valid_tickers(ticker_csv_path)
        self.company_to_ticker = self._build_company_mappings()

        # Initialize spaCy for advanced NLP
        self._init_spacy()

        logger.info(
            f"Initialized ImprovedNER with {len(self.valid_tickers)} valid tickers")

    def _load_valid_tickers(self, ticker_csv_path: str = None) -> Set[str]:
        """Load valid ticker symbols from CSV"""
        valid_tickers = set()

        if ticker_csv_path and Path(ticker_csv_path).exists():
            try:
                df = pd.read_csv(ticker_csv_path)
                if 'symbol' in df.columns:
                    valid_tickers = set(df['symbol'].str.upper().dropna())
                elif 'Symbol' in df.columns:
                    valid_tickers = set(df['Symbol'].str.upper().dropna())
                logger.info(
                    f"Loaded {len(valid_tickers)} tickers from {ticker_csv_path}")
            except Exception as e:
                logger.error(f"Error loading ticker CSV: {e}")

        # Add some common tickers if none loaded
        if not valid_tickers:
            valid_tickers = {
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA',
                'JPM', 'V', 'JNJ', 'WMT', 'PG', 'MA', 'HD', 'DIS', 'BAC',
                'ADBE', 'NFLX', 'CRM', 'PFE', 'CSCO', 'INTC', 'ORCL',
                'BRK.A', 'BRK.B', 'C', 'F', 'T', 'X'  # Include single letters
            }

        return valid_tickers

    def _build_company_mappings(self) -> Dict[str, str]:
        """Build company name to ticker mappings"""
        # Common company mappings
        mappings = {
            'apple': 'AAPL',
            'apple inc': 'AAPL',
            'apple incorporated': 'AAPL',
            'microsoft': 'MSFT',
            'microsoft corp': 'MSFT',
            'microsoft corporation': 'MSFT',
            'amazon': 'AMZN',
            'amazon.com': 'AMZN',
            'google': 'GOOGL',
            'alphabet': 'GOOGL',
            'meta': 'META',
            'meta platforms': 'META',
            'facebook': 'META',
            'tesla': 'TSLA',
            'tesla motors': 'TSLA',
            'nvidia': 'NVDA',
            'berkshire': 'BRK.B',
            'berkshire hathaway': 'BRK.B',
            'jp morgan': 'JPM',
            'jpmorgan': 'JPM',
            'jpmorgan chase': 'JPM',
            'bank of america': 'BAC',
            'wells fargo': 'WFC',
            'goldman sachs': 'GS',
            'morgan stanley': 'MS',
            'citigroup': 'C',
            'cisco': 'CSCO',
            'intel': 'INTC',
            'oracle': 'ORCL',
            'salesforce': 'CRM',
        }

        return mappings

    def _init_spacy(self):
        """Initialize spaCy for entity recognition"""
        try:
            self.nlp = spacy.load("en_core_web_sm", disable=["parser"])
            self.has_spacy = True
        except:
            logger.warning("SpaCy not available, using regex-only extraction")
            self.nlp = None
            self.has_spacy = False

    def extract_tickers(self, text: str, title: str = "") -> List[Tuple[str, float]]:
        """
        Extract ticker symbols with confidence scores

        Args:
            text: Article content
            title: Article title

        Returns:
            List of (ticker, confidence) tuples
        """
        # Combine title and text for analysis
        full_text = f"{title} {text}"

        # Results with confidence scores
        ticker_scores = defaultdict(float)

        # 1. High confidence patterns (0.9)
        high_conf_patterns = ['parenthetical', 'dollar_sign']
        for pattern_name in high_conf_patterns:
            pattern = PATTERNS[pattern_name]
            for match in pattern.finditer(full_text):
                ticker = match.group(1).upper()
                if self._is_valid_ticker(ticker):
                    ticker_scores[ticker] = max(ticker_scores[ticker], 0.9)
                    self.extraction_stats[pattern_name] += 1
                    if self.debug:
                        logger.debug(f"Found {ticker} via {pattern_name}")

        # 2. Medium-high confidence patterns (0.8)
        medium_high_patterns = ['exchange',
                                'stock_mention', 'action', 'analyst']
        for pattern_name in medium_high_patterns:
            pattern = PATTERNS[pattern_name]
            for match in pattern.finditer(full_text):
                ticker = match.group(1).upper()
                if self._is_valid_ticker(ticker):
                    ticker_scores[ticker] = max(ticker_scores[ticker], 0.8)
                    self.extraction_stats[pattern_name] += 1

        # 3. Medium confidence patterns (0.7)
        medium_patterns = ['trading']
        for pattern_name in medium_patterns:
            pattern = PATTERNS[pattern_name]
            for match in pattern.finditer(full_text):
                ticker = match.group(1).upper()
                if self._is_valid_ticker(ticker) and self._has_financial_context(full_text, match.start()):
                    ticker_scores[ticker] = max(ticker_scores[ticker], 0.7)
                    self.extraction_stats[pattern_name] += 1

        # 4. Company name resolution (0.8)
        text_lower = full_text.lower()
        for company_name, ticker in self.company_to_ticker.items():
            if company_name in text_lower and self._is_valid_ticker(ticker):
                ticker_scores[ticker] = max(ticker_scores[ticker], 0.8)
                self.extraction_stats['company_name'] += 1

        # 5. Handle possessive forms (e.g., "Apple's")
        for match in PATTERNS['possessive'].finditer(full_text):
            company = match.group(1).lower()
            if company in self.company_to_ticker:
                ticker = self.company_to_ticker[company]
                if self._is_valid_ticker(ticker):
                    ticker_scores[ticker] = max(ticker_scores[ticker], 0.8)
                    self.extraction_stats['possessive'] += 1

        # 6. Use spaCy for additional entity recognition
        if self.has_spacy and self.nlp:
            doc = self.nlp(full_text)
            for ent in doc.ents:
                if ent.label_ == "ORG":
                    org_lower = ent.text.lower()
                    if org_lower in self.company_to_ticker:
                        ticker = self.company_to_ticker[org_lower]
                        if self._is_valid_ticker(ticker):
                            ticker_scores[ticker] = max(
                                ticker_scores[ticker], 0.7)
                            self.extraction_stats['spacy_org'] += 1

        # Convert to sorted list
        results = [(ticker, score) for ticker, score in ticker_scores.items()]
        results.sort(key=lambda x: x[1], reverse=True)

        return results

    def _is_valid_ticker(self, ticker: str) -> bool:
        """Check if ticker is valid"""
        ticker = ticker.upper()

        # Exclude common names and non-ticker words
        if ticker in EXCLUDED_NAMES:
            return False

        # If we have a valid ticker list, check it
        if self.valid_tickers:
            return ticker in self.valid_tickers

        # Otherwise, apply basic rules
        # Must be 1-5 letters, possibly with .A or .B suffix
        if not re.match(r'^[A-Z]{1,5}(?:\.[A-Z])?$', ticker):
            return False

        # Single letters need special validation
        if len(ticker) == 1:
            # Only allow known single-letter tickers
            return ticker in {'C', 'F', 'T', 'V', 'X', 'K', 'M', 'O'}

        return True

    def _has_financial_context(self, text: str, position: int, window: int = 50) -> bool:
        """Check if position has financial context nearby"""
        start = max(0, position - window)
        end = min(len(text), position + window)
        context = text[start:end].lower()

        # Check for financial words
        words = set(context.split())
        return bool(words & FINANCIAL_CONTEXTS)

    def test_extraction(self):
        """Test the extraction with known examples"""
        test_cases = [
            ("Apple (AAPL) stock rose 5%", [("AAPL", 0.9)]),
            ("Buy MSFT shares on NYSE", [("MSFT", 0.8)]),
            ("BRK.B trading at $350", [("BRK.B", 0.7)]),
            ("$TSLA surged after earnings", [("TSLA", 0.9)]),
            ("Microsoft announced new products", [("MSFT", 0.8)]),
            ("Tim Cook said Apple is doing well", [
             ("AAPL", 0.8)]),  # Should get AAPL, not COOK
            # Single letter with context
            ("C stock gained on banking news", [("C", 0.8)]),
        ]

        print("Running NER extraction tests...")
        passed = 0

        for text, expected in test_cases:
            results = self.extract_tickers(text)
            result_tickers = {ticker for ticker, _ in results}
            expected_tickers = {ticker for ticker, _ in expected}

            if expected_tickers.issubset(result_tickers):
                passed += 1
                print(f"✅ PASSED: '{text}' → {results}")
            else:
                print(f"❌ FAILED: '{text}' → {results}, expected {expected}")

        print(f"\nTest Summary: {passed}/{len(test_cases)} passed")

        # Print extraction statistics
        print("\nExtraction Statistics:")
        for method, count in sorted(self.extraction_stats.items()):
            print(f"  {method}: {count}")

        return passed == len(test_cases)

    def handle_symbols_array(self, symbols_raw):
        """Handle various formats of symbols array from dataframe"""
        import numpy as np

        if symbols_raw is None:
            return []
        elif isinstance(symbols_raw, np.ndarray):
            symbols_list = symbols_raw.tolist()
            if symbols_list and isinstance(symbols_list[0], (list, np.ndarray)):
                symbols_list = [
                    item for sublist in symbols_list for item in sublist]
            return [str(s).strip() for s in symbols_list if s]
        elif isinstance(symbols_raw, list):
            return [str(s).strip() for s in symbols_raw if s]
        elif isinstance(symbols_raw, str):
            if ',' in symbols_raw:
                return [s.strip() for s in symbols_raw.split(',') if s.strip()]
            return [symbols_raw.strip()] if symbols_raw.strip() else []
        else:
            try:
                return [str(symbols_raw).strip()]
            except:
                return []


# Integration function for your pipeline
def create_improved_ner(ticker_csv_path: str = None) -> ImprovedNER:
    """Create an instance of the improved NER"""
    return ImprovedNER(ticker_csv_path=ticker_csv_path, debug=False)


if __name__ == "__main__":
    # Test the improved NER
    ner = ImprovedNER(debug=True)
    ner.test_extraction()

# Alias for compatibility
UnifiedNER = ImprovedNER