"""
Advanced Financial NER using pre-trained models with CSV integration
Leverages master_ticker_list, ticker_sector, and ticker_alias_table
"""

import re
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Set, Optional
from pathlib import Path
import logging
from collections import defaultdict
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
import torch
from fuzzywuzzy import fuzz
from functools import lru_cache

logger = logging.getLogger(__name__)


class PretrainedFinancialNER:
    """
    Financial NER using FinBERT-NER with comprehensive CSV integration
    """

    def __init__(self,
                 master_ticker_csv: str = None,
                 ticker_sector_csv: str = None,
                 ticker_alias_csv: str = None,
                 use_gpu: bool = True):

        # Define exchange suffixes BEFORE loading data
        self.exchange_suffixes = ['.US', '.NY', '.O', '.OQ', '.N', '.A', '.TO']
        """
        Initialize with CSV files and pre-trained model
        
        Args:
            master_ticker_csv: Path to master_ticker_list.csv
            ticker_sector_csv: Path to ticker_sector.csv  
            ticker_alias_csv: Path to ticker_alias_table.csv
            use_gpu: Whether to use GPU for model inference
        """

        # Set paths
        base_path = Path("/content/FinancialSentimentAnalysis/data")
        self.master_ticker_csv = Path(
            master_ticker_csv or base_path / "master_ticker_list.csv")
        self.ticker_sector_csv = Path(
            ticker_sector_csv or base_path / "ticker_sector.csv")
        self.ticker_alias_csv = Path(
            ticker_alias_csv or base_path / "ticker_alias_table.csv")

        # Load all data
        self._load_ticker_data()

        # Initialize pre-trained NER model
        self._init_ner_model(use_gpu)

        # Compile regex patterns
        self._compile_patterns()

        # Statistics
        self.extraction_stats = defaultdict(int)

        logger.info(f"Initialized PretrainedFinancialNER with:")
        logger.info(f"  - {len(self.ticker_to_company)} tickers")
        logger.info(f"  - {len(self.company_to_ticker)} company mappings")
        logger.info(f"  - {len(self.alias_to_ticker)} aliases")

    def _load_ticker_data(self):
        """Load and process all CSV files"""

        # 1. Load master ticker list
        self.ticker_to_company = {}
        self.company_to_ticker = {}
        self.ticker_set = set()

        if self.master_ticker_csv.exists():
            df = pd.read_csv(self.master_ticker_csv, dtype=str)
            df['symbol'] = df['symbol'].str.upper().str.strip()
            df['company_name'] = df['company_name'].str.strip()

            for _, row in df.iterrows():
                symbol = self._clean_symbol(row['symbol'])
                company = row['company_name'].lower()

                self.ticker_to_company[symbol] = row['company_name']
                self.ticker_set.add(symbol)

                # Create multiple company mappings
                self._add_company_variations(company, symbol)

        # 2. Load sector data (if different from master)
        self.ticker_to_sector = {}
        if self.ticker_sector_csv.exists():
            try:
                df_sector = pd.read_csv(self.ticker_sector_csv, dtype=str)
                for _, row in df_sector.iterrows():
                    symbol = self._clean_symbol(row['symbol'])
                    self.ticker_to_sector[symbol] = row.get(
                        'sector', 'Unknown')

                    # Add any new tickers not in master
                    if symbol not in self.ticker_set:
                        self.ticker_set.add(symbol)
                        if 'company_name' in row:
                            company = row['company_name'].lower()
                            self.ticker_to_company[symbol] = row['company_name']
                            self._add_company_variations(company, symbol)
            except Exception as e:
                logger.warning(f"Error loading sector data: {e}")

        # 3. Load alias table
        self.alias_to_ticker = {}
        self.ticker_aliases = defaultdict(set)

        if self.ticker_alias_csv.exists():
            try:
                df_alias = pd.read_csv(self.ticker_alias_csv, dtype=str)
                for _, row in df_alias.iterrows():
                    symbol = self._clean_symbol(row['symbol'])
                    aliases = str(row['aliases']).split('|')

                    for alias in aliases:
                        alias_clean = alias.strip().lower()
                        if alias_clean:
                            self.alias_to_ticker[alias_clean] = symbol
                            self.ticker_aliases[symbol].add(alias_clean)
            except Exception as e:
                logger.warning(f"Error loading alias data: {e}")

        logger.info(f"Loaded {len(self.ticker_set)} unique tickers")

    def _add_company_variations(self, company: str, symbol: str):
        """Add various company name variations for better matching"""

        # Full company name
        self.company_to_ticker[company] = symbol

        # Remove common suffixes
        suffixes = [
            ' inc', ' incorporated', ' corp', ' corporation', ' company',
            ' limited', ' ltd', ' plc', ' llc', ' lp', ' co', ' & co',
            ' holdings', ' holding', ' group', ' enterprises'
        ]

        cleaned = company
        for suffix in suffixes:
            if cleaned.endswith(suffix):
                cleaned = cleaned[:-len(suffix)].strip()
                if cleaned and len(cleaned) > 2:
                    self.company_to_ticker[cleaned] = symbol

        # Add without punctuation
        no_punct = re.sub(r'[^\w\s]', ' ', company).strip()
        if no_punct != company:
            self.company_to_ticker[no_punct] = symbol

        # First significant word (if multi-word and distinctive)
        words = company.split()
        if len(words) >= 2 and len(words[0]) > 4:
            # Avoid common words
            if words[0] not in ['the', 'general', 'american', 'international', 'united']:
                self.company_to_ticker[words[0]] = symbol

    def _init_ner_model(self, use_gpu: bool):
        """Initialize the pre-trained FinBERT-NER model"""

        # Try different model names
        model_names = [
            "yiyanghkust/finbert-ner",  # Original
            "ProsusAI/finbert",          # Alternative FinBERT
            "dslim/bert-base-NER"        # General BERT NER as fallback
        ]

        self.ner_pipeline = None

        for model_name in model_names:
            try:
                # Check device
                device = 0 if use_gpu and torch.cuda.is_available() else -1

                # Load model and tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForTokenClassification.from_pretrained(
                    model_name)

                # Create pipeline
                self.ner_pipeline = pipeline(
                    "ner",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device=device,
                    aggregation_strategy="simple"
                )

                logger.info(f"Loaded {model_name} model (device: {device})")
                break

            except Exception as e:
                logger.warning(f"Failed to load {model_name}: {e}")
                continue

        if self.ner_pipeline is None:
            logger.warning(
                "All NER models failed to load. Using pattern-based extraction only.")

    def _compile_patterns(self):
        """Compile regex patterns for extraction"""

        # Exchange suffixes to remove
        self.exchange_suffixes = ['.US', '.NY', '.O', '.OQ', '.N', '.A', '.TO']

        # Direct ticker patterns
        self.patterns = {
            # Parenthetical: (AAPL), (BRK.B)
            'parenthetical': re.compile(r'\(([A-Z]{1,5}(?:\.[A-Z])?)\)', re.IGNORECASE),

            # Dollar sign: $AAPL, $BRK.B
            'dollar': re.compile(r'\$([A-Z]{1,5}(?:\.[A-Z])?)\b', re.IGNORECASE),

            # Stock mentions: AAPL stock, MSFT shares
            'stock': re.compile(
                r'\b([A-Z]{1,5}(?:\.[A-Z])?)\s+(?:stock|shares?|equity|securities?)\b',
                re.IGNORECASE
            ),

            # Exchange: AAPL on NYSE
            'exchange': re.compile(
                r'\b([A-Z]{1,5}(?:\.[A-Z])?)\s+(?:on|at|from)\s+(?:the\s+)?'
                r'(?:NYSE|NASDAQ|AMEX|OTC)\b',
                re.IGNORECASE
            ),

            # Trading: AAPL rose, MSFT fell
            'trading': re.compile(
                r'\b([A-Z]{1,5}(?:\.[A-Z])?)\s+'
                r'(?:rose|fell|gained?|lost|surged?|plunged?|jumped?|traded?|closed?)\b',
                re.IGNORECASE
            ),

            # Price target: AAPL PT, MSFT price target
            'analyst': re.compile(
                r'\b([A-Z]{1,5}(?:\.[A-Z])?)\s+'
                r'(?:PT|price\s+target|rating|upgraded?|downgraded?)\b',
                re.IGNORECASE
            )
        }

        # Financial context words
        self.financial_contexts = {
            'stock', 'share', 'equity', 'ticker', 'symbol', 'trading',
            'earnings', 'revenue', 'profit', 'dividend', 'market',
            'analyst', 'investor', 'portfolio', 'valuation', 'ipo'
        }

    def _clean_symbol(self, symbol: str) -> str:
        """Clean ticker symbol by removing exchange suffixes"""

        symbol = symbol.upper().strip()

        # Remove $ prefix if present
        if symbol.startswith('$'):
            symbol = symbol[1:]

        # Remove exchange suffixes
        for suffix in self.exchange_suffixes:
            if symbol.endswith(suffix):
                return symbol[:-len(suffix)]

        # Handle special cases
        # Convert BRK-B to BRK.B (standard format)
        if '-' in symbol and symbol.split('-')[0] in self.ticker_set:
            parts = symbol.split('-')
            symbol = f"{parts[0]}.{parts[1]}"

        return symbol

    def extract_tickers(self, text: str, title: str = "") -> List[Tuple[str, float]]:
        """
        Extract tickers using pre-trained NER + pattern matching + CSV data

        Returns: List of (ticker, confidence) tuples
        """

        if not text and not title:
            return []

        full_text = f"{title} {text}"
        ticker_scores = defaultdict(float)

        # 1. Use pre-trained NER model
        if self.ner_pipeline:
            self._extract_with_finbert(full_text, ticker_scores)

        # 2. Pattern-based extraction
        self._extract_with_patterns(full_text, ticker_scores)

        # 3. Alias matching
        self._extract_with_aliases(full_text, ticker_scores)

        # 4. Filter and validate
        validated_scores = self._validate_tickers(ticker_scores)

        # Sort by confidence
        results = sorted(validated_scores.items(),
                         key=lambda x: x[1], reverse=True)

        return results

    def _extract_with_finbert(self, text: str, ticker_scores: Dict[str, float]):
        """Extract organizations using FinBERT-NER and map to tickers"""

        try:
            # Run NER
            entities = self.ner_pipeline(text)

            for entity in entities:
                # Look for organization entities
                if entity['entity_group'] in ['ORG', 'ORGANIZATION']:
                    org_text = entity['word'].strip()
                    confidence = entity['score']

                    # Try to map to ticker
                    ticker = self._org_to_ticker(org_text)
                    if ticker:
                        ticker_scores[ticker] = max(
                            ticker_scores[ticker], confidence * 0.9)
                        self.extraction_stats['finbert_ner'] += 1

                    # Also check if it's a ticker itself
                    cleaned = self._clean_symbol(org_text)
                    if cleaned in self.ticker_set:
                        ticker_scores[cleaned] = max(
                            ticker_scores[cleaned], confidence * 0.95)
                        self.extraction_stats['finbert_direct'] += 1

        except Exception as e:
            logger.debug(f"FinBERT extraction error: {e}")

    def _org_to_ticker(self, org_name: str) -> Optional[str]:
        """Map organization name to ticker symbol"""

        org_lower = org_name.lower().strip()

        # Direct lookup
        if org_lower in self.company_to_ticker:
            return self.company_to_ticker[org_lower]

        # Alias lookup
        if org_lower in self.alias_to_ticker:
            return self.alias_to_ticker[org_lower]

        # Fuzzy matching for close matches
        if len(org_lower) > 3:
            # Check against company names
            for company, ticker in self.company_to_ticker.items():
                if fuzz.ratio(org_lower, company) > 85:
                    return ticker

            # Check against aliases
            for alias, ticker in self.alias_to_ticker.items():
                if fuzz.ratio(org_lower, alias) > 85:
                    return ticker

        return None

    def _extract_with_patterns(self, text: str, ticker_scores: Dict[str, float]):
        """Extract using regex patterns"""

        # High confidence patterns
        for match in self.patterns['parenthetical'].finditer(text):
            ticker = self._clean_symbol(match.group(1))
            if ticker in self.ticker_set:
                ticker_scores[ticker] = max(ticker_scores[ticker], 0.95)
                self.extraction_stats['parenthetical'] += 1

        for match in self.patterns['dollar'].finditer(text):
            ticker = self._clean_symbol(match.group(1))
            if ticker in self.ticker_set:
                ticker_scores[ticker] = max(ticker_scores[ticker], 0.95)
                self.extraction_stats['dollar'] += 1

        # Medium confidence patterns
        for pattern_name in ['stock', 'exchange', 'analyst']:
            for match in self.patterns[pattern_name].finditer(text):
                ticker = self._clean_symbol(match.group(1))
                if ticker in self.ticker_set:
                    ticker_scores[ticker] = max(ticker_scores[ticker], 0.85)
                    self.extraction_stats[pattern_name] += 1

        # Lower confidence with context check
        for match in self.patterns['trading'].finditer(text):
            ticker = self._clean_symbol(match.group(1))
            if ticker in self.ticker_set and self._has_financial_context(text, match.start()):
                ticker_scores[ticker] = max(ticker_scores[ticker], 0.75)
                self.extraction_stats['trading'] += 1

    def _extract_with_aliases(self, text: str, ticker_scores: Dict[str, float]):
        """Extract using alias matching"""

        text_lower = text.lower()

        # Check each alias
        for alias, ticker in self.alias_to_ticker.items():
            # Use word boundaries for better matching
            pattern = r'\b' + re.escape(alias) + r'\b'
            if re.search(pattern, text_lower):
                ticker_scores[ticker] = max(ticker_scores[ticker], 0.8)
                self.extraction_stats['alias'] += 1

        # Check company names
        for company, ticker in self.company_to_ticker.items():
            if len(company) > 3:  # Avoid short matches
                pattern = r'\b' + re.escape(company) + r'\b'
                if re.search(pattern, text_lower):
                    ticker_scores[ticker] = max(ticker_scores[ticker], 0.8)
                    self.extraction_stats['company_name'] += 1

    def _has_financial_context(self, text: str, position: int, window: int = 50) -> bool:
        """Check if position has financial context"""

        start = max(0, position - window)
        end = min(len(text), position + window)
        context = text[start:end].lower()

        return any(term in context for term in self.financial_contexts)

    def _validate_tickers(self, ticker_scores: Dict[str, float]) -> Dict[str, float]:
        """Validate and filter tickers"""

        # Excluded patterns
        excluded = {
            'CEO', 'CFO', 'CTO', 'COO', 'CMO', 'IPO', 'SEC', 'FDA', 'EPA',
            'API', 'AI', 'ML', 'UI', 'UX', 'IT', 'HR', 'PR', 'US', 'UK'
        }

        validated = {}
        for ticker, score in ticker_scores.items():
            # Skip excluded
            if ticker in excluded:
                continue

            # Single letter tickers need special validation
            if len(ticker) == 1:
                # Only accept known single-letter tickers with high confidence
                if ticker in self.ticker_set and score >= 0.8:
                    validated[ticker] = score
            else:
                # Multi-letter tickers
                if ticker in self.ticker_set:
                    validated[ticker] = score

        return validated

    @lru_cache(maxsize=1000)
    def get_company_info(self, ticker: str) -> Dict[str, str]:
        """Get company information for a ticker"""

        ticker = self._clean_symbol(ticker)

        return {
            'symbol': ticker,
            'company_name': self.ticker_to_company.get(ticker, ''),
            'sector': self.ticker_to_sector.get(ticker, ''),
            'aliases': list(self.ticker_aliases.get(ticker, set()))
        }

    def test_extraction(self):
        """Test the extraction system"""

        test_cases = [
            ("Apple (AAPL) reported record earnings", ["AAPL"]),
            ("Microsoft stock rose 5% on strong cloud revenue", ["MSFT"]),
            ("Buy $TSLA before earnings", ["TSLA"]),
            ("Berkshire Hathaway (BRK.B) annual meeting", ["BRK.B"]),
            ("Amazon and Google face antitrust scrutiny", ["AMZN", "GOOGL"]),
            ("Tim Cook announced Apple's new products", ["AAPL"]),
            ("Bank of America upgraded to buy", ["BAC"]),
            ("C stock gained on banking results", ["C"]),
        ]

        print("\n🧪 Testing Pre-trained Financial NER")
        print("=" * 70)

        passed = 0
        for text, expected in test_cases:
            results = self.extract_tickers(text)
            found = [ticker for ticker, _ in results]

            # Check if all expected tickers found
            all_found = all(exp in found for exp in expected)
            no_false_positives = 'COOK' not in found and 'TIM' not in found

            if all_found and no_false_positives:
                passed += 1
                print(f"✅ '{text[:40]}...'")
                print(f"   Found: {found}")
            else:
                print(f"❌ '{text[:40]}...'")
                print(f"   Found: {found}, Expected: {expected}")

        print(f"\nResults: {passed}/{len(test_cases)} passed")

        # Print extraction statistics
        print("\n📊 Extraction Statistics:")
        for method, count in sorted(self.extraction_stats.items()):
            print(f"   {method}: {count}")

        return passed == len(test_cases)


# Integration wrapper for your pipeline
class FinancialNERWrapper(PretrainedFinancialNER):
    """Wrapper to match UnifiedNER interface and expose `use_gpu`."""

    def __init__(
        self,
        ticker_csv_path: str | None = None,
        use_gpu: bool = True,
        **kwargs,
    ):
        # Forward everything to PretrainedFinancialNER
        super().__init__(
            master_ticker_csv=ticker_csv_path,
            use_gpu=use_gpu,
            **kwargs,
        )
        # Compatibility attributes
        self.symbol_dict = {t: self.ticker_to_company.get(
            t, t) for t in self.ticker_set}

    def extract_symbols(self, article: Dict, min_confidence: float = 0.6,
                        use_metadata: bool = True) -> List[Tuple[str, float]]:
        """Match UnifiedNER interface"""

        title = article.get('title', '')
        content = article.get('content', '')

        # Get results from pre-trained NER
        results = self.extract_tickers(content, title)

        # Filter by confidence
        filtered = [(t, c) for t, c in results if c >= min_confidence]

        # Add metadata symbols if requested
        if use_metadata and 'symbols' in article:
            symbols = article['symbols']
            if isinstance(symbols, (list, np.ndarray)):
                for sym in symbols:
                    if isinstance(sym, str):
                        cleaned = self._clean_symbol(sym)
                        if cleaned in self.ticker_set and cleaned not in [t for t, _ in filtered]:
                            filtered.append((cleaned, 1.0))
                            self.extraction_stats['metadata'] += 1

        return filtered

    def clean_symbol(self, symbol: str) -> str:
        """Compatibility method"""
        return self._clean_symbol(symbol)

    def normalize_symbols_list(self, symbols: List[str]) -> List[str]:
        """Compatibility method"""
        return [self._clean_symbol(s) for s in symbols if s]

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


def install_finbert_ner():
    """Install the pre-trained NER in your pipeline"""

    print("🚀 Installing Pre-trained Financial NER")
    print("=" * 60)

    # Check if required packages are installed
    try:
        import transformers
        from fuzzywuzzy import fuzz
    except ImportError:
        print("📦 Installing required packages...")
        import subprocess
        subprocess.check_call(
            ['pip', 'install', 'transformers', 'fuzzywuzzy', 'python-Levenshtein'])

    # Replace the UnifiedNER with our pre-trained version
    import sys
    sys.path.insert(0, '/content/FinancialSentimentAnalysis')

    import core.ner
    core.ner.UnifiedNER = FinancialNERWrapper

    print("✅ Pre-trained Financial NER installed!")
    print("\n📝 The system now uses:")
    print("   - FinBERT-NER for organization extraction")
    print("   - Your CSV files for ticker mapping")
    print("   - Advanced pattern matching")
    print("   - Fuzzy matching for company names")

    return True


if __name__ == "__main__":
    # Test the system
    ner = PretrainedFinancialNER()
    ner.test_extraction()

    # Install it
    print("\n" + "="*60)
    install_finbert_ner()
