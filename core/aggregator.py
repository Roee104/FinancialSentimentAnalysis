# core/aggregator.py
"""
Fixed aggregation module with context-aware chunk-to-ticker assignment
"""

import pandas as pd
import re
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import logging

from config.settings import SENTIMENT_CONFIG, DATA_DIR

logger = logging.getLogger(__name__)


class Aggregator:
    """Handles sentiment aggregation with context-aware assignment"""

    def __init__(self,
                 method: str = None,
                 threshold: float = None,
                 sector_lookup: Dict[str, str] = None):
        """
        Initialize aggregator

        Args:
            method: Aggregation method (default/majority/conf_weighted)
            threshold: Threshold for sentiment classification
            sector_lookup: Optional custom sector lookup dict
        """
        self.method = method or SENTIMENT_CONFIG["method"]
        self.threshold = threshold or SENTIMENT_CONFIG["threshold"]

        # Load sector lookup
        if sector_lookup:
            self.sector_lookup = sector_lookup
        else:
            self.sector_lookup = self._load_sector_lookup()

        logger.info(
            f"Initialized Aggregator (method={self.method}, threshold={self.threshold})")

    def _load_sector_lookup(self) -> Dict[str, str]:
        """Load sector lookup from multiple sources"""
        sector_lookup = {}

        # Try to load S&P 500 sectors
        try:
            sp500 = self._load_sp500_sectors()
            sector_lookup.update(sp500)
        except Exception as e:
            logger.warning(f"Could not load S&P 500 sectors: {e}")

        # Try to load full ticker_sector.csv
        try:
            full_sectors = self._load_full_sectors()
            sector_lookup.update(full_sectors)
        except Exception as e:
            logger.warning(f"Could not load ticker_sector.csv: {e}")

        logger.info(f"Loaded sector lookup with {len(sector_lookup)} entries")
        return sector_lookup

    def _load_sp500_sectors(self) -> Dict[str, str]:
        """Load S&P 500 GICS sectors from Wikipedia"""
        try:
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            df = pd.read_html(url)[0]
            df["Symbol"] = df["Symbol"].str.replace(
                ".", "-", regex=False).str.strip()
            return dict(zip(df["Symbol"], df["GICS Sector"]))
        except Exception as e:
            logger.debug(f"S&P 500 load error: {e}")
            return {}

    def _load_full_sectors(self) -> Dict[str, str]:
        """Load full ticker sector mapping"""
        try:
            csv_path = DATA_DIR / "ticker_sector.csv"
            df = pd.read_csv(csv_path, dtype=str)
            return dict(zip(df["symbol"].str.upper(), df["sector"]))
        except Exception as e:
            logger.debug(f"Ticker sector load error: {e}")
            return {}

    def find_ticker_in_chunk(self, chunk: str, ticker: str, company_name: Optional[str] = None) -> bool:
        """
        Check if a ticker/company is mentioned in a chunk

        Args:
            chunk: Text chunk
            ticker: Ticker symbol
            company_name: Optional company name

        Returns:
            True if ticker or company is mentioned in chunk
        """
        chunk_upper = chunk.upper()

        # Check for ticker symbol (with word boundaries)
        ticker_pattern = r'\b' + re.escape(ticker) + r'\b'
        if re.search(ticker_pattern, chunk_upper):
            return True

        # Check for company name if provided
        if company_name:
            # Simple check - can be enhanced
            if company_name.lower() in chunk.lower():
                return True

            # Check for partial company name (first 2 words)
            company_words = company_name.split()
            if len(company_words) >= 2:
                partial_name = ' '.join(company_words[:2])
                if partial_name.lower() in chunk.lower():
                    return True

        return False

    def assign_chunks_to_tickers(self,
                                 chunks: List[str],
                                 predictions: List[Dict],
                                 tickers: List[str],
                                 ticker_to_company: Optional[Dict[str, str]] = None) -> Dict[str, List[Tuple]]:
        """
        Context-aware assignment of chunks to tickers

        Args:
            chunks: Text chunks
            predictions: Sentiment predictions for chunks
            tickers: List of ticker symbols
            ticker_to_company: Optional mapping of ticker to company name

        Returns:
            Dict mapping ticker to list of (chunk_idx, sentiment, confidence) tuples
        """
        ticker_chunks = defaultdict(list)

        for chunk_idx, (chunk, pred) in enumerate(zip(chunks, predictions)):
            # For each ticker, check if it's mentioned in this chunk
            for ticker in tickers:
                company_name = ticker_to_company.get(
                    ticker) if ticker_to_company else None

                if self.find_ticker_in_chunk(chunk, ticker, company_name):
                    ticker_chunks[ticker].append((
                        chunk_idx,
                        pred["label"],
                        pred["confidence"],
                        chunk  # Keep chunk text for debugging
                    ))

        # Log assignment statistics
        total_assignments = sum(len(v) for v in ticker_chunks.values())
        logger.debug(
            f"Assigned {total_assignments} chunks to {len(ticker_chunks)} tickers")

        return ticker_chunks

    def aggregate_article(self,
                          chunks: List[str],
                          predictions: List[Dict],
                          symbols: List[str],
                          ticker_to_company: Optional[Dict[str, str]] = None) -> Dict:
        """
        Aggregate sentiments for a full article with context-aware assignment

        Args:
            chunks: Text chunks
            predictions: Sentiment predictions for chunks
            symbols: Extracted ticker symbols
            ticker_to_company: Optional ticker to company name mapping

        Returns:
            Dict with ticker, sector, and overall sentiments
        """
        # Context-aware chunk assignment
        ticker_chunks_map = self.assign_chunks_to_tickers(
            chunks, predictions, symbols, ticker_to_company
        )

        # Convert to format expected by compute_ticker_sentiment
        ticker_chunks_list = []
        for ticker, chunk_data in ticker_chunks_map.items():
            for chunk_idx, label, confidence, chunk_text in chunk_data:
                ticker_chunks_list.append((ticker, label, confidence))

        # Aggregate by ticker (only chunks that mention each ticker)
        ticker_sentiments = self.compute_ticker_sentiment(ticker_chunks_list)

        # Aggregate by sector
        sector_sentiments = self.compute_sector_sentiment(ticker_sentiments)

        # Overall article sentiment (from ALL chunks, not just ticker-assigned ones)
        overall_label, overall_conf = self.compute_article_sentiment_from_chunks(
            predictions)

        return {
            "ticker_sentiments": [{"symbol": t, **info} for t, info in ticker_sentiments.items()],
            "sector_sentiments": [{"sector": s, **info} for s, info in sector_sentiments.items()],
            "overall_sentiment": overall_label,
            "overall_confidence": overall_conf,
            "chunk_assignments": {
                ticker: len(chunks) for ticker, chunks in ticker_chunks_map.items()
            }
        }

    def compute_ticker_sentiment(self,
                                 ticker_chunks: List[Tuple[str, str, float]]) -> Dict[str, Dict]:
        """
        Aggregate chunk-level sentiments per ticker

        Args:
            ticker_chunks: List of (ticker, label, confidence) tuples

        Returns:
            Dict mapping ticker to sentiment info
        """
        data = defaultdict(lambda: {
            'pos_count': 0, 'neg_count': 0, 'neu_count': 0,
            'pos_conf': 0.0, 'neg_conf': 0.0, 'neu_conf': 0.0
        })

        for ticker, label, conf in ticker_chunks:
            d = data[ticker]
            if label == 'Positive':
                d['pos_count'] += 1
                d['pos_conf'] += conf
            elif label == 'Negative':
                d['neg_count'] += 1
                d['neg_conf'] += conf
            else:
                d['neu_count'] += 1
                d['neu_conf'] += conf

        results = {}
        for tick, d in data.items():
            pos, neg, neu = d['pos_count'], d['neg_count'], d['neu_count']
            tot_chunks = pos + neg + neu

            if tot_chunks == 0:
                continue

            tot_conf = d['pos_conf'] + d['neg_conf'] + d['neu_conf']

            if self.method == "majority":
                if pos > neg and pos > neu:
                    label = 'Positive'
                elif neg > pos and neg > neu:
                    label = 'Negative'
                else:
                    label = 'Neutral'
                score = (pos - neg) / tot_chunks

            elif self.method == "conf_weighted":
                score = ((d['pos_conf'] - d['neg_conf']) /
                         tot_conf) if tot_conf else 0.0
                if score > self.threshold:
                    label = 'Positive'
                elif score < -self.threshold:
                    label = 'Negative'
                else:
                    label = 'Neutral'

            else:  # default
                score = (pos - neg) / tot_chunks
                if score > self.threshold:
                    label = 'Positive'
                elif score < -self.threshold:
                    label = 'Negative'
                else:
                    label = 'Neutral'

            avg_conf = tot_conf / tot_chunks if tot_chunks else 0.0
            results[tick] = {
                'score': score,
                'label': label,
                'confidence': avg_conf,
                'chunk_count': tot_chunks
            }

        return results

    def compute_sector_sentiment(self,
                                 ticker_sentiments: Dict[str, Dict]) -> Dict[str, Dict]:
        """
        Aggregate ticker-level scores into sectors

        Args:
            ticker_sentiments: Dict of ticker sentiments

        Returns:
            Dict mapping sector to sentiment info
        """
        agg = defaultdict(
            lambda: {'score_sum': 0.0, 'conf_sum': 0.0, 'count': 0})

        for tick, info in ticker_sentiments.items():
            sector = self.sector_lookup.get(tick, 'Unknown')
            agg[sector]['score_sum'] += info['score']
            agg[sector]['conf_sum'] += info['confidence']
            agg[sector]['count'] += 1

        results = {}
        for sector, d in agg.items():
            cnt = d['count']
            if cnt == 0:
                continue

            avg_score = d['score_sum']/cnt
            avg_conf = d['conf_sum']/cnt

            if avg_score > self.threshold:
                label = 'Positive'
            elif avg_score < -self.threshold:
                label = 'Negative'
            else:
                label = 'Neutral'

            results[sector] = {
                'score': avg_score,
                'label': label,
                'confidence': avg_conf,
                'weight': cnt
            }

        return results

    def compute_article_sentiment_from_chunks(self,
                                              predictions: List[Dict]) -> Tuple[str, float]:
        """
        Compute overall article sentiment from ALL chunks

        Args:
            predictions: All chunk predictions

        Returns:
            Tuple of (label, confidence)
        """
        if not predictions:
            return 'Neutral', 0.0

        # Count sentiments
        sentiment_counts = defaultdict(int)
        confidence_sum = defaultdict(float)

        for pred in predictions:
            label = pred['label']
            conf = pred['confidence']
            sentiment_counts[label] += 1
            confidence_sum[label] += conf

        total_chunks = len(predictions)

        if self.method == "majority":
            # Simple majority vote
            label = max(sentiment_counts.items(), key=lambda x: x[1])[0]
            avg_conf = confidence_sum[label] / sentiment_counts[label]

        elif self.method == "conf_weighted":
            # Confidence-weighted score
            total_conf = sum(confidence_sum.values())
            score = ((confidence_sum['Positive'] - confidence_sum['Negative']) /
                     total_conf) if total_conf else 0.0

            if score > self.threshold:
                label = 'Positive'
            elif score < -self.threshold:
                label = 'Negative'
            else:
                label = 'Neutral'

            avg_conf = total_conf / total_chunks

        else:  # default
            # Count-based with threshold
            pos_count = sentiment_counts['Positive']
            neg_count = sentiment_counts['Negative']
            score = (pos_count - neg_count) / total_chunks

            if score > self.threshold:
                label = 'Positive'
            elif score < -self.threshold:
                label = 'Negative'
            else:
                label = 'Neutral'

            avg_conf = sum(confidence_sum.values()) / total_chunks

        return label, avg_conf

    def compute_article_sentiment(self,
                                  ticker_sentiments: Dict[str, Dict]) -> Tuple[str, float]:
        """
        Legacy method - compute from ticker sentiments
        Kept for backward compatibility
        """
        if not ticker_sentiments:
            return 'Neutral', 0.0

        scores = [v['score'] for v in ticker_sentiments.values()]
        confs = [v['confidence'] for v in ticker_sentiments.values()]

        avg_score = sum(scores)/len(scores)
        avg_conf = sum(confs)/len(confs)

        if avg_score > self.threshold:
            label = 'Positive'
        elif avg_score < -self.threshold:
            label = 'Negative'
        else:
            label = 'Neutral'

        return label, avg_conf
