# core/aggregator.py
"""
Aggregation module for sentiment scores at ticker, sector, and article levels
"""

import pandas as pd
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import logging

from config.settings import SENTIMENT_CONFIG, DATA_DIR

logger = logging.getLogger(__name__)


class Aggregator:
    """Handles sentiment aggregation at multiple levels"""

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

    def aggregate_article(self,
                          chunks: List[str],
                          predictions: List[Dict],
                          symbols: List[str]) -> Dict:
        """
        Aggregate sentiments for a full article

        Args:
            chunks: Text chunks
            predictions: Sentiment predictions for chunks
            symbols: Extracted ticker symbols

        Returns:
            Dict with ticker, sector, and overall sentiments
        """
        # Build ticker-chunk assignments
        ticker_chunks = []
        for chunk, pred in zip(chunks, predictions):
            for symbol in symbols:
                ticker_chunks.append(
                    (symbol, pred["label"], pred["confidence"])
                )

        # Aggregate by ticker
        ticker_sentiments = self.compute_ticker_sentiment(ticker_chunks)

        # Aggregate by sector
        sector_sentiments = self.compute_sector_sentiment(ticker_sentiments)

        # Overall article sentiment
        overall_label, overall_conf = self.compute_article_sentiment(
            ticker_sentiments)

        return {
            "ticker_sentiments": [{"symbol": t, **info} for t, info in ticker_sentiments.items()],
            "sector_sentiments": [{"sector": s, **info} for s, info in sector_sentiments.items()],
            "overall_sentiment": overall_label,
            "overall_confidence": overall_conf
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
            tot_conf = d['pos_conf'] + d['neg_conf'] + d['neu_conf']

            if self.method == "majority":
                if pos > neg and pos > neu:
                    label = 'Positive'
                elif neg > pos and neg > neu:
                    label = 'Negative'
                else:
                    label = 'Neutral'
                score = (pos - neg) / tot_chunks if tot_chunks else 0.0

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
                score = (pos - neg) / tot_chunks if tot_chunks else 0.0
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
                'confidence': avg_conf
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
            avg_score = d['score_sum']/cnt if cnt else 0.0
            avg_conf = d['conf_sum']/cnt if cnt else 0.0

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

    def compute_article_sentiment(self,
                                  ticker_sentiments: Dict[str, Dict]) -> Tuple[str, float]:
        """
        Aggregate overall article sentiment from ticker sentiments

        Args:
            ticker_sentiments: Dict of ticker sentiments

        Returns:
            Tuple of (label, confidence)
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
