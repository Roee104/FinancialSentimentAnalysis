# core/aggregator.py
"""
Fixed aggregation module with context-aware chunk-to-ticker assignment
"""

import pandas as pd
import re
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import logging
import numpy as np
from difflib import SequenceMatcher
import json
from pathlib import Path
import time
from rapidfuzz import fuzz


from config.settings import SENTIMENT_CONFIG, DATA_DIR, SECTOR_CACHE_FILE, NER_CONFIG

logger = logging.getLogger(__name__)


class Aggregator:
    """Handles sentiment aggregation with context-aware assignment"""

    def __init__(
        self,
        method: str = None,
        threshold: float = None,
        sector_lookup: Dict[str, str] = None,
        use_distance_weighting: bool = True,
    ):
        """
        Initialize aggregator

        Args:
            method: Aggregation method (default/majority/conf_weighted)
            threshold: Threshold for sentiment classification
            sector_lookup: Optional custom sector lookup dict
            use_distance_weighting: Whether to use distance-based weighting
        """
        self.method = method or SENTIMENT_CONFIG["method"]
        self.threshold = threshold or SENTIMENT_CONFIG["threshold"]
        self.use_distance_weighting = use_distance_weighting

        # Load sector lookup
        if sector_lookup:
            self.sector_lookup = sector_lookup
        else:
            self.sector_lookup = self._load_sector_lookup()

        logger.info(
            f"Initialized Aggregator (method={self.method}, threshold={self.threshold})"
        )

    def _load_sector_lookup(self) -> Dict[str, str]:
        """Load sector lookup from cache or sources"""
        sector_lookup = {}

        # Check cache first
        if NER_CONFIG.get("cache_sectors", True):
            cached_sectors = self._load_cached_sectors()
            if cached_sectors:
                sector_lookup.update(cached_sectors)
                logger.info(f"Loaded {len(cached_sectors)} sectors from cache")
                return sector_lookup

        # Load fresh data
        logger.info("Loading sector data from sources...")

        # Try to load full ticker_sector.csv
        try:
            full_sectors = self._load_full_sectors()
            sector_lookup.update(full_sectors)
        except Exception as e:
            logger.warning(f"Could not load ticker_sector.csv: {e}")

        # Try to load S&P 500 sectors (only if not enough data)
        if len(sector_lookup) < 100:
            try:
                sp500 = self._load_sp500_sectors()
                sector_lookup.update(sp500)
                # Cache the S&P 500 data
                if NER_CONFIG.get("cache_sectors", True):
                    self._save_sector_cache(sp500)
            except Exception as e:
                logger.warning(f"Could not load S&P 500 sectors: {e}")

        logger.info(f"Loaded sector lookup with {len(sector_lookup)} entries")
        return sector_lookup

    def _load_cached_sectors(self) -> Optional[Dict[str, str]]:
        """Load cached sector data if fresh enough"""
        if not SECTOR_CACHE_FILE.exists():
            return None

        try:
            # Check cache age
            cache_age = time.time() - SECTOR_CACHE_FILE.stat().st_mtime
            cache_ttl = NER_CONFIG.get("sector_cache_ttl", 86400)  # 24 hours

            if cache_age > cache_ttl:
                logger.info("Sector cache expired")
                return None

            # Load cache
            with open(SECTOR_CACHE_FILE, "r") as f:
                return json.load(f)

        except Exception as e:
            logger.warning(f"Failed to load sector cache: {e}")
            return None

    def _save_sector_cache(self, sectors: Dict[str, str]):
        """Save sector data to cache"""
        try:
            SECTOR_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(SECTOR_CACHE_FILE, "w") as f:
                json.dump(sectors, f, indent=2)
            logger.info(f"Saved {len(sectors)} sectors to cache")
        except Exception as e:
            logger.warning(f"Failed to save sector cache: {e}")

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

    def find_ticker_mentions(
        self, chunk: str, ticker: str, company_name: Optional[str] = None
    ) -> List[Dict]:
        """
        Find all mentions of a ticker/company in a chunk with positions

        Returns a list of dictionaries:
            {
              "text": str,          # matched text
              "position": int,      # character index within chunk
              "length": int,
              "confidence": float,  # 0-1
              "type": str           # ticker | company_full | company_partial
            }
        """
        mentions: List[Dict] = []
        chunk_lower = chunk.lower()
        chunk_upper = chunk.upper()

        # ---- 1. exact ticker symbol  -----------------------------------
        ticker_pattern = r"\b" + re.escape(ticker) + r"\b"
        for match in re.finditer(ticker_pattern, chunk_upper):
            mentions.append(
                {
                    "text": ticker,
                    "position": match.start(),
                    "length": len(ticker),
                    "confidence": 0.90,
                    "type": "ticker",
                }
            )

        # ---- 2. company name (exact + fuzzy) ---------------------------
        if company_name:
            company_lower = company_name.lower()

            # 2-a full company name (exact substring)
            if company_lower in chunk_lower:
                pos = chunk_lower.find(company_lower)
                mentions.append(
                    {
                        "text": company_name,
                        "position": pos,
                        "length": len(company_name),
                        "confidence": 0.80,
                        "type": "company_full",
                    }
                )

            # 2-b fuzzy / partial match using rapidfuzz
            #     Only attempt if company name ≥ 2 words
            similarity = fuzz.partial_ratio(company_lower, chunk_lower)
            if similarity > 85:  # 85 % similarity threshold
                company_words = company_name.split()
                if len(company_words) >= 2:
                    partial_name = " ".join(company_words[:2]).lower()

                    if partial_name in chunk_lower and len(partial_name) > 5:
                        pos = chunk_lower.find(partial_name)
                        mentions.append(
                            {
                                "text": partial_name,
                                "position": pos,
                                "length": len(partial_name),
                                "confidence": 0.60 * (similarity / 100.0),
                                "type": "company_partial",
                            }
                        )

        return mentions

    def calculate_chunk_ticker_weight(
        self, chunk: str, mentions: List[Dict]
    ) -> float:
        """
        Calculate weight for chunk-ticker assignment based on mentions

        Args:
            chunk: Text chunk
            mentions: List of mention dicts

        Returns:
            Weight score (0-1)
        """
        if not mentions:
            return 0.0

        # Base weight from number of mentions
        mention_weight = min(len(mentions) * 0.2, 0.8)

        # Distance weight (prefer mentions near beginning)
        chunk_length = len(chunk)
        distance_scores = []

        for mention in mentions:
            position_ratio = (
                mention["position"] / chunk_length if chunk_length > 0 else 0
            )
            # Higher score for mentions earlier in chunk
            distance_score = 1.0 - (position_ratio * 0.5)
            distance_scores.append(distance_score * mention["confidence"])

        avg_distance_score = np.mean(distance_scores) if distance_scores else 0

        # Combine weights
        final_weight = (mention_weight + avg_distance_score) / 2

        return min(final_weight, 1.0)

    def assign_chunks_to_tickers(
        self,
        chunks: List[str],
        predictions: List[Dict],
        tickers: List[Tuple[str, float]],
        ticker_to_company: Optional[Dict[str, str]] = None,
    ) -> Dict[str, List[Tuple]]:
        """
        Context-aware assignment of chunks to tickers with distance weighting

        Args:
            chunks: Text chunks
            predictions: Sentiment predictions for chunks
            tickers: List of (ticker, confidence) tuples
            ticker_to_company: Optional mapping of ticker to company name

        Returns:
            Dict mapping ticker to list of (chunk_idx, sentiment, confidence, weight) tuples
        """
        ticker_chunks = defaultdict(list)

        for chunk_idx, (chunk, pred) in enumerate(zip(chunks, predictions)):
            # For each ticker, check mentions and calculate weight
            for ticker, ticker_conf in tickers:
                company_name = (
                    ticker_to_company.get(
                        ticker) if ticker_to_company else None
                )

                mentions = self.find_ticker_mentions(
                    chunk, ticker, company_name)

                if mentions:
                    weight = self.calculate_chunk_ticker_weight(
                        chunk, mentions)

                    # Apply ticker confidence to weight
                    final_weight = weight * ticker_conf

                    ticker_chunks[ticker].append(
                        (
                            chunk_idx,
                            pred["label"],
                            pred["confidence"],
                            final_weight,
                            len(mentions),  # Number of mentions for debugging
                        )
                    )

        # Log assignment statistics
        total_assignments = sum(len(v) for v in ticker_chunks.values())
        logger.debug(
            f"Assigned {total_assignments} weighted chunks to {len(ticker_chunks)} tickers"
        )

        return ticker_chunks

    def aggregate_article(
        self,
        chunks: List[str],
        predictions: List[Dict],
        symbols: List[Tuple[str, float]],
        ticker_to_company: Optional[Dict[str, str]] = None,
    ) -> Dict:
        """
        Aggregate sentiments for a full article with context-aware assignment

        Args:
            chunks: Text chunks
            predictions: Sentiment predictions for chunks
            symbols: List of (ticker, confidence) tuples
            ticker_to_company: Optional ticker to company name mapping

        Returns:
            Dict with ticker, sector, and overall sentiments
        """
        # Context-aware chunk assignment with weights
        ticker_chunks_map = self.assign_chunks_to_tickers(
            chunks, predictions, symbols, ticker_to_company
        )

        # Convert to format for aggregation
        ticker_chunks_list = []
        for ticker, chunk_data in ticker_chunks_map.items():
            for chunk_idx, label, confidence, weight, mention_count in chunk_data:
                ticker_chunks_list.append((ticker, label, confidence, weight))

        # Aggregate by ticker using weighted sentiments
        ticker_sentiments = self.compute_ticker_sentiment(ticker_chunks_list)

        # Aggregate by sector
        sector_sentiments = self.compute_sector_sentiment(ticker_sentiments)

        # Overall article sentiment (from ALL chunks, not just ticker-assigned ones)
        overall_label, overall_conf = self.compute_article_sentiment_from_chunks(
            predictions
        )

        return {
            "ticker_sentiments": [
                {"symbol": t, **info} for t, info in ticker_sentiments.items()
            ],
            "sector_sentiments": [
                {"sector": s, **info} for s, info in sector_sentiments.items()
            ],
            "overall_sentiment": overall_label,
            "overall_confidence": overall_conf,
            "chunk_assignments": {
                ticker: len(chunks) for ticker, chunks in ticker_chunks_map.items()
            },
        }

    def compute_ticker_sentiment(
        self, ticker_chunks: List[Tuple[str, str, float, float]]
    ) -> Dict[str, Dict]:
        """
        Aggregate chunk-level sentiments per ticker with distance weighting

        Args:
            ticker_chunks: List of (ticker, label, confidence, weight) tuples

        Returns:
            Dict mapping ticker to sentiment info
        """
        data = defaultdict(
            lambda: {
                "pos_count": 0,
                "neg_count": 0,
                "neu_count": 0,
                "pos_conf": 0.0,
                "neg_conf": 0.0,
                "neu_conf": 0.0,
                "pos_weight": 0.0,
                "neg_weight": 0.0,
                "neu_weight": 0.0,
                "total_weight": 0.0,
            }
        )

        for ticker, label, conf, weight in ticker_chunks:
            d = data[ticker]
            d["total_weight"] += weight

            if label == "Positive":
                d["pos_count"] += 1
                d["pos_conf"] += conf
                d["pos_weight"] += weight
            elif label == "Negative":
                d["neg_count"] += 1
                d["neg_conf"] += conf
                d["neg_weight"] += weight
            else:
                d["neu_count"] += 1
                d["neu_conf"] += conf
                d["neu_weight"] += weight

        results = {}
        for tick, d in data.items():
            pos, neg, neu = d["pos_count"], d["neg_count"], d["neu_count"]
            tot_chunks = pos + neg + neu

            if tot_chunks == 0:
                continue

            if self.use_distance_weighting and d["total_weight"] > 0:
                # Use weighted scores
                score = (d["pos_weight"] - d["neg_weight"]) / d["total_weight"]
            else:
                # Fallback to count-based
                tot_conf = d["pos_conf"] + d["neg_conf"] + d["neu_conf"]

                if self.method == "majority":
                    if pos > neg and pos > neu:
                        label = "Positive"
                    elif neg > pos and neg > neu:
                        label = "Negative"
                    else:
                        label = "Neutral"
                    score = (pos - neg) / tot_chunks

                elif self.method == "conf_weighted":
                    score = (
                        (d["pos_conf"] - d["neg_conf"]) / tot_conf
                    ) if tot_conf else 0.0

                else:  # default
                    score = (pos - neg) / tot_chunks

            # Determine label based on score
            if score > self.threshold:
                label = "Positive"
            elif score < -self.threshold:
                label = "Negative"
            else:
                label = "Neutral"

            avg_conf = (
                (d["pos_conf"] + d["neg_conf"] + d["neu_conf"]) / tot_chunks
                if tot_chunks
                else 0.0
            )

            results[tick] = {
                "score": score,
                "label": label,
                "confidence": avg_conf,
                "chunk_count": tot_chunks,
                "avg_weight": d["total_weight"] / tot_chunks if tot_chunks else 0,
            }

        return results

    def compute_sector_sentiment(
        self, ticker_sentiments: Dict[str, Dict]
    ) -> Dict[str, Dict]:
        """
        Aggregate ticker-level scores into sectors

        Args:
            ticker_sentiments: Dict of ticker sentiments

        Returns:
            Dict mapping sector to sentiment info
        """
        agg = defaultdict(
            lambda: {"score_sum": 0.0, "conf_sum": 0.0,
                     "count": 0, "weight_sum": 0.0}
        )

        for tick, info in ticker_sentiments.items():
            sector = self.sector_lookup.get(tick, "Unknown")
            agg[sector]["score_sum"] += info["score"]
            agg[sector]["conf_sum"] += info["confidence"]
            agg[sector]["count"] += 1
            agg[sector]["weight_sum"] += info.get("avg_weight", 1.0)

        results = {}
        for sector, d in agg.items():
            cnt = d["count"]
            if cnt == 0:
                continue

            avg_score = d["score_sum"] / cnt
            avg_conf = d["conf_sum"] / cnt

            if avg_score > self.threshold:
                label = "Positive"
            elif avg_score < -self.threshold:
                label = "Negative"
            else:
                label = "Neutral"

            results[sector] = {
                "score": avg_score,
                "label": label,
                "confidence": avg_conf,
                "weight": cnt,
                "avg_ticker_weight": d["weight_sum"] / cnt,
            }

        return results

    def compute_article_sentiment_from_chunks(
        self, predictions: List[Dict]
    ) -> Tuple[str, float]:
        """
        Compute overall article sentiment from ALL chunks

        Args:
            predictions: All chunk predictions

        Returns:
            Tuple of (label, confidence)
        """
        if not predictions:
            return "Neutral", 0.0

        # Count sentiments
        sentiment_counts = defaultdict(int)
        confidence_sum = defaultdict(float)

        for pred in predictions:
            label = pred["label"]
            conf = pred["confidence"]
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
            score = (
                (confidence_sum["Positive"] -
                 confidence_sum["Negative"]) / total_conf
            ) if total_conf else 0.0

            if score > self.threshold:
                label = "Positive"
            elif score < -self.threshold:
                label = "Negative"
            else:
                label = "Neutral"

            avg_conf = total_conf / total_chunks

        else:  # default
            # Count-based with threshold
            pos_count = sentiment_counts["Positive"]
            neg_count = sentiment_counts["Negative"]
            score = (pos_count - neg_count) / total_chunks

            if score > self.threshold:
                label = "Positive"
            elif score < -self.threshold:
                label = "Negative"
            else:
                label = "Neutral"

            avg_conf = sum(confidence_sum.values()) / total_chunks

        return label, avg_conf

    # Unit test
    def test_distance_weighting(self):
        """Test distance-based weighting functionality"""
        test_chunks = [
            "Apple (AAPL) reported strong earnings growth.",
            "The market reacted positively to the news.",
            "Microsoft shares declined on regulatory concerns.",
        ]

        test_predictions = [
            {"label": "Positive", "confidence": 0.9},
            {"label": "Positive", "confidence": 0.8},
            {"label": "Negative", "confidence": 0.85},
        ]

        test_tickers = [("AAPL", 0.95), ("MSFT", 0.9)]

        try:
            result = self.aggregate_article(
                chunks=test_chunks,
                predictions=test_predictions,
                symbols=test_tickers,
                ticker_to_company={"AAPL": "Apple Inc",
                                   "MSFT": "Microsoft Corporation"},
            )

            # Check structure
            assert "ticker_sentiments" in result
            assert "overall_sentiment" in result
            assert len(result["ticker_sentiments"]) > 0

            # AAPL should be positive (mentioned in positive chunk)
            aapl_sentiment = next(
                (t for t in result["ticker_sentiments"]
                 if t["symbol"] == "AAPL"), None
            )
            assert aapl_sentiment is not None
            assert aapl_sentiment["label"] == "Positive"

            logger.info("✅ Distance weighting test passed")
            return True

        except Exception as e:
            logger.error(f"❌ Distance weighting test failed: {e}")
            return False
