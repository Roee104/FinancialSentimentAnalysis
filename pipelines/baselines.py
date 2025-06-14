# pipelines/baselines.py
"""
Baseline models for comparison (VADER, etc.)
"""

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from typing import Dict, Optional
import logging

from pipelines.base_pipeline import BasePipeline
from core.ner import UnifiedNER
from core.text_processor import TextProcessor
from config.settings import VADER_CONFIG

logger = logging.getLogger(__name__)


class VADERBaseline(BasePipeline):
    """VADER baseline for sentiment analysis comparison"""

    def __init__(self,
                 vader_threshold: float = None,
                 **kwargs):
        """
        Initialize VADER baseline

        Args:
            vader_threshold: Threshold for pos/neg classification
            **kwargs: Base pipeline arguments
        """
        super().__init__(**kwargs)
        self.vader_threshold = vader_threshold or VADER_CONFIG["threshold"]
        logger.info(f"VADER baseline with threshold: {self.vader_threshold}")

    def initialize_components(self):
        """Initialize VADER and other components"""
        logger.info("Initializing VADER baseline components...")

        # Initialize VADER
        logger.info("Loading VADER...")
        self.vader = SentimentIntensityAnalyzer()

        # Initialize NER (same as main pipeline)
        logger.info("Loading Enhanced NER...")
        self.ner = UnifiedNER(ticker_csv_path=self.ticker_csv)

        # Text processor not needed for VADER
        self.text_processor = None

        logger.info("VADER baseline initialized")

    def process_article(self, row: pd.Series) -> Optional[Dict]:
        """Process article with VADER"""
        try:
            # Extract basic info
            date = row.get("date")
            title = str(row.get("title", ""))
            content = str(row.get("content", ""))

            # Skip if empty
            if not title and not content:
                return None

            # Skip if already processed
            if self.resume and title in self.processed_titles:
                self.stats['skipped'] += 1
                return None

            full_text = f"{title}. {content}"

            # Get VADER scores
            scores = self.vader.polarity_scores(full_text)

            # Convert to label
            sentiment_label = self._vader_sentiment_to_label(
                scores['compound'])
            self.stats['sentiment_dist'][sentiment_label] += 1

            # Extract tickers (same as main pipeline)
            article_dict = {
                "title": title,
                "content": content,
                "symbols": self.ner.handle_symbols_array(row.get("symbols", []))
            }

            symbols = self.ner.extract_symbols(article_dict)

            # Update ticker stats
            if symbols:
                self.stats['articles_with_tickers'] += 1
                self.stats['total_tickers_found'] += len(symbols)
            else:
                self.stats['articles_without_tickers'] += 1

            # Build record
            record = {
                "date": date.isoformat() if hasattr(date, "isoformat") else str(date),
                "title": title[:500],
                "overall_sentiment": sentiment_label,
                "vader_scores": scores,
                "compound_score": scores['compound'],
                "tickers": symbols,
                "ticker_count": len(symbols)
            }

            self.stats['processed'] += 1
            return record

        except Exception as e:
            logger.error(f"Error processing article: {str(e)[:100]}")
            self.stats['errors'] += 1
            return None

    def _vader_sentiment_to_label(self, compound_score: float) -> str:
        """Convert VADER compound score to sentiment label"""
        if compound_score >= self.vader_threshold:
            return "Positive"
        elif compound_score <= -self.vader_threshold:
            return "Negative"
        else:
            return "Neutral"
