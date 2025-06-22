# pipelines/baselines.py
"""
Baseline models for comparison (VADER, etc.).
"""
from __future__ import annotations

# ───────────────────────────────  logging & config  ──────────────────────────
import logging
import nltk
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from config.settings import CACHE_DIR, VADER_CONFIG
from core.ner import UnifiedNER
from pipelines.base_pipeline import BasePipeline

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────


class VADERBaseline(BasePipeline):
    """VADER baseline for sentiment-analysis comparison."""

    def __init__(self, vader_threshold: float | None = None, **kwargs):
        super().__init__(**kwargs)
        self.vader_threshold = vader_threshold or VADER_CONFIG["threshold"]
        logger.info("VADER baseline threshold: %.3f", self.vader_threshold)

    # -------------------------------------------------------------------- init
    def initialize_components(self) -> None:
        logger.info("Initialising VADER baseline components …")

        # ---- VADER SENTIMENT -------------------------------------------------
        lex_home = CACHE_DIR / "nltk"
        nltk.data.path.append(str(lex_home))
        try:
            nltk.data.find("sentiment/vader_lexicon.zip")
        except LookupError:
            nltk.download("vader_lexicon", download_dir=str(
                lex_home), quiet=True)

        self.vader = SentimentIntensityAnalyzer()

        # ---- NER same as main pipeline --------------------------------------
        logger.info("Loading Enhanced NER …")
        self.ner = UnifiedNER(ticker_csv_path=self.ticker_csv)

        # Text-processor not needed
        self.text_processor = None
        logger.info("VADER baseline ready.")

    # -------------------------------------------------------------------- core
    def process_article(self, row: pd.Series) -> Optional[Dict]:
        try:
            title = str(row.get("title", ""))
            content = str(row.get("content", ""))
            if not title and not content:
                return None

            article_hash = self._get_article_hash(title, content)
            if self.resume and article_hash in self.processed_hashes:
                self.stats["skipped"] += 1
                return None

            full_text = f"{title}. {content}"
            scores = self.vader.polarity_scores(full_text)
            sentiment_label = self._vader_sentiment_to_label(
                scores["compound"])
            self.stats["sentiment_dist"][sentiment_label] += 1

            article_dict = {
                "title": title,
                "content": content,
                "symbols": self.ner.handle_symbols_array(row.get("symbols", [])),
            }
            symbols = [sym for sym,
                       _ in self.ner.extract_symbols(article_dict)]

            if symbols:
                self.stats["articles_with_tickers"] += 1
                self.stats["total_tickers_found"] += len(symbols)
            else:
                self.stats["articles_without_tickers"] += 1

            self.stats["processed"] += 1
            return {
                "date": str(row.get("date")),
                "title": title[:500],
                "article_hash": article_hash,
                "overall_sentiment": sentiment_label,
                "vader_scores": scores,
                "compound_score": scores["compound"],
                "tickers": symbols,
                "ticker_count": len(symbols),
            }

        except Exception as exc:  # noqa: BLE001
            logger.error("Error processing article: %s", str(exc)[:100]) 
            self.stats["errors"] += 1
            return None

    # -------------------------------------------------------------------- util
    def _vader_sentiment_to_label(self, compound: float) -> str:
        if compound >= self.vader_threshold:
            return "Positive"
        if compound <= -self.vader_threshold:
            return "Negative"
        return "Neutral"
