# data/validator.py
"""
Data validation and quality control module
"""

import re
from typing import Dict, List, Optional, Tuple
import logging

from config.settings import DATA_COLLECTION, LOW_QUALITY_PATTERNS

logger = logging.getLogger(__name__)


class DataValidator:
    """Handles data validation and quality control"""

    def __init__(self, config: dict = None):
        """
        Initialize validator

        Args:
            config: Configuration dict (uses settings if None)
        """
        self.config = config or DATA_COLLECTION
        logger.info("Initialized DataValidator")

    def validate_article_quality(self, article: Dict) -> bool:
        """
        Filter out low-quality articles based on multiple criteria

        Args:
            article: Dictionary containing article data

        Returns:
            bool: True if article meets quality standards
        """
        is_valid, _ = self.validate_article_quality_with_reason(article)
        return is_valid

    def validate_article_quality_with_reason(self, article: Dict) -> Tuple[bool, str]:
        """
        Filter out low-quality articles and return reason if invalid

        Args:
            article: Dictionary containing article data

        Returns:
            Tuple of (is_valid, reason)
        """
        content = article.get("content", "").strip()
        title = article.get("title", "").strip()

        # Check minimum length requirements
        if len(content.split()) < self.config["min_content_words"]:
            return False, "min_content_words"

        if len(title.split()) < self.config["min_title_words"]:
            return False, "min_title_words"

        # Check for empty or placeholder content
        if not content or content.lower() in [
            "",
            "n/a",
            "no content",
            "content not available",
        ]:
            return False, "empty_content"

        if not title or title.lower() in ["", "n/a", "untitled"]:
            return False, "empty_title"

        # Check for auto-generated or low-quality content patterns
        content_lower = content.lower()
        title_lower = title.lower()

        for pattern in LOW_QUALITY_PATTERNS:
            if pattern in content_lower or pattern in title_lower:
                return False, "low_quality_pattern"

        # Check for repetitive content
        words = content.split()
        if len(words) > 0:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.3:  # Too many repeated words
                return False, "repetitive_content"

        # Check for suspiciously short sentences
        sentences = content.split(".")
        if len(sentences) > 1:
            avg_sentence_length = sum(len(s.split()) for s in sentences) / len(
                sentences
            )
            if avg_sentence_length < 3:
                return False, "short_sentences"

        # Check if article has meaningful symbols
        symbols = article.get("symbols", [])
        if not symbols:
            # Look for potential ticker symbols in content
            potential_tickers = re.findall(
                r"\b[A-Z]{2,5}\b", content + " " + title)
            if len(potential_tickers) < 1:
                return False, "no_symbols"

        return True, "valid"

    def detect_near_duplicates(
        self, article: Dict, existing_articles: List[Dict], check_last_n: int = 100
    ) -> bool:
        """
        Simple duplicate detection based on title and content similarity

        Args:
            article: New article to check
            existing_articles: List of already collected articles
            check_last_n: Number of recent articles to check against

        Returns:
            bool: True if article is likely a duplicate
        """
        new_title = article.get("title", "").lower().strip()
        new_content = article.get("content", "").lower().strip()

        # Check against last N articles for efficiency
        check_articles = (
            existing_articles[-check_last_n:]
            if len(existing_articles) > check_last_n
            else existing_articles
        )

        for existing in check_articles:
            existing_title = existing.get("title", "").lower().strip()
            existing_content = existing.get("content", "").lower().strip()

            # Title similarity
            if new_title and existing_title:
                title_words_new = set(new_title.split())
                title_words_existing = set(existing_title.split())
                if title_words_new and title_words_existing:
                    title_similarity = len(
                        title_words_new & title_words_existing
                    ) / max(len(title_words_new), 1)
                    if title_similarity > 0.8:
                        return True

            # Content similarity (first 200 words)
            new_words = set(new_content.split()[:200])
            existing_words = set(existing_content.split()[:200])
            if new_words and existing_words:
                content_similarity = len(new_words & existing_words) / len(
                    new_words | existing_words
                )
                if content_similarity > self.config["max_duplicate_threshold"]:
                    return True

        return False

    def validate_symbols(self, symbols: List[str]) -> List[str]:
        """
        Validate and clean symbol list

        Args:
            symbols: List of ticker symbols

        Returns:
            Cleaned list of symbols
        """
        valid_symbols = []

        for symbol in symbols:
            if isinstance(symbol, str):
                # Basic validation
                symbol = symbol.strip().upper()
                if 1 <= len(symbol) <= 5 and symbol.isalpha():
                    valid_symbols.append(symbol)

        return valid_symbols

    def validate_date(self, date_str: str) -> bool:
        """
        Validate date string

        Args:
            date_str: Date string to validate

        Returns:
            bool: True if valid date
        """
        try:
            # Try common date formats
            from datetime import datetime

            for fmt in ["%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%d/%m/%Y"]:
                try:
                    datetime.strptime(date_str, fmt)
                    return True
                except ValueError:
                    continue
            return False
        except:
            return False

    def validate_sentiment_scores(self, scores: Dict) -> bool:
        """
        Validate sentiment score dictionary

        Args:
            scores: Dict with sentiment scores

        Returns:
            bool: True if valid scores
        """
        required_keys = {"pos", "neu", "neg"}

        if not isinstance(scores, dict):
            return False

        if not required_keys.issubset(scores.keys()):
            return False

        # Check if scores sum to approximately 1
        total = sum(scores.get(key, 0) for key in required_keys)
        if abs(total - 1.0) > 0.01:
            return False

        # Check if all scores are between 0 and 1
        for key in required_keys:
            score = scores.get(key, 0)
            if not (0 <= score <= 1):
                return False

        return True
