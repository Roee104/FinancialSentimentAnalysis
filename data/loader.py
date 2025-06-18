# data/loader.py
"""
Data loading and collection module for financial news
"""

import time
import requests
import pandas as pd
import numpy as np
from transformers import AutoTokenizer
import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Union, Set
import logging
from pathlib import Path
import hashlib
from tenacity import retry, stop_after_attempt, wait_exponential
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from unittest.mock import Mock, patch

from config.settings import (
    DATA_COLLECTION,
    EODHD_API_TOKEN,
    MODELS,
    COLLECTION_TAGS,
    LOW_QUALITY_PATTERNS,
    DATA_DIR,
    CHECKPOINTS_DIR,
    QUALITY_FILTER_STATS,
    CACHE_DIR,
)
from data.validator import DataValidator

logger = logging.getLogger(__name__)


class NewsDataCollector:
    """Handles collection of financial news data from EODHD API"""

    def __init__(self, api_token: str = None, config: dict = None):
        """
        Initialize data collector

        Args:
            api_token: EODHD API token
            config: Configuration dict (uses settings if None)
        """
        self.api_token = api_token or EODHD_API_TOKEN
        if not self.api_token:
            raise ValueError(
                "EODHD_API_TOKEN not set. Please set it in your environment or .env file")

        self.config = config or DATA_COLLECTION
        self.validator = DataValidator()

        # Initialize tokenizer for token counting
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                MODELS["bert_tokenizer"],
                cache_dir=MODELS.get("cache_dir", CACHE_DIR / "models")
            )
        except Exception as e:
            logger.warning(f"Could not load tokenizer: {e}")
            self.tokenizer = None

        # Statistics
        self.quality_stats = {
            "total_fetched": 0,
            "quality_filtered": 0,
            "duplicate_filtered": 0,
            "final_count": 0,
            "failed_tags": [],
            "filter_reasons": QUALITY_FILTER_STATS.copy(),
        }

        # Efficient duplicate detection with rolling window
        self.seen_hashes = set()
        self.recent_articles = deque(
            maxlen=self.config.get("duplicate_check_window", 1000)
        )

        # State persistence
        self.state_dir = CHECKPOINTS_DIR / "crawler_state"
        self.state_dir.mkdir(parents=True, exist_ok=True)

        # CSV for quality filter stats
        self.stats_file = DATA_DIR / "quality_filter_stats.csv"

        logger.info("Initialized NewsDataCollector")

    def _get_state_file(self, tag: str) -> Path:
        """Get state file path for a tag"""
        safe_tag = tag.replace(" ", "_").replace("/", "_")
        return self.state_dir / f"{safe_tag}_state.json"

    def _load_state(self, tag: str) -> Dict:
        """Load crawler state for a tag"""
        state_file = self._get_state_file(tag)
        if state_file.exists():
            try:
                with open(state_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load state for {tag}: {e}")
        return {
            "last_offset": 0,
            "collected_count": 0,
            "retry_count": 0,
            "last_timestamp": None,
        }

    def _save_state(self, tag: str, state: Dict):
        """Save crawler state for a tag"""
        state_file = self._get_state_file(tag)
        try:
            with open(state_file, "w") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save state for {tag}: {e}")

    def _save_quality_stats(self):
        """Save quality filter statistics to CSV"""
        try:
            stats_df = pd.DataFrame([
                {
                    'reason': reason,
                    'count': count,
                    'percentage': count / self.quality_stats['total_fetched'] * 100
                    if self.quality_stats['total_fetched'] > 0 else 0
                }
                for reason, count in self.quality_stats['filter_reasons'].items()
            ])
            stats_df.to_csv(self.stats_file, index=False)
            logger.info(f"Saved quality stats to {self.stats_file}")
        except Exception as e:
            logger.error(f"Failed to save quality stats: {e}")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    def fetch_page(self, tag: str, offset: int) -> Optional[List[Dict]]:
        """
        Fetch one page of articles for a given tag

        Args:
            tag: Search tag
            offset: Pagination offset

        Returns:
            List of articles or None if error
        """
        params = {
            "t": tag,
            "from": self.config["from_date"],
            "to": self.config["to_date"],
            "limit": self.config["batch_size"],
            "offset": offset,
            "api_token": self.api_token,
            "fmt": "json",
        }

        try:
            response = requests.get(
                self.config["base_url"], params=params, timeout=30
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(
                f"Error fetching page at offset {offset} for tag '{tag}': {e}")
            raise
        except json.JSONDecodeError as e:
            logger.error(
                f"Error parsing JSON for tag '{tag}' at offset {offset}: {e}")
            raise

    def _add_to_duplicate_check(self, article_hash: str, article: Dict):
        """Add article to duplicate detection structures"""
        self.seen_hashes.add(article_hash)
        self.recent_articles.append(article)

    def _is_duplicate(self, article: Dict) -> bool:
        """Check if article is duplicate using O(1) hash lookup + O(n) similarity for recent window"""
        # Generate hash
        article_hash = hashlib.md5(
            f"{article.get('title', '')}_{article.get('content', '')}".encode()
        ).hexdigest()

        # Quick hash check
        if article_hash in self.seen_hashes:
            return True

        # Check similarity against recent window only
        if self.validator.detect_near_duplicates(article, list(self.recent_articles)):
            return True

        return False

    def collect_articles_for_tag(self, tag: str) -> List[Dict]:
        """
        Collect articles for a single tag with state persistence

        Args:
            tag: Tag to search for

        Returns:
            List of validated articles
        """
        logger.info(f"Processing tag: '{tag}'")

        # Load state
        state = self._load_state(tag)
        offset = state["last_offset"]
        tag_collected = {}
        tag_quality_filtered = 0
        tag_duplicate_filtered = 0
        consecutive_failures = 0
        pages_fetched = 0

        # Continue from saved state
        if state["collected_count"] > 0:
            logger.info(
                f"Resuming from offset {offset}, already collected {state['collected_count']}"
            )

        while (
            len(tag_collected) + state["collected_count"]
            < self.config["target_per_tag"]
            and consecutive_failures < self.config["max_retries"]
            and pages_fetched < self.config.get("max_pages_per_tag", 50)
        ):

            try:
                # Fetch page with retry
                data = self.fetch_page(tag, offset)

                if data is None or not data:
                    logger.info(f"No more articles found for tag: '{tag}'")
                    break

                # Reset failure counter on success
                consecutive_failures = 0
                pages_fetched += 1

                new_count = 0
                for art in data:
                    self.quality_stats["total_fetched"] += 1

                    link = art.get("link")
                    if not link:
                        continue

                    # Create article hash
                    article_hash = hashlib.md5(
                        f"{art.get('title', '')}_{art.get('content', '')}".encode()
                    ).hexdigest()

                    if article_hash in tag_collected:
                        continue

                    # Quality validation with reason tracking using validator
                    is_valid, reason = self.validator.validate_article_quality_with_reason(
                        art)
                    if not is_valid:
                        self.quality_stats["quality_filtered"] += 1
                        self.quality_stats["filter_reasons"][reason] += 1
                        tag_quality_filtered += 1
                        continue

                    # Efficient duplicate detection
                    if self._is_duplicate(art):
                        self.quality_stats["duplicate_filtered"] += 1
                        tag_duplicate_filtered += 1
                        continue

                    # Validate and clean symbols
                    raw_symbols = art.get("symbols", [])
                    clean_symbols = self.validator.validate_symbols(
                        raw_symbols)

                    # Add valid article
                    article_data = {
                        "date": art.get("date"),
                        "title": art.get("title", ""),
                        "content": art.get("content", ""),
                        "symbols": clean_symbols,
                        "tags": art.get("tags", []),
                        "sentiment": art.get("sentiment", {}),
                        "tag_source": tag,
                        "link": link,
                        "article_hash": article_hash,
                    }

                    tag_collected[article_hash] = article_data
                    self._add_to_duplicate_check(article_hash, article_data)
                    new_count += 1

                logger.info(
                    f"  Offset {offset}: fetched {len(data)}, "
                    f"+{new_count} valid, total = {len(tag_collected) + state['collected_count']}"
                )

                # Update state
                state["last_offset"] = offset + self.config["batch_size"]
                state["collected_count"] += new_count
                state["last_timestamp"] = datetime.now().isoformat()
                self._save_state(tag, state)

                if (
                    len(tag_collected) + state["collected_count"]
                    >= self.config["target_per_tag"]
                ):
                    break

                offset += self.config["batch_size"]
                time.sleep(self.config["sleep_sec"])

            except Exception as e:
                consecutive_failures += 1
                logger.warning(
                    f"Attempt {consecutive_failures}/{self.config['max_retries']} failed: {e}"
                )

                if consecutive_failures < self.config["max_retries"]:
                    # Exponential backoff
                    sleep_time = self.config["initial_retry_delay"] * (
                        self.config["backoff_factor"] ** consecutive_failures
                    )
                    logger.info(f"Retrying after {sleep_time:.1f} seconds...")
                    time.sleep(sleep_time)
                else:
                    logger.error(f"Max retries reached for tag '{tag}'")
                    self.quality_stats["failed_tags"].append(tag)
                    break

        # Summary
        tag_articles = list(tag_collected.values())
        logger.info(f"✅ Tag '{tag}' completed:")
        logger.info(f"   Collected: {len(tag_articles)} articles")
        logger.info(f"   Quality filtered: {tag_quality_filtered}")
        logger.info(f"   Duplicate filtered: {tag_duplicate_filtered}")
        logger.info(f"   Failed attempts: {consecutive_failures}")

        return tag_articles

    def collect_all_articles(
        self,
        tags: List[str] = None,
        checkpoint_interval: int = 10,
        use_async: bool = False,
    ) -> List[Dict]:
        """
        Collect articles for all tags

        Args:
            tags: List of tags (uses default if None)
            checkpoint_interval: Save checkpoint every N tags
            use_async: Use async fetching (experimental)

        Returns:
            List of all collected articles
        """
        tags = tags or COLLECTION_TAGS
        all_articles = []

        logger.info(f"🚀 Starting data collection for {len(tags)} tags...")
        logger.info(f"Target per tag: {self.config['target_per_tag']}")
        logger.info(
            f"Date range: {self.config['from_date']} to {self.config['to_date']}")

        for tag_idx, tag in enumerate(tags, 1):
            logger.info(f"\n[{tag_idx}/{len(tags)}] {tag}")

            tag_articles = self.collect_articles_for_tag(tag)
            all_articles.extend(tag_articles)

            # Save checkpoint
            if tag_idx % checkpoint_interval == 0:
                self.save_checkpoint(all_articles, f"tags_1_to_{tag_idx}")
                self._save_quality_stats()  # Save stats after each checkpoint

        # Final statistics
        self.quality_stats["final_count"] = len(all_articles)
        logger.info(f"\n🎯 Collection Summary:")
        logger.info(
            f"Total articles fetched: {self.quality_stats['total_fetched']}")
        logger.info(
            f"Quality filtered out: {self.quality_stats['quality_filtered']}")
        logger.info(
            f"Duplicates filtered out: {self.quality_stats['duplicate_filtered']}")
        logger.info(
            f"Final collection size: {self.quality_stats['final_count']}")

        # Log filter reasons
        logger.info("\nQuality filter breakdown:")
        for reason, count in self.quality_stats["filter_reasons"].items():
            if count > 0:
                pct = count / self.quality_stats['total_fetched'] * \
                    100 if self.quality_stats['total_fetched'] > 0 else 0
                logger.info(f"  {reason}: {count} ({pct:.1f}%)")

        if self.quality_stats["failed_tags"]:
            logger.warning(
                f"Failed tags: {', '.join(self.quality_stats['failed_tags'])}")

        # Save final quality stats
        self._save_quality_stats()

        return all_articles

    def process_for_dataset(self, articles: List[Dict]) -> pd.DataFrame:
        """
        Process articles for final dataset

        Args:
            articles: List of article dicts

        Returns:
            DataFrame ready for saving
        """
        logger.info("Processing articles for final dataset...")

        for art in articles:
            # Add sentiment label
            art["sentiment_label"] = self._label_by_max_prob(art["sentiment"])

            # Add token count
            if self.tokenizer:
                full_text = art["title"] + "\n\n" + art["content"]
                art["token_count"] = self._get_token_count(full_text)
            else:
                full_text = art["title"] + "\n\n" + art["content"]
                art["token_count"] = len(full_text.split())

            # Validate sentiment scores if present
            if "sentiment" in art and isinstance(art["sentiment"], dict):
                if not self.validator.validate_sentiment_scores(art["sentiment"]):
                    logger.warning(
                        f"Invalid sentiment scores for article {art.get('article_hash', 'unknown')}")
                    # Default to neutral
                    art["sentiment"] = {"pos": 0.0, "neu": 1.0, "neg": 0.0}

        # Build DataFrame
        df = pd.DataFrame(articles)
        df = df[
            [
                "date",
                "title",
                "content",
                "symbols",
                "tags",
                "tag_source",
                "sentiment_label",
                "sentiment",
                "token_count",
                "link",
                "article_hash",
            ]
        ]

        return df

    def save_checkpoint(self, articles: List[Dict], tag: str):
        """Save checkpoint of collected articles"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_file = (
            CHECKPOINTS_DIR /
            f"checkpoint_{tag.replace(' ', '_')}_{timestamp}.json"
        )

        # Save quality stats too
        checkpoint_data = {
            "tag": tag,
            "count": len(articles),
            "timestamp": timestamp,
            "quality_stats": self.quality_stats,
            "articles": articles[-100:],  # Save last 100 as sample
        }

        with open(checkpoint_file, "w", encoding="utf-8") as f:
            json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Checkpoint saved: {checkpoint_file}")

    def _label_by_max_prob(self, sent_dict: Dict) -> str:
        """Label sentiment by maximum probability"""
        if not sent_dict or not isinstance(sent_dict, dict):
            return "Neutral"

        neg = sent_dict.get("neg", 0.0)
        neu = sent_dict.get("neu", 0.0)
        pos = sent_dict.get("pos", 0.0)

        label, _ = max(
            [("Negative", neg), ("Neutral", neu), ("Positive", pos)],
            key=lambda x: x[1],
        )
        return label

    def _get_token_count(self, text: str) -> int:
        """Count tokens in text"""
        try:
            return len(self.tokenizer.tokenize(text))
        except Exception:
            return len(text.split())

    # Unit test with mocking
    def test_fetch_page(self):
        """Test fetch_page functionality with mock"""
        try:
            # Mock the requests.get call
            with patch("requests.get") as mock_get:
                # Setup mock response
                mock_response = Mock()
                mock_response.json.return_value = [
                    {
                        "title": "Test Article",
                        "content": "Test content with enough words " * 20,  # Make it pass validation
                        "link": "http://test.com/1",
                        "date": "2024-01-01",
                        "symbols": ["AAPL"],
                    }
                ]
                mock_response.raise_for_status = Mock()
                mock_get.return_value = mock_response

                # Test
                test_tag = "earnings"
                result = self.fetch_page(test_tag, 0)

                assert result is not None, "Fetch returned None"
                assert isinstance(result, list), "Fetch didn't return a list"
                assert len(result) == 1, "Mock should return 1 article"
                logger.info(
                    f"✅ fetch_page test passed: got {len(result)} articles")
                return True

        except Exception as e:
            logger.error(f"❌ fetch_page test failed: {e}")
            return False


class DataLoader:
    """Handles loading of processed data files"""

    @staticmethod
    def load_parquet(filepath: Union[str, Path], engine: str = "pyarrow") -> pd.DataFrame:
        """Load parquet file with error handling"""
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")

        try:
            df = pd.read_parquet(filepath, engine=engine)
            logger.info(f"Loaded {len(df)} records from {filepath}")
            return df
        except Exception as e:
            logger.error(f"Error loading parquet: {e}")
            # Try alternative engine
            alt_engine = "fastparquet" if engine == "pyarrow" else "pyarrow"
            logger.info(f"Trying alternative engine: {alt_engine}")
            return pd.read_parquet(filepath, engine=alt_engine)

    @staticmethod
    def load_jsonl(filepath: Union[str, Path]) -> List[Dict]:
        """Load JSONL file"""
        filepath = Path(filepath)
        results = []

        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning(f"Skipping invalid JSON line")
                    continue

        logger.info(f"Loaded {len(results)} records from {filepath}")
        return results

    @staticmethod
    def save_jsonl(data: List[Dict], filepath: Union[str, Path]):
        """Save data to JSONL file"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w", encoding="utf-8") as f:
            for record in data:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        logger.info(f"Saved {len(data)} records to {filepath}")

    # Unit test
    @staticmethod
    def test_jsonl_io():
        """Test JSONL I/O functionality"""
        test_data = [
            {"id": 1, "text": "test article 1"},
            {"id": 2, "text": "test article 2"}
        ]
        test_file = Path("/tmp/test_loader.jsonl")

        try:
            DataLoader.save_jsonl(test_data, test_file)
            loaded = DataLoader.load_jsonl(test_file)
            assert len(loaded) == len(test_data), "Length mismatch"
            assert loaded[0]["id"] == 1, "Data mismatch"
            test_file.unlink()  # Clean up
            logger.info("✅ JSONL I/O test passed")
            return True
        except Exception as e:
            logger.error(f"❌ JSONL I/O test failed: {e}")
            return False


# Main collection function
def main():
    """Run data collection"""
    import argparse

    parser = argparse.ArgumentParser(description="Collect financial news data")
    parser.add_argument('--config', type=str, help='Path to YAML config file')
    parser.add_argument('--tags', nargs='+', help='Custom tags to collect')
    parser.add_argument('--output', type=Path,
                        default=DATA_DIR / "financial_news_2020_2025_100k.parquet",
                        help='Output parquet file')
    parser.add_argument('--test', action='store_true', help='Run tests only')

    args = parser.parse_args()

    # Load config
    from utils.config_loader import load_config
    config = load_config(args.config)

    # Initialize collector
    try:
        collector = NewsDataCollector(config=config.get("data_collection"))
    except ValueError as e:
        logger.error(str(e))
        return 1

    # Run tests if requested
    if args.test:
        logger.info("Running data loader tests...")
        collector.test_fetch_page()
        DataLoader.test_jsonl_io()
        return 0

    # Collect articles
    tags = args.tags or config.get("collection_tags", COLLECTION_TAGS)
    articles = collector.collect_all_articles(tags=tags)

    if not articles:
        logger.error("No articles collected")
        return 1

    # Process to DataFrame
    df = collector.process_for_dataset(articles)

    # Save to parquet
    df.to_parquet(args.output, index=False)
    logger.info(f"✅ Dataset saved to {args.output}")

    # Save metadata
    metadata = {
        "collection_date": datetime.now().isoformat(),
        "total_articles": len(df),
        "date_range": f"{config['data_collection']['from_date']} to {config['data_collection']['to_date']}",
        "tags_used": tags,
        "quality_stats": collector.quality_stats,
        "columns": list(df.columns),
        "sentiment_distribution": df["sentiment_label"].value_counts().to_dict(),
        "avg_token_count": float(df["token_count"].mean()),
    }

    metadata_file = args.output.parent / f"{args.output.stem}_metadata.json"
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    logger.info(f"✅ Metadata saved to {metadata_file}")

    # Display statistics
    logger.info(f"\n📊 Dataset Statistics:")
    logger.info(f"Articles by sentiment:")
    logger.info(df["sentiment_label"].value_counts())
    logger.info(f"\nToken count statistics:")
    logger.info(df["token_count"].describe())

    return 0


if __name__ == "__main__":
    exit(main())
