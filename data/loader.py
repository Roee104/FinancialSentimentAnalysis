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
from typing import Dict, List, Optional, Union
import logging
from pathlib import Path
import hashlib
from tenacity import retry, stop_after_attempt, wait_exponential
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor

from config.settings import (
    DATA_COLLECTION, EODHD_API_TOKEN, MODELS,
    COLLECTION_TAGS, LOW_QUALITY_PATTERNS, DATA_DIR, CHECKPOINTS_DIR
)
from data.validator import DataValidator

logger = logging.getLogger(__name__)


class NewsDataCollector:
    """Handles collection of financial news data from EODHD API"""

    def __init__(self,
                 api_token: str = None,
                 config: dict = None):
        """
        Initialize data collector

        Args:
            api_token: EODHD API token
            config: Configuration dict (uses settings if None)
        """
        self.api_token = api_token or EODHD_API_TOKEN
        self.config = config or DATA_COLLECTION
        self.validator = DataValidator()

        # Initialize tokenizer for token counting
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                MODELS["bert_tokenizer"])
        except Exception as e:
            logger.warning(f"Could not load tokenizer: {e}")
            self.tokenizer = None

        # Statistics
        self.quality_stats = {
            'total_fetched': 0,
            'quality_filtered': 0,
            'duplicate_filtered': 0,
            'final_count': 0,
            'failed_tags': []
        }

        # State persistence
        self.state_dir = CHECKPOINTS_DIR / "crawler_state"
        self.state_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Initialized NewsDataCollector")

    def _get_state_file(self, tag: str) -> Path:
        """Get state file path for a tag"""
        safe_tag = tag.replace(' ', '_').replace('/', '_')
        return self.state_dir / f"{safe_tag}_state.json"

    def _load_state(self, tag: str) -> Dict:
        """Load crawler state for a tag"""
        state_file = self._get_state_file(tag)
        if state_file.exists():
            try:
                with open(state_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load state for {tag}: {e}")
        return {
            'last_offset': 0,
            'collected_count': 0,
            'retry_count': 0,
            'last_timestamp': None
        }

    def _save_state(self, tag: str, state: Dict):
        """Save crawler state for a tag"""
        state_file = self._get_state_file(tag)
        try:
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save state for {tag}: {e}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10)
    )
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
            "fmt": "json"
        }

        try:
            response = requests.get(
                self.config["base_url"],
                params=params,
                timeout=30
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
        offset = state['last_offset']
        tag_collected = {}
        tag_quality_filtered = 0
        tag_duplicate_filtered = 0
        consecutive_failures = 0
        pages_fetched = 0

        # Continue from saved state
        if state['collected_count'] > 0:
            logger.info(
                f"Resuming from offset {offset}, already collected {state['collected_count']}")

        while (len(tag_collected) + state['collected_count'] < self.config["target_per_tag"] and
               consecutive_failures < self.config["max_retries"] and
               pages_fetched < self.config.get("max_pages_per_tag", 50)):

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
                    self.quality_stats['total_fetched'] += 1

                    link = art.get("link")
                    if not link:
                        continue

                    # Create article hash for deduplication
                    article_hash = hashlib.md5(
                        f"{art.get('title', '')}_{art.get('content', '')}".encode()
                    ).hexdigest()

                    if article_hash in tag_collected:
                        continue

                    # Quality validation
                    if not self.validator.validate_article_quality(art):
                        self.quality_stats['quality_filtered'] += 1
                        tag_quality_filtered += 1
                        continue

                    # Duplicate detection
                    if self.validator.detect_near_duplicates(
                        art, list(tag_collected.values())
                    ):
                        self.quality_stats['duplicate_filtered'] += 1
                        tag_duplicate_filtered += 1
                        continue

                    # Add valid article
                    tag_collected[article_hash] = {
                        "date": art.get("date"),
                        "title": art.get("title", ""),
                        "content": art.get("content", ""),
                        "symbols": art.get("symbols", []),
                        "tags": art.get("tags", []),
                        "sentiment": art.get("sentiment", {}),
                        "tag_source": tag,
                        "link": link,
                        "article_hash": article_hash
                    }
                    new_count += 1

                logger.info(f"  Offset {offset}: fetched {len(data)}, "
                            f"+{new_count} valid, total = {len(tag_collected) + state['collected_count']}")

                # Update state
                state['last_offset'] = offset + self.config["batch_size"]
                state['collected_count'] += new_count
                state['last_timestamp'] = datetime.now().isoformat()
                self._save_state(tag, state)

                if len(tag_collected) + state['collected_count'] >= self.config["target_per_tag"]:
                    break

                offset += self.config["batch_size"]
                time.sleep(self.config["sleep_sec"])

            except Exception as e:
                consecutive_failures += 1
                logger.warning(
                    f"Attempt {consecutive_failures}/{self.config['max_retries']} failed: {e}")

                if consecutive_failures < self.config["max_retries"]:
                    # Exponential backoff
                    sleep_time = self.config["initial_retry_delay"] * \
                        (self.config["backoff_factor"] ** consecutive_failures)
                    logger.info(f"Retrying after {sleep_time:.1f} seconds...")
                    time.sleep(sleep_time)
                else:
                    logger.error(f"Max retries reached for tag '{tag}'")
                    self.quality_stats['failed_tags'].append(tag)
                    break

        # Summary
        tag_articles = list(tag_collected.values())
        logger.info(f"✅ Tag '{tag}' completed:")
        logger.info(f"   Collected: {len(tag_articles)} articles")
        logger.info(f"   Quality filtered: {tag_quality_filtered}")
        logger.info(f"   Duplicate filtered: {tag_duplicate_filtered}")
        logger.info(f"   Failed attempts: {consecutive_failures}")

        return tag_articles

    async def fetch_page_async(self, session: aiohttp.ClientSession, tag: str, offset: int) -> Optional[List[Dict]]:
        """Async version of fetch_page"""
        params = {
            "t": tag,
            "from": self.config["from_date"],
            "to": self.config["to_date"],
            "limit": self.config["batch_size"],
            "offset": offset,
            "api_token": self.api_token,
            "fmt": "json"
        }

        try:
            async with session.get(self.config["base_url"], params=params, timeout=30) as response:
                response.raise_for_status()
                return await response.json()
        except Exception as e:
            logger.error(
                f"Async fetch error for tag '{tag}' at offset {offset}: {e}")
            return None

    def collect_all_articles(self,
                             tags: List[str] = None,
                             checkpoint_interval: int = 10,
                             use_async: bool = False) -> List[Dict]:
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

        # Final statistics
        self.quality_stats['final_count'] = len(all_articles)
        logger.info(f"\n🎯 Collection Summary:")
        logger.info(
            f"Total articles fetched: {self.quality_stats['total_fetched']}")
        logger.info(
            f"Quality filtered out: {self.quality_stats['quality_filtered']}")
        logger.info(
            f"Duplicates filtered out: {self.quality_stats['duplicate_filtered']}")
        logger.info(
            f"Final collection size: {self.quality_stats['final_count']}")

        if self.quality_stats['failed_tags']:
            logger.warning(
                f"Failed tags: {', '.join(self.quality_stats['failed_tags'])}")

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

        # Build DataFrame
        df = pd.DataFrame(articles)
        df = df[[
            "date", "title", "content", "symbols", "tags", "tag_source",
            "sentiment_label", "sentiment", "token_count", "link", "article_hash"
        ]]

        return df

    def save_checkpoint(self, articles: List[Dict], tag: str):
        """Save checkpoint of collected articles"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_file = CHECKPOINTS_DIR / \
            f"checkpoint_{tag.replace(' ', '_')}_{timestamp}.json"

        # Save quality stats too
        checkpoint_data = {
            'tag': tag,
            'count': len(articles),
            'timestamp': timestamp,
            'quality_stats': self.quality_stats,
            'articles': articles[-100:]  # Save last 100 as sample
        }

        with open(checkpoint_file, 'w', encoding='utf-8') as f:
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
            key=lambda x: x[1]
        )
        return label

    def _get_token_count(self, text: str) -> int:
        """Count tokens in text"""
        try:
            return len(self.tokenizer.tokenize(text))
        except Exception:
            return len(text.split())

    # Unit test
    def test_fetch_page(self):
        """Test fetch_page functionality"""
        try:
            # Test with a small request
            test_tag = self.config.get('collection_tags', ['earnings'])[0]
            result = self.fetch_page(test_tag, 0)
            assert result is not None, "Fetch returned None"
            assert isinstance(result, list), "Fetch didn't return a list"
            logger.info(
                f"✅ fetch_page test passed: got {len(result)} articles")
            return True
        except Exception as e:
            logger.error(f"❌ fetch_page test failed: {e}")
            return False


class DataLoader:
    """Handles loading of processed data files"""

    @staticmethod
    def load_parquet(filepath: Union[str, Path],
                     engine: str = "pyarrow") -> pd.DataFrame:
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

        with open(filepath, 'r', encoding='utf-8') as f:
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

        with open(filepath, 'w', encoding='utf-8') as f:
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


# Main collection function for backward compatibility
def main():
    """Run data collection"""
    # Load config
    from utils.config_loader import load_config
    config = load_config()

    collector = NewsDataCollector(config=config.get('data_collection'))

    # Run tests first
    logger.info("Running data loader tests...")
    collector.test_fetch_page()
    DataLoader.test_jsonl_io()

    # Collect articles
    articles = collector.collect_all_articles()

    if not articles:
        logger.error("No articles collected")
        return

    # Process to DataFrame
    df = collector.process_for_dataset(articles)

    # Save to parquet
    output_file = Path(config.get('input_parquet', DATA_DIR /
                       "financial_news_2020_2025_100k.parquet"))
    df.to_parquet(output_file, index=False)
    logger.info(f"✅ Dataset saved to {output_file}")

    # Save metadata
    metadata = {
        "collection_date": datetime.now().isoformat(),
        "total_articles": len(df),
        "date_range": f"{config['data_collection']['from_date']} to {config['data_collection']['to_date']}",
        "tags_used": config.get('collection_tags', COLLECTION_TAGS),
        "quality_stats": collector.quality_stats,
        "columns": list(df.columns)
    }

    metadata_file = DATA_DIR / "collection_metadata.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    logger.info(f"✅ Metadata saved to {metadata_file}")

    # Display statistics
    logger.info(f"\n📊 Dataset Statistics:")
    logger.info(f"Articles by sentiment:")
    logger.info(df['sentiment_label'].value_counts())
    logger.info(f"\nToken count statistics:")
    logger.info(df['token_count'].describe())


if __name__ == "__main__":
    main()
