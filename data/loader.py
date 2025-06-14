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
            'final_count': 0
        }

        logger.info("Initialized NewsDataCollector")

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
            return None
        except json.JSONDecodeError as e:
            logger.error(
                f"Error parsing JSON for tag '{tag}' at offset {offset}: {e}")
            return None

    def collect_articles_for_tag(self, tag: str) -> List[Dict]:
        """
        Collect articles for a single tag

        Args:
            tag: Tag to search for

        Returns:
            List of validated articles
        """
        logger.info(f"Processing tag: '{tag}'")
        offset = 0
        tag_collected = {}  # Use link as key for deduplication
        tag_quality_filtered = 0
        tag_duplicate_filtered = 0

        while len(tag_collected) < self.config["target_per_tag"]:
            # Fetch page
            data = self.fetch_page(tag, offset)

            if data is None:
                logger.info("Retrying after error...")
                time.sleep(5)
                continue

            if not data:
                logger.info(f"No more articles found for tag: '{tag}'")
                break

            new_count = 0
            for art in data:
                self.quality_stats['total_fetched'] += 1

                link = art.get("link")
                if not link or link in tag_collected:
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
                tag_collected[link] = {
                    "date": art.get("date"),
                    "title": art.get("title", ""),
                    "content": art.get("content", ""),
                    "symbols": art.get("symbols", []),
                    "tags": art.get("tags", []),
                    "sentiment": art.get("sentiment", {}),
                    "tag_source": tag,
                    "link": link
                }
                new_count += 1

            logger.info(f"  Offset {offset}: fetched {len(data)}, "
                        f"+{new_count} valid, total = {len(tag_collected)}")

            if len(tag_collected) >= self.config["target_per_tag"]:
                break

            offset += self.config["batch_size"]
            time.sleep(self.config["sleep_sec"])

        # Summary
        tag_articles = list(tag_collected.values())
        logger.info(f"✅ Tag '{tag}' completed:")
        logger.info(f"   Collected: {len(tag_articles)} articles")
        logger.info(f"   Quality filtered: {tag_quality_filtered}")
        logger.info(f"   Duplicate filtered: {tag_duplicate_filtered}")

        return tag_articles

    def collect_all_articles(self,
                             tags: List[str] = None,
                             checkpoint_interval: int = 10) -> List[Dict]:
        """
        Collect articles for all tags

        Args:
            tags: List of tags (uses default if None)
            checkpoint_interval: Save checkpoint every N tags

        Returns:
            List of all collected articles
        """
        tags = tags or COLLECTION_TAGS
        all_articles = []

        logger.info(f"Starting data collection for {len(tags)} tags...")
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
            "sentiment_label", "sentiment", "token_count", "link"
        ]]

        return df

    def save_checkpoint(self, articles: List[Dict], tag: str):
        """Save checkpoint of collected articles"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_file = CHECKPOINTS_DIR / \
            f"checkpoint_{tag.replace(' ', '_')}_{timestamp}.json"

        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump({
                'tag': tag,
                'count': len(articles),
                'timestamp': timestamp,
                'articles': articles[-100:]  # Save last 100 as sample
            }, f, indent=2, ensure_ascii=False)

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


# Main collection function for backward compatibility
def main():
    """Run data collection"""
    collector = NewsDataCollector()

    # Collect articles
    articles = collector.collect_all_articles()

    if not articles:
        logger.error("No articles collected")
        return

    # Process to DataFrame
    df = collector.process_for_dataset(articles)

    # Save to parquet
    output_file = DATA_DIR / "financial_news_2020_2025_100k.parquet"
    df.to_parquet(output_file, index=False)
    logger.info(f"✅ Dataset saved to {output_file}")

    # Save metadata
    metadata = {
        "collection_date": datetime.now().isoformat(),
        "total_articles": len(df),
        "date_range": f"{DATA_COLLECTION['from_date']} to {DATA_COLLECTION['to_date']}",
        "tags_used": COLLECTION_TAGS,
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
