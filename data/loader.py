# ────────────────────────────────────────────────────────────────────────────
#  data/loader.py
#  Data loading and collection module for financial news
# ────────────────────────────────────────────────────────────────────────────
from __future__ import annotations
from data.validator import DataValidator
from unittest.mock import Mock, patch
from transformers import AutoTokenizer
from tenacity import retry, stop_after_attempt, wait_exponential
import requests
import pandas as pd
from typing import Dict, List, Optional, Set, Union
from pathlib import Path
from datetime import datetime
from collections import deque
import time
import json
import hashlib


# ----------------------------------------------------------------- logging FIRST
import logging
import logging.config
from config.settings import LOGGING_CONFIG
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------- stdlib

# ----------------------------------------------------------------- third-party

# ----------------------------------------------------------------- internal
from config.settings import (
    CACHE_DIR,
    CHECKPOINTS_DIR,
    COLLECTION_TAGS,
    DATA_COLLECTION,
    DATA_DIR,
    EODHD_API_TOKEN,
    LOW_QUALITY_PATTERNS,  
    MODELS,
    QUALITY_FILTER_STATS,
)

# ────────────────────────────────────────────────────────────────────────────


class NewsDataCollector:
    """Handles collection of financial news data from EODHD API"""

    def __init__(self, api_token: str | None = None, config: dict | None = None):
        self.api_token = api_token or EODHD_API_TOKEN
        if not self.api_token:
            raise ValueError(
                "EODHD_API_TOKEN not set. Please set it in your environment or .env file"
            )

        self.config = config or DATA_COLLECTION
        self.validator = DataValidator()

        # Initialise tokenizer (cached)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                MODELS["bert_tokenizer"],
                cache_dir=MODELS.get("cache_dir", CACHE_DIR / "models"),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not load tokenizer: %s", exc)
            self.tokenizer = None

        # Stats
        self.quality_stats = {
            "total_fetched": 0,
            "quality_filtered": 0,
            "duplicate_filtered": 0,
            "final_count": 0,
            "failed_tags": [],
            "filter_reasons": QUALITY_FILTER_STATS.copy(),
        }

        # Duplicate detection helpers
        self.seen_hashes: Set[str] = set()
        self.recent_articles: deque = deque(
            maxlen=self.config.get("duplicate_check_window", 1000)
        )

        # State persistence
        self.state_dir = CHECKPOINTS_DIR / "crawler_state"
        self.state_dir.mkdir(parents=True, exist_ok=True)

        # CSV for quality filter stats
        self.stats_file = DATA_DIR / "quality_filter_stats.csv"

        logger.info("Initialized NewsDataCollector")

    # --------------------------------------------------------------------- helpers
    def _get_state_file(self, tag: str) -> Path:
        safe_tag = tag.replace(" ", "_").replace("/", "_")
        return self.state_dir / f"{safe_tag}_state.json"

    def _load_state(self, tag: str) -> Dict:
        fp = self._get_state_file(tag)
        if fp.exists():
            try:
                return json.loads(fp.read_text())
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to load state for %s: %s", tag, exc)
        return dict(last_offset=0, collected_count=0, retry_count=0, last_timestamp=None)

    def _save_state(self, tag: str, state: Dict) -> None:
        try:
            self._get_state_file(tag).write_text(json.dumps(state, indent=2))
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to save state for %s: %s", tag, exc)

    def _save_quality_stats(self) -> None:
        try:
            df = pd.DataFrame(
                {
                    "reason": list(self.quality_stats["filter_reasons"]),
                    "count": list(self.quality_stats["filter_reasons"].values()),
                }
            )
            total = self.quality_stats["total_fetched"] or 1
            df["percentage"] = df["count"] * 100 / total
            df.to_csv(self.stats_file, index=False)
            logger.info("Saved quality stats → %s", self.stats_file)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to save quality stats: %s", exc)

    # --------------------------------------------------------------------- network
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    def fetch_page(self, tag: str, offset: int) -> Optional[List[Dict]]:
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
            resp = requests.get(
                self.config["base_url"], params=params, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            logger.error("Error fetching tag %s offset %s: %s",
                         tag, offset, exc)
            raise
        except json.JSONDecodeError as exc:
            logger.error(
                "JSON parse error for tag %s offset %s: %s", tag, offset, exc)
            raise

    # --------------------------------------------------------------------- de-dupe
    def _is_duplicate(self, article: Dict) -> bool:
        article_hash = hashlib.md5(
            f"{article.get('title', '')}_{article.get('content', '')}".encode()
        ).hexdigest()

        if article_hash in self.seen_hashes:
            return True

        if self.validator.detect_near_duplicates(article, list(self.recent_articles)):
            return True

        return False

    def _add_to_duplicate_check(self, article_hash: str, article: Dict) -> None:
        self.seen_hashes.add(article_hash)
        self.recent_articles.append(article)

    # --------------------------------------------------------------------- main tag loop
    def collect_articles_for_tag(self, tag: str) -> List[Dict]:
        logger.info("Processing tag: %s", tag)
        state = self._load_state(tag)
        offset = state["last_offset"]

        tag_collected: dict[str, Dict] = {}
        tag_quality_filtered = tag_duplicate_filtered = 0
        consecutive_failures = pages_fetched = 0

        if state["collected_count"]:
            logger.info("Resuming @ offset %s (collected %s)",
                        offset, state["collected_count"])

        while (
            len(tag_collected) +
                state["collected_count"] < self.config["target_per_tag"]
            and consecutive_failures < self.config["max_retries"]
            and pages_fetched < self.config.get("max_pages_per_tag", 50)
        ):
            try:
                data = self.fetch_page(tag, offset)
                if not data:
                    logger.info("No more results for %s", tag)
                    break

                consecutive_failures = 0
                pages_fetched += 1
                new_count = 0

                for art in data:
                    self.quality_stats["total_fetched"] += 1
                    if not art.get("link"):
                        continue

                    article_hash = hashlib.md5(
                        f"{art.get('title', '')}_{art.get('content', '')}".encode()
                    ).hexdigest()

                    if article_hash in tag_collected:
                        continue

                    # Quality validation
                    is_valid, reason = self.validator.validate_article_quality_with_reason(
                        art)
                    if not is_valid:
                        self.quality_stats["quality_filtered"] += 1
                        self.quality_stats["filter_reasons"][reason] += 1
                        tag_quality_filtered += 1
                        continue

                    # Duplicate detection
                    if self._is_duplicate(art):
                        self.quality_stats["duplicate_filtered"] += 1
                        tag_duplicate_filtered += 1
                        continue

                    # Symbol cleaning
                    clean_symbols = self.validator.validate_symbols(
                        art.get("symbols", []))

                    tag_collected[article_hash] = {
                        "date": art.get("date"),
                        "title": art.get("title", ""),
                        "content": art.get("content", ""),
                        "symbols": clean_symbols,
                        "tags": art.get("tags", []),
                        "sentiment": art.get("sentiment", {}),
                        "tag_source": tag,
                        "link": art["link"],
                        "article_hash": article_hash,
                    }
                    self._add_to_duplicate_check(
                        article_hash, tag_collected[article_hash])
                    new_count += 1

                logger.info(
                    "Offset %s: fetched %s, +%s valid (total %s)",
                    offset,
                    len(data),
                    new_count,
                    len(tag_collected) + state["collected_count"],
                )

                # Update state
                state.update(
                    last_offset=offset + self.config["batch_size"],
                    collected_count=state["collected_count"] + new_count,
                    last_timestamp=datetime.now().isoformat(),
                )
                self._save_state(tag, state)

                if len(tag_collected) + state["collected_count"] >= self.config["target_per_tag"]:
                    break

                offset += self.config["batch_size"]
                time.sleep(self.config["sleep_sec"])

            except Exception as exc:  # noqa: BLE001
                consecutive_failures += 1
                logger.warning("Attempt %s/%s failed: %s",
                               consecutive_failures, self.config["max_retries"], exc)
                if consecutive_failures < self.config["max_retries"]:
                    backoff = self.config["initial_retry_delay"] * \
                        (self.config["backoff_factor"] ** consecutive_failures)
                    logger.info("Retrying in %.1f s", backoff)
                    time.sleep(backoff)
                else:
                    logger.error("Max retries reached for tag %s", tag)
                    self.quality_stats["failed_tags"].append(tag)
                    break

        logger.info(
            "✅ Tag '%s' done. +%s collected, %s quality-filtered, %s duplicates",
            tag,
            len(tag_collected),
            tag_quality_filtered,
            tag_duplicate_filtered,
        )
        return list(tag_collected.values())

    # --------------------------------------------------------------------- all tags wrapper
    def collect_all_articles(
        self,
        tags: List[str] | None = None,
        checkpoint_interval: int = 10,
        use_async: bool = False,  # kept for future, not implemented
    ) -> List[Dict]:
        tags = tags or COLLECTION_TAGS
        all_articles: List[Dict] = []

        logger.info("🚀 Collecting %s tags (%s per tag)",
                    len(tags), self.config["target_per_tag"])
        logger.info("Date range: %s → %s",
                    self.config["from_date"], self.config["to_date"])

        for idx, tag in enumerate(tags, 1):
            logger.info("\n[%s/%s] %s", idx, len(tags), tag)
            all_articles.extend(self.collect_articles_for_tag(tag))

            if idx % checkpoint_interval == 0:
                self.save_checkpoint(all_articles, f"tags_1_to_{idx}")
                self._save_quality_stats()

        # Summary
        self.quality_stats["final_count"] = len(all_articles)
        logger.info("🎯 Total fetched: %s", self.quality_stats["total_fetched"])
        logger.info("    Quality-filtered: %s",
                    self.quality_stats["quality_filtered"])
        logger.info("    Duplicates: %s",
                    self.quality_stats["duplicate_filtered"])
        logger.info("    Final: %s", self.quality_stats["final_count"])
        self._save_quality_stats()

        return all_articles

    # --------------------------------------------------------------------- dataframe + save helpers
    def process_for_dataset(self, articles: List[Dict]) -> pd.DataFrame:
        logger.info("Processing %s articles into DataFrame …", len(articles))
        for art in articles:
            art["sentiment_label"] = self._label_by_max_prob(
                art.get("sentiment", {}))
            full_text = f"{art['title']}\n\n{art['content']}"
            art["token_count"] = (
                len(self.tokenizer.tokenize(full_text)
                    ) if self.tokenizer else len(full_text.split())
            )
            if "sentiment" in art and isinstance(art["sentiment"], dict):
                if not self.validator.validate_sentiment_scores(art["sentiment"]):
                    art["sentiment"] = {"pos": 0.0, "neu": 1.0, "neg": 0.0}

        df = pd.DataFrame(articles)
        return df[
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

    def save_checkpoint(self, articles: List[Dict], tag: str) -> None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fp = CHECKPOINTS_DIR / f"checkpoint_{tag.replace(' ', '_')}_{ts}.json"
        fp.write_text(
            json.dumps(
                {
                    "tag": tag,
                    "count": len(articles),
                    "timestamp": ts,
                    "quality_stats": self.quality_stats,
                    "articles": articles[-100:],
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        logger.info("Checkpoint saved → %s", fp)

    # --------------------------------------------------------------------- utility helpers
    @staticmethod
    def _label_by_max_prob(sent_dict: Dict) -> str:
        if not sent_dict or not isinstance(sent_dict, dict):
            return "Neutral"
        neg, neu, pos = sent_dict.get("neg", 0.0), sent_dict.get(
            "neu", 0.0), sent_dict.get("pos", 0.0)
        return max([("Negative", neg), ("Neutral", neu), ("Positive", pos)], key=lambda x: x[1])[0]

    # --------------------------------------------------------------------- quick unit tests (unchanged)
    def test_fetch_page(self) -> bool:  # unchanged
        try:
            with patch("requests.get") as mock_get:
                mock_resp = Mock()
                mock_resp.json.return_value = [
                    {
                        "title": "Test Article",
                        "content": "Test content with enough words " * 20,
                        "link": "http://test.com/1",
                        "date": "2024-01-01",
                        "symbols": ["AAPL"],
                    }
                ]
                mock_resp.raise_for_status = Mock()
                mock_get.return_value = mock_resp

                res = self.fetch_page("earnings", 0)
                assert res and isinstance(res, list) and len(res) == 1
                logger.info("✅ fetch_page test passed")
                return True
        except Exception as exc:  # noqa: BLE001
            logger.error("❌ fetch_page test failed: %s", exc)
            return False


class DataLoader:
    """Loading helpers for processed datasets (unchanged)"""

    @staticmethod
    def load_parquet(filepath: Union[str, Path], engine: str = "pyarrow") -> pd.DataFrame:
        fp = Path(filepath)
        if not fp.exists():
            raise FileNotFoundError(fp)
        try:
            df = pd.read_parquet(fp, engine=engine)
            logger.info("Loaded %s records from %s", len(df), fp)
            return df
        except Exception as exc:
            alt = "fastparquet" if engine == "pyarrow" else "pyarrow"
            logger.warning("PyArrow read failed (%s). Trying %s …", exc, alt)
            return pd.read_parquet(fp, engine=alt)

    @staticmethod
    def load_jsonl(filepath: Union[str, Path]) -> List[Dict]:
        fp = Path(filepath)
        return [json.loads(line) for line in fp.read_text(encoding="utf-8").splitlines() if line.strip()]

    @staticmethod
    def save_jsonl(data: List[Dict], filepath: Union[str, Path]) -> None:
        fp = Path(filepath)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text("\n".join(json.dumps(rec, ensure_ascii=False)
                      for rec in data))
        logger.info("Saved %s records → %s", len(data), fp)

    @staticmethod
    def test_jsonl_io() -> bool:  # unchanged
        test_data = [{"id": 1}, {"id": 2}]
        fp = Path("/tmp/test_loader.jsonl")
        try:
            DataLoader.save_jsonl(test_data, fp)
            loaded = DataLoader.load_jsonl(fp)
            assert len(loaded) == 2
            fp.unlink()
            logger.info("✅ JSONL I/O test passed")
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error("❌ JSONL I/O test failed: %s", exc)
            return False


# ------------------------------------------------------------------------- CLI
def main() -> int:
    import argparse
    from utils.config_loader import load_config

    p = argparse.ArgumentParser(description="Collect financial news data")
    p.add_argument("--config", type=str, help="YAML config override")
    p.add_argument("--tags", nargs="+", help="Custom tag list")
    p.add_argument(
        "--output",
        type=Path,
        default=DATA_DIR / "financial_news_2020_2025_100k.parquet",
        help="Destination parquet file",
    )
    p.add_argument("--test", action="store_true",
                   help="Run internal tests and exit")

    args = p.parse_args()
    cfg = load_config(args.config)

    if args.test:
        logger.info("Running loader tests …")
        NewsDataCollector().test_fetch_page()
        DataLoader.test_jsonl_io()
        return 0

    collector = NewsDataCollector(config=cfg["data_collection"])
    tags = args.tags or cfg.get("collection_tags", COLLECTION_TAGS)

    df = collector.process_for_dataset(collector.collect_all_articles(tags))
    df.to_parquet(args.output, index=False)
    logger.info("✅ Dataset saved → %s", args.output)

    metadata = {
        "collection_date": datetime.now().isoformat(),
        "total_articles": len(df),
        "date_range": f"{cfg['data_collection']['from_date']}→{cfg['data_collection']['to_date']}",
        "tags_used": tags,
        "quality_stats": collector.quality_stats,
        "columns": list(df.columns),
        "sentiment_distribution": df["sentiment_label"].value_counts().to_dict(),
        "avg_token_count": float(df["token_count"].mean()),
    }
    meta_fp = args.output.with_name(f"{args.output.stem}_metadata.json")
    meta_fp.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    logger.info("✅ Metadata saved → %s", meta_fp)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
