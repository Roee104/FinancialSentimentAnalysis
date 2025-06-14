# pipelines/base_pipeline.py
"""
Fixed base pipeline with context-aware sentiment assignment
"""

import json
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging
from tqdm import tqdm
import gc
import os

from config.settings import (
    INPUT_PARQUET, PROCESSED_OUTPUT, MASTER_TICKER_LIST,
    PIPELINE_CONFIG, TEXT_PROCESSING
)

logger = logging.getLogger(__name__)


class BasePipeline(ABC):
    """Abstract base class for sentiment analysis pipelines"""

    def __init__(self,
                 input_path: str = None,
                 output_path: str = None,
                 ticker_csv: str = None,
                 batch_size: int = None,
                 resume: bool = True,
                 **kwargs):
        """
        Initialize base pipeline

        Args:
            input_path: Input data file path
            output_path: Output file path
            ticker_csv: Ticker list CSV path
            batch_size: Processing batch size
            resume: Whether to resume from existing output
            **kwargs: Additional configuration
        """
        self.input_path = Path(input_path or INPUT_PARQUET)
        self.output_path = Path(output_path or PROCESSED_OUTPUT)
        self.ticker_csv = Path(ticker_csv or MASTER_TICKER_LIST)

        # Configuration
        self.config = PIPELINE_CONFIG.copy()
        self.config.update(kwargs)
        self.batch_size = batch_size or self.config["batch_size"]
        self.resume = resume

        # Components (to be initialized by subclasses)
        self.sentiment_analyzer = None
        self.ner = None
        self.text_processor = None
        self.aggregator = None

        # Load ticker to company mapping
        self.ticker_to_company = self._load_ticker_to_company()

        # Statistics
        self.stats = {
            'processed': 0,
            'errors': 0,
            'skipped': 0,
            'sentiment_dist': {'Positive': 0, 'Neutral': 0, 'Negative': 0},
            'articles_with_tickers': 0,
            'articles_without_tickers': 0,
            'total_tickers_found': 0,
            'total_chunks_assigned': 0
        }

        # Resume tracking
        self.processed_titles = set()
        self.processed_count = 0

        logger.info(f"Initialized {self.__class__.__name__}")

    def _load_ticker_to_company(self) -> Dict[str, str]:
        """Load ticker to company name mapping"""
        try:
            if self.ticker_csv.exists():
                df = pd.read_csv(self.ticker_csv, dtype=str)
                mapping = dict(
                    zip(df['symbol'].str.upper(), df['company_name']))
                logger.info(f"Loaded {len(mapping)} ticker-company mappings")
                return mapping
        except Exception as e:
            logger.warning(f"Could not load ticker-company mapping: {e}")
        return {}

    @abstractmethod
    def initialize_components(self):
        """Initialize pipeline components - must be implemented by subclasses"""
        pass

    def load_processed_articles(self):
        """Load already processed articles for resuming"""
        if self.resume and self.output_path.exists():
            logger.info("Loading existing processed articles...")
            with open(self.output_path, 'r') as f:
                for line in f:
                    try:
                        record = json.loads(line)
                        self.processed_titles.add(record.get('title', ''))
                        self.processed_count += 1
                    except:
                        continue
            logger.info(
                f"Found {self.processed_count} already processed articles")

    def load_data(self, max_articles: Optional[int] = None,
                  start_from: int = 0) -> pd.DataFrame:
        """Load input data"""
        logger.info(f"Loading data from {self.input_path}")
        df = pd.read_parquet(self.input_path, engine="pyarrow")

        # Apply start_from
        if start_from > 0:
            df = df.iloc[start_from:]
            logger.info(f"Starting from article {start_from}")

        # Limit if requested
        if max_articles:
            df = df.head(max_articles)

        logger.info(f"Loaded {len(df)} articles")
        return df

    def process_article(self, row: pd.Series) -> Optional[Dict]:
        """
        Process a single article with context-aware sentiment assignment

        Args:
            row: DataFrame row containing article data

        Returns:
            Processed article dict or None if error
        """
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

            # Limit content length
            if len(content) > TEXT_PROCESSING["max_content_length"]:
                content = content[:TEXT_PROCESSING["max_content_length"]]

            full_text = f"{title}\n\n{content}"

            # Process text into chunks
            chunks = self.text_processor.split_to_chunks(full_text)
            if not chunks:
                return None

            # Limit chunks
            if len(chunks) > TEXT_PROCESSING["max_chunks"]:
                chunks = chunks[:TEXT_PROCESSING["max_chunks"]]

            # Extract tickers
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

            # Get sentiment predictions for chunks
            predictions = self.sentiment_analyzer.predict(
                chunks,
                batch_size=self.config.get("sentiment_batch_size", 8)
            )

            # Context-aware aggregation
            result = self.aggregator.aggregate_article(
                chunks=chunks,
                predictions=predictions,
                symbols=symbols,
                ticker_to_company=self.ticker_to_company
            )

            # Update sentiment stats
            self.stats['sentiment_dist'][result['overall_sentiment']] += 1

            # Track chunk assignments
            total_assigned = sum(result['chunk_assignments'].values())
            self.stats['total_chunks_assigned'] += total_assigned

            # Build final record
            record = {
                "date": date.isoformat() if hasattr(date, "isoformat") else str(date),
                "title": title[:500],  # Limit title length
                "content": content[:1000],  # Store truncated content
                "overall_sentiment": result['overall_sentiment'],
                "overall_confidence": float(result['overall_confidence']),
                "tickers": result['ticker_sentiments'],
                "sectors": result['sector_sentiments'],
                "ticker_count": len(symbols),
                "chunk_count": len(chunks),
                "chunks_assigned": result['chunk_assignments']
            }

            return record

        except Exception as e:
            logger.error(f"Error processing article: {str(e)[:100]}")
            self.stats['errors'] += 1
            return None

    def process_batch(self, batch_df: pd.DataFrame) -> List[Dict]:
        """Process a batch of articles"""
        results = []

        for idx, row in batch_df.iterrows():
            result = self.process_article(row)
            if result:
                results.append(result)
                self.stats['processed'] += 1

        return results

    def run(self,
            max_articles: Optional[int] = None,
            start_from: int = 0,
            checkpoint_interval: Optional[int] = None):
        """
        Run the pipeline

        Args:
            max_articles: Maximum number of articles to process
            start_from: Starting index
            checkpoint_interval: Save checkpoint every N batches
        """
        logger.info(f"Starting {self.__class__.__name__} pipeline")
        logger.info(f"Batch size: {self.batch_size}")
        logger.info(f"Max articles: {max_articles or 'All'}")

        # Initialize components
        self.initialize_components()

        # Load processed articles for resume
        self.load_processed_articles()

        # Load data
        df = self.load_data(max_articles, start_from)

        # Determine output mode
        mode = 'a' if (self.resume and self.output_path.exists()) else 'w'

        # Process in batches
        total_batches = (len(df) + self.batch_size - 1) // self.batch_size

        with open(self.output_path, mode, encoding='utf-8') as fout:
            for batch_idx in tqdm(range(0, len(df), self.batch_size),
                                  total=total_batches,
                                  desc="Processing batches"):

                # Get batch
                batch_df = df.iloc[batch_idx:batch_idx + self.batch_size]

                # Process batch
                try:
                    results = self.process_batch(batch_df)

                    # Write results
                    for record in results:
                        fout.write(json.dumps(
                            record, ensure_ascii=False) + "\n")

                        # Update processed titles
                        if self.resume:
                            self.processed_titles.add(record['title'])

                    # Flush to disk
                    fout.flush()

                except Exception as e:
                    logger.error(f"Error in batch at {batch_idx}: {e}")
                    self.stats['errors'] += 1

                # Memory cleanup
                if (batch_idx // self.batch_size) % 5 == 0:
                    gc.collect()

                # Progress update
                if (batch_idx // self.batch_size) % 10 == 0 and batch_idx > 0:
                    self._print_progress()

                # Checkpoint
                if checkpoint_interval and (batch_idx // self.batch_size) % checkpoint_interval == 0:
                    self._save_checkpoint(batch_idx)

        # Final report
        self._print_final_report()

    def _print_progress(self):
        """Print progress update"""
        total = self.stats['processed'] + self.processed_count
        logger.info(f"Progress: {total} total articles processed")
        if self.stats['processed'] > 0:
            logger.info(
                f"Recent sentiment: {dict(self.stats['sentiment_dist'])}")

            # Show chunk assignment rate
            if self.stats['total_chunks_assigned'] > 0:
                avg_assigned = self.stats['total_chunks_assigned'] / \
                    self.stats['processed']
                logger.info(
                    f"Average chunks assigned per article: {avg_assigned:.1f}")

    def _print_final_report(self):
        """Print final statistics"""
        print("\n" + "="*50)
        print("📊 PIPELINE STATISTICS")
        print("="*50)
        print(f"New articles processed: {self.stats['processed']}")
        print(f"Previously processed: {self.processed_count}")
        print(
            f"Total processed: {self.stats['processed'] + self.processed_count}")
        print(f"Skipped: {self.stats['skipped']}")
        print(f"Errors: {self.stats['errors']}")

        if self.stats['processed'] > 0:
            print("\nSentiment Distribution:")
            total = self.stats['processed']
            for sentiment, count in self.stats['sentiment_dist'].items():
                pct = count / total * 100 if total > 0 else 0
                print(f"  {sentiment}: {count} ({pct:.1f}%)")

            print(f"\nTicker Coverage:")
            print(f"  Articles with tickers: {self.stats['articles_with_tickers']} " +
                  f"({self.stats['articles_with_tickers']/total*100:.1f}%)")
            print(f"  Average tickers/article: " +
                  f"{self.stats['total_tickers_found']/total:.2f}")

            print(f"\nContext-Aware Assignment:")
            print(
                f"  Total chunks assigned: {self.stats['total_chunks_assigned']}")
            print(f"  Average chunks assigned/article: " +
                  f"{self.stats['total_chunks_assigned']/total:.2f}")

        # NER stats
        if self.ner:
            ner_stats = self.ner.get_extraction_stats()
            if ner_stats:
                print("\nNER Extraction Methods:")
                for method, count in sorted(ner_stats.items(),
                                            key=lambda x: x[1],
                                            reverse=True)[:10]:
                    print(f"  {method}: {count}")

        print(f"\n✅ Pipeline completed! Results: {self.output_path}")

    def _save_checkpoint(self, batch_idx: int):
        """Save checkpoint (can be extended by subclasses)"""
        checkpoint = {
            'batch_idx': batch_idx,
            'stats': self.stats,
            'timestamp': datetime.now().isoformat()
        }
        checkpoint_path = self.output_path.parent / \
            f"checkpoint_{batch_idx}.json"
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
