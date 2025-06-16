
"""
Main pipeline implementation for financial sentiment analysis
"""

import logging
from typing import Optional

from pipelines.base_pipeline import BasePipeline
from core.sentiment import UnifiedSentimentAnalyzer
from core.ner import UnifiedNER
from core.text_processor import TextProcessor
from core.aggregator import Aggregator
from config.settings import SENTIMENT_CONFIG

logger = logging.getLogger(__name__)


class FinancialSentimentPipeline(BasePipeline):
    """Main financial sentiment analysis pipeline"""

    def __init__(self,
                 sentiment_mode: str = "optimized",
                 aggregation_method: str = "conf_weighted",
                 aggregation_threshold: float = 0.1,
                 use_distance_weighting: bool = True,
                 **kwargs):
        """
        Initialize main pipeline

        Args:
            sentiment_mode: Sentiment analysis mode (standard/optimized/calibrated)
            aggregation_method: Method for aggregating sentiments
            aggregation_threshold: Threshold for sentiment classification
            use_distance_weighting: Whether to use distance-based weighting in aggregation
            **kwargs: Additional configuration
        """
        super().__init__(**kwargs)

        self.sentiment_mode = sentiment_mode
        self.aggregation_method = aggregation_method
        self.aggregation_threshold = aggregation_threshold
        self.use_distance_weighting = use_distance_weighting

        logger.info(f"Pipeline configured with:")
        logger.info(f"  Sentiment mode: {sentiment_mode}")
        logger.info(
            f"  Aggregation: {aggregation_method} (threshold={aggregation_threshold})")
        logger.info(f"  Distance weighting: {use_distance_weighting}")

    def initialize_components(self):
        """Initialize all pipeline components"""
        logger.info("Initializing pipeline components...")

        # Initialize sentiment analyzer
        logger.info(
            f"Loading sentiment analyzer ({self.sentiment_mode} mode)...")
        self.sentiment_analyzer = UnifiedSentimentAnalyzer(
            mode=self.sentiment_mode,
            device=self.config.get("device"),
            batch_size=self.config.get("sentiment_batch_size", 16)
        )

        # Initialize NER
        logger.info("Loading Enhanced NER...")
        self.ner = UnifiedNER(ticker_csv_path=self.ticker_csv)

        # Initialize text processor
        logger.info("Loading text processor...")
        self.text_processor = TextProcessor()

        # Initialize aggregator
        logger.info("Loading aggregator...")
        self.aggregator = Aggregator(
            method=self.aggregation_method,
            threshold=self.aggregation_threshold,
            use_distance_weighting=self.use_distance_weighting
        )

        logger.info("All components initialized successfully")


class OptimizedPipeline(FinancialSentimentPipeline):
    """Optimized pipeline with bias correction"""

    def __init__(self, **kwargs):
        # Remove use_distance_weighting from kwargs if present
        kwargs.pop('use_distance_weighting', None)

        super().__init__(
            sentiment_mode="optimized",
            aggregation_method="conf_weighted",
            aggregation_threshold=0.1,
            use_distance_weighting=True,
            **kwargs
        )


class StandardPipeline(FinancialSentimentPipeline):
    """Standard pipeline without optimizations"""

    def __init__(self, **kwargs):
        # Remove use_distance_weighting from kwargs if present
        kwargs.pop('use_distance_weighting', None)

        super().__init__(
            sentiment_mode="standard",
            aggregation_method="default",
            aggregation_threshold=0.1,
            use_distance_weighting=False,
            **kwargs
        )


class CalibratedPipeline(FinancialSentimentPipeline):
    """Calibrated pipeline with advanced bias reduction"""

    def __init__(self, **kwargs):
        # Remove use_distance_weighting from kwargs if present
        kwargs.pop('use_distance_weighting', None)

        super().__init__(
            sentiment_mode="calibrated",
            aggregation_method="conf_weighted",
            aggregation_threshold=0.05,
            use_distance_weighting=True,
            **kwargs
        )


# Factory function for creating pipelines
def create_pipeline(pipeline_type: str = "optimized", **kwargs) -> BasePipeline:
    """
    Factory function to create pipeline instances

    Args:
        pipeline_type: Type of pipeline (standard/optimized/calibrated)
        **kwargs: Pipeline configuration

    Returns:
        Pipeline instance
    """
    pipelines = {
        "standard": StandardPipeline,
        "optimized": OptimizedPipeline,
        "calibrated": CalibratedPipeline,
        "main": FinancialSentimentPipeline
    }

    if pipeline_type not in pipelines:
        raise ValueError(f"Unknown pipeline type: {pipeline_type}")

    return pipelines[pipeline_type](**kwargs)
