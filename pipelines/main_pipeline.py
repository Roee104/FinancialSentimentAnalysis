"""
Main pipeline implementation for financial sentiment analysis
(Revised to use the FinBERT‑based `FinancialNERWrapper` instead of the regex‑only
`ImprovedNER`).

"""

import logging
from typing import Optional
import functools

from pipelines.base_pipeline import BasePipeline
from core.sentiment import UnifiedSentimentAnalyzer
from core.text_processor import TextProcessor
from core.aggregator import Aggregator
from config.settings import SENTIMENT_CONFIG

# ▶ NEW: bring in the FinBERT‑powered NER wrapper
from core.pretrained_financial_ner import FinancialNERWrapper

logger = logging.getLogger(__name__)


class FinancialSentimentPipeline(BasePipeline):
    """Main financial sentiment analysis pipeline"""

    def __init__(self,
                 sentiment_mode: str = "optimized",
                 aggregation_method: str = "conf_weighted",
                 agg_method: str = None,  # CLI alias
                 aggregation_threshold: float = 0.1,
                 use_distance_weighting: bool = True,
                 ner_use_gpu: Optional[bool] = True,
                 **kwargs):
        """Initialize the main pipeline.

        Args:
            sentiment_mode: Sentiment analysis mode (standard/optimized/calibrated)
            aggregation_method: Method for aggregating sentiments.
            agg_method: CLI alias for *aggregation_method* (kept for backward‑compat).
            aggregation_threshold: Threshold used by the aggregator.
            use_distance_weighting: Whether to apply distance weighting in aggregation.
            ner_use_gpu: Force‑override GPU usage for the NER model.  *None* autodetects.
            **kwargs: Anything accepted by `BasePipeline` (paths, batch_size, …).
        """
        super().__init__(**kwargs)

        self.sentiment_mode = sentiment_mode

        # handle "distance_weighted" alias coming from CLI tooling
        if (agg_method or aggregation_method) == "distance_weighted":
            self.aggregation_method = "conf_weighted"
            self.use_distance_weighting = True
        else:
            self.aggregation_method = agg_method or aggregation_method
            self.use_distance_weighting = use_distance_weighting

        self.aggregation_threshold = aggregation_threshold
        self.ner_use_gpu = ner_use_gpu

        logger.info("Pipeline configured with:")
        logger.info(f"  Sentiment mode: {sentiment_mode}")
        logger.info(
            f"  Aggregation: {self.aggregation_method} (threshold={aggregation_threshold})")
        logger.info(f"  Distance weighting: {self.use_distance_weighting}")

    def initialize_components(self):
        """Instantiate and wire all sub‑modules."""
        logger.info("Initializing pipeline components…")

        # ───────────────────────── Sentiment ──────────────────────────
        logger.info(
            f"Loading sentiment analyzer ({self.sentiment_mode} mode)…")
        adapter_path = self.config.get(
            "sentiment_config", {}).get("adapter_path")
        if adapter_path:
            logger.info(f"Using PEFT adapter: {adapter_path}")

        self.sentiment_analyzer = UnifiedSentimentAnalyzer(
            mode=self.sentiment_mode,
            device=self.config.get("device"),
            batch_size=self.config.get(
                "sentiment_config", {}).get("batch_size", 16),
            adapter_path=adapter_path,
        )

        # ─────────────────────────── NER ──────────────────────────────
        logger.info("Loading FinBERT‑powered NER (FinancialNERWrapper)…")
        # Pass GPU preference; the wrapper itself will call torch.cuda.is_available()
        self.ner = FinancialNERWrapper(
            ticker_csv_path=str(self.ticker_csv),
            use_gpu=self.ner_use_gpu,
        )

        # ──────────────────────── Text processor ──────────────────────
        logger.info("Loading text processor…")
        self.text_processor = TextProcessor()

        # ───────────────────────── Aggregator ─────────────────────────
        logger.info("Loading aggregator…")
        self.aggregator = Aggregator(
            method=self.aggregation_method,
            threshold=self.aggregation_threshold,
            use_distance_weighting=self.use_distance_weighting,
        )

        logger.info("All components initialized successfully")


class OptimizedPipeline(FinancialSentimentPipeline):
    """Pipeline tuned for GPU + bias‑corrected, confidence‑weighted aggregation."""

    def __init__(self, aggregation_method: Optional[str] = None, **kwargs):
        kwargs.pop("use_distance_weighting", None)  # avoid duplicate kwarg
        super().__init__(
            sentiment_mode="optimized",
            aggregation_method=aggregation_method or "conf_weighted",
            aggregation_threshold=0.1,
            use_distance_weighting=True,
            **kwargs,
        )


class StandardPipeline(FinancialSentimentPipeline):
    """Reference pipeline – no optimisation layers."""

    def __init__(self, **kwargs):
        kwargs.pop("use_distance_weighting", None)
        super().__init__(
            sentiment_mode="standard",
            aggregation_method="default",
            aggregation_threshold=0.1,
            use_distance_weighting=False,
            **kwargs,
        )


class CalibratedPipeline(FinancialSentimentPipeline):
    """Pipeline with advanced bias‑reduction (temperature scaling, etc.)."""

    def __init__(self, **kwargs):
        kwargs.pop("use_distance_weighting", None)
        super().__init__(
            sentiment_mode="calibrated",
            aggregation_method="conf_weighted",
            aggregation_threshold=0.05,
            use_distance_weighting=True,
            **kwargs,
        )


# ───────────────────────── Factory helper ───────────────────────────

def create_pipeline(pipeline_type: str = "optimized", agg_method: str = None, **kwargs) -> BasePipeline:
    """Return a ready‑to‑run pipeline instance.

    Args:
        pipeline_type: one of {standard, optimized, calibrated, main}
        agg_method: alias so CLI users can pass `--agg_method`.
        **kwargs: forwarded to the pipeline constructor(s).
    """
    pipelines = {
        "standard": StandardPipeline,
        "optimized": OptimizedPipeline,
        "calibrated": CalibratedPipeline,
        "main": FinancialSentimentPipeline,
    }

    # harmonise alias → canonical arg name
    if agg_method and "aggregation_method" not in kwargs:
        kwargs["aggregation_method"] = agg_method

    if pipeline_type not in pipelines:
        raise ValueError(f"Unknown pipeline type: {pipeline_type}")

    factory = pipelines[pipeline_type]

    # If the factory is a functools.partial, drop duplicate kwarg keys
    if isinstance(factory, functools.partial):
        for k in list(kwargs):
            if k in factory.keywords:
                kwargs.pop(k)

    return factory(**kwargs)
