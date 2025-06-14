# pipelines/__init__.py
"""Pipeline implementations"""

from pipelines.base_pipeline import BasePipeline
from pipelines.main_pipeline import (
    FinancialSentimentPipeline,
    OptimizedPipeline,
    StandardPipeline,
    CalibratedPipeline,
    create_pipeline
)
from pipelines.baselines import VADERBaseline

__all__ = [
    'BasePipeline',
    'FinancialSentimentPipeline',
    'OptimizedPipeline',
    'StandardPipeline',
    'CalibratedPipeline',
    'VADERBaseline',
    'create_pipeline'
]
