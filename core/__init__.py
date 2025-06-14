# core/__init__.py
"""Core functionality modules"""

from core.sentiment import UnifiedSentimentAnalyzer
from core.ner import UnifiedNER
from core.text_processor import TextProcessor
from core.aggregator import Aggregator

__all__ = [
    'UnifiedSentimentAnalyzer',
    'UnifiedNER',
    'TextProcessor',
    'Aggregator'
]
