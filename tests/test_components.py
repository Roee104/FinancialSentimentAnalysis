# tests/test_components.py
"""
Test suite for core components
"""


from core.ner import UnifiedNER
from core.sentiment import UnifiedSentimentAnalyzer
from core.aggregator import Aggregator
from core.text_processor import TextProcessor


class TestNER:
    """Test NER functionality"""

    def test_ticker_extraction(self):
        """Test ticker extraction from text"""
        ner = UnifiedNER()

        test_cases = [
            ("Apple (AAPL) stock rose 5%", ["AAPL"]),
            ("Buy MSFT shares on NYSE", ["MSFT"]),
            ("BRK.B trading at $350", ["BRK.B"]),
            ("$aapl is up today", ["AAPL"]),  # Test case-insensitive
        ]

        for text, expected_tickers in test_cases:
            article = {"title": text, "content": ""}
            results = ner.extract_symbols(article)
            found_symbols = [sym for sym, _ in results]

            for expected in expected_tickers:
                assert expected in found_symbols, f"Expected {expected} in {found_symbols}"

    def test_confidence_scores(self):
        """Test confidence scoring"""
        ner = UnifiedNER()

        article = {
            "title": "Apple (AAPL) announces earnings",
            "content": "MSFT stock mentioned in passing"
        }

        results = ner.extract_symbols(article)

        # AAPL should have higher confidence (parenthetical mention)
        aapl_conf = next((conf for sym, conf in results if sym == "AAPL"), 0)
        msft_conf = next((conf for sym, conf in results if sym == "MSFT"), 0)

        assert aapl_conf > msft_conf

    def test_alias_resolution():
        ner = UnifiedNER()
        article = {"title": "Apple launches new MacBook", "content": ""}
        res = ner.extract_symbols(article)
        symbols = {s for s, _ in res}
        assert "AAPL" in symbols

    def test_single_letter_filter():
        ner = UnifiedNER()
        article_bad = {"title": "Plan C was discussed", "content": ""}
        article_good = {"title": "C stock surges on NYSE", "content": ""}
        assert "C" not in {s for s, _ in ner.extract_symbols(article_bad)}
        assert "C" in {s for s, _ in ner.extract_symbols(article_good)}


class TestSentimentAnalyzer:
    """Test sentiment analysis"""

    def test_batch_prediction(self):
        """Test batch processing"""
        analyzer = UnifiedSentimentAnalyzer(mode="standard")

        texts = [
            "Apple reported record earnings",
            "The company faces bankruptcy",
            "Markets remained stable today"
        ]

        results = analyzer.predict(texts, batch_size=2)

        assert len(results) == len(texts)
        assert all("label" in r for r in results)
        assert all(r["label"] in ["Positive", "Neutral", "Negative"]
                   for r in results)

    def test_modes(self):
        """Test different sentiment modes"""
        text = ["The company reported mixed results with some challenges"]

        standard = UnifiedSentimentAnalyzer(mode="standard")
        optimized = UnifiedSentimentAnalyzer(mode="optimized")

        standard_result = standard.predict(text)[0]
        optimized_result = optimized.predict(text)[0]

        # Should have valid results from both
        assert standard_result["label"] in ["Positive", "Neutral", "Negative"]
        assert optimized_result["label"] in ["Positive", "Neutral", "Negative"]


class TestAggregator:
    """Test aggregation logic"""

    def test_distance_weighting(self):
        """Test distance-based weighting"""
        aggregator = Aggregator(use_distance_weighting=True)

        chunks = [
            "Apple (AAPL) reported strong earnings.",
            "The market reacted positively.",
            "Microsoft faces challenges."
        ]

        predictions = [
            {"label": "Positive", "confidence": 0.9},
            {"label": "Positive", "confidence": 0.8},
            {"label": "Negative", "confidence": 0.85}
        ]

        tickers = [("AAPL", 0.95), ("MSFT", 0.9)]

        result = aggregator.aggregate_article(
            chunks=chunks,
            predictions=predictions,
            symbols=tickers,
            ticker_to_company={"AAPL": "Apple Inc",
                               "MSFT": "Microsoft Corporation"}
        )

        # Check structure
        assert "ticker_sentiments" in result
        assert "overall_sentiment" in result

        # AAPL should be positive (mentioned in positive chunk)
        aapl_sentiment = next(
            (t for t in result["ticker_sentiments"] if t["symbol"] == "AAPL"),
            None
        )
        assert aapl_sentiment is not None
        assert aapl_sentiment["label"] == "Positive"


class TestTextProcessor:
    """Test text processing"""

    def test_chunk_splitting(self):
        """Test text chunking"""
        processor = TextProcessor()

        text = """Apple reported earnings. The results were strong; revenue increased significantly.
        However, the company faces challenges in the Chinese market."""

        chunks = processor.split_to_chunks(text)

        assert len(chunks) > 1
        assert all(len(chunk.split()) >= 3 for chunk in chunks)
