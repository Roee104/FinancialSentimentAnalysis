# core/sentiment.py
"""
Fixed unified sentiment analysis module - using standard FinBERT with optional light optimization
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import logging

from config.settings import MODELS, SENTIMENT_CONFIG

logger = logging.getLogger(__name__)


class UnifiedSentimentAnalyzer:
    """
    Unified sentiment analyzer with multiple modes:
    - standard: Original FinBERT (recommended)
    - optimized: Light bias correction
    - calibrated: Stronger calibration (use with caution)
    """

    def __init__(self,
                 mode: str = "standard",
                 model_name: str = None,
                 device: str = None,
                 batch_size: int = None,
                 **kwargs):
        """
        Initialize sentiment analyzer

        Args:
            mode: Analysis mode - "standard", "optimized", or "calibrated"
            model_name: Model to use (defaults to config)
            device: Device to use (auto-detect if None)
            batch_size: Batch size for inference
            **kwargs: Override config parameters
        """
        self.mode = mode
        self.model_name = model_name or MODELS["finbert"]

        # Load config with overrides
        self.config = SENTIMENT_CONFIG.copy()
        self.config.update(kwargs)

        # Fixed batch size (not dynamic)
        self.batch_size = batch_size or self.config.get("batch_size", 16)

        # Load model and tokenizer
        logger.info(f"Loading {self.model_name} in {mode} mode...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name)

        # Set device
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)
        self.model.eval()

        # Label mapping - CORRECT for yiyanghkust/finbert-tone
        self.id2label = {0: "Neutral", 1: "Positive", 2: "Negative"}
        self.label2id = {v: k for k, v in self.id2label.items()}

        # Statistics tracking
        self.stats = defaultdict(int)

        logger.info(f"Initialized {mode} sentiment analyzer on {self.device}")
        logger.info(f"Batch size: {self.batch_size}")

    def predict(self,
                texts: List[str],
                batch_size: int = None,
                max_length: int = None) -> List[Dict]:
        """
        Main prediction method that routes to appropriate implementation
        """
        # Use instance batch_size if not overridden
        batch_size = batch_size or self.batch_size

        if self.mode == "standard":
            return self._predict_standard(texts, batch_size, max_length)
        elif self.mode == "optimized":
            return self._predict_optimized(texts, batch_size, max_length)
        elif self.mode == "calibrated":
            return self._predict_calibrated(texts, batch_size, max_length)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def _predict_standard(self,
                          texts: List[str],
                          batch_size: int = None,
                          max_length: int = None) -> List[Dict]:
        """Standard FinBERT prediction - no modifications"""
        batch_size = batch_size or self.batch_size
        max_length = max_length or self.config["max_length"]

        results = []

        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            try:
                # Tokenize
                enc = self.tokenizer(batch,
                                     padding=True,
                                     truncation=True,
                                     max_length=max_length,
                                     return_tensors="pt")

                input_ids = enc["input_ids"].to(self.device)
                attention_mask = enc["attention_mask"].to(self.device)

                # Predict
                with torch.no_grad():
                    outputs = self.model(input_ids=input_ids,
                                         attention_mask=attention_mask)
                    logits = outputs.logits
                    probs = F.softmax(logits, dim=-1)

                    # Build results
                    for idx, text in enumerate(batch):
                        prob_values = probs[idx].cpu().numpy()
                        predicted_class = prob_values.argmax()

                        label = self.id2label[predicted_class]
                        conf = float(prob_values[predicted_class])

                        score_dict = {
                            "Neutral": float(prob_values[0]),
                            "Positive": float(prob_values[1]),
                            "Negative": float(prob_values[2])
                        }

                        results.append({
                            "text": text,
                            "label": label,
                            "confidence": conf,
                            "scores": score_dict
                        })

                        self.stats[f'{label.lower()}_predicted'] += 1

            except Exception as e:
                logger.error(f"Error in batch prediction: {e}")
                # Add empty results for failed batch
                for text in batch:
                    results.append({
                        "text": text,
                        "label": "Neutral",
                        "confidence": 0.0,
                        "scores": {"Neutral": 1.0, "Positive": 0.0, "Negative": 0.0}
                    })
                    self.stats['errors'] += 1

        return results

    def _predict_optimized(self,
                           texts: List[str],
                           batch_size: int = None,
                           max_length: int = None) -> List[Dict]:
        """Optimized prediction with LIGHT bias correction"""
        batch_size = batch_size or self.batch_size
        max_length = max_length or self.config["max_length"]

        results = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            try:
                enc = self.tokenizer(batch,
                                     padding=True,
                                     truncation=True,
                                     max_length=max_length,
                                     return_tensors="pt")

                input_ids = enc["input_ids"].to(self.device)
                attention_mask = enc["attention_mask"].to(self.device)

                with torch.no_grad():
                    outputs = self.model(input_ids=input_ids,
                                         attention_mask=attention_mask)

                    # Apply LIGHT correction to logits (not probabilities)
                    logits = outputs.logits
                    corrected_logits = self._apply_logit_correction(logits)

                    # Get probabilities from corrected logits
                    corrected_probs = F.softmax(corrected_logits, dim=-1)

                    # Build results
                    for idx, text in enumerate(batch):
                        prob_values = corrected_probs[idx].cpu().numpy()
                        predicted_class = prob_values.argmax()

                        label = self.id2label[predicted_class]
                        conf = float(prob_values[predicted_class])

                        # Track if changed from original
                        orig_class = outputs.logits[idx].argmax().item()
                        if orig_class != predicted_class:
                            self.stats[f'changed_{self.id2label[orig_class].lower()}_to_{label.lower()}'] += 1

                        score_dict = {
                            "Neutral": float(prob_values[0]),
                            "Positive": float(prob_values[1]),
                            "Negative": float(prob_values[2])
                        }

                        results.append({
                            "text": text,
                            "label": label,
                            "confidence": conf,
                            "scores": score_dict,
                            "original_label": self.id2label[orig_class],
                            "bias_corrected": orig_class != predicted_class
                        })

            except Exception as e:
                logger.error(f"Error in optimized batch prediction: {e}")
                for text in batch:
                    results.append({
                        "text": text,
                        "label": "Neutral",
                        "confidence": 0.0,
                        "scores": {"Neutral": 1.0, "Positive": 0.0, "Negative": 0.0}
                    })
                    self.stats['errors'] += 1

        return results

    def _apply_logit_correction(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply LIGHT correction to logits to reduce neutral bias
        Working with logits is more numerically stable than probabilities
        """
        corrected_logits = logits.clone()

        # Light adjustments - index mapping: 0=Neutral, 1=Positive, 2=Negative
        corrected_logits[:, 0] -= 0.2  # Slightly reduce neutral tendency
        corrected_logits[:, 1] += 0.05  # Very slight boost to positive
        corrected_logits[:, 2] += 0.05  # Very slight boost to negative

        return corrected_logits

    def _predict_calibrated(self,
                            texts: List[str],
                            batch_size: int = None,
                            max_length: int = None) -> List[Dict]:
        """Calibrated prediction with stronger bias reduction"""
        batch_size = batch_size or self.batch_size
        max_length = max_length or self.config["max_length"]

        results = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            try:
                enc = self.tokenizer(batch,
                                     padding=True,
                                     truncation=True,
                                     max_length=max_length,
                                     return_tensors="pt")

                input_ids = enc["input_ids"].to(self.device)
                attention_mask = enc["attention_mask"].to(self.device)

                with torch.no_grad():
                    outputs = self.model(input_ids=input_ids,
                                         attention_mask=attention_mask)

                    # Apply stronger correction
                    logits = outputs.logits
                    corrected_logits = logits.clone()

                    # Stronger adjustments
                    corrected_logits[:, 0] -= 0.4  # Reduce neutral more
                    corrected_logits[:, 1] += 0.1  # Boost positive
                    corrected_logits[:, 2] += 0.1  # Boost negative

                    corrected_probs = F.softmax(corrected_logits, dim=-1)

                    # Build results with additional logic
                    for idx, text in enumerate(batch):
                        prob_values = corrected_probs[idx].cpu().numpy()

                        # Apply minimum confidence difference requirement
                        sorted_indices = prob_values.argsort()[::-1]
                        top_prob = prob_values[sorted_indices[0]]
                        second_prob = prob_values[sorted_indices[1]]

                        if (sorted_indices[0] != 0 and  # Not neutral
                                top_prob - second_prob >= self.config["min_confidence_diff"]):
                            predicted_class = sorted_indices[0]
                        elif (sorted_indices[1] != 0 and
                                second_prob - prob_values[0] >= self.config["min_confidence_diff"]):
                            predicted_class = sorted_indices[1]
                        else:
                            predicted_class = 0  # Default to neutral

                        label = self.id2label[predicted_class]
                        conf = float(prob_values[predicted_class])

                        self.stats[f'{label.lower()}_classified'] += 1

                        score_dict = {
                            "Neutral": float(prob_values[0]),
                            "Positive": float(prob_values[1]),
                            "Negative": float(prob_values[2])
                        }

                        results.append({
                            "text": text,
                            "label": label,
                            "confidence": conf,
                            "scores": score_dict,
                            "calibrated": True
                        })

            except Exception as e:
                logger.error(f"Error in calibrated batch prediction: {e}")
                for text in batch:
                    results.append({
                        "text": text,
                        "label": "Neutral",
                        "confidence": 0.0,
                        "scores": {"Neutral": 1.0, "Positive": 0.0, "Negative": 0.0}
                    })
                    self.stats['errors'] += 1

        return results

    def get_stats(self) -> Dict[str, int]:
        """Get processing statistics"""
        return dict(self.stats)

    def reset_stats(self):
        """Reset statistics"""
        self.stats.clear()

    # Unit tests
    def test_batch_prediction(self):
        """Test batch prediction functionality"""
        test_texts = [
            "Apple reported record earnings, beating all estimates.",
            "The company faces significant regulatory challenges.",
            "Markets remained stable during the trading session.",
            "Bankruptcy proceedings have begun for the troubled firm.",
            "The merger was completed successfully.",
        ]

        try:
            # Test with small batch size
            results = self.predict(test_texts, batch_size=2)

            assert len(results) == len(
                test_texts), f"Expected {len(test_texts)} results, got {len(results)}"

            # Check result structure
            for result in results:
                assert "label" in result, "Missing label in result"
                assert "confidence" in result, "Missing confidence in result"
                assert "scores" in result, "Missing scores in result"
                assert result["label"] in ["Positive", "Neutral",
                                           "Negative"], f"Invalid label: {result['label']}"
                assert 0 <= result["confidence"] <= 1, f"Invalid confidence: {result['confidence']}"

            logger.info("✅ Batch prediction test passed")
            return True

        except Exception as e:
            logger.error(f"❌ Batch prediction test failed: {e}")
            return False


# Convenience functions for backward compatibility
def create_finbert_analyzer(**kwargs) -> UnifiedSentimentAnalyzer:
    """Create standard FinBERT analyzer"""
    return UnifiedSentimentAnalyzer(mode="standard", **kwargs)


def create_optimized_analyzer(**kwargs) -> UnifiedSentimentAnalyzer:
    """Create optimized analyzer with light bias correction"""
    return UnifiedSentimentAnalyzer(mode="optimized", **kwargs)


def create_calibrated_analyzer(**kwargs) -> UnifiedSentimentAnalyzer:
    """Create calibrated analyzer with stronger bias reduction"""
    return UnifiedSentimentAnalyzer(mode="calibrated", **kwargs)
