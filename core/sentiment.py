# core/sentiment.py
"""
Unified sentiment analysis module combining FinBERT with optimizations
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
    - standard: Original FinBERT
    - optimized: With bias correction
    - calibrated: With advanced calibration
    """

    def __init__(self,
                 mode: str = "optimized",
                 model_name: str = None,
                 device: str = None,
                 **kwargs):
        """
        Initialize sentiment analyzer

        Args:
            mode: Analysis mode - "standard", "optimized", or "calibrated"
            model_name: Model to use (defaults to config)
            device: Device to use (auto-detect if None)
            **kwargs: Override config parameters
        """
        self.mode = mode
        self.model_name = model_name or MODELS["finbert"]

        # Load config with overrides
        self.config = SENTIMENT_CONFIG.copy()
        self.config.update(kwargs)

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

        # Label mapping
        self.id2label = self.model.config.id2label
        self.label2id = {v: k for k, v in self.id2label.items()}

        # Statistics tracking
        self.stats = defaultdict(int)

        logger.info(f"Initialized {mode} sentiment analyzer on {self.device}")

    def predict(self,
                texts: List[str],
                batch_size: int = None,
                max_length: int = None) -> List[Dict]:
        """
        Main prediction method that routes to appropriate implementation
        """
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
        """Standard FinBERT prediction"""
        batch_size = batch_size or self.config["batch_size"]
        max_length = max_length or self.config["max_length"]

        results = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

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
                confidences, indices = torch.max(probs, dim=1)

                # Build results
                for idx, text in enumerate(batch):
                    label_id = indices[idx].item()
                    label = self.id2label[label_id].title()
                    conf = confidences[idx].item()

                    score_dict = {
                        self.id2label[j].title(): probs[idx][j].item()
                        for j in range(probs.size(1))
                    }

                    results.append({
                        "text": text,
                        "label": label,
                        "confidence": conf,
                        "scores": score_dict
                    })

        return results

    def _predict_optimized(self,
                           texts: List[str],
                           batch_size: int = None,
                           max_length: int = None) -> List[Dict]:
        """Optimized prediction with bias correction"""
        batch_size = batch_size or self.config["batch_size"]
        max_length = max_length or self.config["max_length"]

        results = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

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

                # Get original probabilities
                original_probs = F.softmax(outputs.logits, dim=-1)

                # Apply bias correction
                corrected_probs = self._apply_bias_correction(original_probs)

                # Classify
                for idx, text in enumerate(batch):
                    prob_neg = corrected_probs[idx][0].item()
                    prob_neu = corrected_probs[idx][1].item()
                    prob_pos = corrected_probs[idx][2].item()

                    # Get original prediction for comparison
                    orig_max_idx = torch.argmax(original_probs[idx]).item()
                    original_label = ["Negative",
                                      "Neutral", "Positive"][orig_max_idx]

                    # Determine final classification
                    label, confidence = self._classify_with_threshold(
                        prob_neg, prob_neu, prob_pos
                    )

                    # Track changes
                    if original_label != label:
                        self.stats[f'changed_{original_label.lower()}_to_{label.lower()}'] += 1

                    score_dict = {
                        "Negative": prob_neg,
                        "Neutral": prob_neu,
                        "Positive": prob_pos
                    }

                    results.append({
                        "text": text,
                        "label": label,
                        "confidence": confidence,
                        "scores": score_dict,
                        "original_label": original_label,
                        "bias_corrected": original_label != label
                    })

        return results

    def _predict_calibrated(self,
                            texts: List[str],
                            batch_size: int = None,
                            max_length: int = None) -> List[Dict]:
        """Calibrated prediction with advanced bias reduction"""
        batch_size = batch_size or self.config["batch_size"]
        max_length = max_length or self.config["max_length"]

        results = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

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

                # Get original probabilities
                original_probs = F.softmax(outputs.logits, dim=-1)

                # Apply stronger bias correction
                adjusted_probs = self._reduce_neutral_bias(original_probs)

                # Classify with confidence thresholding
                classifications = self._classify_with_confidence_threshold(
                    adjusted_probs)

                # Build results
                for idx, (text, (label, confidence)) in enumerate(zip(batch, classifications)):
                    # Get all probabilities
                    prob_neg = adjusted_probs[idx][0].item()
                    prob_neu = adjusted_probs[idx][1].item()
                    prob_pos = adjusted_probs[idx][2].item()

                    # Track original
                    original_max_idx = torch.argmax(original_probs[idx]).item()
                    original_label = ["Negative", "Neutral",
                                      "Positive"][original_max_idx]

                    if original_label != label:
                        self.stats[f'changed_{original_label.lower()}_to_{label.lower()}'] += 1

                    score_dict = {
                        "Negative": prob_neg,
                        "Neutral": prob_neu,
                        "Positive": prob_pos
                    }

                    results.append({
                        "text": text,
                        "label": label,
                        "confidence": confidence,
                        "scores": score_dict,
                        "original_label": original_label,
                        "calibrated": original_label != label
                    })

        return results

    def _apply_bias_correction(self, probs: torch.Tensor) -> torch.Tensor:
        """Apply optimized bias correction"""
        corrected_probs = probs.clone()

        # Apply corrections from config
        corrected_probs[:, 0] *= self.config["neg_boost"]    # Negative
        corrected_probs[:, 1] *= self.config["neutral_penalty"]  # Neutral
        corrected_probs[:, 2] *= self.config["pos_boost"]    # Positive

        # Renormalize
        corrected_probs = F.softmax(corrected_probs, dim=1)

        return corrected_probs

    def _reduce_neutral_bias(self, probs: torch.Tensor) -> torch.Tensor:
        """Apply stronger neutral bias reduction for calibrated mode"""
        adjusted_probs = probs.clone()

        # Stronger reduction
        adjusted_probs[:, 1] *= self.config["neutral_penalty"]
        adjusted_probs[:, 0] *= self.config["neg_boost"]
        adjusted_probs[:, 2] *= self.config["pos_boost"]

        # Renormalize
        adjusted_probs = F.softmax(adjusted_probs, dim=1)

        return adjusted_probs

    def _classify_with_threshold(self, prob_neg: float, prob_neu: float,
                                 prob_pos: float) -> Tuple[str, float]:
        """Classify with confidence threshold"""
        threshold = self.config["min_confidence_diff"]

        if prob_pos > prob_neg and prob_pos > prob_neu:
            if prob_pos - max(prob_neg, prob_neu) >= threshold:
                return "Positive", prob_pos
            else:
                return "Neutral", prob_neu
        elif prob_neg > prob_pos and prob_neg > prob_neu:
            if prob_neg - max(prob_pos, prob_neu) >= threshold:
                return "Negative", prob_neg
            else:
                return "Neutral", prob_neu
        else:
            return "Neutral", prob_neu

    def _classify_with_confidence_threshold(self, probs: torch.Tensor) -> List[Tuple[str, float]]:
        """Classify batch with confidence thresholding"""
        results = []

        for i in range(probs.size(0)):
            prob_neg = probs[i][0].item()
            prob_neu = probs[i][1].item()
            prob_pos = probs[i][2].item()

            # Sort probabilities
            sorted_probs = sorted([
                ('Negative', prob_neg),
                ('Neutral', prob_neu),
                ('Positive', prob_pos)
            ], key=lambda x: x[1], reverse=True)

            top_label, top_prob = sorted_probs[0]
            second_label, second_prob = sorted_probs[1]

            # Check confidence difference
            confidence_diff = top_prob - second_prob

            if (top_label != 'Neutral' and
                    confidence_diff >= self.config["min_confidence_diff"]):
                final_label = top_label
                final_confidence = top_prob
            elif (second_label != 'Neutral' and
                  (top_prob - prob_neu) >= self.config["min_confidence_diff"]):
                final_label = second_label
                final_confidence = second_prob
            else:
                final_label = 'Neutral'
                final_confidence = prob_neu

            results.append((final_label, final_confidence))
            self.stats[f'{final_label.lower()}_classified'] += 1

        return results

    def get_stats(self) -> Dict[str, int]:
        """Get processing statistics"""
        return dict(self.stats)

    def reset_stats(self):
        """Reset statistics"""
        self.stats.clear()


# Convenience functions for backward compatibility
def create_finbert_analyzer(**kwargs) -> UnifiedSentimentAnalyzer:
    """Create standard FinBERT analyzer"""
    return UnifiedSentimentAnalyzer(mode="standard", **kwargs)


def create_optimized_analyzer(**kwargs) -> UnifiedSentimentAnalyzer:
    """Create optimized analyzer with bias correction"""
    return UnifiedSentimentAnalyzer(mode="optimized", **kwargs)


def create_calibrated_analyzer(**kwargs) -> UnifiedSentimentAnalyzer:
    """Create calibrated analyzer with advanced bias reduction"""
    return UnifiedSentimentAnalyzer(mode="calibrated", **kwargs)
