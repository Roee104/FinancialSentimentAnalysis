# core/sentiment.py
"""
Unified sentiment-analysis module (FinBERT) with optional light optimisation.
Downloads the model once into <repo>/.cache/models.
"""
from __future__ import annotations
from config.settings import CACHE_DIR, MODELS, SENTIMENT_CONFIG
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch.nn.functional as F
import torch
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

# ───────────────────────────────  logging at import  ──────────────────────────
import logging
logger = logging.getLogger(__name__)

# ───────────────────────────────  std / 3-rd party  ───────────────────────────


# ───────────────────────────────  internal config  ────────────────────────────

# ──────────────────────────────────────────────────────────────────────────────

class UnifiedSentimentAnalyzer:
    """
    Modes  
      • **standard**   – raw FinBERT probabilities  
      • **optimized**  – light bias correction  
      • **calibrated** – stronger neutral-bias reduction  
    """

    def __init__(
        self,
        mode: str = "standard",
        model_name: str | None = None,
        device: str | None = None,
        batch_size: int | None = None,
        **kwargs,
    ):
        self.mode = mode
        self.model_name = model_name or MODELS["finbert"]

        # ---- config overrides ------------------------------------------------
        self.config = SENTIMENT_CONFIG.copy()
        self.config.update(kwargs)
        self.batch_size = batch_size or self.config.get("batch_size", 16)

        # ---- model & tokenizer (cached) --------------------------------------
        cache_home = CACHE_DIR / "models"
        logger.info("Loading %s (cache → %s) …", self.model_name, cache_home)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=str(cache_home),
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=3,
            cache_dir=str(cache_home),
        )

        # ---- device ----------------------------------------------------------
        self.device = torch.device(device) if device else (
            torch.device("cuda") if torch.cuda.is_available(
            ) else torch.device("cpu")
        )
        self.model.to(self.device).eval()

        # ---- label maps & stats ---------------------------------------------
        self.id2label = {0: "Neutral", 1: "Positive", 2: "Negative"}
        self.label2id = {v: k for k, v in self.id2label.items()}
        self.stats: defaultdict[str, int] = defaultdict(int)

        logger.info("Initialised %s mode (batch=%s, device=%s)",
                    mode, self.batch_size, self.device)

    # ────────────────────────────────  API  ───────────────────────────────────
    def predict(
        self,
        texts: List[str],
        batch_size: int | None = None,
        max_length: int | None = None,
    ) -> List[Dict]:
        batch_size = batch_size or self.batch_size

        if self.mode == "standard":
            return self._predict_standard(texts, batch_size, max_length)
        if self.mode == "optimized":
            return self._predict_optimized(texts, batch_size, max_length)
        if self.mode == "calibrated":
            return self._predict_calibrated(texts, batch_size, max_length)
        raise ValueError(f"Unknown mode: {self.mode}")

    # ------------------------------------------------------------------------
    def _predict_standard(
        self, texts: List[str], batch_size: int, max_length: Optional[int]
    ) -> List[Dict]:
        max_length = max_length or self.config["max_length"]
        return self._batched_predict(texts, batch_size, max_length)

    def _predict_optimized(
        self, texts: List[str], batch_size: int, max_length: Optional[int]
    ) -> List[Dict]:
        max_length = max_length or self.config["max_length"]
        return self._batched_predict(texts, batch_size, max_length, light_bias=True)

    def _predict_calibrated(
        self, texts: List[str], batch_size: int, max_length: Optional[int]
    ) -> List[Dict]:
        max_length = max_length or self.config["max_length"]
        return self._batched_predict(texts, batch_size, max_length, strong_bias=True)

    # ------------------------------------------------------------------------
    def _batched_predict(
        self,
        texts: List[str],
        batch_size: int,
        max_length: int,
        light_bias: bool = False,
        strong_bias: bool = False,
    ) -> List[Dict]:
        results: List[Dict] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            try:
                enc = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                ).to(self.device)

                with torch.no_grad():
                    logits = self.model(**enc).logits

                    # optional bias corrections
                    if light_bias or strong_bias:
                        logits = logits.clone()
                        logits[:, 0] -= 0.2 if light_bias else 0.4  # neutral ↓
                        # positive ↑
                        logits[:, 1] += 0.05 if light_bias else 0.1
                        # negative ↑
                        logits[:, 2] += 0.05 if light_bias else 0.1

                    probs = F.softmax(logits, dim=-1).cpu().numpy()

                for idx, text in enumerate(batch):
                    p = probs[idx]
                    pred_idx = p.argmax()
                    label = self.id2label[pred_idx]

                    results.append(
                        dict(
                            text=text,
                            label=label,
                            confidence=float(p[pred_idx]),
                            scores={
                                "Neutral": float(p[0]),
                                "Positive": float(p[1]),
                                "Negative": float(p[2]),
                            },
                        )
                    )
                    self.stats[f"{label.lower()}_predicted"] += 1
            except Exception as exc:  # noqa: BLE001
                logger.error("Batch prediction error: %s", exc)
                for text in batch:
                    results.append(
                        dict(
                            text=text,
                            label="Neutral",
                            confidence=0.0,
                            scores={"Neutral": 1.0,
                                    "Positive": 0.0, "Negative": 0.0},
                        )
                    )
                self.stats["errors"] += len(batch)

        return results

    # --------------------------------------------------------------------- misc
    def get_stats(self) -> Dict[str, int]:
        return dict(self.stats)

    def reset_stats(self) -> None:
        self.stats.clear()

    # --------------------------------------------------------------------- tests (unchanged)
    def test_batch_prediction(self) -> bool:  # unchanged
        test_texts = [
            "Apple reported record earnings, beating estimates.",
            "The company faces significant regulatory challenges.",
        ]
        try:
            res = self.predict(test_texts, batch_size=1)
            assert len(res) == 2 and all("label" in r for r in res)
            logger.info("✅ sentiment batch test passed")
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error("❌ sentiment batch test failed: %s", exc)
            return False


# Convenience creators
def create_finbert_analyzer(**kw) -> UnifiedSentimentAnalyzer:
    return UnifiedSentimentAnalyzer(mode="standard", **kw)


def create_optimized_analyzer(**kw) -> UnifiedSentimentAnalyzer:
    return UnifiedSentimentAnalyzer(mode="optimized", **kw)


def create_calibrated_analyzer(**kw) -> UnifiedSentimentAnalyzer:
    return UnifiedSentimentAnalyzer(mode="calibrated", **kw)
