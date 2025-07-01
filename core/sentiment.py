# core/sentiment.py
"""
Unified sentiment-analysis module (FinBERT) with optional light optimisation.
Downloads the model once into <repo>/.cache/models.
Supports loading PEFT adapters for fine-tuned models.
"""
from __future__ import annotations
from config.settings import CACHE_DIR, MODELS, SENTIMENT_CONFIG
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch.nn.functional as F
import torch
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from pathlib import Path
import json

# ───────────────────────────────  logging at import  ──────────────────────────
import logging
logger = logging.getLogger(__name__)

# ───────────────────────────────  PEFT support  ───────────────────────────────
try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    logger.warning("PEFT not available - adapter loading disabled")

# ──────────────────────────────────────────────────────────────────────────────

# Model mappings for adapters
ADAPTER_MODEL_MAP = {
    "finbert": "yiyanghkust/finbert-tone",
    "deberta-fin": "mrm8488/deberta-v3-ft-financial-news-sentiment-analysis",
    "deberta": "mrm8488/deberta-v3-ft-financial-news-sentiment-analysis",
}


class UnifiedSentimentAnalyzer:
    """
    Modes  
      • **standard**   – raw FinBERT probabilities  
      • **optimized**  – light bias correction  
      • **calibrated** – stronger neutral-bias reduction  

    Supports loading PEFT adapters for fine-tuned models.
    """

    def __init__(
        self,
        mode: str = "standard",
        model_name: str | None = None,
        device: str | None = None,
        batch_size: int | None = None,
        adapter_path: str | None = None,
        **kwargs,
    ):
        self.mode = mode
        self.adapter_path = adapter_path

        # Determine which base model to use based on adapter path
        if adapter_path:
            adapter_path_str = str(adapter_path).lower()

            # Check adapter config to determine base model
            adapter_config_path = Path(adapter_path) / "adapter_config.json"
            if adapter_config_path.exists():
                try:
                    with open(adapter_config_path, 'r') as f:
                        adapter_config = json.load(f)
                    base_model_name = adapter_config.get(
                        "base_model_name_or_path", "")

                    # Map to our known models
                    if "deberta" in base_model_name.lower():
                        self.model_name = ADAPTER_MODEL_MAP["deberta-fin"]
                        logger.info(
                            f"Detected DeBERTa adapter, using model: {self.model_name}")
                    elif "finbert" in base_model_name.lower():
                        self.model_name = MODELS["finbert"]
                        logger.info(
                            f"Detected FinBERT adapter, using model: {self.model_name}")
                    else:
                        # Fallback to path-based detection
                        if "deberta" in adapter_path_str:
                            self.model_name = ADAPTER_MODEL_MAP["deberta-fin"]
                        else:
                            self.model_name = model_name or MODELS["finbert"]
                except Exception as e:
                    logger.warning(f"Could not read adapter config: {e}")
                    # Fallback to path-based detection
                    if "deberta" in adapter_path_str:
                        self.model_name = ADAPTER_MODEL_MAP["deberta-fin"]
                    else:
                        self.model_name = model_name or MODELS["finbert"]
            else:
                # Fallback to path-based detection
                if "deberta" in adapter_path_str:
                    self.model_name = ADAPTER_MODEL_MAP["deberta-fin"]
                else:
                    self.model_name = model_name or MODELS["finbert"]
        else:
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

        # ---- load PEFT adapter if provided -----------------------------------
        if self.adapter_path and PEFT_AVAILABLE:
            adapter_path = Path(self.adapter_path)
            if adapter_path.exists():
                logger.info(f"Loading PEFT adapter from {adapter_path}")
                try:
                    self.model = PeftModel.from_pretrained(
                        self.model, str(adapter_path))
                    logger.info("✅ PEFT adapter loaded successfully")
                except Exception as e:
                    logger.error(f"Failed to load PEFT adapter: {e}")
                    raise
            else:
                logger.warning(f"Adapter path not found: {adapter_path}")
        elif self.adapter_path and not PEFT_AVAILABLE:
            logger.error(
                "PEFT adapter requested but PEFT library not installed!")

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

        logger.info("Initialised %s mode (batch=%s, device=%s, adapter=%s)",
                    mode, self.batch_size, self.device,
                    "loaded" if self.adapter_path else "none")

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
        results = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            try:
                encoded = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                input_ids = encoded["input_ids"].to(self.device)
                attn_mask = encoded["attention_mask"].to(self.device)

                with torch.no_grad():
                    outputs = self.model(
                        input_ids=input_ids, attention_mask=attn_mask)
                    logits = outputs.logits

                probs = F.softmax(logits, dim=-1)
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

    # ---- optimized mode (light de-biasing) ----------------------------------
    def _predict_optimized(
        self, texts: List[str], batch_size: int, max_length: Optional[int]
    ) -> List[Dict]:
        results = self._predict_standard(texts, batch_size, max_length)

        for r in results:
            # reduce neutral
            n_score = r["scores"]["Neutral"]
            p_score = r["scores"]["Positive"]
            neg_score = r["scores"]["Negative"]

            n_adj = n_score * 0.8
            p_adj = p_score * 1.1
            neg_adj = neg_score * 1.1
            total_adj = n_adj + p_adj + neg_adj

            r["scores"]["Neutral"] = n_adj / total_adj
            r["scores"]["Positive"] = p_adj / total_adj
            r["scores"]["Negative"] = neg_adj / total_adj

            # re-predict
            best_idx = max(
                range(3), key=lambda i: list(r["scores"].values())[i])
            r["label"] = list(r["scores"].keys())[best_idx]
            r["confidence"] = list(r["scores"].values())[best_idx]

        return results

    # ---- calibrated mode (stronger) -----------------------------------------
    def _predict_calibrated(
        self, texts: List[str], batch_size: int, max_length: Optional[int]
    ) -> List[Dict]:
        results = self._predict_standard(texts, batch_size, max_length)

        for r in results:
            n_score = r["scores"]["Neutral"]
            p_score = r["scores"]["Positive"]
            neg_score = r["scores"]["Negative"]

            n_adj = n_score * 0.65
            p_adj = p_score * 1.15
            neg_adj = neg_score * 1.15
            total_adj = n_adj + p_adj + neg_adj

            r["scores"]["Neutral"] = n_adj / total_adj
            r["scores"]["Positive"] = p_adj / total_adj
            r["scores"]["Negative"] = neg_adj / total_adj

            best_idx = max(
                range(3), key=lambda i: list(r["scores"].values())[i])
            r["label"] = list(r["scores"].keys())[best_idx]
            r["confidence"] = list(r["scores"].values())[best_idx]

        return results

    # --------------------------------------------------------------------- misc
    def get_stats(self) -> Dict[str, int]:
        return dict(self.stats)

    def reset_stats(self) -> None:
        self.stats.clear()

    # --------------------------------------------------------------------- tests
    def test_batch_prediction(self) -> bool:
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
