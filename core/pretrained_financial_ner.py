# core/pretrained_financial_ner.py
import torch
import pandas as pd
import re
from typing import Dict, List, Tuple, Optional
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from rapidfuzz import process, fuzz


class PretrainedFinancialNER:
    """
    Uses FinBERT-NER to extract tickers from financial text, with alias mapping,
    fuzzy fallback, token cleanup, and regex backup.
    """

    def __init__(self,
               model_name: str = "yiyanghkust/finbert-ner",
               alias_path: str = "data/ticker_alias_table.csv",
               master_ticker_path: str = "data/master_ticker_list.csv"):
        self.device = 0 if torch.cuda.is_available() else -1

        # Load FinBERT-NER model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name)
        self.ner_pipeline = pipeline("ner", model=self.model, tokenizer=self.tokenizer,
                                     device=self.device, aggregation_strategy="simple")

        # Load alias → ticker mapping
        alias_df = pd.read_csv(alias_path)
        self.alias_to_ticker: Dict[str, str] = {}
        for _, row in alias_df.iterrows():
            ticker = row["symbol"].strip().upper()
            aliases = row["aliases"].split("|")
            for alias in aliases:
                clean_alias = alias.strip().lower()
                if clean_alias:
                    self.alias_to_ticker[clean_alias] = ticker

        # For fuzzy fallback
        self.alias_keys = list(self.alias_to_ticker.keys())

        # Load master ticker list
        ticker_df = pd.read_csv(master_ticker_path)
        self.valid_tickers: set = set(
            ticker_df["symbol"].str.strip().str.upper().tolist())

    def _clean_token(self, word: str) -> str:
        """Cleans and normalizes entity tokens."""
        return re.sub(r"[^a-zA-Z0-9.\-& ]", "", word.lower().strip())

    def _fuzzy_match(self, name: str, score_cutoff: int = 85) -> Optional[str]:
        """Fuzzy matches a name to the alias table."""
        match, score, _ = process.extractOne(
            name, self.alias_keys, scorer=fuzz.token_sort_ratio)
        if score >= score_cutoff:
            return self.alias_to_ticker.get(match)
        return None

    def _regex_fallback(self, text: str) -> List[Tuple[str, float]]:
        """Simple regex to capture ticker-like tokens."""
        candidates = re.findall(r"\b[A-Z]{2,5}\b", text)
        deduped = set([sym.strip().upper()
                      for sym in candidates if sym.strip().upper() in self.valid_tickers])
        # assign default low confidence
        return [(ticker, 0.6) for ticker in deduped]

    def extract_symbols(self, article: Dict, min_confidence: float = 0.6, use_metadata: bool = True) -> List[Tuple[str, float]]:
        """
        Extracts valid ticker symbols from a financial article using FinBERT-NER,
        alias matching, fuzzy fallback, and optional regex backup.

        Returns:
            List of (ticker, confidence) tuples.
        """
        VALID_ENTITY_GROUPS = {"ORG", "COMP", "FIN_INST", "TICKER"}
        text = article.get("title", "") + " " + article.get("content", "")
        ner_results = self.ner_pipeline(text)

        seen = {}
        matched_any = False

        for ent in ner_results:
            if ent.get("entity_group") not in VALID_ENTITY_GROUPS:
                continue
            if ent.get("score", 0.0) < min_confidence:
                continue

            # Get original span from raw text if available
            span = text[ent["start"]:ent["end"]
                        ] if "start" in ent and "end" in ent else ent["word"]
            clean_word = self._clean_token(span)

            if not clean_word:
                continue

            # Try exact match
            ticker = self.alias_to_ticker.get(clean_word)

            # Try fuzzy if not found
            if not ticker:
                ticker = self._fuzzy_match(clean_word)

            if not ticker or ticker not in self.valid_tickers:
                continue

            matched_any = True
            prev_score = seen.get(ticker, 0)
            if ent["score"] > prev_score:
                seen[ticker] = ent["score"]

        # If nothing matched, try regex fallback
        if not matched_any:
            fallback = self._regex_fallback(text)
            for ticker, score in fallback:
                if ticker not in seen:
                    seen[ticker] = score

        return [(ticker, conf) for ticker, conf in seen.items()]
