# core/text_processor.py
"""
Text processing utilities for sentence and clause splitting
"""

from nltk.tokenize import sent_tokenize
import nltk
import re
import logging
from typing import List

from config.settings import TEXT_PROCESSING

logger = logging.getLogger(__name__)

# Download punkt if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    logger.info("Downloading NLTK punkt tokenizer...")
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)


# Regex for splitting on semicolons and key conjunctions
CLAUSE_DELIM_RE = re.compile(
    r"\s*;\s*"  # split on semicolons
    r"|\s+(?:and|but|while|however)\s+",  # split on conjunctions
    flags=re.IGNORECASE
)


class TextProcessor:
    """Handles text processing for sentiment analysis"""

    def __init__(self, config: dict = None):
        """
        Initialize text processor

        Args:
            config: Configuration dict (uses settings if None)
        """
        self.config = config or TEXT_PROCESSING
        logger.info("Initialized TextProcessor")

    def split_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using NLTK

        Args:
            text: Full text or paragraph

        Returns:
            List of sentences
        """
        try:
            return sent_tokenize(text)
        except Exception as e:
            logger.warning(f"Error in sentence splitting: {e}")
            # Fallback to simple splitting
            return text.split('. ')

    def split_clauses(self, sentence: str, min_length_for_commas: int = None) -> List[str]:
        """
        Break a sentence into smaller clause fragments

        Args:
            sentence: A single sentence string
            min_length_for_commas: Length threshold for comma splitting

        Returns:
            List of clause-level fragments
        """
        min_length_for_commas = min_length_for_commas or self.config["min_length_for_commas"]

        parts = CLAUSE_DELIM_RE.split(sentence)
        final = []

        for part in parts:
            part = part.strip()
            if not part:
                continue

            # If fragment is long and contains commas, split on commas
            if len(part) > min_length_for_commas and ',' in part:
                for sub in map(str.strip, part.split(',')):
                    if sub:
                        final.append(sub)
            else:
                final.append(part)

        return final

    def split_to_chunks(self,
                        text: str,
                        min_clause_words: int = None,
                        min_length_for_commas: int = None) -> List[str]:
        """
        Convert full text into sentence- and clause-level chunks

        Args:
            text: Full article or paragraph
            min_clause_words: Minimum number of words per chunk
            min_length_for_commas: Threshold for comma-based splitting

        Returns:
            List of cleaned text chunks
        """
        min_clause_words = min_clause_words or self.config["min_clause_words"]
        min_length_for_commas = min_length_for_commas or self.config["min_length_for_commas"]

        chunks = []

        for sent in self.split_sentences(text):
            for clause in self.split_clauses(sent, min_length_for_commas):
                # Keep only clauses with enough words
                if len(clause.split()) >= min_clause_words:
                    chunks.append(clause)

        return chunks

    def preprocess_text(self, text: str) -> str:
        """
        Basic text preprocessing

        Args:
            text: Raw text

        Returns:
            Preprocessed text
        """
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)

        # Remove special characters but keep financial symbols
        text = re.sub(r'[^\w\s\$\%\.\,\;\:\!\?\-\(\)]', '', text)

        return text.strip()
