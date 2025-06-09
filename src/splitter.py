"""
src/splitter.py

Provides sentence and clause splitting utilities for aspect-based sentiment analysis.
Uses NLTK for sentence tokenization and regex-based clause splitting,
with a word-count–based filter to retain meaningful chunks.
"""

import nltk
import re

# Uncomment if running for the first time to download punkt
# nltk.download('punkt')

from nltk.tokenize import sent_tokenize

# Regex for splitting on semicolons and key conjunctions
CLAUSE_DELIM_RE = re.compile(
    r"\s*;\s*"               # split on semicolons
    r"|\s+(?:and|but|while|however)\s+",  # split on conjunctions
    flags=re.IGNORECASE
)


def split_sentences(text: str) -> list[str]:
    """
    Split raw text into sentences using NLTK's Punkt tokenizer.

    Args:
        text (str): Full text or paragraph.
    Returns:
        List[str]: Sentences extracted from text.
    """
    return sent_tokenize(text)


def split_clauses(sentence: str, min_length_for_commas: int = 40) -> list[str]:
    """
    Break a sentence into smaller clause fragments.

    1. Split on semicolons and core conjunctions.
    2. Further split long fragments (over min_length_for_commas chars) on commas.

    Args:
        sentence (str): A single sentence string.
        min_length_for_commas (int): Length threshold for comma splitting.
    Returns:
        List[str]: Clause-level fragments.
    """
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


def split_to_chunks(
    text: str,
    min_clause_words: int = 3,
    min_length_for_commas: int = 40
) -> list[str]:
    """
    Convert full text into sentence- and clause-level chunks,
    filtering by a minimum word count to keep meaningful fragments.

    Args:
        text (str): Full article or paragraph.
        min_clause_words (int): Minimum number of words per chunk.
        min_length_for_commas (int): Threshold for comma-based splitting.
    Returns:
        List[str]: Cleaned list of text chunks.
    """
    chunks = []
    for sent in split_sentences(text):
        for clause in split_clauses(sent, min_length_for_commas):
            # Keep only clauses with enough words
            if len(clause.split()) >= min_clause_words:
                chunks.append(clause)
    return chunks


if __name__ == "__main__":
    sample = (
        "Apple (AAPL) released its earnings; revenue rose 5%, "
        "but guidance was light, and investors were underwhelmed. "
        "Meanwhile, Microsoft saw strong cloud growth."
    )
    print("Sample Chunks:")
    for chunk in split_to_chunks(sample):
        print("➤", chunk)
