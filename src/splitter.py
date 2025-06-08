"""
splitter.py

Provides sentence and clause splitting utilities for aspect-based sentiment analysis.
Uses NLTK for sentence tokenization and regex-based clause splitting.
"""

import nltk
import re

# Ensure the Punkt tokenizer models are available. Uncomment on first run:
nltk.download('punkt')

from nltk.tokenize import sent_tokenize

# Regex for splitting on semicolons and coordinating conjunctions
CLAUSE_DELIM_RE = re.compile(
    r'\s*;\s*'                          # semicolons as boundaries
    r'|\s+(?:and|but|while|however)\s+',  # coordinating conjunctions
    flags=re.IGNORECASE
)


def split_sentences(text: str) -> list[str]:
    """
    Split raw text into sentences using NLTK's PunktSentenceTokenizer.

    Args:
        text (str): Full text or paragraph.

    Returns:
        list[str]: List of sentence strings.
    """
    return sent_tokenize(text)


def split_clauses(sentence: str, min_length_for_commas: int = 40) -> list[str]:
    """
    Break a sentence into smaller clauses:
    1) Split on semicolons and core conjunctions (and, but, while, however).
    2) If a fragment is longer than `min_length_for_commas`, further split on commas.

    Args:
        sentence (str): A single sentence string.
        min_length_for_commas (int): Minimum length to allow comma-based splitting.

    Returns:
        list[str]: List of clause fragments.
    """
    parts = CLAUSE_DELIM_RE.split(sentence)
    final = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        # Further split long clauses on commas
        if len(part) > min_length_for_commas and ',' in part:
            for sub in map(str.strip, part.split(',')):
                if sub:
                    final.append(sub)
        else:
            final.append(part)
    return final


def split_to_chunks(text: str,
                    min_clause_length: int = 20,
                    min_length_for_commas: int = 40) -> list[str]:
    """
    Convert full text into sentence- and clause-level chunks,
    filtering out fragments shorter than `min_clause_length`.

    Args:
        text (str): Full article or paragraph.
        min_clause_length (int): Minimum length of a clause to keep.
        min_length_for_commas (int): Min length for comma-based splitting.

    Returns:
        list[str]: Cleaned list of text chunks.
    """
    chunks = []
    for sent in split_sentences(text):
        for clause in split_clauses(sent, min_length_for_commas):
            if len(clause) >= min_clause_length:
                chunks.append(clause)
    return chunks


if __name__ == "__main__":
    sample = (
        "Apple (AAPL) released its earnings; revenue rose 5%, "
        "but guidance was light, and investors were underwhelmed. "
        "Meanwhile, Microsoft saw strong cloud growth."
    )
    for chunk in split_to_chunks(sample):
        print("➤", chunk)
