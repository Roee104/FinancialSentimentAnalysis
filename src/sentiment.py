"""
sentiment.py

Provides FinBERT-based sentiment analysis for financial news chunks.
Utilizes the "yiyanghkust/finbert-tone" model to produce Positive, Neutral, or Negative labels
with associated confidence scores. Designed for batch inference and GPU/CPU flexibility.
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
from typing import List, Dict


class FinBERTSentimentAnalyzer:
    """
    Wrapper for FinBERT sentiment analysis.

    - Loads the specified FinBERT model and tokenizer.
    - Predicts sentiment labels and confidence scores for batches of text.
    """

    def __init__(self,
                 model_name: str = "yiyanghkust/finbert-tone",
                 device: str = None):
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

        # Determine device: user-specified or auto GPU/CPU
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        # Mapping from model output IDs to label strings
        self.id2label = self.model.config.id2label

    def predict(self,
                texts: List[str],
                batch_size: int = 16,
                max_length: int = 512) -> List[Dict]:
        """
        Perform sentiment analysis on a list of text chunks.

        Args:
            texts (List[str]): Text chunks to classify.
            batch_size (int): Number of texts per inference batch.
            max_length (int): Max tokens per text (for truncation).

        Returns:
            List[Dict]: A list of dicts with keys:
                - text: original chunk
                - label: 'Positive'|'Neutral'|'Negative'
                - confidence: float, probability of predicted label
                - scores: dict of all label probabilities
        """
        results = []
        # Iterate in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            # Tokenize with padding and truncation
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
                logits = outputs.logits  # shape (batch_size, num_labels)
                probs = F.softmax(logits, dim=-1)
                # Get top probabilities and indices
                confidences, indices = torch.max(probs, dim=1)

                # Build result entries
                for idx, text in enumerate(batch):
                    label_id = indices[idx].item()
                    label = self.id2label[label_id].title()
                    conf = confidences[idx].item()
                    # all label probabilities
                    score_dict = {self.id2label[j]: probs[idx][j].item()
                                  for j in range(probs.size(1))}
                    results.append({
                        "text": text,
                        "label": label,
                        "confidence": conf,
                        "scores": score_dict
                    })
        return results


if __name__ == "__main__":
    # Quick functional test
    analyzer = FinBERTSentimentAnalyzer()
    sample_chunks = [
        "Apple (AAPL) released its earnings and beat revenue expectations.",
        "Investors are worried about slowing growth.",
        "The stock price remained flat."
    ]
    predictions = analyzer.predict(sample_chunks, batch_size=8)
    for pred in predictions:
        print(pred)
