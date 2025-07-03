import json
from pathlib import Path
from torch.utils.data import Dataset
import torch


class MultitaskDataset(Dataset):
    def __init__(self, path, tokenizer, label2id, ticker_label2id, max_length=512):
        self.samples = []
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.ticker_label2id = ticker_label2id
        self.max_length = max_length

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)

                # Tokenize
                enc = tokenizer(
                    data["text"],
                    truncation=True,
                    max_length=max_length,
                    padding="max_length"
                )

                overall = data["overall_label"]
                tickers = data.get("ticker_sentiments", {})

                ticker_ids = []
                ticker_mask = []

                for tkr, sentiment in tickers.items():
                    label_id = ticker_label2id.get(sentiment)
                    if label_id is not None:
                        ticker_ids.append(label_id)
                        ticker_mask.append(1)

                # Pad or truncate
                max_tickers = 10  # limit max tickers per article
                if len(ticker_ids) > max_tickers:
                    ticker_ids = ticker_ids[:max_tickers]
                    ticker_mask = ticker_mask[:max_tickers]
                else:
                    pad_len = max_tickers - len(ticker_ids)
                    ticker_ids += [0] * pad_len
                    ticker_mask += [0] * pad_len

                self.samples.append({
                    "input_ids": enc["input_ids"],
                    "attention_mask": enc["attention_mask"],
                    "overall_label": label2id[overall],
                    "ticker_labels": ticker_ids,
                    "ticker_mask": ticker_mask
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        return {
            "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(item["attention_mask"], dtype=torch.long),
            "overall_labels": torch.tensor(item["overall_label"], dtype=torch.long),
            "ticker_labels": torch.tensor(item["ticker_labels"], dtype=torch.long),
            "ticker_mask": torch.tensor(item["ticker_mask"], dtype=torch.bool)
        }
