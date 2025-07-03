import torch
import torch.nn as nn
from transformers import AutoModel, PreTrainedModel, AutoConfig


class MultitaskSentimentModel(nn.Module):
    def __init__(
        self,
        model_name: str,
        class_weights: dict,
        label2id: dict,
        ticker_label2id: dict,
        hidden_dropout_prob: float = 0.1,
    ):
        super().__init__()
        self.model_name = model_name
        self.label2id = label2id
        self.ticker_label2id = ticker_label2id
        self.id2label = {v: k for k, v in label2id.items()}
        self.ticker_id2label = {v: k for k, v in ticker_label2id.items()}

        # Base model (e.g., FinBERT or DeBERTa)
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size

        # Dropout
        self.dropout = nn.Dropout(hidden_dropout_prob)

        # Overall sentiment head
        self.overall_head = nn.Linear(hidden_size, len(label2id))

        # Ticker-level sentiment head (per article, multi-ticker)
        self.ticker_head = nn.Linear(hidden_size, len(ticker_label2id))

        # Class weights for loss
        self.overall_loss_fn = nn.CrossEntropyLoss(
            weight=self._make_tensor_weights(
                class_weights["overall"], label2id)
        )
        self.ticker_loss_fn = nn.CrossEntropyLoss(
            weight=self._make_tensor_weights(
                class_weights["ticker"], ticker_label2id)
        )

    def _make_tensor_weights(self, weight_dict, label2id):
        weights = [weight_dict[label]
                   for label in sorted(label2id, key=label2id.get)]
        return torch.tensor(weights, dtype=torch.float)

    def forward(
        self,
        input_ids,
        attention_mask,
        overall_labels=None,
        ticker_labels=None,
        ticker_mask=None,
    ):
        # ───── encode ─────
        outputs = self.backbone(input_ids=input_ids,
                                attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]  # [CLS] token
        pooled_output = self.dropout(pooled_output)

        # ───── heads ─────
        overall_logits = self.overall_head(pooled_output)
        ticker_logits = self.ticker_head(pooled_output)

        output = {
            "overall_logits": overall_logits,
            "ticker_logits": ticker_logits,
        }

        if overall_labels is not None and ticker_labels is not None:
            # ───── compute losses ─────
            overall_loss = self.overall_loss_fn(overall_logits, overall_labels)

            if ticker_mask is not None:
                ticker_loss = self.ticker_loss_fn(
                    ticker_logits[ticker_mask], ticker_labels[ticker_mask]
                )
            else:
                ticker_loss = self.ticker_loss_fn(ticker_logits, ticker_labels)

            output["loss"] = overall_loss + ticker_loss
            output["overall_loss"] = overall_loss
            output["ticker_loss"] = ticker_loss

        return output
