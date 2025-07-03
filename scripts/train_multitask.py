import json
import torch
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    HfArgumentParser,
)
from datasets import load_metric
from models.multitask_model import MultitaskSentimentModel
from datasets.multitask_dataset import MultitaskDataset


def load_class_weights(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_label2id():
    return {"Positive": 0, "Neutral": 1, "Negative": 2}


class DummyArgs:
    model_type: str = "yiyanghkust/finbert-pretrain"
    train_file: str = "data/train_ready_multitask.jsonl"
    class_weights_file: str = "data/class_weights.json"
    output_dir: str = "models/finbert-multitask-v1"
    ticker_loss_weight: float = 1.0
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 8
    save_steps: int = 500
    logging_steps: int = 100
    warmup_steps: int = 200
    weight_decay: float = 0.01


def main():
    args = DummyArgs()

    label2id = get_label2id()
    class_weights = load_class_weights(args.class_weights_file)

    tokenizer = AutoTokenizer.from_pretrained(args.model_type)

    dataset = MultitaskDataset(
        args.train_file, tokenizer, label2id, label2id  # same label set for both
    )

    model = MultitaskSentimentModel(
        model_name=args.model_type,
        class_weights=class_weights,
        label2id=label2id,
        ticker_label2id=label2id,  # shared for now
    )

    def compute_loss(model, inputs, return_outputs=False):
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            overall_labels=inputs["overall_labels"],
            ticker_labels=inputs["ticker_labels"],
            ticker_mask=inputs["ticker_mask"],
        )
        return (outputs["loss"], outputs) if return_outputs else outputs["loss"]

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        num_train_epochs=args.num_train_epochs,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        evaluation_strategy="no",
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        save_total_limit=1,
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        compute_loss=compute_loss,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"✅ Model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
