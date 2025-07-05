import json
import torch
import argparse
import logging
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel


logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger("evaluate_on_test")

LABEL2ID = {"Positive": 0, "Neutral": 1, "Negative": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

HF_ID = {
    "finbert": "ProsusAI/finbert",
    "deberta-fin": "mrm8488/deberta-v3-ft-financial-news-sentiment-analysis",
}


def load_test_dataset(path: Path) -> Dataset:
    samples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            label = item["true_overall"]
            if label == "Mixed":
                label = "Neutral"
            if label in LABEL2ID:
                samples.append({
                    "text": item["content"],
                    "label": LABEL2ID[label]
                })
    return Dataset.from_list(samples)


def evaluate(model_name: str, model_dir: Path, test_path: Path):
    base_model_name = HF_ID[model_name]
    LOG.info("Loading base model %s from %s", model_name, model_dir)

    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name, num_labels=3)
    model = PeftModel.from_pretrained(base_model, model_dir)
    model.eval().cuda()

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    test_ds = load_test_dataset(test_path)

    def tokenize(example):
        return tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)
    test_ds = test_ds.map(tokenize, batched=False)
    test_ds.set_format(type="torch", columns=[
                       "input_ids", "attention_mask", "label"])

    LOG.info("Running predictions on test set...")
    preds, labels = [], []
    for batch in torch.utils.data.DataLoader(test_ds, batch_size=32):
        input_ids = batch["input_ids"].cuda()
        attn_mask = batch["attention_mask"].cuda()
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attn_mask)
            pred = outputs.logits.argmax(dim=-1).cpu().tolist()
        preds.extend(pred)
        labels.extend(batch["label"].tolist())

    print("Classification Report:")
    print(classification_report(labels, preds, target_names=[
          "Positive", "Neutral", "Negative"], digits=4))

    matrix = confusion_matrix(labels, preds)
    plt.figure()
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Positive", "Neutral", "Negative"],
                yticklabels=["Positive", "Neutral", "Negative"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(model_dir / "confusion_matrix.png")

    trainer_state_path = model_dir / "trainer_state.json"
    if trainer_state_path.exists():
        with open(trainer_state_path) as f:
            state_data = json.load(f)

        train_loss = [x["loss"]
                      for x in state_data["log_history"] if "loss" in x]
        val_loss = [x["eval_loss"]
                    for x in state_data["log_history"] if "eval_loss" in x]

        plt.figure()
        plt.plot(train_loss, label="Train Loss")
        plt.plot(val_loss, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Train vs Validation Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(model_dir / "loss_plot.png")
        plt.show()
    else:
        print("Could not find trainer_state.json in:", trainer_state_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["finbert", "deberta-fin"], required=True,
                        help="Model shortcut name (finbert or deberta-fin)")
    parser.add_argument("--model_dir", type=Path, required=True,
                        help="Path to trained LoRA adapter directory")
    parser.add_argument("--test", type=Path, required=True,
                        help="Path to test.jsonl")
    args = parser.parse_args()

    evaluate(args.model, args.model_dir, args.test)
