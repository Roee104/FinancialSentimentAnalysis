```markdown
# 🧾 FinancialSentimentAnalysis

**FinancialSentimentAnalysis** is a research-grade NLP pipeline for analyzing the **overall sentiment** of financial-news articles. It classifies full-length articles (headline + body) as **Positive**, **Neutral**, or **Negative** using a fine-tuned [FinBERT-tone](https://huggingface.co/ProsusAI/finbert) model. Designed for financial-domain sentiment research, it emphasizes confidence-aware predictions and high generalization on noisy real-world data.

---

## 🧭 Project Scope

- **Input:** A single financial-news article (`headline + full text`).
- **Output:** One **overall sentiment label**: `Positive`, `Neutral`, or `Negative`, with an associated **confidence score**.
- **Note:** Ticker-level and sector-level sentiment extraction were explored but excluded from this final version to ensure quality and reproducibility.

---

## 🧱 Folder Structure

```

├── .gitignore
├── pyproject.toml / setup.py / requirements.txt
├── config/                  # Global settings and config overrides
│   ├── settings.py
│   ├── production.yaml
├── core/                    # Core logic for sentiment and NER
│   ├── sentiment.py         # FinBERT prediction wrapper
│   ├── text\_processor.py    # Cleansing and token preparation
│   ├── pretrained\_financial\_ner.py (unused in final run)
├── pipelines/               # Pipelines for inference and evaluation
│   ├── main\_pipeline.py     # Final optimized pipeline
│   ├── baselines.py         # FinBERT standard and VADER baselines
├── scripts/                 # Entrypoint scripts
│   ├── run\_pipeline.py      # Main CLI script for inference
│   ├── evaluate\_on\_test.py  # Runs evaluation on gold standard
│   ├── update\_plots.py      # Refreshes confusion matrix, reliability, etc.
├── data/
│   ├── loader.py            # Loads and formats JSONL/parquet datasets
│   ├── master\_ticker\_list.csv / ticker\_sector.csv 
├── analysis/
│   ├── visualization.py     # Generates plots for results and metrics
├── notebooks/
│   ├── EDA\_interim\_stage.ipynb
├── presentations/
│   ├── Interim\_Report.pdf / Project\_Proposal.pdf
├── tests/                   # Smoke and unit tests
│   ├── test\_pipeline\_smoke.py

````

---

## ⚙️ Installation

### 1. Clone and Setup

```bash
git clone https://github.com/Roee104/FinancialSentimentAnalysis.git
cd FinancialSentimentAnalysis
````

### 2. Virtual Environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### Run Sentiment Inference

```bash
python scripts/run_pipeline.py \
    --pipeline optimized \
    --max-articles 300 \
    --batch-size 8
```

> Input: A preprocessed JSONL or parquet dataset defined in config.
> Output: A JSONL file with `id`, `headline`, `text`, `overall_sentiment`, and `confidence`.

---

## 🧪 Example Output

```json
{
  "id": "article_2083",
  "headline": "Fed expected to hold rates steady amid inflation worries",
  "text": "The Federal Reserve is likely to pause rate hikes...",
  "overall_sentiment": "Neutral",
  "confidence": 0.8512
}
```


## 📉 Known Limitations

* **No per-ticker or per-sector sentiment.** This was removed to focus on high-precision document-level classification.
* **No rationale/span extraction.** Model does not provide explanation or localization of sentiment-bearing phrases.
* **May degrade on non-financial or sarcastic articles.** Tuned specifically for serious financial tone.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

```
```
