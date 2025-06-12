# annotation_gpt_full.py

import os
import json
import random
import time
from collections import defaultdict
from tqdm import tqdm
import openai

# 1. Ensure your OpenAI API key is set in the environment
#    export OPENAI_API_KEY="your_api_key_here"
openai.api_key = os.getenv("OPENAI_API_KEY")

# 2. Configuration
INPUT_PATH = "data/processed_articles.jsonl"
OUTPUT_PATH = "data/gold_fulltext_annotations.jsonl"
MODEL = "gpt-4"               # or "gpt-3.5-turbo"
SAMPLE_SIZE_PER_LABEL = 100   # e.g., 100 Positive, 100 Neutral, 100 Negative


def load_articles(path):
    """
    Load articles from JSONL and inject an article_id based on line number.
    """
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            art = json.loads(line)
            art["article_id"] = idx
            yield art


def stratified_sample(articles, label_key="overall_sentiment", n_per_label=100):
    """
    Sample up to n_per_label articles for each sentiment label.
    """
    buckets = defaultdict(list)
    for art in articles:
        buckets[art[label_key]].append(art)
    sample = []
    for label_items in buckets.values():
        sample.extend(random.sample(
            label_items, min(n_per_label, len(label_items))))
    random.shuffle(sample)
    return sample


def build_prompt(article):
    """
    Construct a prompt including the injected article_id, full text, and tickers.
    """
    title = article.get("title", "")
    full_text = article.get("content", "")
    tickers = [t["symbol"] for t in article.get("tickers", [])]
    prompt = f"""
You are a financial news sentiment annotation assistant.
Given the article ID, headline, and full text below, output a JSON object with keys:
- "article_id": the ID
- "true_ticker_sentiments": a dict mapping each ticker to its sentiment ("Positive", "Neutral", or "Negative")
- "true_overall": the overall sentiment ("Positive", "Neutral", or "Negative")

Provide ONLY the JSON object in your response (no extra commentary).

Article ID: {article['article_id']}
Headline: {title}

Full Text:
{full_text}

Tickers to label: {tickers}
"""
    return prompt


def annotate_with_gpt(sample):
    """
    Annotate each sampled article by sending the full-text prompt to the GPT model.
    """
    output = []
    for art in tqdm(sample, desc="Annotating"):
        prompt = build_prompt(art)
        for attempt in range(3):
            try:
                resp = openai.ChatCompletion.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user",   "content": prompt}
                    ],
                    temperature=0.0
                )
                text = resp.choices[0].message.content.strip()
                obj = json.loads(text)
                output.append(obj)
                break
            except Exception:
                time.sleep(2 ** attempt)
        else:
            # If all retries fail, record a placeholder
            output.append({
                "article_id": art["article_id"],
                "true_ticker_sentiments": {},
                "true_overall": None
            })
    return output


def save_annotations(annotations, path):
    """
    Save the list of annotation dicts to a JSONL file.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ann in annotations:
            f.write(json.dumps(ann, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    # Load and sample articles
    articles = list(load_articles(INPUT_PATH))
    sample = stratified_sample(articles, n_per_label=SAMPLE_SIZE_PER_LABEL)
    # Annotate with GPT
    annotations = annotate_with_gpt(sample)
    # Save annotations
    save_annotations(annotations, OUTPUT_PATH)
    print(f"✅ Saved {len(annotations)} annotations to {OUTPUT_PATH}")
