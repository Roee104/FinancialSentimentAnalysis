# article_length_analysis.py

import json
import os
import statistics
from tqdm import tqdm

INPUT_PATH = "data/processed_articles.jsonl"

def load_articles(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)

def compute_length_stats(path):
    char_counts = []
    word_counts = []
    for art in tqdm(load_articles(path), desc="Measuring lengths"):
        text = art.get("title", "") + "\n\n" + art.get("content", "")
        char_counts.append(len(text))
        word_counts.append(len(text.split()))
    stats = {
        "num_articles": len(char_counts),
        "avg_chars": statistics.mean(char_counts),
        "median_chars": statistics.median(char_counts),
        "min_chars": min(char_counts),
        "max_chars": max(char_counts),
        "avg_words": statistics.mean(word_counts),
        "median_words": statistics.median(word_counts),
        "min_words": min(word_counts),
        "max_words": max(word_counts),
    }
    return stats

if __name__ == "__main__":
    stats = compute_length_stats(INPUT_PATH)
    os.makedirs("data", exist_ok=True)
    out_path = "data/article_length_stats.json"
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"✅ Length stats saved to {out_path}")
    print(json.dumps(stats, indent=2))