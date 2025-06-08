import time
import requests
import pandas as pd
from transformers import AutoTokenizer, logging
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# 1. Configuration
API_TOKEN = os.getenv("EODHD_API_TOKEN")
BASE_URL = "https://eodhd.com/api/news"

# Date range for articles.
FROM_DATE = "2020-01-01"
TO_DATE = "2025-06-07"

# Target number of articles per tag.
TARGET_PER_TAG = 2000

# Pagination batch size for each API call (500 is safe).
BATCH_SIZE = 500

# Sleep between requests to avoid hitting rate limits.
SLEEP_SEC = 0.3

# Broad tags to query for diversified coverage.
TAGS = [
    'earnings release', 'earnings results', 'class action', 'financial results',
    'insider transactions', 'price target', 'quarterly results', 'revenue growth',
    'initial public offering', 'institutional ownership', 'share price', 'research analysis and reports'
]

# Helper function: fetch one page of articles for a given tag and offset


def fetch_page(tag, offset):
    params = {
        "t": tag,
        "from": FROM_DATE,
        "to": TO_DATE,
        "limit": BATCH_SIZE,
        "offset": offset,
        "api_token": API_TOKEN,
        "fmt": "json"
    }
    response = requests.get(BASE_URL, params=params)
    response.raise_for_status()
    return response.json()


# Initialize collected articles list
all_articles = []

# 4. Main collection loop: iterate tags, fetch pages, dedupe per tag
for tag in TAGS:
    print(f"→ Pulling articles for tag: {tag}")
    offset = 0
    tag_collected = {}  # dedupe *per tag* using link as key

    while len(tag_collected) < TARGET_PER_TAG:
        try:
            data = fetch_page(tag, offset)
        except Exception as e:
            print(f"Error fetching page at offset {offset} for tag {tag}: {e}")
            time.sleep(5)
            continue

        if not data:
            print(f"No more articles found for tag: {tag}")
            break  # no more articles for this tag

        new_count = 0
        for art in data:
            link = art.get("link")
            if not link:
                continue  # skip if no link
            if link not in tag_collected:
                tag_collected[link] = {
                    "date":      art.get("date"),
                    "title":     art.get("title", ""),
                    "content":   art.get("content", ""),
                    "symbols":   art.get("symbols", []),
                    "tags":      art.get("tags", []),
                    "sentiment": art.get("sentiment", {}),
                    "tag_source": tag  # keep track of where it came from
                }
                new_count += 1

        print(
            f"  fetched {len(data)} articles, +{new_count} new, total for this tag = {len(tag_collected)}")

        if len(tag_collected) >= TARGET_PER_TAG:
            break  # reached target for this tag

        offset += BATCH_SIZE
        time.sleep(SLEEP_SEC)  # throttle requests

    # add all articles for this tag to global list
    all_articles.extend(tag_collected.values())
    print(f"✅ Collected {len(tag_collected)} articles for tag '{tag}'.")

print(f"🎯 Total collected articles across all tags: {len(all_articles)}")

# Sentiment labeling helper


def label_by_max_prob(sent_dict):
    if not sent_dict or not isinstance(sent_dict, dict):
        return "Neutral"
    neg = sent_dict.get("neg", 0.0)
    neu = sent_dict.get("neu", 0.0)
    pos = sent_dict.get("pos", 0.0)
    label, _ = max(
        [("Negative", neg), ("Neutral", neu), ("Positive", pos)],
        key=lambda x: x[1]
    )
    return label


# Initialize tokenizer for token count computation
logging.set_verbosity_error()
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Function to count tokens


def get_token_count(text):
    return len(tokenizer.tokenize(text))


# Apply token counting and sentiment labeling
for art in all_articles:
    full_text = art["title"] + "\n\n" + art["content"]
    art["token_count"] = get_token_count(full_text)
    art["sentiment_label"] = label_by_max_prob(art["sentiment"])

# Build final DataFrame
df = pd.DataFrame(all_articles)
df = df[[
    "date", "title", "content", "symbols", "tags", "tag_source",
    "sentiment_label", "sentiment", "token_count"
]]

# Save to Parquet
df.to_parquet("data/financial_news_2020_2025.parquet", index=False)
print("✅ Dataset saved to data/financial_news_2020_2025.parquet")
