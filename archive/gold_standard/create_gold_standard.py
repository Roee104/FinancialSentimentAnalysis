import json
import time
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from collections import Counter
from openai import OpenAI
from tqdm import tqdm
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GoldStandardGenerator:
    def __init__(self, api_key: str, model: str = "gpt-4-turbo-preview"):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.cost_tracker = {
            'input_tokens': 0,
            'output_tokens': 0,
            'api_calls': 0,
            'estimated_cost': 0.0
        }

    def create_annotation_prompt(self, article: Dict) -> str:
        return f"""You are a financial sentiment analysis expert. Analyze this financial news article and provide detailed sentiment labels.

ARTICLE:
Title: {article.get('title', '')}
Content: {article.get('content', '')}

INSTRUCTIONS:
1. Determine the overall article sentiment (Positive/Neutral/Negative)
2. For EACH company/ticker mentioned:
   - Identify the ticker symbol
   - Determine sentiment toward that specific company
   - Provide confidence (0.0-1.0)
   - Give brief rationale (1 sentence)

SENTIMENT GUIDELINES:
- Positive: growth, beats expectations, upgrades, partnerships, positive outlook
- Negative: losses, misses expectations, downgrades, layoffs, negative outlook  
- Neutral: factual reporting, mixed signals, no clear direction

IMPORTANT:
- If multiple companies have different sentiments, overall = predominant sentiment
- Consider the weight/importance of each company in the article
- Be precise about which sentiment applies to which company

OUTPUT FORMAT (valid JSON):
{{
    "overall_sentiment": "Positive|Neutral|Negative",
    "overall_confidence": 0.85,
    "overall_rationale": "Brief explanation",
    "ticker_sentiments": {{
        "AAPL": {{
            "sentiment": "Positive",
            "confidence": 0.90,
            "rationale": "Reported record earnings beating analyst expectations"
        }},
        "MSFT": {{
            "sentiment": "Negative",
            "confidence": 0.75,
            "rationale": "Facing regulatory scrutiny over antitrust concerns"
        }}
    }},
    "key_sentences": [
        "Apple reported its best quarter ever with revenue up 15%",
        "Microsoft faces potential fines from EU regulators"
    ]
}}

Return ONLY the JSON object, no additional text."""

    def call_gpt4(self, prompt: str, max_retries: int = 3) -> Optional[Dict]:
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a financial sentiment analysis expert. Always return valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=1000
                )
                usage = response.usage
                self.cost_tracker['input_tokens'] += usage.prompt_tokens
                self.cost_tracker['output_tokens'] += usage.completion_tokens
                self.cost_tracker['api_calls'] += 1
                content = response.choices[0].message.content
                return json.loads(content)
            except Exception as e:
                logger.warning(f"GPT error: {e}")
                time.sleep(5)
        return None

    def estimate_cost(self) -> float:
        return (self.cost_tracker['input_tokens'] / 1000) * 0.01 + \
               (self.cost_tracker['output_tokens'] / 1000) * 0.03

    def annotate_article(self, article: Dict) -> Optional[Dict]:
        prompt = self.create_annotation_prompt(article)
        annotation = self.call_gpt4(prompt)
        if annotation:
            annotation['article_hash'] = article.get('article_hash')
            annotation['title'] = article.get('title')
            annotation['content'] = article.get('content', '')[:500]
            annotation['annotator'] = self.model
            annotation['annotation_timestamp'] = datetime.now().isoformat()
            annotation['true_overall'] = annotation['overall_sentiment']
            annotation['found_tickers'] = list(article.get('tickers', []))
        return annotation

    def validate_annotations(self, annotations: List[Dict]) -> Dict:
        stats = {
            "total": len(annotations),
            "missing_overall": 0,
            "missing_ticker_sentiments": 0,
            "low_confidence": 0,
            "ticker_mismatches": 0,
        }
        for ann in annotations:
            if not ann.get("overall_sentiment"):
                stats["missing_overall"] += 1
            if not ann.get("ticker_sentiments"):
                stats["missing_ticker_sentiments"] += 1
            if ann.get("overall_confidence", 0.0) < 0.50:
                stats["low_confidence"] += 1
        return stats


def main():
    import argparse
    import os

    parser = argparse.ArgumentParser(description='Round 2 annotation script')
    parser.add_argument('--api-key', type=str, default=os.getenv('OPENAI_API_KEY'),
                        help='OpenAI API key')
    parser.add_argument('--input', type=str, default='data/round2_candidates_3000.jsonl',
                        help='Input candidate file')
    parser.add_argument('--output', type=str, default='data/3000_gold_standard_round2.jsonl',
                        help='Output annotation file')
    parser.add_argument('--max-cost', type=float, default=25.0,
                        help='Max token cost in USD')
    args = parser.parse_args()

    if not args.api_key:
        raise ValueError("Set OPENAI_API_KEY or use --api-key")

    generator = GoldStandardGenerator(api_key=args.api_key)

    # Load candidate articles
    with open(args.input, "r", encoding="utf-8") as f:
        articles = [json.loads(line) for line in f]

    logger.info(f"Loaded {len(articles)} candidate articles")

    # Check existing
    existing_hashes = set()
    if Path(args.output).exists():
        with open(args.output, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    ann = json.loads(line)
                    existing_hashes.add(ann.get("article_hash"))
                except:
                    continue

    logger.info(f"Found {len(existing_hashes)} existing hashes in output")

    new_articles = [a for a in articles if a.get("article_hash") not in existing_hashes]
    logger.info(f"{len(new_articles)} articles remaining for annotation")

    # Annotate
    annotations = []
    with open(args.output, "a", encoding="utf-8") as f:
        for i, article in enumerate(tqdm(new_articles)):
            if generator.estimate_cost() >= args.max_cost:
                logger.warning("Stopping — cost limit reached")
                break
            ann = generator.annotate_article(article)
            if ann:
                annotations.append(ann)
                f.write(json.dumps(ann) + "\n")
                f.flush()
            time.sleep(1)

    stats = generator.validate_annotations(annotations)
    logger.info("Validation stats:")
    for k, v in stats.items():
        logger.info(f"  {k}: {v}")
    logger.info(f"Total cost: ${generator.estimate_cost():.2f}")


if __name__ == "__main__":
    main()
