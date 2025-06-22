# scripts/create_gold_standard.py
"""
Create gold standard dataset using GPT-4 for financial sentiment analysis
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from openai import OpenAI
from tqdm import tqdm
import logging
import hashlib
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GoldStandardGenerator:
    def __init__(self, api_key: str, model: str = "gpt-4-turbo-preview"):
        """Initialize with OpenAI API key"""
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.cost_tracker = {
            'input_tokens': 0,
            'output_tokens': 0,
            'api_calls': 0,
            'estimated_cost': 0.0
        }

    def create_annotation_prompt(self, article: Dict) -> str:
        """Create detailed prompt for GPT-4 annotation"""
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
        """Call GPT-4 with retry logic"""
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a financial sentiment analysis expert. Always return valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,  # Low temperature for consistency
                    max_tokens=1000
                )

                # Track usage
                usage = response.usage
                self.cost_tracker['input_tokens'] += usage.prompt_tokens
                self.cost_tracker['output_tokens'] += usage.completion_tokens
                self.cost_tracker['api_calls'] += 1

                # Parse response
                content = response.choices[0].message.content
                return json.loads(content)

            except json.JSONDecodeError as e:
                logger.warning(
                    f"JSON parse error on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
            except Exception as e:
                if "rate_limit" in str(e).lower():
                    logger.warning("Rate limit hit, waiting...")
                    time.sleep(60)
                else:
                    logger.error(f"API error on attempt {attempt + 1}: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)

        return None

    def estimate_cost(self) -> float:
        """Estimate current costs"""
        input_cost = (self.cost_tracker['input_tokens'] / 1000) * 0.01
        output_cost = (self.cost_tracker['output_tokens'] / 1000) * 0.03
        return input_cost + output_cost

    def select_articles_for_annotation(
        self,
        input_file: Path,
        n_samples: int = 300,
    ) -> pd.DataFrame:
        """
        Smart sampling of articles for annotation.

        • “Complex”  = any article that mentions ≥ 1 ticker  
        • “Two-ticker” and “Single-ticker” buckets are unchanged.  
        • Bucket sizes are derived from n_samples (1⁄3 each, rounded down).  
        • If a bucket has fewer rows than requested, we fill the short-fall
          with a random sample from the remaining pool so that the returned
          DataFrame contains *at most* n_samples rows.
        """
        logger.info("Selecting articles for annotation…")

        # ----- load ----------------------------------------------------------------
        articles = []
        with open(input_file, "r", encoding="utf-8") as f:
            for ln in f:
                try:
                    articles.append(json.loads(ln))
                except json.JSONDecodeError:
                    continue

        df = pd.DataFrame(articles)
        logger.info("Loaded %d articles", len(df))

        # ----- features ------------------------------------------------------------
        df["ticker_count"] = df["tickers"].apply(len)

        bucket_target = max(1, n_samples // 3)        # per-bucket goal
        samples: list[pd.DataFrame] = []

        # 1️⃣  complex  (≥ 1 ticker) -------------------------------------------------
        complex_df = df[df["ticker_count"] >= 1]
        if len(complex_df) > 0:
            sel = self._balanced_sample(complex_df, bucket_target)
            samples.append(sel)
            logger.info("Selected %d complex articles (≥ 1 ticker)", len(sel))

        # 2️⃣  two–ticker -----------------------------------------------------------
        two_df = df[df["ticker_count"] == 2]
        if len(two_df) > 0:
            sel = self._balanced_sample(two_df, bucket_target)
            samples.append(sel)
            logger.info("Selected %d two-ticker articles", len(sel))

        # 3️⃣  single-ticker --------------------------------------------------------
        single_df = df[df["ticker_count"] == 1]
        if len(single_df) > 0:
            sel = self._balanced_sample(single_df, bucket_target)
            samples.append(sel)
            logger.info("Selected %d single-ticker articles", len(sel))

        # ---------------------------------------------------------------------------
        if samples:
            selected = pd.concat(samples, ignore_index=True).drop_duplicates(
                subset="article_hash"
            )
        else:
            selected = pd.DataFrame(columns=df.columns)

        # Top-up with random rows (if any short-fall)
        if len(selected) < n_samples:
            remainder = df[~df["article_hash"].isin(selected["article_hash"])]
            extra_needed = n_samples - len(selected)
            if extra_needed > 0 and len(remainder) > 0:
                extra = remainder.sample(n=min(extra_needed, len(remainder)))
                selected = pd.concat([selected, extra], ignore_index=True)

        # Final report --------------------------------------------------------------
        logger.info("Total articles selected: %d", len(selected))
        logger.info(
            "Sentiment distribution: %s",
            selected["overall_sentiment"].value_counts().to_dict(),
        )
        logger.info(
            "Ticker distribution: %s",
            selected["ticker_count"].value_counts().to_dict(),
        )
        return selected

    def _balanced_sample(self, df: pd.DataFrame, n: int) -> pd.DataFrame:
        """Sample with balanced sentiment distribution"""
        sentiment_counts = df['overall_sentiment'].value_counts()
        samples_per_sentiment = n // 3

        samples = []
        for sentiment in ['Positive', 'Neutral', 'Negative']:
            if sentiment in sentiment_counts:
                sentiment_df = df[df['overall_sentiment'] == sentiment]
                sample_size = min(samples_per_sentiment, len(sentiment_df))
                samples.append(sentiment_df.sample(n=sample_size))

        return pd.concat(samples, ignore_index=True)

    def annotate_article(self, article: Dict) -> Optional[Dict]:
        """Annotate a single article"""
        prompt = self.create_annotation_prompt(article)
        annotation = self.call_gpt4(prompt)

        if annotation:
            # Add metadata
            annotation['article_hash'] = article.get('article_hash')
            annotation['title'] = article.get('title')
            annotation['content'] = article.get(
                'content', '')[:500]  # First 500 chars
            annotation['annotator'] = self.model
            annotation['annotation_timestamp'] = datetime.now().isoformat()

            # Map overall sentiment for compatibility
            annotation['true_overall'] = annotation['overall_sentiment']

            # Extract tickers that were found
            annotation['found_tickers'] = list(article.get('tickers', []))

        return annotation

    def validate_annotations(self, annotations: List[Dict]) -> Dict:
        """Validate annotation quality (robust to strange ticker formats)."""
        stats = {
            "total": len(annotations),
            "missing_overall": 0,
            "missing_ticker_sentiments": 0,
            "low_confidence": 0,
            "ticker_mismatches": 0,
        }

        # ---------- helper --------------------------------------------------
        def _normalise(seq) -> set[str]:
            """
            Return a *set of uppercase ticker strings* from:
              • a list of strings          → ["AAPL", "msft"]
              • a list of dicts            → [{"symbol": "MMP"}, ...]
              • a dict-of-dicts (ticker_sentiments) → keys
            Anything empty / None is ignored.
            """
            if isinstance(seq, dict):        # e.g. ticker_sentiments
                seq = seq.keys()

            out: set[str] = set()
            for item in seq or []:
                if isinstance(item, dict):
                    sym = item.get("ticker") or item.get("symbol")
                else:
                    sym = item
                if sym:
                    out.add(str(sym).upper())
            return out
        # --------------------------------------------------------------------

        for ann in annotations:
            # ---------- overall sentiment ----------
            if not ann.get("overall_sentiment"):
                stats["missing_overall"] += 1

            # ---------- ticker-level sentiments ----------
            ticker_sents = ann.get("ticker_sentiments", {})
            if not ticker_sents:
                stats["missing_ticker_sentiments"] += 1

            # ---------- confidence threshold ----------
            if ann.get("overall_confidence", 0.0) < 0.50:
                stats["low_confidence"] += 1

            # ---------- ticker consistency ----------
            found = _normalise(ann.get("found_tickers", []))
            annotated = _normalise(ticker_sents)

            if found and annotated and found != annotated:
                stats["ticker_mismatches"] += 1

        return stats

    def run_consensus_annotation(self, article: Dict, n_runs: int = 3) -> Optional[Dict]:
        """Run multiple annotations and take consensus"""
        annotations = []

        for i in range(n_runs):
            ann = self.annotate_article(article)
            if ann:
                annotations.append(ann)
            time.sleep(1)  # Rate limiting

        if not annotations:
            return None

        # Take consensus
        overall_sentiments = [a['overall_sentiment'] for a in annotations]
        overall_consensus = Counter(overall_sentiments).most_common(1)[0][0]

        # Average confidence
        avg_confidence = np.mean(
            [a.get('overall_confidence', 0.5) for a in annotations])

        # For ticker sentiments, take majority vote
        all_tickers = set()
        for ann in annotations:
            all_tickers.update(ann.get('ticker_sentiments', {}).keys())

        ticker_consensus = {}
        for ticker in all_tickers:
            ticker_sentiments = []
            ticker_confidences = []

            for ann in annotations:
                if ticker in ann.get('ticker_sentiments', {}):
                    ticker_sentiments.append(
                        ann['ticker_sentiments'][ticker]['sentiment'])
                    ticker_confidences.append(
                        ann['ticker_sentiments'][ticker]['confidence'])

            if ticker_sentiments:
                consensus_sentiment = Counter(
                    ticker_sentiments).most_common(1)[0][0]
                avg_confidence = np.mean(ticker_confidences)
                ticker_consensus[ticker] = {
                    'sentiment': consensus_sentiment,
                    'confidence': float(avg_confidence),
                    'agreement': len(set(ticker_sentiments)) == 1
                }

        # Return consensus annotation
        consensus = annotations[0].copy()  # Use first as template
        consensus['overall_sentiment'] = overall_consensus
        consensus['overall_confidence'] = float(avg_confidence)
        consensus['ticker_sentiments'] = ticker_consensus
        consensus['consensus_runs'] = n_runs

        return consensus


def main():
    """Main function to create gold standard dataset"""
    import argparse
    import os

    parser = argparse.ArgumentParser(
        description='Create gold standard dataset')
    parser.add_argument('--api-key', type=str, default=os.getenv('OPENAI_API_KEY'),
                        help='OpenAI API key')
    parser.add_argument('--input', type=str, default='data/processed_articles_optimized.jsonl',
                        help='Input processed articles file')
    parser.add_argument('--output', type=str, default='data/gold_standard_annotations.jsonl',
                        help='Output gold standard file')
    parser.add_argument('--n-samples', type=int, default=300,
                        help='Number of articles to annotate')
    parser.add_argument('--consensus', action='store_true',
                        help='Use consensus annotation (3x cost)')
    parser.add_argument('--max-cost', type=float, default=30.0,
                        help='Maximum cost in USD')

    args = parser.parse_args()

    if not args.api_key:
        raise ValueError(
            "OpenAI API key required. Set OPENAI_API_KEY or use --api-key")

    # Initialize generator
    generator = GoldStandardGenerator(args.api_key)

    # Select articles
    selected_articles = generator.select_articles_for_annotation(
        Path(args.input),
        args.n_samples
    )

    # Annotate articles
    annotations = []
    output_path = Path(args.output)
    output_path.parent.mkdir(exist_ok=True)

    # Resume capability
    existing_hashes = set()
    if output_path.exists():
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    ann = json.loads(line)
                    existing_hashes.add(ann.get('article_hash'))
                    annotations.append(ann)
                except:
                    continue

    logger.info(f"Found {len(existing_hashes)} existing annotations")

    # Process articles
    with open(output_path, 'a', encoding='utf-8') as f:
        for idx, row in tqdm(selected_articles.iterrows(), total=len(selected_articles)):
            # Check cost
            current_cost = generator.estimate_cost()
            if current_cost >= args.max_cost:
                logger.warning(f"Reached cost limit: ${current_cost:.2f}")
                break

            # Skip if already annotated
            if row.get('article_hash') in existing_hashes:
                continue

            # Prepare article dict
            article = row.to_dict()

            # Annotate
            if args.consensus and idx < 50:  # Consensus for first 50
                annotation = generator.run_consensus_annotation(article)
            else:
                annotation = generator.annotate_article(article)

            if annotation:
                annotations.append(annotation)
                f.write(json.dumps(annotation) + '\n')
                f.flush()

            # Rate limiting
            time.sleep(0.5)

            # Progress update
            if (idx + 1) % 10 == 0:
                cost = generator.estimate_cost()
                logger.info(
                    f"Progress: {idx + 1}/{len(selected_articles)}, Cost: ${cost:.2f}")

    # Final validation
    logger.info("\nValidating annotations...")
    validation_stats = generator.validate_annotations(annotations)
    for key, value in validation_stats.items():
        logger.info(f"  {key}: {value}")

    # Final cost
    final_cost = generator.estimate_cost()
    logger.info(f"\nFinal annotation statistics:")
    logger.info(f"  Total annotations: {len(annotations)}")
    logger.info(f"  Total API calls: {generator.cost_tracker['api_calls']}")
    logger.info(f"  Estimated cost: ${final_cost:.2f}")
    logger.info(f"  Output file: {output_path}")

    # Create subset for manual review
    review_subset = np.random.choice(
        annotations, size=min(30, len(annotations)), replace=False)
    review_path = output_path.parent / "gold_standard_for_review.jsonl"
    with open(review_path, 'w', encoding='utf-8') as f:
        for ann in review_subset:
            f.write(json.dumps(ann) + '\n')
    logger.info(f"  Review subset: {review_path}")


if __name__ == "__main__":
    main()
