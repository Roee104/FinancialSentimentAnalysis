"""
aggregator.py

Aggregates chunk-level FinBERT sentiment results into per-ticker,
per-sector, and overall article sentiment scores.

New in this version:

- compute_ticker_sentiment(..., method, threshold):
    • method="default": (pos-neg)/total with ±threshold for Neutral
    • method="majority": label by chunk-count majority, score=(pos-neg)/total
    • method="conf_weighted": score=(Σconf_pos-Σconf_neg)/Σconf_all,
      label by ±threshold

- compute_sector_sentiment and compute_article_sentiment use a global
  SECTOR_LOOKUP loaded once at import time.
"""
import pandas as pd
import yfinance as yf
from typing import List, Dict, Tuple
from collections import defaultdict


def load_sector_lookup() -> Dict[str, str]:
    """
    Load S&P 500 tickers and their GICS sectors from Wikipedia.

    Returns:
        Dict mapping ticker symbol -> GICS sector string.
    """
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    df_list = pd.read_html(url)
    df = df_list[0]
    df['Symbol'] = df['Symbol'].str.replace('.', '-', regex=False).str.strip()
    return dict(zip(df['Symbol'], df['GICS Sector']))


# Load sector lookup once at module import
SECTOR_LOOKUP = load_sector_lookup()


def compute_ticker_sentiment(
    ticker_chunks: List[Tuple[str, str, float]],
    method: str = "conf_weighted",
    threshold: float = 0.1
) -> Dict[str, Dict]:
    """
    Aggregate chunk-level sentiment for each ticker.

    Args:
        ticker_chunks: list of (ticker, label, confidence)
        method: one of "default", "majority", "conf_weighted"
        threshold: Neutral cutoff for score in "default" or "conf_weighted"

    Returns:
        Dict mapping ticker -> {score, label, confidence_avg}
    """
    data = defaultdict(lambda: {
        'pos_count': 0, 'neg_count': 0, 'neu_count': 0,
        'pos_conf': 0.0, 'neg_conf': 0.0, 'neu_conf': 0.0
    })
    for ticker, label, conf in ticker_chunks:
        d = data[ticker]
        if label == 'Positive':
            d['pos_count'] += 1
            d['pos_conf'] += conf
        elif label == 'Negative':
            d['neg_count'] += 1
            d['neg_conf'] += conf
        else:
            d['neu_count'] += 1
            d['neu_conf'] += conf

    results = {}
    for ticker, d in data.items():
        pos, neg, neu = d['pos_count'], d['neg_count'], d['neu_count']
        total_chunks = pos + neg + neu
        total_conf = d['pos_conf'] + d['neg_conf'] + d['neu_conf']

        if method == "majority":
            if pos > neg and pos > neu:
                label = 'Positive'
            elif neg > pos and neg > neu:
                label = 'Negative'
            else:
                label = 'Neutral'
            score = (pos - neg) / total_chunks if total_chunks else 0.0

        elif method == "conf_weighted":
            pos_c = d['pos_conf']
            neg_c = d['neg_conf']
            score = ((pos_c - neg_c) / total_conf) if total_conf else 0.0
            if score > threshold:
                label = 'Positive'
            elif score < -threshold:
                label = 'Negative'
            else:
                label = 'Neutral'

        else:  # default
            score = (pos - neg) / total_chunks if total_chunks else 0.0
            if score > threshold:
                label = 'Positive'
            elif score < -threshold:
                label = 'Negative'
            else:
                label = 'Neutral'

        avg_conf = total_conf / total_chunks if total_chunks else 0.0
        results[ticker] = {'score': score,
                           'label': label, 'confidence': avg_conf}
    return results


def compute_sector_sentiment(
    ticker_sentiments: Dict[str, Dict],
    threshold: float = 0.1
) -> Dict[str, Dict]:
    """
    Aggregate ticker-level sentiments into sector-level.
    filling Unknown sectors on the fly via yfinance.

    Args:
        ticker_sentiments: mapping ticker -> {'score','label','confidence'}
        threshold: Neutral cutoff for avg_score

    Returns:
        Dict mapping sector -> {score, label, confidence, weight}
    """
    agg = defaultdict(lambda: {"score_sum": 0.0, "conf_sum": 0.0, "count": 0})
    for ticker, info in ticker_sentiments.items():
        sector = SECTOR_LOOKUP.get(ticker)
        if not sector or sector == "Unknown":
            try:
                info_yf = yf.Ticker(ticker).info
                sector = info_yf.get("sector") or "Unknown"
            except Exception:
                sector = "Unknown"
            SECTOR_LOOKUP[ticker] = sector  # cache

        agg[sector]["score_sum"] += info["score"]
        agg[sector]["conf_sum"] += info["confidence"]
        agg[sector]["count"] += 1

    results = {}
    for sector, d in agg.items():
        cnt = d["count"]
        avg_score = d["score_sum"] / cnt if cnt else 0.0
        avg_conf = d["conf_sum"] / cnt if cnt else 0.0
        if avg_score > threshold:
            label = "Positive"
        elif avg_score < -threshold:
            label = "Negative"
        else:
            label = "Neutral"
        results[sector] = {
            "score":      avg_score,
            "label":      label,
            "confidence": avg_conf,
            "weight":     cnt
        }
    return results


def compute_article_sentiment(
    ticker_sentiments: Dict[str, Dict],
    threshold: float = 0.1
) -> Tuple[str, float]:
    """
    Compute overall article sentiment from ticker-level scores.

    Args:
        ticker_sentiments: mapping ticker -> {'score','label','confidence'}
        threshold: Neutral cutoff for avg_score

    Returns:
        (label, avg_confidence)
    """
    if not ticker_sentiments:
        return 'Neutral', 0.0
    scores = [v['score'] for v in ticker_sentiments.values()]
    confs = [v['confidence'] for v in ticker_sentiments.values()]
    avg_score = sum(scores) / len(scores)
    avg_conf = sum(confs) / len(confs)
    if avg_score > threshold:
        label = 'Positive'
    elif avg_score < -threshold:
        label = 'Negative'
    else:
        label = 'Neutral'
    return label, avg_conf


if __name__ == '__main__':
    sample = [
        ('AAPL', 'Positive', 0.95), ('AAPL',
                                     'Neutral', 0.8), ('AAPL', 'Positive', 0.9),
        ('MSFT', 'Negative', 0.85), ('MSFT', 'Negative', 0.8)
    ]
    for m in ['default', 'majority', 'conf_weighted']:
        print(f"--- method: {m}")
        res = compute_ticker_sentiment(sample, method=m, threshold=0.1)
        print("Ticker-level:", res)
        sec = compute_sector_sentiment(res, threshold=0.1)
        print("Sector-level:", sec)
        art = compute_article_sentiment(res, threshold=0.1)
        print("Article-level:", art)
