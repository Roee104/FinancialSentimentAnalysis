# src/aggregator.py

"""
Aggregates chunk-level FinBERT sentiment into per-ticker,
per-sector, and overall article scores.

- Uses a full ticker->sector map (ticker_sector.csv + S&P 500).
- Defaults to conf_weighted method at threshold=0.1.
"""

import pandas as pd
from collections import defaultdict
from typing import Dict, List, Tuple

# 1) Load S&P 500 GICS sectors from Wikipedia


def load_sp500_sectors() -> Dict[str, str]:
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    df = pd.read_html(url)[0]
    df["Symbol"] = df["Symbol"].str.replace(".", "-", regex=False).str.strip()
    return dict(zip(df["Symbol"], df["GICS Sector"]))

# 2) Load your full ticker_sector.csv


def load_full_sectors(
    csv_path: str = "data/ticker_sector.csv"
) -> Dict[str, str]:
    df = pd.read_csv(csv_path, dtype=str)
    return dict(zip(df["symbol"].str.upper(), df["sector"]))


# Build a master sector lookup once
SP500 = load_sp500_sectors()
FULL = load_full_sectors()
SECTOR_LOOKUP = {**FULL, **SP500}  # FULL overrides SP500 for duplicates


def compute_ticker_sentiment(
    ticker_chunks: List[Tuple[str, str, float]],
    method: str = "conf_weighted",
    threshold: float = 0.1
) -> Dict[str, Dict]:
    """
    Aggregates chunk-level sentiments per ticker.
    Default: confidence-weighted at ±0.1 threshold.
    Methods: default / majority / conf_weighted
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
    for tick, d in data.items():
        pos, neg, neu = d['pos_count'], d['neg_count'], d['neu_count']
        tot_chunks = pos + neg + neu
        tot_conf = d['pos_conf'] + d['neg_conf'] + d['neu_conf']

        if method == "majority":
            if pos > neg and pos > neu:
                label = 'Positive'
            elif neg > pos and neg > neu:
                label = 'Negative'
            else:
                label = 'Neutral'
            score = (pos - neg) / tot_chunks if tot_chunks else 0.0

        elif method == "conf_weighted":
            score = ((d['pos_conf'] - d['neg_conf']) /
                     tot_conf) if tot_conf else 0.0
            if score > threshold:
                label = 'Positive'
            elif score < -threshold:
                label = 'Negative'
            else:
                label = 'Neutral'

        else:  # default
            score = (pos - neg) / tot_chunks if tot_chunks else 0.0
            if score > threshold:
                label = 'Positive'
            elif score < -threshold:
                label = 'Negative'
            else:
                label = 'Neutral'

        avg_conf = tot_conf / tot_chunks if tot_chunks else 0.0
        results[tick] = {'score': score,
                         'label': label, 'confidence': avg_conf}

    return results


def compute_sector_sentiment(
    ticker_sentiments: Dict[str, Dict],
    threshold: float = 0.1
) -> Dict[str, Dict]:
    """
    Aggregates ticker-level scores into sectors using SECTOR_LOOKUP.
    """
    agg = defaultdict(lambda: {'score_sum': 0.0, 'conf_sum': 0.0, 'count': 0})
    for tick, info in ticker_sentiments.items():
        sector = SECTOR_LOOKUP.get(tick, 'Unknown')
        agg[sector]['score_sum'] += info['score']
        agg[sector]['conf_sum'] += info['confidence']
        agg[sector]['count'] += 1

    results = {}
    for sector, d in agg.items():
        cnt = d['count']
        avg_score = d['score_sum']/cnt if cnt else 0.0
        avg_conf = d['conf_sum']/cnt if cnt else 0.0
        if avg_score > threshold:
            label = 'Positive'
        elif avg_score < -threshold:
            label = 'Negative'
        else:
            label = 'Neutral'
        results[sector] = {
            'score':      avg_score,
            'label':      label,
            'confidence': avg_conf,
            'weight':     cnt
        }
    return results


def compute_article_sentiment(
    ticker_sentiments: Dict[str, Dict],
    threshold: float = 0.1
) -> Tuple[str, float]:
    """
    Aggregates overall article sentiment from ticker-level scores.
    """
    if not ticker_sentiments:
        return 'Neutral', 0.0
    scores = [v['score'] for v in ticker_sentiments.values()]
    confs = [v['confidence'] for v in ticker_sentiments.values()]
    avg_score = sum(scores)/len(scores)
    avg_conf = sum(confs)/len(confs)
    if avg_score > threshold:
        label = 'Positive'
    elif avg_score < -threshold:
        label = 'Negative'
    else:
        label = 'Neutral'
    return label, avg_conf
