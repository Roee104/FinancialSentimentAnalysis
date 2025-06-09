"""
aggregator.py

Aggregates chunk-level FinBERT sentiment results into per-ticker,
per-sector, and overall article sentiment scores.

Functions:
- load_sector_lookup: fetches S&P 500 ticker-to-sector mapping from Wikipedia.
- compute_ticker_sentiment: given a list of (ticker, label, confidence) tuples,
  returns final label, score, and confidence per ticker.
- compute_sector_sentiment: maps tickers to GICS sectors and aggregates their scores.
- compute_article_sentiment: aggregates ticker-level results into overall sentiment.
"""
import pandas as pd
from typing import List, Dict, Tuple
from collections import defaultdict

# 1. Dynamically load symbol -> sector mapping from Wikipedia's S&P 500 list


def load_sector_lookup() -> Dict[str, str]:
    """
    Load S&P 500 tickers and their GICS sectors from Wikipedia.

    Returns:
        Dict mapping ticker symbol -> GICS sector string.
    """
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    # Read all tables; the first table is the constituents list
    df_list = pd.read_html(url)
    df = df_list[0]
    # Ensure the Symbol column matches our ticker format (remove dots)
    df['Symbol'] = df['Symbol'].str.replace('.', '-', regex=False).str.strip()
    # Create mapping
    return dict(zip(df['Symbol'], df['GICS Sector']))


# Initialize sector lookup once\ n
SECTOR_LOOKUP = load_sector_lookup()


def compute_ticker_sentiment(
    ticker_chunks: List[Tuple[str, str, float]]
) -> Dict[str, Dict]:
    """
    Aggregate chunk-level sentiment for each ticker.

    Args:
        ticker_chunks: list of tuples (ticker, label, confidence)
                         where label in {'Positive','Neutral','Negative'}.

    Returns:
        Dict mapping ticker -> {
            'score': float ([-1,1] weighted sentiment),
            'label': str (final label),
            'confidence': float (average confidence)
        }
    """
    ticker_data = defaultdict(
        lambda: {'pos': 0.0, 'neg': 0.0, 'neu': 0.0, 'conf_sum': 0.0, 'count': 0})
    for ticker, label, conf in ticker_chunks:
        data = ticker_data[ticker]
        data['conf_sum'] += conf
        data['count'] += 1
        if label == 'Positive':
            data['pos'] += 1
        elif label == 'Negative':
            data['neg'] += 1
        else:
            data['neu'] += 1

    results = {}
    for ticker, data in ticker_data.items():
        pos, neg, neu = data['pos'], data['neg'], data['neu']
        total = pos + neg + neu
        score = (pos - neg) / total if total > 0 else 0.0
        if score > 0.2:
            label = 'Positive'
        elif score < -0.2:
            label = 'Negative'
        else:
            label = 'Neutral'
        confidence = data['conf_sum'] / data['count']
        results[ticker] = {'score': score,
                           'label': label, 'confidence': confidence}
    return results


def compute_sector_sentiment(
    ticker_sentiments: Dict[str, Dict]
) -> Dict[str, Dict]:
    """
    Aggregate ticker-level sentiments into sector-level.

    Args:
        ticker_sentiments: mapping ticker -> {'score', 'label', 'confidence'}

    Returns:
        Dict mapping sector -> {
            'score': float, 'label': str, 'confidence': float, 'weight': float
        }
    """
    sector_data = defaultdict(
        lambda: {'score_sum': 0.0, 'weight_sum': 0.0, 'conf_sum': 0.0})
    for ticker, info in ticker_sentiments.items():
        sector = SECTOR_LOOKUP.get(ticker, 'Unknown')
        score = info['score']
        conf = info['confidence']
        weight = 1.0
        sector_data[sector]['score_sum'] += score * weight
        sector_data[sector]['weight_sum'] += weight
        sector_data[sector]['conf_sum'] += conf * weight

    sector_results = {}
    for sector, data in sector_data.items():
        avg_score = data['score_sum'] / \
            data['weight_sum'] if data['weight_sum'] else 0.0
        avg_conf = data['conf_sum'] / \
            data['weight_sum'] if data['weight_sum'] else 0.0
        if avg_score > 0.2:
            label = 'Positive'
        elif avg_score < -0.2:
            label = 'Negative'
        else:
            label = 'Neutral'
        sector_results[sector] = {
            'score': avg_score,
            'label': label,
            'confidence': avg_conf,
            'weight': data['weight_sum']
        }
    return sector_results


def compute_article_sentiment(
    ticker_sentiments: Dict[str, Dict]
) -> Tuple[str, float]:
    """
    Compute overall article sentiment from ticker-level scores.

    Args:
        ticker_sentiments: mapping ticker -> {'score', 'label', 'confidence'}

    Returns:
        (label, confidence) for the entire article.
    """
    if not ticker_sentiments:
        return 'Neutral', 0.0
    total, conf_sum = 0.0, 0.0
    for info in ticker_sentiments.values():
        total += info['score']
        conf_sum += info['confidence']
    count = len(ticker_sentiments)
    avg_score = total / count
    avg_conf = conf_sum / count
    if avg_score > 0.2:
        label = 'Positive'
    elif avg_score < -0.2:
        label = 'Negative'
    else:
        label = 'Neutral'
    return label, avg_conf


if __name__ == '__main__':
    sample_chunks = [
        ('AAPL', 'Positive', 0.95),
        ('AAPL', 'Neutral', 0.80),
        ('AAPL', 'Positive', 0.90),
        ('MSFT', 'Negative', 0.85),
        ('MSFT', 'Negative', 0.80)
    ]
    ticker_res = compute_ticker_sentiment(sample_chunks)
    print('Ticker-level:', ticker_res)
    sector_res = compute_sector_sentiment(ticker_res)
    print('Sector-level:', sector_res)
    article_label, article_conf = compute_article_sentiment(ticker_res)
    print('Article-level:', article_label, article_conf)
