# Financial Sentiment Analysis Pipeline

A modular, production-ready pipeline for sentiment analysis of financial news articles using FinBERT and enhanced NER.

## 🏗️ Project Structure

```
FinancialSentimentAnalysis/
│
├── config/
│   ├── __init__.py
│   └── settings.py          # Centralized configuration
│
├── core/
│   ├── __init__.py
│   ├── sentiment.py         # Unified sentiment analysis
│   ├── ner.py              # Enhanced NER system
│   ├── aggregator.py       # Sentiment aggregation
│   └── text_processor.py   # Text preprocessing
│
├── data/
│   ├── __init__.py
│   ├── loader.py           # Data loading and collection
│   ├── builder.py          # Ticker list builders
│   └── validator.py        # Data validation
│
├── pipelines/
│   ├── __init__.py
│   ├── base_pipeline.py    # Base pipeline class
│   ├── main_pipeline.py    # Main implementations
│   └── baselines.py        # Baseline models (VADER)
│
├── analysis/
│   ├── __init__.py
│   ├── comparison.py       # Results comparison
│   ├── visualization.py    # Plotting functions
│   └── evaluation.py       # Evaluation metrics
│
├── utils/
│   ├── __init__.py
│   ├── helpers.py          # General utilities
│   └── colab_utils.py      # Colab-specific utilities
│
├── scripts/
│   ├── run_pipeline.py     # Main entry point
│   ├── run_experiments.py  # Run all experiments
│   └── setup_data.py       # Data setup
│
├── notebooks/
│   └── main_analysis.ipynb # Colab notebook
│
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py    # Unit tests
│
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/yourusername/FinancialSentimentAnalysis.git
cd FinancialSentimentAnalysis

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt')"
```

### 2. Data Collection (Optional)

```bash
# Collect financial news data
python data/loader.py
```

### 3. Run Pipeline

```bash
# Run optimized pipeline (recommended)
python scripts/run_pipeline.py --pipeline optimized

# Run with custom settings
python scripts/run_pipeline.py \
    --pipeline optimized \
    --batch-size 100 \
    --threshold 0.15 \
    --output results/my_results.jsonl
```

### 4. Run All Experiments

```bash
# Run complete comparison
python scripts/run_experiments.py
```

## 📊 Pipeline Options

### Sentiment Modes

1. **Standard**: Original FinBERT without modifications
2. **Optimized**: With bias correction (recommended)
3. **Calibrated**: Advanced bias reduction

### Aggregation Methods

1. **default**: Simple count-based
2. **majority**: Majority voting
3. **conf_weighted**: Confidence-weighted (recommended)

### Example Commands

```bash
# Standard pipeline (baseline)
python scripts/run_pipeline.py --pipeline standard

# VADER baseline
python scripts/run_pipeline.py --pipeline vader --vader-threshold 0.05

# Optimized with custom threshold
python scripts/run_pipeline.py --pipeline optimized --threshold 0.15

# Calibrated with majority voting
python scripts/run_pipeline.py --pipeline calibrated --method majority
```

## 🔧 Configuration

Edit `config/settings.py` to customize:

- API tokens
- Model parameters
- Processing thresholds
- Batch sizes
- Output directories

## 📈 Key Features

### Enhanced NER

- Context-aware ticker extraction
- Exchange suffix handling (.US, .TO, etc.)
- Company name recognition
- Confidence scoring

### Sentiment Analysis

- Multiple modes (standard/optimized/calibrated)
- Bias reduction techniques
- Batch processing optimization
- GPU support

### Aggregation

- Ticker-level sentiment
- Sector-level rollup
- Article-level summary
- Configurable methods

### Analysis Tools

- Results comparison
- Visualization generation
- Performance metrics
- Statistical reports

## 🎯 Results

The optimized pipeline achieves:

- **Neutral bias reduction**: From ~80% to ~63% (16.6pp improvement)
- **Ticker coverage**: 87.5% of articles
- **Average tickers/article**: 3.01
- **Processing speed**: ~1000 articles/minute on GPU

## 📝 Output Format

Each processed article produces a JSON record:

```json
{
  "date": "2024-01-15",
  "title": "Apple Reports Q4 Earnings...",
  "overall_sentiment": "Positive",
  "overall_confidence": 0.89,
  "tickers": [
    {
      "symbol": "AAPL",
      "label": "Positive",
      "score": 0.75,
      "confidence": 0.92
    }
  ],
  "sectors": [
    {
      "sector": "Technology",
      "label": "Positive",
      "score": 0.68,
      "confidence": 0.88
    }
  ]
}
```

## 🧪 Testing

```bash
# Run tests
python -m pytest tests/

# Test specific component
python -m pytest tests/test_pipeline.py::test_sentiment_analysis
```

## 📊 Visualization

The analysis module generates:

- Sentiment distribution plots
- Ticker coverage analysis
- Neutral bias reduction charts
- Performance comparisons

## 🚀 Google Colab

For Google Colab usage:

```python
# Clone repository
!git clone https://github.com/yourusername/FinancialSentimentAnalysis.git
%cd FinancialSentimentAnalysis

# Install requirements
!pip install -r requirements.txt

# Run pipeline
!python scripts/run_pipeline.py --batch-size 50 --max-articles 1000
```

## 📄 License

MIT License - see LICENSE file for details

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📧 Contact

For questions or issues, please open a GitHub issue or contact the maintainers.
