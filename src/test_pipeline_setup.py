# src/test_pipeline_setup.py

"""
Quick test to verify pipeline components work before running full pipeline.
Run this first to catch any issues early.
"""

import os
import sys
import pandas as pd
import torch
from datetime import datetime


def test_imports():
    """Test all required imports"""
    print("📚 Testing imports...")

    modules = {
        'transformers': 'FinBERT',
        'vaderSentiment': 'VADER',
        'nltk': 'NLTK',
        'tqdm': 'Progress bars',
        'pandas': 'Data processing',
        'numpy': 'Numerical',
        'matplotlib': 'Plotting',
        'seaborn': 'Advanced plotting'
    }

    failed = []
    for module, name in modules.items():
        try:
            __import__(module)
            print(f"  ✅ {name} ({module})")
        except ImportError:
            print(f"  ❌ {name} ({module})")
            failed.append(module)

    return len(failed) == 0


def test_data_access():
    """Test data file access"""
    print("\n📁 Testing data access...")

    data_file = "data/financial_news_2020_2025_100k.parquet"

    if not os.path.exists(data_file):
        print(f"  ❌ Data file not found: {data_file}")
        return False

    try:
        # Try loading a small sample
        df = pd.read_parquet(data_file, engine='pyarrow')
        print(f"  ✅ Data file accessible")
        print(f"     Shape: {df.shape}")
        print(f"     Columns: {list(df.columns)}")

        # Check for required columns
        required_cols = ['title', 'content', 'symbols']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            print(f"  ⚠️  Missing columns: {missing}")

        return True

    except Exception as e:
        print(f"  ❌ Error loading data: {e}")
        return False


def test_pipeline_components():
    """Test individual pipeline components"""
    print("\n🔧 Testing pipeline components...")

    tests = []

    # Test 1: Splitter
    try:
        from splitter import split_to_chunks
        test_text = "This is a test. It has multiple sentences! And it works?"
        chunks = split_to_chunks(test_text)
        assert len(chunks) > 0
        print("  ✅ Splitter")
        tests.append(True)
    except Exception as e:
        print(f"  ❌ Splitter: {e}")
        tests.append(False)

    # Test 2: NER
    try:
        from ner import EnhancedNER
        if os.path.exists("data/master_ticker_list.csv"):
            ner = EnhancedNER()
            print("  ✅ Enhanced NER")
            tests.append(True)
        else:
            print("  ⚠️  Enhanced NER (missing ticker list)")
            tests.append(True)  # Not critical
    except Exception as e:
        print(f"  ❌ Enhanced NER: {e}")
        tests.append(False)

    # Test 3: Sentiment
    try:
        from sentiment import FinBERTSentimentAnalyzer
        print("  ✅ FinBERT Sentiment (import successful)")
        tests.append(True)
    except Exception as e:
        print(f"  ❌ FinBERT Sentiment: {e}")
        tests.append(False)

    # Test 4: Aggregator
    try:
        from aggregator import compute_ticker_sentiment
        print("  ✅ Aggregator")
        tests.append(True)
    except Exception as e:
        print(f"  ❌ Aggregator: {e}")
        tests.append(False)

    return all(tests)


def test_memory_and_gpu():
    """Test memory and GPU availability"""
    print("\n💾 Testing system resources...")

    # Check GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  ✅ GPU: {gpu_name} ({gpu_memory:.1f} GB)")

        # Test GPU memory
        try:
            test_tensor = torch.randn(1000, 1000).cuda()
            del test_tensor
            torch.cuda.empty_cache()
            print("  ✅ GPU memory test passed")
        except:
            print("  ⚠️  GPU memory limited")
    else:
        print("  ℹ️  No GPU available (CPU mode)")

    # Check RAM
    try:
        import psutil
        ram = psutil.virtual_memory()
        print(
            f"  ℹ️  RAM: {ram.total/1e9:.1f} GB total, {ram.available/1e9:.1f} GB available")
    except:
        print("  ℹ️  RAM info not available")

    return True


def test_small_pipeline():
    """Run pipeline on tiny sample to test full flow"""
    print("\n🧪 Testing mini pipeline...")

    try:
        # Load tiny sample
        df = pd.read_parquet(
            "data/financial_news_2020_2025_100k.parquet", engine='pyarrow')
        sample = df.head(5)

        # Import components
        from splitter import split_to_chunks
        from sentiment import FinBERTSentimentAnalyzer
        from ner import load_symbol_list, get_combined_symbols

        # Initialize
        print("  Initializing components...")
        analyzer = FinBERTSentimentAnalyzer()
        ticker_dict = {}  # Empty dict for test

        # Process one article
        row = sample.iloc[0]
        title = str(row.get('title', ''))
        content = str(row.get('content', ''))[:500]  # Limit content

        print(f"  Processing: {title[:50]}...")

        # Split
        chunks = split_to_chunks(f"{title}\n\n{content}")
        print(f"  ✅ Split into {len(chunks)} chunks")

        # Sentiment (just first chunk)
        if chunks:
            pred = analyzer.predict([chunks[0]], batch_size=1)
            print(f"  ✅ Sentiment: {pred[0]['label']}")

        print("  ✅ Mini pipeline successful!")
        return True

    except Exception as e:
        print(f"  ❌ Mini pipeline failed: {e}")
        return False


def main():
    """Run all tests"""

    print("🔍 PIPELINE SETUP TEST")
    print("="*50)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python: {sys.version.split()[0]}")
    print("="*50)

    tests = [
        ("Imports", test_imports),
        ("Data Access", test_data_access),
        ("Components", test_pipeline_components),
        ("Resources", test_memory_and_gpu),
        ("Mini Pipeline", test_small_pipeline)
    ]

    results = []

    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ {name} test crashed: {e}")
            results.append((name, False))

    # Summary
    print("\n" + "="*50)
    print("📋 TEST SUMMARY")
    print("="*50)

    all_passed = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{name:15} {status}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n✨ All tests passed! Ready to run pipeline.")
        print("\nNext steps:")
        print("1. For small test: python src/pipeline_optimized.py --max-articles 100")
        print("2. For full run: python src/run_experiments.py")
    else:
        print("\n⚠️  Some tests failed. Please fix issues before running pipeline.")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
