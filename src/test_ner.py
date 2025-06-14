# test_ner.py

"""
Test script for the enhanced NER system using real data
"""

import pandas as pd
import numpy as np
import json
from ner import EnhancedNER, get_enhanced_symbols


def clean_symbols(symbols_raw):
    """Clean and normalize symbols from pandas data"""
    try:
        # Handle various types of symbols data
        if symbols_raw is None:
            return []
        elif isinstance(symbols_raw, (list, tuple)):
            return list(symbols_raw)
        elif isinstance(symbols_raw, np.ndarray):
            return symbols_raw.tolist()
        elif isinstance(symbols_raw, str):
            # Handle string representations
            symbols_raw = symbols_raw.strip()
            if not symbols_raw or symbols_raw.lower() in ['nan', 'none', 'null']:
                return []
            elif symbols_raw.startswith('[') and symbols_raw.endswith(']'):
                try:
                    return eval(symbols_raw)
                except:
                    return []
            else:
                return [symbols_raw]
        elif pd.isna(symbols_raw):
            return []
        else:
            return [str(symbols_raw)]
    except:
        return []


def test_with_real_data():
    """Test enhanced NER with your actual collected data"""

    # Load your collected data
    try:
        df = pd.read_parquet("data/financial_news_2020_2025_100k.parquet")
        print(f"✅ Loaded {len(df)} articles for testing")
    except FileNotFoundError:
        print("❌ Could not find collected data. Make sure data_loader.py ran successfully.")
        return

    # Debug: Check what the symbols column looks like
    print(f"\nDebugging symbols column:")
    print(f"Symbols column type: {type(df['symbols'].iloc[0])}")
    print(f"First 5 symbols entries:")
    for i in range(min(5, len(df))):
        symbols = df['symbols'].iloc[i]
        print(f"  [{i}] Type: {type(symbols)}, Value: {symbols}")

    # Initialize enhanced NER
    ner = EnhancedNER()

    # Test on a sample of articles
    sample_size = min(100, len(df))
    sample_df = df.sample(n=sample_size, random_state=42)

    print(f"\nTesting Enhanced NER on {sample_size} articles...")

    results = {
        'total_articles': 0,
        'articles_with_metadata_symbols': 0,
        'articles_with_extracted_symbols': 0,
        'articles_with_no_symbols_old': 0,
        'articles_with_no_symbols_new': 0,
        'total_symbols_metadata': 0,
        'total_symbols_extracted': 0,
        'improvement_cases': []
    }

    for idx, (_, row) in enumerate(sample_df.iterrows()):
        results['total_articles'] += 1

        # Clean symbols properly
        symbols_clean = clean_symbols(row.get('symbols'))

        # Create article dict
        article = {
            'title': str(row['title']) if pd.notna(row['title']) else "",
            'content': str(row['content']) if pd.notna(row['content']) else "",
            'symbols': symbols_clean
        }

        # Original metadata symbols
        original_symbols = symbols_clean
        if original_symbols and len(original_symbols) > 0:
            results['articles_with_metadata_symbols'] += 1
            results['total_symbols_metadata'] += len(original_symbols)
        else:
            results['articles_with_no_symbols_old'] += 1

        # Enhanced extraction
        try:
            enhanced_symbols = get_enhanced_symbols(
                article=article,
                ner_extractor=ner,
                min_confidence=0.6,
                use_metadata=True
            )
        except Exception as e:
            print(f"Error processing article {idx}: {e}")
            enhanced_symbols = []

        if enhanced_symbols:
            results['articles_with_extracted_symbols'] += 1
            results['total_symbols_extracted'] += len(enhanced_symbols)
        else:
            results['articles_with_no_symbols_new'] += 1

        # Check for improvements
        if (not original_symbols or len(original_symbols) == 0) and enhanced_symbols:
            improvement_case = {
                'title': article['title'][:100] + "..." if len(article['title']) > 100 else article['title'],
                'original_symbols': original_symbols,
                'enhanced_symbols': enhanced_symbols
            }
            results['improvement_cases'].append(improvement_case)

        # Show progress every 20 articles
        if (idx + 1) % 20 == 0:
            print(f"  Processed {idx + 1}/{sample_size} articles...")

    # Print results
    print("\n" + "="*60)
    print("ENHANCED NER TEST RESULTS")
    print("="*60)

    print(f"Total articles tested: {results['total_articles']}")
    print(
        f"Articles with metadata symbols: {results['articles_with_metadata_symbols']} ({results['articles_with_metadata_symbols']/results['total_articles']*100:.1f}%)")
    print(
        f"Articles with enhanced symbols: {results['articles_with_extracted_symbols']} ({results['articles_with_extracted_symbols']/results['total_articles']*100:.1f}%)")

    print(f"\nBEFORE Enhancement:")
    print(
        f"  Articles with NO symbols: {results['articles_with_no_symbols_old']} ({results['articles_with_no_symbols_old']/results['total_articles']*100:.1f}%)")

    print(f"\nAFTER Enhancement:")
    print(
        f"  Articles with NO symbols: {results['articles_with_no_symbols_new']} ({results['articles_with_no_symbols_new']/results['total_articles']*100:.1f}%)")

    improvement = results['articles_with_no_symbols_old'] - \
        results['articles_with_no_symbols_new']
    improvement_pct = improvement / results['total_articles'] * 100
    print(
        f"\n🎯 IMPROVEMENT: {improvement} articles now have symbols ({improvement_pct:.1f}% improvement)")

    print(f"\nSymbol extraction stats:")
    if results['total_articles'] > 0:
        print(
            f"  Average symbols per article (metadata): {results['total_symbols_metadata']/results['total_articles']:.1f}")
        print(
            f"  Average symbols per article (enhanced): {results['total_symbols_extracted']/results['total_articles']:.1f}")

    # Show some improvement examples
    if results['improvement_cases']:
        print(f"\n📈 IMPROVEMENT EXAMPLES (showing first 5):")
        for i, case in enumerate(results['improvement_cases'][:5], 1):
            print(f"\n{i}. Title: {case['title']}")
            print(f"   Before: {case['original_symbols'] or 'No symbols'}")
            print(f"   After:  {case['enhanced_symbols']}")

    # Show extraction method statistics
    stats = ner.get_extraction_stats()
    if stats:
        print(f"\n📊 EXTRACTION METHOD BREAKDOWN:")
        for method, count in sorted(stats.items(), key=lambda x: x[1], reverse=True):
            print(f"  {method}: {count}")

    return results


def test_specific_examples():
    """Test on specific challenging examples"""

    ner = EnhancedNER()

    challenging_articles = [
        {
            "title": "Technology Sector Shows Strong Growth",
            "content": "Major technology companies reported strong quarterly results. Apple saw revenue growth of 15% while Microsoft cloud services expanded significantly. Tesla delivery numbers exceeded expectations despite supply chain challenges.",
            "symbols": []
        },
        {
            "title": "Market Analysis: Healthcare Stocks Rally",
            "content": "Johnson & Johnson announced positive trial results for its new drug. Pfizer reported better than expected vaccine revenue. UnitedHealth Group raised its full-year guidance.",
            "symbols": []
        },
        {
            "title": "Banking Sector Update",
            "content": "JPMorgan Chase reported strong trading revenue. Bank of America saw increased loan demand. Wells Fargo improved its credit loss provisions.",
            "symbols": []
        }
    ]

    print("\n" + "="*60)
    print("TESTING CHALLENGING EXAMPLES")
    print("="*60)

    for i, article in enumerate(challenging_articles, 1):
        print(f"\nExample {i}:")
        print(f"Title: {article['title']}")
        print(f"Content: {article['content'][:100]}...")

        enhanced_symbols = get_enhanced_symbols(
            article=article,
            ner_extractor=ner,
            min_confidence=0.5,  # Lower threshold for testing
            use_metadata=True
        )

        print(f"Enhanced extraction: {enhanced_symbols}")

        # Show detailed confidence scores
        full_text = article['title'] + "\n\n" + article['content']
        detailed = ner.extract_symbols_with_confidence(full_text)
        if detailed:
            print("Confidence breakdown:")
            for symbol, conf in detailed:
                print(f"  {symbol}: {conf:.2f}")


def debug_symbols_column():
    """Debug function to understand the symbols column format"""
    try:
        df = pd.read_parquet("data/financial_news_2020_2025_100k.parquet")
        print("DEBUGGING SYMBOLS COLUMN:")
        print("=" * 40)

        print(f"DataFrame shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")

        if 'symbols' in df.columns:
            print(f"\nSymbols column info:")
            print(f"Data type: {df['symbols'].dtype}")
            print(f"Non-null count: {df['symbols'].count()}")
            print(f"Null count: {df['symbols'].isnull().sum()}")

            # Sample of different symbol entries
            print(f"\nFirst 10 symbol entries:")
            for i in range(min(10, len(df))):
                symbols = df['symbols'].iloc[i]
                print(f"  [{i}] {type(symbols).__name__}: {repr(symbols)}")

            # Check unique types
            symbol_types = df['symbols'].apply(
                lambda x: type(x).__name__).value_counts()
            print(f"\nSymbol entry types:")
            print(symbol_types)
        else:
            print("No 'symbols' column found!")

    except Exception as e:
        print(f"Error debugging: {e}")


if __name__ == "__main__":
    print("🧪 Testing Enhanced NER System")
    print("=" * 40)

    # Debug symbols column first
    debug_symbols_column()

    # Test 1: Real data comparison
    test_with_real_data()

    # Test 2: Challenging examples
    test_specific_examples()

    print("\n✅ Testing completed!")
