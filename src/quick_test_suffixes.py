# quick_test_suffixes.py

import pandas as pd
import numpy as np
from ner import EnhancedNER, get_enhanced_symbols, normalize_symbols_list, clean_symbol

def test_suffix_handling():
    """Test the exchange suffix handling functionality"""
    
    print("🧪 Testing Exchange Suffix Handling")
    print("=" * 40)
    
    # Test individual symbol cleaning
    test_symbols = ["AAPL.US", "MSFT.TO", "GOOGL", "TSLA.L", "AMZN.F"]
    print("Testing individual symbol cleaning:")
    for symbol in test_symbols:
        cleaned = clean_symbol(symbol)
        print(f"  {symbol} → {cleaned}")
    
    # Test list normalization
    symbol_list = ["AAPL.US", "MSFT.US", "AAPL.MX", "GOOGL", "TSLA.US"]
    normalized = normalize_symbols_list(symbol_list)
    print(f"\nTesting list normalization:")
    print(f"  Original: {symbol_list}")
    print(f"  Normalized: {normalized}")
    
    # Test with numpy array (like your real data)
    np_array = np.array(["AAPL.US", "MSFT.US", "GOOGL.US"])
    np_normalized = normalize_symbols_list(np_array.tolist())
    print(f"\nTesting numpy array:")
    print(f"  Original: {np_array}")
    print(f"  Normalized: {np_normalized}")

def test_with_real_sample():
    """Test with a small sample of real data"""
    
    print("\n🔍 Testing with Real Data Sample")
    print("=" * 40)
    
    try:
        df = pd.read_parquet("data/financial_news_2020_2025_100k.parquet")
        print(f"Loaded {len(df)} articles")
        
        # Test with first 5 articles
        ner = EnhancedNER()
        
        for i in range(min(5, len(df))):
            row = df.iloc[i]
            
            # Get original symbols
            original_symbols = row['symbols'].tolist() if hasattr(row['symbols'], 'tolist') else list(row['symbols'])
            
            # Clean symbols
            cleaned_symbols = normalize_symbols_list(original_symbols)
            
            print(f"\nArticle {i+1}:")
            print(f"  Title: {row['title'][:60]}...")
            print(f"  Original symbols: {original_symbols}")
            print(f"  Cleaned symbols: {cleaned_symbols}")
            
            # Test enhanced extraction
            article = {
                'title': row['title'],
                'content': row['content'],
                'symbols': original_symbols
            }
            
            enhanced = get_enhanced_symbols(article, ner, min_confidence=0.6)
            print(f"  Enhanced extraction: {enhanced}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_suffix_handling()
    test_with_real_sample()
    print("\n✅ Testing completed!")