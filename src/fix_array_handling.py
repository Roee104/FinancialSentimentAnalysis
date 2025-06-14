# fix_array_handling.py

"""
Fix the numpy array issue in ner.py by adding a helper function
"""

import os
import fileinput
import sys

def add_array_handling_to_ner():
    """Add array handling to ner.py"""
    
    print("🔧 Fixing array handling in ner.py...")
    
    # The fix to add at the beginning of get_enhanced_symbols function
    fix_code = '''    # Handle numpy array for metadata_symbols
    if metadata_symbols is not None:
        if hasattr(metadata_symbols, '__len__') and not isinstance(metadata_symbols, str):
            # It's array-like, convert to list
            if hasattr(metadata_symbols, 'tolist'):
                metadata_symbols = metadata_symbols.tolist()
            else:
                metadata_symbols = list(metadata_symbols)
'''
    
    # Read the file
    with open('src/ner.py', 'r') as f:
        content = f.read()
    
    # Check if fix already applied
    if "Handle numpy array for metadata_symbols" in content:
        print("✅ Fix already applied to ner.py")
        return
    
    # Find the location to insert the fix
    search_str = "if use_metadata:\n        metadata_symbols = article.get(\"symbols\", [])\n        if metadata_symbols:"
    
    if search_str in content:
        # Replace the problematic line
        new_content = content.replace(
            search_str,
            "if use_metadata:\n        metadata_symbols = article.get(\"symbols\", [])\n        " + fix_code.strip() + "\n        if metadata_symbols and len(metadata_symbols) > 0:"
        )
        
        # Write back
        with open('src/ner.py', 'w') as f:
            f.write(new_content)
        
        print("✅ Fixed ner.py")
    else:
        print("❌ Could not find the target code in ner.py")
        print("   Attempting alternative fix...")
        
        # Alternative: Add at the beginning of the function
        import_pos = content.find("def get_enhanced_symbols(")
        if import_pos != -1:
            # Find the end of the function definition
            func_start = content.find(":", import_pos)
            if func_start != -1:
                # Insert after the docstring
                docstring_end = content.find('"""', func_start + 1)
                if docstring_end != -1:
                    docstring_end = content.find('"""', docstring_end + 3) + 3
                    
                    new_content = (
                        content[:docstring_end] + 
                        "\n    # Fix for numpy array handling\n" +
                        "    if 'symbols' in article:\n" +
                        "        symbols = article['symbols']\n" +
                        "        if hasattr(symbols, 'tolist'):\n" +
                        "            article['symbols'] = symbols.tolist()\n" +
                        content[docstring_end:]
                    )
                    
                    with open('src/ner.py', 'w') as f:
                        f.write(new_content)
                    
                    print("✅ Applied alternative fix to ner.py")

def test_fix():
    """Test if the fix works"""
    print("\n🧪 Testing the fix...")
    
    try:
        import numpy as np
        from ner import get_enhanced_symbols, EnhancedNER
        
        # Create test article with numpy array
        test_article = {
            'title': 'Test Article',
            'content': 'Apple (AAPL) reported earnings.',
            'symbols': np.array(['AAPL', 'MSFT'])  # This would cause the error
        }
        
        ner = EnhancedNER()
        result = get_enhanced_symbols(test_article, ner)
        print(f"✅ Test passed! Result: {result}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")

def check_recent_results():
    """Check the actual sentiment distribution in recent results"""
    print("\n📊 Checking actual sentiment distribution...")
    
    import json
    from collections import defaultdict
    
    # Check last 1000 articles
    sentiments = defaultdict(int)
    count = 0
    
    with open('data/processed_articles.jsonl', 'r') as f:
        # Go to end and read backwards
        lines = f.readlines()
        for line in reversed(lines[-1000:]):
            try:
                data = json.loads(line)
                sentiments[data['overall_sentiment']] += 1
                count += 1
            except:
                continue
    
    print(f"\nLast {count} articles sentiment distribution:")
    for sent, cnt in sentiments.items():
        print(f"  {sent}: {cnt} ({cnt/count*100:.1f}%)")

if __name__ == "__main__":
    # Apply the fix
    add_array_handling_to_ner()
    
    # Test it
    test_fix()
    
    # Check recent results
    check_recent_results()