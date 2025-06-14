# src/ner_patched.py

"""
Patched version of get_enhanced_symbols that handles numpy arrays properly
"""

from ner import EnhancedNER, normalize_symbols_list
import numpy as np
from typing import List, Dict

def get_enhanced_symbols_patched(
    article: dict,
    ner_extractor: EnhancedNER,
    min_confidence: float = 0.6,
    use_metadata: bool = True
) -> List[str]:
    """
    Patched version that handles numpy arrays in metadata symbols
    """
    all_symbols = set()
    
    # Start with metadata symbols if available and requested
    if use_metadata:
        metadata_symbols = article.get("symbols", [])
        
        # FIX: Handle numpy arrays and other formats
        if metadata_symbols is not None:
            # Convert numpy array to list
            if isinstance(metadata_symbols, np.ndarray):
                metadata_symbols = metadata_symbols.tolist()
            # Handle single string
            elif isinstance(metadata_symbols, str):
                metadata_symbols = [metadata_symbols] if metadata_symbols else []
            # Ensure it's a list
            elif not isinstance(metadata_symbols, list):
                try:
                    metadata_symbols = list(metadata_symbols)
                except:
                    metadata_symbols = []
        else:
            metadata_symbols = []
        
        # Now safely check if we have symbols
        if metadata_symbols and len(metadata_symbols) > 0:
            # Clean exchange suffixes from metadata symbols
            clean_metadata = normalize_symbols_list(metadata_symbols)
            all_symbols.update(clean_metadata)
    
    # Extract from text
    title = article.get("title", "")
    content = article.get("content", "")
    full_text = f"{title}\n\n{content}"
    
    if full_text.strip():
        text_symbols = ner_extractor.extract_symbols_with_confidence(full_text)
        
        # Filter by confidence threshold
        for symbol, confidence in text_symbols:
            if confidence >= min_confidence:
                all_symbols.add(symbol)
    
    return sorted(list(all_symbols))

# Monkey patch the original module
import sys
if 'ner' in sys.modules:
    sys.modules['ner'].get_enhanced_symbols = get_enhanced_symbols_patched
    print("✅ Patched get_enhanced_symbols function")