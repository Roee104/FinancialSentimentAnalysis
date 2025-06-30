#!/usr/bin/env python3
"""
Integrate the pre-trained NER into your pipeline
"""
import core.ner
from core.pretrained_financial_ner import FinancialNERWrapper
import sys
sys.path.append('/content/FinancialSentimentAnalysis')

# Import the wrapper from your new file

# Replace the original UnifiedNER with the pre-trained version
core.ner.UnifiedNER = FinancialNERWrapper

# Make it permanent by updating the module
sys.modules['core.ner'].UnifiedNER = FinancialNERWrapper

print("✅ Pre-trained Financial NER integrated successfully!")

# Test it immediately
print("\n🧪 Testing the integration...")

ner = FinancialNERWrapper()

test_cases = [
    "Apple (AAPL) reported strong earnings",
    "Microsoft stock rose 5%",
    "Tim Cook announced new products",  # Should find AAPL, not COOK
    "Bank of America upgraded to buy"   # Should find BAC via alias
]

for test in test_cases:
    article = {"title": test, "content": ""}
    results = ner.extract_symbols(article)
    print(f"\n'{test}'")
    print(f"→ Extracted: {results}")

print("\n✅ Integration complete! Your pipeline now uses pre-trained NER.")
