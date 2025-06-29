"""Run DeBERTa LoRA fine-tuning"""

import subprocess
import sys
import json
from pathlib import Path
from collections import Counter

# Check gold standard
root = Path.cwd()
gold_file = root / "data" / "3000_gold_standard.jsonl"
if not gold_file.exists():
    print(f"❌ Gold standard file not found: {gold_file}")
    sys.exit(1)

# Analyze file format and labels
print("✓ Gold standard file found")
print("📋 Analyzing file format and labels...")
try:
    label_counter = Counter()
    with open(gold_file, 'r') as f:
        for i, line in enumerate(f):
            if i == 0:  # Check first line format
                sample = json.loads(line)
                if 'content' not in sample or 'true_overall' not in sample:
                    print(f"❌ ERROR: Gold standard missing required fields!")
                    sys.exit(1)
                print(f"✓ File format validated")

            article = json.loads(line)
            label_counter[article.get('true_overall', 'NO_LABEL')] += 1

    print(f"\n📊 Label distribution:")
    for label, count in label_counter.most_common():
        print(f"   {label}: {count}")

    if 'Mixed' in label_counter:
        print(
            f"\n⚠️  Note: {label_counter['Mixed']} 'Mixed' labels will be mapped to 'Neutral'")

except Exception as e:
    print(f"❌ Error checking file: {e}")
    sys.exit(1)

print("\n🚀 Starting DeBERTa LoRA fine-tuning...")
print("📦 Using model: mrm8488/deberta-v3-ft-financial-news-sentiment-analysis")

# Run the training command
cmd = [
    sys.executable, "-m", "scripts.run_experiments",
    "--model", "deberta-fin",
    "--lora",
    "--epochs", "2",
    "--lr", "2e-5",
    "--rank", "8",
    "--alpha", "32",
    "--gold", str(gold_file)
]

print(f"\nCommand: {' '.join(cmd)}")
result = subprocess.run(cmd, capture_output=True, text=True)

if result.returncode == 0:
    print("\n✅ Training completed successfully!")
    # Extract adapter path from stderr
    if "[✓] Saved adapter to" in result.stderr:
        for line in result.stderr.split('\n'):
            if "[✓] Saved adapter to" in line:
                print(f"📁 {line}")
    print("\nFull output:")
    print(result.stdout)
else:
    print("\n❌ Training failed!")
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    sys.exit(1)
