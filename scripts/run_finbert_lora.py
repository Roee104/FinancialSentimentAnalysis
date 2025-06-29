"""Run FinBERT LoRA fine-tuning"""

import subprocess
import sys
import json
from pathlib import Path
from collections import Counter

# Setup directories
root = Path.cwd()
models_dir = root / "models" / "lora"
models_dir.mkdir(parents=True, exist_ok=True)
(models_dir / "finbert").mkdir(exist_ok=True)
(models_dir / "deberta-fin").mkdir(exist_ok=True)
(root / "experiments").mkdir(exist_ok=True)

# Check gold standard
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
                    print(f"   Found fields: {list(sample.keys())}")
                    print(f"   Required: 'content' and 'true_overall'")
                    sys.exit(1)
                print(
                    f"✓ File format validated - {len(sample['content'])} chars in first article")

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

print("\n🚀 Starting FinBERT LoRA fine-tuning...")

# Run the training command
cmd = [
    sys.executable, "-m", "scripts.run_experiments",
    "--model", "finbert",
    "--lora",
    "--epochs", "2",
    "--lr", "2e-5",
    "--rank", "8",
    "--alpha", "32",
    "--gold", str(gold_file)
]

print(f"Command: {' '.join(cmd)}")
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
