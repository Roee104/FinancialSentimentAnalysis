"""Run FinBERT LoRA fine-tuning"""

import subprocess
import sys
from pathlib import Path

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

print("✓ Gold standard file found")
print("🚀 Starting FinBERT LoRA fine-tuning...")

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
    print("✅ Training completed successfully!")
    print("STDOUT:", result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
else:
    print("❌ Training failed!")
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    sys.exit(1)
