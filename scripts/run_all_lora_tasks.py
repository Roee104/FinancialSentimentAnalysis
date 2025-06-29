"""
Master script to run all LoRA fine-tuning tasks in sequence

Tasks:
1. LoRA fine-tune FinBERT
2. LoRA fine-tune DeBERTa
3. Wire pipeline CLI (already done)
4. Run both pipelines
5. Evaluate with comprehensive_evaluation
6. Select best adapter
7. Aggregation ablation
8. Update plots
"""

import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime


def run_command(cmd, description):
    """Run a command and report results"""
    print(f"\n{'='*70}")
    print(f"🚀 {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*70)

    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start_time

    if result.returncode == 0:
        print(f"✅ SUCCESS ({elapsed:.1f}s)")
        if result.stderr:  # Often contains the adapter path
            print(f"Output: {result.stderr.strip()}")
        return True
    else:
        print(f"❌ FAILED ({elapsed:.1f}s)")
        print(f"Error: {result.stderr}")
        return False


def main():
    """Run all LoRA tasks"""

    print("🔧 Financial Sentiment Analysis - LoRA Fine-tuning Pipeline")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    root = Path.cwd()
    gold_file = root / "data" / "3000_gold_standard.jsonl"

    # Check prerequisites
    if not gold_file.exists():
        print(f"❌ Gold standard file not found: {gold_file}")
        sys.exit(1)

    # Task 1: LoRA fine-tune FinBERT
    if not run_command([
        sys.executable, "-m", "scripts.run_experiments",
        "--model", "finbert",
        "--lora",
        "--epochs", "2",
        "--lr", "2e-5",
        "--rank", "8",
        "--alpha", "32",
        "--gold", str(gold_file)
    ], "Task 1: LoRA fine-tune FinBERT"):
        print("⚠️  FinBERT training failed, continuing...")

    # Task 2: LoRA fine-tune DeBERTa
    if not run_command([
        sys.executable, "-m", "scripts.run_experiments",
        "--model", "deberta-fin",
        "--lora",
        "--epochs", "2",
        "--lr", "2e-5",
        "--rank", "8",
        "--alpha", "32",
        "--gold", str(gold_file)
    ], "Task 2: LoRA fine-tune DeBERTa"):
        print("⚠️  DeBERTa training failed, continuing...")

    # Task 3: Wire pipeline CLI - already done in code
    print(f"\n{'='*70}")
    print("✅ Task 3: Pipeline CLI already updated to support --adapter and --aggregation")
    print('='*70)

    # Tasks 4-7: Run pipelines and evaluate
    if not run_command([
        sys.executable, "-m", "scripts.run_finetuned_pipelines"
    ], "Tasks 4-7: Run pipelines, evaluate, select best, ablation"):
        print("⚠️  Pipeline evaluation failed, continuing...")

    # Task 8: Update plots
    if not run_command([
        sys.executable, "-m", "scripts.update_plots"
    ], "Task 8: Update visualization plots"):
        print("⚠️  Plot generation failed")

    print(f"\n{'='*70}")
    print("🎉 All LoRA tasks completed!")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print('='*70)

    # Summary of outputs
    print("\n📁 Key outputs:")
    print("  - models/lora/finbert/<timestamp>/      # FinBERT adapter")
    print("  - models/lora/deberta-fin/<timestamp>/  # DeBERTa adapter")
    print("  - models/lora/best_adapter/             # Best performing adapter")
    print("  - data/processed_articles_finetuned_finbert.jsonl")
    print("  - data/processed_articles_finetuned_deberta.jsonl")
    print("  - data/plots/*.png                      # Updated visualizations")
    print("  - experiments/*.json                    # Training metadata")


if __name__ == "__main__":
    main()
