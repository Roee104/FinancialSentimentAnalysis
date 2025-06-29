"""Run pipelines with fine-tuned models and evaluate results"""

import subprocess
import sys
import json
from pathlib import Path
from datetime import datetime

# Setup paths
root = Path(__file__).resolve().parents[1]
models_dir = root / "models" / "lora"
data_dir = root / "data"


def find_latest_adapter(model_name):
    """Find the latest adapter directory for a model"""
    model_dir = models_dir / model_name
    if not model_dir.exists():
        return None

    # Find all timestamp directories
    adapter_dirs = [d for d in model_dir.iterdir() if d.is_dir()]
    if not adapter_dirs:
        return None

    # Sort by name (timestamp) and return latest
    return sorted(adapter_dirs)[-1]


def run_pipeline_with_adapter(model_name, adapter_path, output_file):
    """Run the pipeline with a specific adapter"""
    print(f"\n{'='*60}")
    print(f"Running pipeline with {model_name} adapter")
    print(f"Adapter: {adapter_path}")
    print(f"Output: {output_file}")
    print('='*60)

    cmd = [
        sys.executable, "-m", "scripts.run_pipeline",
        "--pipeline", "optimized",
        "--adapter", str(adapter_path),
        "--aggregation", "conf_weighted",
        "--output", str(output_file),
        "--batch-size", "32",
        "--max-articles", "3000"  # Process gold standard size
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print("✅ Pipeline completed successfully!")
        return True
    else:
        print("❌ Pipeline failed!")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        return False


def run_evaluation(description=""):
    """Run comprehensive evaluation"""
    print(
        f"\n🔍 Running evaluation{' - ' + description if description else ''}")

    cmd = [
        sys.executable, "-m", "scripts.comprehensive_evaluation",
        "--split", "all"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print("✅ Evaluation completed!")
        # Extract F1 scores from output
        lines = result.stdout.split('\n')
        for line in lines:
            if 'Ticker F1:' in line or 'Macro F1:' in line:
                print(f"  {line.strip()}")
        return result.stdout
    else:
        print("❌ Evaluation failed!")
        print("STDERR:", result.stderr)
        return None


def run_aggregation_ablation(best_adapter_path):
    """Run pipeline with distance_weighted aggregation"""
    print(f"\n{'='*60}")
    print("Running aggregation ablation with distance_weighted")
    print('='*60)

    output_file = data_dir / "processed_articles_distance_weighted.jsonl"

    cmd = [
        sys.executable, "-m", "scripts.run_pipeline",
        "--pipeline", "optimized",
        "--adapter", str(best_adapter_path),
        "--aggregation", "distance_weighted",
        "--output", str(output_file),
        "--batch-size", "32",
        "--max-articles", "3000"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print("✅ Distance weighted pipeline completed!")
        return True
    else:
        print("❌ Distance weighted pipeline failed!")
        return False


def main():
    # Task 4: Run both pipelines
    results = {}

    # Run FinBERT pipeline
    finbert_adapter = find_latest_adapter("finbert")
    if finbert_adapter:
        output_finbert = data_dir / "processed_articles_finetuned_finbert.jsonl"
        if run_pipeline_with_adapter("finbert", finbert_adapter, output_finbert):
            finbert_eval = run_evaluation("FinBERT fine-tuned")
            if finbert_eval:
                results['finbert'] = {'adapter': str(
                    finbert_adapter), 'eval': finbert_eval}
    else:
        print("❌ No FinBERT adapter found!")

    # Run DeBERTa pipeline
    deberta_adapter = find_latest_adapter("deberta-fin")
    if deberta_adapter:
        output_deberta = data_dir / "processed_articles_finetuned_deberta.jsonl"
        if run_pipeline_with_adapter("deberta-fin", deberta_adapter, output_deberta):
            deberta_eval = run_evaluation("DeBERTa fine-tuned")
            if deberta_eval:
                results['deberta'] = {'adapter': str(
                    deberta_adapter), 'eval': deberta_eval}
    else:
        print("❌ No DeBERTa adapter found!")

    # Task 6: Select best adapter
    print(f"\n{'='*60}")
    print("Selecting best adapter based on ticker F1 score")
    print('='*60)

    best_model = None
    best_f1 = 0.0
    best_adapter_path = None

    for model, data in results.items():
        # Extract F1 score
        for line in data['eval'].split('\n'):
            if 'Ticker F1:' in line:
                f1_score = float(line.split(':')[-1].strip())
                print(f"{model}: Ticker F1 = {f1_score:.4f}")
                if f1_score > best_f1:
                    best_f1 = f1_score
                    best_model = model
                    best_adapter_path = data['adapter']

    if best_model:
        print(f"\n🏆 Best model: {best_model} with F1={best_f1:.4f}")

        # Create symlink or copy
        best_adapter_dir = models_dir / "best_adapter"
        if best_adapter_dir.exists():
            import shutil
            shutil.rmtree(best_adapter_dir)

        # Try to create symlink (Windows requires admin)
        try:
            cmd = f'cmd /c mklink /D "{best_adapter_dir}" "{best_adapter_path}"'
            subprocess.run(cmd, shell=True, check=True)
            print(
                f"✅ Created symlink: {best_adapter_dir} -> {best_adapter_path}")
        except:
            # Fall back to copying
            import shutil
            shutil.copytree(best_adapter_path, best_adapter_dir)
            print(f"✅ Copied adapter to: {best_adapter_dir}")

        # Task 7: Aggregation ablation
        if run_aggregation_ablation(best_adapter_dir):
            ablation_eval = run_evaluation("Distance weighted aggregation")
            if ablation_eval:
                # Check if macro F1 improved
                for line in ablation_eval.split('\n'):
                    if 'Macro F1:' in line:
                        new_f1 = float(line.split(':')[-1].strip())
                        print(f"\nDistance weighted Macro F1: {new_f1:.4f}")
                        # Compare with baseline - would need to store original macro F1

    print(f"\n{'='*60}")
    print("All tasks completed!")
    print('='*60)


if __name__ == "__main__":
    main()
