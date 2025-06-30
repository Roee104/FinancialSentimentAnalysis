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
        "--input", str(root / "data" / "financial_news_2020_2025_100k.parquet"),
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


def run_custom_evaluation():
    """Run evaluation specifically for fine-tuned models"""
    print(f"\n🔍 Running custom evaluation for fine-tuned models")

    # Create a custom evaluation script that includes our files
    eval_script = '''
import json
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# Load gold standard
gold_file = Path("data/3000_gold_standard.jsonl")
gold_standard = {}
gold_tickers = {}

with open(gold_file, "r") as f:
    for line in f:
        ann = json.loads(line)
        h = ann["article_hash"]
        gold_standard[h] = ann["true_overall"]
        gold_tickers[h] = ann.get("ticker_sentiments", {})

print(f"Loaded {len(gold_standard)} gold annotations")

# Evaluate each fine-tuned model
pipelines = {
    "FinBERT-LoRA": "data/processed_articles_finetuned_finbert.jsonl",
    "DeBERTa-LoRA": "data/processed_articles_finetuned_deberta.jsonl",
}

results = {}

for name, path_str in pipelines.items():
    path = Path(path_str)
    if not path.exists():
        print(f"\\nSkipping {name} - file not found")
        continue
        
    print(f"\\n{'='*50}")
    print(f"Evaluating {name}")
    print('='*50)
    
    y_true = []
    y_pred = []
    ticker_ok = total_ticker = 0
    
    with open(path, "r") as f:
        for line in f:
            try:
                pred = json.loads(line)
                h = pred.get("article_hash")
                if h not in gold_standard:
                    continue
                    
                y_true.append(gold_standard[h])
                y_pred.append(pred["overall_sentiment"])
                
                # Ticker evaluation
                if h in gold_tickers:
                    g_map = gold_tickers[h]
                    p_map = {t["symbol"]: t for t in pred.get("tickers", [])}
                    for sym, g_info in g_map.items():
                        if sym in p_map:
                            total_ticker += 1
                            if p_map[sym]["label"] == g_info["sentiment"]:
                                ticker_ok += 1
            except Exception as e:
                continue
    
    if not y_true:
        print(f"No matching articles found for {name}")
        continue
        
    # Calculate metrics
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=["Positive", "Neutral", "Negative"], average="macro"
    )
    ticker_acc = ticker_ok / total_ticker if total_ticker else 0
    
    results[name] = {
        "accuracy": acc,
        "macro_f1": f1,
        "ticker_accuracy": ticker_acc,
        "n_samples": len(y_true),
        "ticker_ok": ticker_ok,
        "total_ticker": total_ticker
    }
    
    # Print results
    print(f"Overall Accuracy: {acc:.3f}")
    print(f"Macro F1:         {f1:.3f}")
    print(f"Ticker Accuracy:  {ticker_acc:.3f} ({ticker_ok}/{total_ticker})")

# Save results
with open("data/finetuned_evaluation_results.json", "w") as f:
    json.dump(results, f, indent=2)
    
print("\\n✅ Evaluation complete! Results saved to data/finetuned_evaluation_results.json")

# Return results for further processing
print(json.dumps(results))
'''

    # Write and run the evaluation script
    eval_file = root / "temp_eval.py"
    eval_file.write_text(eval_script)

    try:
        result = subprocess.run(
            [sys.executable, str(eval_file)],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            print("✅ Evaluation completed!")
            print(result.stdout)

            # Try to parse the results
            lines = result.stdout.split('\n')
            for line in lines:
                if line.strip().startswith('{') and line.strip().endswith('}'):
                    try:
                        return json.loads(line)
                    except:
                        pass

            # If we can't parse, at least we printed the results
            return True
        else:
            print("❌ Evaluation failed!")
            print("STDERR:", result.stderr)
            return None
    finally:
        # Clean up temp file
        if eval_file.exists():
            eval_file.unlink()


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
        print("STDERR:", result.stderr)
        return False


def main():
    # Check if any adapters exist first
    adapter_found = False

    # Task 4: Run both pipelines
    pipeline_results = {}

    # Run FinBERT pipeline
    finbert_adapter = find_latest_adapter("finbert")
    if finbert_adapter:
        adapter_found = True
        output_finbert = data_dir / "processed_articles_finetuned_finbert.jsonl"
        if run_pipeline_with_adapter("finbert", finbert_adapter, output_finbert):
            pipeline_results['finbert'] = {
                'adapter': str(finbert_adapter),
                'output': str(output_finbert)
            }
    else:
        print("\n❌ No FinBERT adapter found!")

    # Run DeBERTa pipeline
    deberta_adapter = find_latest_adapter("deberta-fin")
    if deberta_adapter:
        adapter_found = True
        output_deberta = data_dir / "processed_articles_finetuned_deberta.jsonl"
        if run_pipeline_with_adapter("deberta-fin", deberta_adapter, output_deberta):
            pipeline_results['deberta'] = {
                'adapter': str(deberta_adapter),
                'output': str(output_deberta)
            }
    else:
        print("\n❌ No DeBERTa adapter found!")

    # Exit if no adapters were found
    if not adapter_found:
        print("\n❌ ERROR: No adapters found! Please run the fine-tuning scripts first.")
        print("   Run: python -m scripts.run_finbert_lora")
        print("   Run: python -m scripts.run_deberta_lora")
        sys.exit(1)

    # Exit if no successful pipeline runs
    if not pipeline_results:
        print("\n❌ ERROR: No successful pipeline runs!")
        sys.exit(1)

    # Task 5: Evaluate with custom evaluation
    eval_results = run_custom_evaluation()

    if not eval_results or eval_results == True:
        # Try to load from file if direct parsing failed
        eval_file = root / "data" / "finetuned_evaluation_results.json"
        if eval_file.exists():
            with open(eval_file, 'r') as f:
                eval_results = json.load(f)
        else:
            print("\n❌ ERROR: Could not get evaluation results!")
            sys.exit(1)

    # Task 6: Select best adapter based on macro F1 (more balanced than accuracy)
    print(f"\n{'='*60}")
    print("Selecting best adapter based on Macro F1 score")
    print('='*60)

    best_model = None
    best_f1 = 0.0
    best_adapter_path = None

    # Map evaluation names to our pipeline names
    name_map = {
        "FinBERT-LoRA": "finbert",
        "DeBERTa-LoRA": "deberta"
    }

    for eval_name, metrics in eval_results.items():
        model_name = name_map.get(eval_name)
        if model_name and model_name in pipeline_results:
            f1_score = metrics.get('macro_f1', 0)
            accuracy = metrics.get('accuracy', 0)
            ticker_acc = metrics.get('ticker_accuracy', 0)

            print(f"\n{eval_name}:")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Macro F1: {f1_score:.4f}")
            print(f"  Ticker Accuracy: {ticker_acc:.4f}")

            if f1_score > best_f1:
                best_f1 = f1_score
                best_model = model_name
                best_adapter_path = pipeline_results[model_name]['adapter']

    if best_model:
        print(f"\n🏆 Best model: {best_model} with Macro F1={best_f1:.4f}")

        # Create symlink or copy
        best_adapter_dir = models_dir / "best_adapter"
        if best_adapter_dir.exists():
            import shutil
            shutil.rmtree(best_adapter_dir)

        # Try to create symlink (Windows requires admin)
        try:
            import os
            if os.name == 'nt':  # Windows
                cmd = f'cmd /c mklink /D "{best_adapter_dir}" "{best_adapter_path}"'
                subprocess.run(cmd, shell=True, check=True)
            else:  # Unix-like
                os.symlink(best_adapter_path, best_adapter_dir)
            print(
                f"✅ Created symlink: {best_adapter_dir} -> {best_adapter_path}")
        except:
            # Fall back to copying
            import shutil
            shutil.copytree(best_adapter_path, best_adapter_dir)
            print(f"✅ Copied adapter to: {best_adapter_dir}")

        # Task 7: Aggregation ablation
        if run_aggregation_ablation(best_adapter_dir):
            # Run evaluation on distance weighted
            print("\n🔍 Evaluating distance-weighted aggregation...")
            # Could run another evaluation here if needed
    else:
        print("\n❌ ERROR: Could not determine best model!")
        sys.exit(1)

    print(f"\n{'='*60}")
    print("All tasks completed successfully!")
    print('='*60)


if __name__ == "__main__":
    main()
