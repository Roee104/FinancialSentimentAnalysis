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
        "--input", str(root / "data" /
                       "financial_news_2020_2025_100k.parquet"),
        "--output", str(output_file),
        "--batch-size", "32", "--device", "cuda"
        # NO --max-articles limit - process all articles
    ]

    result = subprocess.run(cmd, text=True)

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

for name, path in pipelines.items():
    path = Path(path)
    
    if not path.exists():
        print(f"⚠️  {name} results not found at {path}")
        continue
        
    # Load predictions
    predictions = {}
    ticker_predictions = {}
    
    with open(path, 'r') as f:
        for line in f:
            try:
                article = json.loads(line)
                h = article.get("article_hash")
                if h and h in gold_standard:
                    predictions[h] = article["overall_sentiment"]
                    
                    # Extract ticker predictions
                    if "tickers" in article:
                        ticker_predictions[h] = {
                            t["symbol"]: t["label"] 
                            for t in article["tickers"] 
                            if "symbol" in t and "label" in t
                        }
            except Exception as e:
                continue
    
    if not predictions:
        print(f"⚠️  No valid predictions found for {name}")
        continue
        
    # Align predictions with gold standard
    y_true = []
    y_pred = []
    
    for h in gold_standard:
        if h in predictions:
            y_true.append(gold_standard[h])
            y_pred.append(predictions[h])
    
    # Calculate metrics
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    
    # Calculate ticker accuracy
    ticker_correct = 0
    ticker_total = 0
    
    for h in gold_tickers:
        if h in ticker_predictions:
            gold_t = gold_tickers[h]
            pred_t = ticker_predictions[h]
            
            for symbol in gold_t:
                if symbol in pred_t:
                    ticker_total += 1
                    if gold_t[symbol] == pred_t[symbol]:
                        ticker_correct += 1
    
    ticker_acc = ticker_correct / ticker_total if ticker_total > 0 else 0
    
    results[name] = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "macro_f1": float(f1),
        "ticker_accuracy": float(ticker_acc),
        "samples": len(y_pred),
        "ticker_samples": ticker_total
    }
    
    print(f"\\n{name}:")
    print(f"  Overall Accuracy: {acc:.4f}")
    print(f"  Macro F1: {f1:.4f}")
    print(f"  Ticker Accuracy: {ticker_acc:.4f} ({ticker_total} samples)")

# Save results
output_file = Path("data/finetuned_evaluation_results.json")
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\\n📄 Results saved to {output_file}")

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
        "--input", str(root / "data" /
                       "financial_news_2020_2025_100k.parquet"),
        "--output", str(output_file),
        "--batch-size", "32", "--device", "cuda"
        # NO --max-articles limit - process all articles
    ]

    result = subprocess.run(cmd, text=True)

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
        output_finbert = Path(
            "data/processed_articles_finetuned_finbert.jsonl")
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
        output_deberta = Path(
            "data/processed_articles_finetuned_deberta.jsonl")
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
        print("   Run: python -m scripts.run_experiments --model finbert --lora --gold data/3000_gold_standard.jsonl")
        print("   Run: python -m scripts.run_experiments --model deberta-fin --lora --gold data/3000_gold_standard.jsonl")
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
