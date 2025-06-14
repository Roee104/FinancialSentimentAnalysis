# src/run_experiments.py

"""
Run all experiments for the interim report:
1. Original pipeline (baseline)
2. VADER baseline
3. Enhanced pipeline with different thresholds
4. Compare all results
"""

import os
import subprocess
import time


def run_command(cmd: str, description: str):
    """Run a command and track timing"""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}")
    print(f"Command: {cmd}")
    
    start_time = time.time()
    try:
        subprocess.run(cmd, shell=True, check=True)
        elapsed = time.time() - start_time
        print(f"✅ Completed in {elapsed:.1f} seconds")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed with error: {e}")
        return False


def main():
    """Run all experiments"""
    
    print("🔬 RUNNING ALL EXPERIMENTS FOR INTERIM REPORT")
    print("=" * 60)
    
    # Check if data exists
    if not os.path.exists("data/financial_news_2020_2025_100k.parquet"):
        print("❌ ERROR: Input data not found!")
        print("Please run: python src/data_loader.py")
        return
    
    experiments = [
        # 1. Run original pipeline (baseline)
        (
            "python src/pipeline.py",
            "Running ORIGINAL pipeline (baseline)"
        ),
        
        # 2. Run VADER baseline
        (
            "python src/vader_baseline.py --threshold 0.05",
            "Running VADER baseline"
        ),
        
        # 3. Run enhanced pipeline with default settings
        (
            "python src/pipeline_updated.py --output data/processed_articles_enhanced.jsonl",
            "Running ENHANCED pipeline (conf_weighted, threshold=0.1)"
        ),
        
        # 4. Run enhanced pipeline with different threshold
        (
            "python src/pipeline_updated.py --threshold 0.15 --output data/processed_articles_enhanced_t15.jsonl",
            "Running ENHANCED pipeline (conf_weighted, threshold=0.15)"
        ),
        
        # 5. Run enhanced pipeline with majority method
        (
            "python src/pipeline_updated.py --method majority --output data/processed_articles_majority.jsonl",
            "Running ENHANCED pipeline (majority method)"
        ),
    ]
    
    # Run experiments
    successful = 0
    failed = 0
    
    for cmd, description in experiments:
        if run_command(cmd, description):
            successful += 1
        else:
            failed += 1
            print("⚠️  Continuing with next experiment...")
    
    # Run comparison
    print(f"\n{'='*60}")
    print("📊 Running comparison analysis")
    print(f"{'='*60}")
    
    # Update compare script to include all files
    comparison_cmd = "python src/compare_results.py"
    run_command(comparison_cmd, "Comparing all results")
    
    # Summary
    print(f"\n{'='*60}")
    print("📋 EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    print(f"Total experiments: {len(experiments)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    print("\n📁 Output files generated:")
    output_files = [
        "data/processed_articles.jsonl",
        "data/vader_baseline_results.jsonl", 
        "data/processed_articles_enhanced.jsonl",
        "data/processed_articles_enhanced_t15.jsonl",
        "data/processed_articles_majority.jsonl",
        "data/comparison_report.txt",
        "data/plots/"
    ]
    
    for file in output_files:
        if os.path.exists(file):
            print(f"  ✅ {file}")
        else:
            print(f"  ❌ {file} (not found)")
    
    print("\n🎯 Next steps:")
    print("1. Review comparison_report.txt for insights")
    print("2. Check plots/ directory for visualizations")
    print("3. Use these results for your interim presentation")
    
    print("\n✨ All experiments completed!")


if __name__ == "__main__":
    main()