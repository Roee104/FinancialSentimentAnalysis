# colab_runner.py
"""
Google Colab optimized runner for the financial sentiment pipeline.
Run this in Colab for best performance.
"""

import os
import sys
import subprocess
import gc
import torch


def setup_colab_environment():
    """Setup Colab environment with proper memory management"""

    print("🔧 Setting up Colab environment...")

    # Check if we're in Colab
    try:
        import google.colab
        IN_COLAB = True
        print("✅ Running in Google Colab")
    except:
        IN_COLAB = False
        print("⚠️  Not running in Colab")

    # Clear GPU cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
    else:
        print("ℹ️  No GPU available, using CPU")

    # Mount Google Drive if in Colab
    if IN_COLAB:
        from google.colab import drive
        drive.mount('/content/drive')
        print("✅ Google Drive mounted")

    # Set memory-efficient environment variables
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    # Garbage collection
    gc.collect()

    return IN_COLAB


def install_requirements():
    """Install required packages efficiently"""

    print("\n📦 Installing requirements...")

    # Essential packages only (skip what Colab already has)
    packages = [
        'transformers==4.36.0',
        'vaderSentiment',
        'python-dotenv',
        'tqdm',
        'yfinance'
    ]

    for package in packages:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", package])

    print("✅ Requirements installed")


def run_pipeline_safely(script_path, args="", max_retries=3):
    """Run a pipeline script with error handling and retries"""

    for attempt in range(max_retries):
        try:
            print(f"\n🚀 Running: python {script_path} {args}")
            print(f"   Attempt {attempt + 1}/{max_retries}")

            # Run with subprocess for better memory management
            result = subprocess.run(
                f"python {script_path} {args}",
                shell=True,
                capture_output=True,
                text=True
            )

            if result.returncode == 0:
                print("✅ Completed successfully")
                return True
            else:
                print(f"❌ Error: {result.stderr}")

                # Clear memory before retry
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        except Exception as e:
            print(f"❌ Exception: {e}")

        if attempt < max_retries - 1:
            print("⏳ Retrying after cleanup...")
            gc.collect()

    return False


def main():
    """Main runner for Colab"""

    print("🎯 Financial Sentiment Analysis Pipeline - Colab Runner")
    print("="*60)

    # Setup environment
    IN_COLAB = setup_colab_environment()

    # Install requirements
    install_requirements()

    # Import required modules
    print("\n📚 Importing modules...")
    import nltk
    nltk.download('punkt', quiet=True)

    # Check data file
    data_file = "data/financial_news_2020_2025_100k.parquet"
    if not os.path.exists(data_file):
        print(f"❌ Data file not found: {data_file}")
        print("Please upload your data file to Colab or mount from Drive")
        return

    print(f"✅ Data file found: {data_file}")

    # Get file size
    file_size_mb = os.path.getsize(data_file) / (1024 * 1024)
    print(f"   File size: {file_size_mb:.1f} MB")

    # Determine optimal batch size based on environment
    if IN_COLAB:
        if torch.cuda.is_available():
            batch_size = 100  # GPU can handle more
            sentiment_batch = 16
        else:
            batch_size = 50   # CPU needs smaller batches
            sentiment_batch = 8
    else:
        batch_size = 100
        sentiment_batch = 16

    print(f"\n⚙️  Optimal settings detected:")
    print(f"   Pipeline batch size: {batch_size}")
    print(f"   Sentiment batch size: {sentiment_batch}")

    # Run experiments
    experiments = [
        # 1. Original pipeline (optimized version)
        ("src/pipeline_optimized.py",
         f"--batch-size {batch_size} --max-articles 5000",
         "Original Pipeline (Baseline)"),

        # 2. VADER baseline
        ("src/vader_baseline.py",
         "--threshold 0.05",
         "VADER Baseline"),

        # 3. Enhanced pipeline
        ("src/pipeline_updated.py",
         f"--output data/processed_articles_enhanced.jsonl",
         "Enhanced Pipeline"),
    ]

    results = []

    for script, args, description in experiments:
        print(f"\n{'='*60}")
        print(f"📊 {description}")
        print(f"{'='*60}")

        success = run_pipeline_safely(script, args)
        results.append((description, success))

        # Clean up after each experiment
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Run comparison
    print(f"\n{'='*60}")
    print("📈 Running comparison analysis")
    print(f"{'='*60}")

    run_pipeline_safely("src/compare_results.py", "")

    # Summary
    print(f"\n{'='*60}")
    print("📋 SUMMARY")
    print(f"{'='*60}")

    for description, success in results:
        status = "✅ Success" if success else "❌ Failed"
        print(f"{description}: {status}")

    print("\n📁 Check these outputs:")
    print("  - data/processed_articles.jsonl")
    print("  - data/comparison_report.txt")
    print("  - data/plots/")

    if IN_COLAB:
        print("\n💡 Tip: Download results to your local machine:")
        print("  from google.colab import files")
        print("  files.download('data/comparison_report.txt')")
        print("  files.download('data/plots/sentiment_distribution_comparison.png')")

    print("\n✨ Pipeline completed!")


if __name__ == "__main__":
    # Run with error handling
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Final cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
