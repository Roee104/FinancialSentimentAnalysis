# archive/gold_standard/setup_gold_standard.py
"""
Setup script for gold standard generation
"""

import os
import sys
from pathlib import Path
import subprocess


def setup_openai_key():
    """Setup OpenAI API key"""
    print("=== OpenAI API Key Setup ===")

    # Check if already set
    if os.getenv('OPENAI_API_KEY'):
        print("✅ OpenAI API key already set in environment")
        return True

    # Prompt for key
    print("\nPlease enter your OpenAI API key:")
    print("(It will be stored in .env file)")
    api_key = input("API Key: ").strip()

    if not api_key:
        print("❌ No API key provided")
        return False

    # Save to .env file
    env_file = Path('.env')
    with open(env_file, 'a') as f:
        f.write(f"\nOPENAI_API_KEY={api_key}\n")

    # Also set for current session
    os.environ['OPENAI_API_KEY'] = api_key

    print("✅ API key saved to .env file")
    return True


def check_dependencies():
    """Check if required packages are installed"""
    try:
        import openai
        print("✅ OpenAI package installed")
        return True
    except ImportError:
        print("❌ OpenAI package not installed")
        print("Installing openai...")
        subprocess.run([sys.executable, "-m", "pip", "install", "openai"])
        return True


def main():
    """Main setup function"""
    print("="*60)
    print("GOLD STANDARD GENERATION SETUP")
    print("="*60)

    # Check dependencies
    if not check_dependencies():
        return

    # Setup API key
    if not setup_openai_key():
        return

    print("\n=== Ready to Generate Gold Standard ===")
    print("\nRun with default settings (300 articles, $30 limit):")
    print("  python scripts/create_gold_standard.py")

    print("\nRun with custom settings:")
    print("  python scripts/create_gold_standard.py --n-samples 500 --max-cost 30")

    print("\nRun with consensus (3x cost but higher quality):")
    print("  python scripts/create_gold_standard.py --consensus --n-samples 100")

    print("\nTo analyze results after generation:")
    print("  python scripts/analyze_gold_standard.py")


if __name__ == "__main__":
    main()
