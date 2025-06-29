"""Update plots after LoRA fine-tuning"""

from analysis.visualization import create_final_summary, plot_reliability_diagram
import sys
from pathlib import Path

# Add project root to path
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))


def main():
    """Generate updated plots after LoRA fine-tuning"""

    print("📊 Generating updated visualizations...")

    # Create output directory
    plots_dir = root / "data" / "plots"
    plots_dir.mkdir(exist_ok=True, parents=True)

    # Generate final summary plots
    print("\n1. Creating final summary plots...")
    try:
        # This should create comparison plots for all pipelines
        create_final_summary()
        print("✅ Final summary plots created")
    except Exception as e:
        print(f"❌ Error creating summary plots: {e}")

    # Generate reliability diagram
    print("\n2. Creating reliability diagram...")
    try:
        # This should create calibration/reliability plots
        plot_reliability_diagram()
        print("✅ Reliability diagram created")
    except Exception as e:
        print(f"❌ Error creating reliability diagram: {e}")

    # List generated plots
    print("\n📁 Generated plots:")
    for plot_file in plots_dir.glob("*.png"):
        print(f"  - {plot_file.name}")

    print("\n✅ All visualizations updated!")


if __name__ == "__main__":
    main()
