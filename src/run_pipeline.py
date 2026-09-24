"""
Main Pipeline Runner -- executes all phases sequentially.
Run this single script to execute the entire dark patterns analysis pipeline.
"""

import sys
import os
import io

# Force UTF-8 output on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_phase(name, func):
    """Run a pipeline phase with error handling."""
    print(f"\n{'+'+ '-'*58 + '+'}")
    print(f"|  PHASE: {name:<49}|")
    print(f"{'+'+ '-'*58 + '+'}\n")
    try:
        result = func()
        print(f"\n[OK] {name} -- COMPLETE\n")
        return result
    except Exception as e:
        print(f"\n[FAIL] {name} -- FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return None


def main():
    print("=" * 60)
    print("  E-COMMERCE DARK PATTERNS DETECTION PIPELINE")
    print("  Full end-to-end analysis")
    print("=" * 60)

    # Phase 1 & 2: Data Collection
    from scrape_all import main as scrape_main
    run_phase("Data Collection (Scraping)", scrape_main)

    # Phase 3: Preprocessing
    from preprocessing import main as preprocess_main
    run_phase("Data Preprocessing & Feature Engineering", preprocess_main)

    # Phase 4: Labeling
    from labeling import main as label_main
    run_phase("Semi-Automated Labeling", label_main)

    # Phase 5: EDA
    from eda import main as eda_main
    run_phase("Exploratory Data Analysis", eda_main)

    # Phase 6: Model Training
    from model import main as model_main
    run_phase("Model Training & Evaluation", model_main)

    # Phase 7: Report
    from generate_report import generate_report
    run_phase("Report Generation", generate_report)

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE -- ALL PHASES FINISHED")
    print("=" * 60)
    print(f"\nOutputs:")
    print(f"  Charts:     src/output/charts/")
    print(f"  Model:      src/output/model_results/")
    print(f"  Report:     src/REPORT.md")
    print(f"  Dashboard:  src/dashboard.html")
    print(f"  Data:       data/")


if __name__ == "__main__":
    main()
