"""Example: Training Visualization

Demonstrates:
- Launching the Streamlit dashboard
- Generating reports
- Comparing runs
- Analyzing tournament results
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.viz import (
    run_advanced_dashboard,
    ReportGenerator,
    compute_tournament_summary,
)
from src.viz.runs_manager import RunManager


def launch_dashboard():
    """Launch the advanced Streamlit dashboard."""
    print("Launching Streamlit dashboard...")
    print("Open http://localhost:8501 in your browser")
    run_advanced_dashboard(runs_dir="runs", refresh_interval=10)


def generate_reports():
    """Generate various reports for a training run."""
    generator = ReportGenerator(output_dir="reports")

    # Generate training report
    report_path = generator.generate_training_report(
        run_dir="runs/example_basic",
        output_filename="training_report.html",
        include_charts=True,
    )
    print(f"Training report: {report_path}")

    # Generate comparison report
    comparison_path = generator.generate_comparison_report(
        run_dirs=["runs/example_basic", "runs/hpo_bayesian"],
        output_filename="comparison_report.html",
    )
    print(f"Comparison report: {comparison_path}")


def compare_runs():
    """Compare multiple training runs."""
    manager = RunManager("runs")
    runs = manager.discover_runs()

    if len(runs) < 2:
        print("Need at least 2 runs to compare")
        return

    # Get fitness data for comparison
    fitness_data = manager.load_fitness_data()

    # Compare top runs
    for run in runs[:3]:
        curves = manager.get_fitness_curves(run.run_id)
        if curves:
            print(f"\nRun: {run.name}")
            print(f"  Best fitness: {curves['best'][-1]:.4f}")
            print(f"  Mean fitness: {curves['mean'][-1]:.4f}")
            print(f"  Generations: {len(curves['best'])}")


if __name__ == "__main__":
    # Choose one:
    # launch_dashboard()
    # generate_reports()
    compare_runs()
