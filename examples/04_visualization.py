"""Example: Training Visualization & Reporting

Demonstrates visualization and reporting using the CURRENT public API:

* Launching the advanced Streamlit dashboard (``src.viz.dashboard``, 8 tabs)
* Generating HTML/Markdown reports with ``ReportGenerator``
* Comparing runs programmatically with ``RunManager.compare_runs``

Note: the dashboard is a long-running web server, so it is *not* called from
the ``__main__`` block by default -- pick whichever you want to run.

    python examples/04_visualization.py            # reports + comparison (no browser)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.viz.dashboard import run_advanced_dashboard   # long-running Streamlit server
from src.viz.reports import ReportGenerator
from src.viz.runs_manager import RunManager


def launch_dashboard(runs_dir: str = "runs"):
    """Launch the advanced Streamlit dashboard (blocks until stopped).

    Open http://localhost:8501 in your browser. Tabs cover fitness, statistics,
    tournament results, run comparison, configuration and monitoring.
    """
    print("Launching Streamlit dashboard... open http://localhost:8501")
    run_advanced_dashboard(runs_dir=runs_dir, refresh_interval=10)


def generate_reports(run_id: str = "example_basic"):
    """Generate training + comparison reports for a run."""
    generator = ReportGenerator(output_dir="reports")

    report_path = generator.generate_training_report(
        run_dir=f"runs/{run_id}",
        output_filename="training_report.html",
        include_charts=True,
    )
    print(f"Training report: {report_path}")


def compare_runs():
    """Compare the most recent runs programmatically."""
    manager = RunManager("runs")
    runs = manager.discover_runs()

    if len(runs) < 2:
        print(f"Only {len(runs)} run(s) found under runs/ -- need at least 2 to compare.")
        return None

    # Compare the two best by peak fitness.
    top_two = [r.run_id for r in sorted(runs, key=lambda r: r.best_fitness or 0)[-2:]]
    comparison = manager.compare_runs(top_two, metric="best")

    print(f"\nComparing runs (metric=best): {top_two}")
    for entry in comparison.get("runs", []):
        print(f"  {entry['id']}: best={entry['best_value']:.3f} "
              f"final={entry['final_value']:.3f} over {entry['length']} generations")

    return comparison


if __name__ == "__main__":
    # Pick one:
    # launch_dashboard()          # interactive Streamlit server (blocks)
    generate_reports()            # needs a run under runs/example_basic first
    compare_runs()                # works on whatever is already in runs/
