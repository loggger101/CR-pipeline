#!/usr/bin/env python3
"""Launch the CR-Pipeline training visualization dashboard.

Usage:
    python scripts/launch_dashboard.py [--runs-dir runs]

Options:
    --runs-dir    Directory containing training runs
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.viz import run_dashboard


def main():
    parser = argparse.ArgumentParser(
        description="Launch CR-Pipeline training dashboard."
    )
    parser.add_argument("--runs-dir", type=str, default="runs",
                       help="Directory containing training runs")
    parser.add_argument("--refresh", type=int, default=5,
                       help="Dashboard refresh interval in seconds")

    args = parser.parse_args()

    print("=" * 60)
    print("CR-Pipeline Training Dashboard")
    print("=" * 60)
    print(f"Training data directory: {args.runs_dir}")
    print(f"Refresh interval: {args.refresh}s")
    print()
    print("To view the dashboard, open:")
    print("    streamlit run scripts/launch_dashboard.py")
    print()
    print("Or run directly:")
    print(f"    streamlit run --server.port 8501 scripts/launch_dashboard.py -- --runs-dir {args.runs_dir}")
    print()

    # Run the dashboard
    run_dashboard(runs_dir=args.runs_dir, refresh_interval=args.refresh)


if __name__ == "__main__":
    main()
