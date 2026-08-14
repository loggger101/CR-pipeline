#!/usr/bin/env python3
"""Launch the CR-Pipeline desktop application.

    python scripts/crp_gui.py [--runs-dir PATH]

Also the entry point PyInstaller freezes into CR-Pipeline.exe.
"""

from __future__ import annotations

import argparse
import multiprocessing
import sys
from pathlib import Path


def _project_root() -> Path:
    """Where runs/ and configs/ live.

    A frozen build runs from a temp extraction directory, so the sources'
    location is meaningless there; use the folder holding the .exe instead so
    runs land somewhere the user can find.
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent.parent


def main() -> int:
    # Must come before anything can spawn a worker. Without it, each pool
    # worker of a frozen build re-executes the .exe from the top and opens
    # another window -- forever.
    multiprocessing.freeze_support()

    root = _project_root()
    if not getattr(sys, "frozen", False):
        sys.path.insert(0, str(root))

    parser = argparse.ArgumentParser(description="CR-Pipeline desktop app")
    parser.add_argument("--runs-dir", type=str, default=None,
                        help="Directory holding training runs "
                             "(default: <project>/runs)")
    args = parser.parse_args()

    try:
        from src.ui import launch
    except ImportError as exc:  # pragma: no cover - startup diagnostics
        print(f"Could not load the UI: {exc}", file=sys.stderr)
        print("Install the dependencies first:  pip install -r requirements.txt",
              file=sys.stderr)
        return 1

    runs_dir = str(Path(args.runs_dir).resolve()) if args.runs_dir else None
    launch(str(root), runs_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
