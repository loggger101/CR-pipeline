#!/usr/bin/env python3
"""Build CR-Pipeline.exe from the desktop app.

    python packaging/build_exe.py

Produces ``dist/CR-Pipeline/CR-Pipeline.exe``. Double-click it, or make a
shortcut to it. Training runs are written to a ``runs`` folder next to the
executable.

The build is large (PyTorch alone is most of it) and takes several minutes.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPEC = PROJECT_ROOT / "packaging" / "crp_gui.spec"


def _is_synced_folder(path: Path) -> bool:
    """Whether ``path`` sits inside a cloud-sync folder.

    A multi-gigabyte bundle written into OneDrive/Dropbox gets uploaded and,
    worse, locked mid-build -- which shows up as a bewildering
    "PermissionError: the process cannot access the file" partway through.
    """
    markers = ("onedrive", "dropbox", "google drive", "icloud")
    text = str(path).lower()
    return any(marker in text for marker in markers)


def default_output_root() -> Path:
    """Where to write build artefacts.

    Defaults to the project, but steps outside it when the project lives in a
    synced folder.
    """
    if not _is_synced_folder(PROJECT_ROOT):
        return PROJECT_ROOT
    base = os.environ.get("LOCALAPPDATA") or tempfile.gettempdir()
    return Path(base) / "CR-Pipeline-build"


def ensure_pyinstaller() -> bool:
    try:
        import PyInstaller  # noqa: F401
        return True
    except ImportError:
        print("PyInstaller is not installed. Install it with:\n"
              "    pip install pyinstaller\n")
        return False


def warn_about_cuda() -> None:
    """Point out the single biggest contributor to bundle size.

    The desktop app only uses torch to read and write checkpoints -- the
    evolved policy itself is pure NumPy -- so a CUDA build contributes several
    gigabytes of libraries the app never calls.
    """
    try:
        import torch
    except ImportError:
        return

    cuda_version = getattr(torch.version, "cuda", None)
    if not cuda_version:
        return

    print(f"\nNote: this environment has a CUDA build of torch "
          f"(cuda {cuda_version}).")
    print("That adds roughly 3 GB of GPU libraries to the bundle, and the app")
    print("only uses torch to load and save checkpoints. For a much smaller")
    print("executable, build in a venv with the CPU-only wheel:")
    print("    pip install torch --index-url "
          "https://download.pytorch.org/whl/cpu\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean", action="store_true",
                        help="Remove previous build artefacts first")
    parser.add_argument("--output", type=str, default=None,
                        help="Where to write build artefacts "
                             "(default: outside any cloud-sync folder)")
    args = parser.parse_args()

    if not ensure_pyinstaller():
        return 1

    warn_about_cuda()

    output_root = Path(args.output) if args.output else default_output_root()
    dist_path = output_root / "dist"
    work_path = output_root / "build"
    bundle = dist_path / "CR-Pipeline"

    if output_root != PROJECT_ROOT:
        print(f"Building outside the project directory ({PROJECT_ROOT} looks "
              f"like a cloud-synced folder).")
    print(f"Output: {bundle}\n")

    if args.clean:
        for target in (dist_path, work_path):
            if target.exists():
                print(f"removing {target}")
                shutil.rmtree(target, ignore_errors=True)
                if target.exists():
                    print(f"  could not fully remove {target}; a running copy "
                          f"of the app or a sync client may be holding files",
                          file=sys.stderr)

    command = [
        sys.executable, "-m", "PyInstaller", str(SPEC), "--noconfirm",
        "--distpath", str(dist_path),
        "--workpath", str(work_path),
    ]
    print("running:", " ".join(command))
    result = subprocess.run(command, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        print("\nBuild failed.", file=sys.stderr)
        return result.returncode

    executable = bundle / "CR-Pipeline.exe"
    if not executable.exists():
        print("\nBuild reported success but the executable is missing.",
              file=sys.stderr)
        return 1

    size_mb = sum(f.stat().st_size for f in bundle.rglob("*") if f.is_file())
    size_mb /= 1024 * 1024
    print(f"\nBuilt {executable}")
    print(f"Bundle size: {size_mb:.0f} MB")
    print("\nDouble-click the executable to open the app. Runs are saved to a "
          "'runs' folder beside it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
