# PyInstaller spec for the CR-Pipeline desktop app.
#
# Build with:  python packaging/build_exe.py
#
# Things this spec has to get right:
#
#  * The entry point calls multiprocessing.freeze_support() before anything
#    else. Without it, every worker the training pool spawns re-executes the
#    .exe from the top and opens another window, without end.
#
#  * PyInstaller ships its own torch hook, which knows which parts of torch are
#    needed. Calling collect_submodules("torch") on top of it asks for hundreds
#    of modules that no longer exist in current torch (the deprecated
#    torch.distributed._sharded_tensor tree, among others) and buries real
#    problems in a wall of "Hidden import not found" errors. Leave torch to the
#    hook.
#
#  * Only the simulation and model packages are needed at runtime; the live
#    game capture modules pull in mss/pyautogui/opencv, which the desktop app
#    never touches.
#
#  * onedir, not onefile: a single-file build unpacks the whole bundle to a temp
#    directory on every launch, which for a torch-sized app costs many seconds
#    each time and repeats for every spawned worker.

import os
from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules

PROJECT_ROOT = Path(SPECPATH).parent

# A windowed build reports startup failures in a message box, which cannot be
# captured. Set CRP_CONSOLE=1 to build a console variant that prints instead.
SHOW_CONSOLE = os.environ.get("CRP_CONSOLE", "") == "1"

# Worker processes import the simulation by name, so it is collected wholesale.
hidden = collect_submodules("src.env.sim")
hidden += collect_submodules("src.models")
hidden += [
    "src.train.trainer",
    "src.train.evaluator",
    "src.ui.app",
    "src.ui.operations",
    "src.ui.jobs",
    "src.ui.arena_canvas",
    "src.ui.chart",
    # Imported by name by matplotlib, so invisible to static analysis.
    "matplotlib.backends.backend_tkagg",
    "matplotlib.backends.backend_agg",
]

a = Analysis(
    [str(PROJECT_ROOT / "scripts" / "crp_gui.py")],
    pathex=[str(PROJECT_ROOT)],
    binaries=[],
    datas=[
        (str(PROJECT_ROOT / "assets"), "assets"),
        (str(PROJECT_ROOT / "configs"), "configs"),
    ],
    hiddenimports=hidden,
    hookspath=[],
    runtime_hooks=[],
    excludes=[
        # Large packages that happen to be installed but are unreachable from
        # the desktop app. TensorFlow in particular gets dragged in by a hook
        # and contributes gigabytes for nothing.
        "tensorflow", "keras", "tensorboard", "jax",
        "streamlit", "plotly", "altair", "pandas",
        "pytest", "black", "flake8", "isort", "IPython", "notebook",
        # Live-game capture; the desktop app does not use it.
        "pyautogui", "mss", "cv2", "onnx", "onnxruntime",
        "src.env.live",
    ],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="CR-Pipeline",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=SHOW_CONSOLE,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="CR-Pipeline",
)
