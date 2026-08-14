@echo off
REM Double-click launcher for the CR-Pipeline desktop app.
REM
REM Uses the Python already installed on this machine, so there is no build
REM step. For a self-contained executable that needs no Python at all, run:
REM     python packaging\build_exe.py --clean

setlocal
cd /d "%~dp0"

REM pythonw runs without a console window. The Windows launcher (py) is the
REM most reliable way to find it; plain pythonw on PATH is the fallback.
where pyw >nul 2>nul && (
    start "" pyw -3 "scripts\crp_gui.py" %*
    goto :eof
)
where pythonw >nul 2>nul && (
    start "" pythonw "scripts\crp_gui.py" %*
    goto :eof
)

echo Could not find pythonw on this machine.
echo Install Python 3.10+ from https://python.org, then run:
echo     python scripts\crp_gui.py
pause

endlocal
