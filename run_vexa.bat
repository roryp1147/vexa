@echo off
rem Launch Vexa without a console window (uses the local .venv).
cd /d "%~dp0"
if exist ".venv\Scripts\pythonw.exe" (
    start "" ".venv\Scripts\pythonw.exe" main.py %*
) else (
    start "" pythonw main.py %*
)
