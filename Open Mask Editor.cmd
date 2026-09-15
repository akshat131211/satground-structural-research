@echo off
cd /d "%~dp0"
".venv\python.exe" -u scripts\edit_masks.py --open-browser
if errorlevel 1 pause
