@echo off
title Eye Control Mouse
cd /d "%~dp0"
echo ===================================================
echo Starting Real-Time Eye Control Mouse Controller...
echo ===================================================
python eye_tracking_mouse.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Running with virtual environment Python fallback...
    d:\aaaassistan_pcb\.venv\Scripts\python.exe eye_tracking_mouse.py
    if %ERRORLEVEL% NEQ 0 pause
)
