@echo off
rem Double-click to set up, download the datasets and train. Extra args are passed on, e.g. START_HERE.bat --no-openbmi
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_and_run.ps1" %*
echo.
pause
