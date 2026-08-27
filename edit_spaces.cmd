@echo off
setlocal
cd /d "%~dp0"
".venv\Scripts\python.exe" select_spaces.py
if errorlevel 1 pause
endlocal
