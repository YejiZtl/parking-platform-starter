@echo off
setlocal
cd /d "%~dp0"
".venv\Scripts\python.exe" training_status.py --watch 5
endlocal
