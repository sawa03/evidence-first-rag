@echo off
cd /d "%~dp0"
python app.py --provider openai --port 8766
pause
