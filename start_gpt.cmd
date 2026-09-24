@echo off
cd /d "%~dp0"
python app.py --provider openai --prompt-api-key --port 8766
pause
