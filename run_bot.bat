@echo off
chcp 65001 >nul
cd /d E:\agent_engineer\AI_clone
set PYTHONIOENCODING=utf-8

python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:11434/api/tags',timeout=3)" >nul 2>nul
if errorlevel 1 (
  echo [WARN] Ollama is not running. Please start Ollama first.
  pause
  exit /b 1
)

echo Personal clone - WeChat auto reply started.
echo Config: bot_config.json  ^(target / interval / cooldown / live^)
echo Stop: run stop_bot.bat, or press Ctrl+C
echo.

python scripts\wechat_screener.py
pause
