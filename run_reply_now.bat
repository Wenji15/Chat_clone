@echo off
chcp 65001 >nul
cd /d E:\agent_engineer\AI_clone
set PYTHONIOENCODING=utf-8
echo Reply once to the current last incoming message (LIVE).
python scripts\wechat_screener.py --once --live
pause
