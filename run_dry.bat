@echo off
chcp 65001 >nul
cd /d E:\agent_engineer\AI_clone
set PYTHONIOENCODING=utf-8
echo Dry run: identify and draft only, no sending.
python scripts\wechat_screener.py --once
pause
