@echo off
chcp 65001 >nul
cd /d %~dp0
echo ============================================
echo  AI Clone 一键流水线
echo ============================================
python scripts\auto_pipeline.py %*
pause
