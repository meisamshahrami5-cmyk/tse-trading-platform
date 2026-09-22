@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo Starting CryptoAI Pro ...
python server.py
pause
