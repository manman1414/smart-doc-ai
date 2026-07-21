@echo off
REM SmartDoc Python AI 一键启动（无需手动 activate）
REM 作者: Cursor Agent / 2026-06-24
cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
  echo [错误] 未找到 venv，请先执行:
  echo   python -m venv venv
  echo   venv\Scripts\pip install -r requirements.txt
  exit /b 1
)

echo Starting SmartDoc AI on http://127.0.0.1:8000 ...
venv\Scripts\python.exe -m uvicorn main:app --host 0.0.0.0 --port 8000
