@echo off
REM 开发模式：代码变更自动重载
REM 作者: Cursor Agent / 2026-06-24
cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
  echo [错误] 未找到 venv，请先安装依赖
  exit /b 1
)

echo Starting SmartDoc AI (reload) on http://127.0.0.1:8000 ...
venv\Scripts\python.exe -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
