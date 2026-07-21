@echo off
REM 作者：yangkunpeng1 / 2026-07-21
cd /d "%~dp0"
if exist "venv\Scripts\python.exe" (
  set PY=venv\Scripts\python.exe
) else (
  set PY=python
)
echo Using: %PY%
"%PY%" -m pip install pdfplumber==0.11.8 PyMuPDF==1.26.4 rapidocr-onnxruntime==1.4.4 opencv-python-headless==4.11.0.86 Pillow==11.3.0
"%PY%" -m unittest tests.test_parser -v
echo EXIT=%ERRORLEVEL%
pause
