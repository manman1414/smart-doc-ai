# SmartDoc AI 一键启动（Python + Node + Web）
# Author: Cursor Agent / 2026-07-06

$ErrorActionPreference = "Continue"
$Root = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.MyCommand.Path }
Set-Location $Root

Write-Host "========================================"
Write-Host "  SmartDoc AI"
Write-Host "========================================"
Write-Host ""

$pythonVenv = Join-Path $Root "server-python\venv\Scripts\python.exe"
$nodeModules = Join-Path $Root "server-node\node_modules"
$webModules = Join-Path $Root "web\node_modules"

if (-not (Test-Path $pythonVenv)) {
    Write-Host "[ERROR] server-python venv not found" -ForegroundColor Red
    Write-Host "  cd server-python"
    Write-Host "  python -m venv venv"
    Write-Host "  .\venv\Scripts\pip install -r requirements.txt"
    Read-Host "Press Enter to exit"
    exit 1
}

if (-not (Test-Path $nodeModules)) {
    Write-Host "[ERROR] server-node dependencies not installed" -ForegroundColor Red
    Write-Host "  cd server-node; yarn install"
    Read-Host "Press Enter to exit"
    exit 1
}

if (-not (Test-Path $webModules)) {
    Write-Host "[ERROR] web dependencies not installed" -ForegroundColor Red
    Write-Host "  cd web; yarn install"
    Read-Host "Press Enter to exit"
    exit 1
}

$yarn = Get-Command yarn -ErrorAction SilentlyContinue
if (-not $yarn) {
    Write-Host "[ERROR] yarn not found in PATH" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "[1/3] Python AI  :8000"
Start-Process cmd.exe -ArgumentList @("/k", "call start-dev.bat") -WorkingDirectory (Join-Path $Root "server-python")

Start-Sleep -Seconds 2

Write-Host "[2/3] Node API   :3000"
Start-Process cmd.exe -ArgumentList @("/k", "yarn dev") -WorkingDirectory (Join-Path $Root "server-node")

Start-Sleep -Seconds 1

Write-Host "[3/3] Web UI     :8001"
Start-Process cmd.exe -ArgumentList @("/k", "yarn dev") -WorkingDirectory (Join-Path $Root "web")

Write-Host ""
Write-Host "Services started in separate windows:"
Write-Host "  Python AI   http://127.0.0.1:8000"
Write-Host "  Node API    http://127.0.0.1:3000"
Write-Host "  Web UI      http://127.0.0.1:8001"
Write-Host ""
Write-Host "LM Studio required on port 11435"
Write-Host "Browser: http://localhost:8001"
Write-Host ""
Read-Host "Press Enter to close this window"
