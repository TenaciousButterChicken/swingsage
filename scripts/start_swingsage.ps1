# SwingSage launcher. Double-clickable via the Desktop shortcut, runnable
# from PowerShell directly. Starts the bundled FastAPI server and opens
# the web UI in the default browser. Close the window (or Ctrl+C) to stop.

$ErrorActionPreference = "Stop"

# Always run from the repo root, regardless of how the shortcut is invoked.
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$venvPython = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Write-Host "ERROR: venv not found at $venvPython" -ForegroundColor Red
    Write-Host "Run .\scripts\setup_windows.ps1 first." -ForegroundColor Yellow
    Read-Host "Press Enter to close"
    exit 1
}

# If the frontend bundle is missing, build it once (first-run only).
$webDist = Join-Path $RepoRoot "web\dist\index.html"
if (-not (Test-Path $webDist)) {
    Write-Host "First-run: building the web UI..." -ForegroundColor Cyan
    Push-Location (Join-Path $RepoRoot "web")
    if (-not (Test-Path "node_modules")) {
        npm install --silent --no-audit --no-fund
    }
    npm run build
    Pop-Location
}

Write-Host ""
Write-Host "  SwingSage" -ForegroundColor Yellow
Write-Host "  http://127.0.0.1:8000" -ForegroundColor Gray
Write-Host "  Close this window to stop the server." -ForegroundColor DarkGray
Write-Host ""

# Open the browser ~2 s after startup so it hits a ready server. Use a
# background job so the main terminal keeps uvicorn in the foreground.
Start-Job -ScriptBlock {
    Start-Sleep -Seconds 2
    Start-Process "http://127.0.0.1:8000/"
} | Out-Null

# Run uvicorn in the foreground. The bound host is 0.0.0.0 so your phone
# on the same wifi can hit http://<pc-ip>:8000 - LM Studio must already
# be running for the coaching stage to succeed.
& $venvPython -m uvicorn server.main:app --host 0.0.0.0 --port 8000 --log-level warning
