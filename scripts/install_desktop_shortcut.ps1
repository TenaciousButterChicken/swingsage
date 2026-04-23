# Creates (or refreshes) a "SwingSage" shortcut on the Desktop that
# launches scripts/start_swingsage.ps1 inside a visible PowerShell window
# and opens the browser. Idempotent - safe to re-run.

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$Launcher = Join-Path $RepoRoot "scripts\start_swingsage.ps1"
$IconPath = Join-Path $RepoRoot "web\public\swingsage.ico"

if (-not (Test-Path $Launcher)) {
    Write-Host "ERROR: launcher script not found: $Launcher" -ForegroundColor Red
    exit 1
}

$DesktopDir = [Environment]::GetFolderPath("Desktop")
$ShortcutPath = Join-Path $DesktopDir "SwingSage.lnk"

$Shell = New-Object -ComObject WScript.Shell
$Shortcut = $Shell.CreateShortcut($ShortcutPath)

# Use powershell.exe directly so the script runs in a normal (visible)
# console window - the user wants to see "Server running, close to stop"
# rather than a hidden process they can't terminate.
$Shortcut.TargetPath = "$Env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe"
$Shortcut.Arguments = "-NoExit -ExecutionPolicy Bypass -File `"$Launcher`""
$Shortcut.WorkingDirectory = $RepoRoot
$Shortcut.WindowStyle = 1  # normal window
$Shortcut.Description = "SwingSage - local golf AI. Runs the full pipeline + web UI."
if (Test-Path $IconPath) {
    $Shortcut.IconLocation = "$IconPath,0"
}
$Shortcut.Save()

Write-Host "Desktop shortcut created:" -ForegroundColor Green
Write-Host "  $ShortcutPath"
Write-Host ""
Write-Host "Double-click it. A console window will open, the server will start,"
Write-Host "and your browser will jump to http://127.0.0.1:8000/ a couple seconds later."
