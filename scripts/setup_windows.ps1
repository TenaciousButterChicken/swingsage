# SwingSage Windows production setup.
# Run from PowerShell in the repo root: .\scripts\setup_windows.ps1
# Idempotent.

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

# ─── Python check ─────────────────────────────────────────────────────
$python = "python"
$ver = & $python --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: python not on PATH. Install Python 3.12 from python.org." -ForegroundColor Red
    exit 1
}
Write-Host "Found: $ver"
if ($ver -notmatch "Python 3\.(1[2-9]|[2-9]\d)") {
    Write-Host "WARNING: Python 3.12+ recommended for cu128 Blackwell support." -ForegroundColor Yellow
}

# ─── venv ─────────────────────────────────────────────────────────────
if (-not (Test-Path ".venv")) {
    Write-Host "Creating .venv..."
    & $python -m venv .venv
}
& ".\.venv\Scripts\Activate.ps1"
python -m pip install --upgrade pip

# ─── PyTorch (cu128 stable) ───────────────────────────────────────────
Write-Host ""
Write-Host "Installing PyTorch 2.10 cu128 (stable Blackwell sm_120)..."
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 `
    --index-url https://download.pytorch.org/whl/cu128

# ─── Verify CUDA ──────────────────────────────────────────────────────
$cudaCheck = python -c "import torch; print('CUDA:', torch.cuda.is_available(), 'Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
Write-Host $cudaCheck

if ($cudaCheck -notmatch "CUDA: True") {
    Write-Host ""
    Write-Host "CUDA not detected. See HANDOFF.md 'PyTorch fallback' section." -ForegroundColor Yellow
    Write-Host "Quick fallback:" -ForegroundColor Yellow
    Write-Host "  pip uninstall -y torch torchvision torchaudio" -ForegroundColor Yellow
    Write-Host "  pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128" -ForegroundColor Yellow
}

# ─── Rest of stack ────────────────────────────────────────────────────
pip install -r requirements-prod-windows.txt

# ─── .env ─────────────────────────────────────────────────────────────
if (-not (Test-Path ".env")) {
    Copy-Item .env.example .env
    Write-Host ""
    Write-Host "Wrote .env. Edit and set VTRACK_SHOTDATA_PATH to:" -ForegroundColor Cyan
    Write-Host "  C:\Users\<you>\AppData\Local\Packages\02ce737d-b4f8-4bbb-92b2-1355681ff1e8_qbntr2denpnae\LocalState\LAON PEOPLE\LPGDLL\ShotData" -ForegroundColor Cyan
}

Write-Host ""
Write-Host "Done. Run smoke tests with: pytest" -ForegroundColor Green
