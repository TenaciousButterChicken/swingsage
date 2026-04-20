#!/usr/bin/env bash
# Mac development environment bootstrap.
# Idempotent: safe to re-run.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "ERROR: this script is for macOS. Use scripts/setup_windows.ps1 on Windows." >&2
    exit 1
fi

PYTHON="${PYTHON:-python3.12}"
if ! command -v "$PYTHON" >/dev/null 2>&1; then
    echo "ERROR: $PYTHON not found. Install via 'brew install python@3.12'." >&2
    exit 1
fi

if [[ ! -d .venv ]]; then
    echo "Creating .venv with $PYTHON..."
    "$PYTHON" -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements-dev-mac.txt

if [[ ! -f .env ]]; then
    cp .env.example .env
    echo "Wrote .env from .env.example — review before running the watcher."
fi

echo
echo "Environment ready. Activate with: source .venv/bin/activate"
echo "Run smoke tests with:               pytest"
echo "Generate sample video with:         python scripts/generate_sample_video.py"
