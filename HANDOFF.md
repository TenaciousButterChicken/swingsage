# HANDOFF — Cross-machine workflow & Windows install gotchas

The Mac is for code authoring. The Windows box is the only place real models
run. This doc is the reference for keeping the two in sync and for surviving
the Windows install gauntlet.

## Day-to-day cross-machine flow

```
Mac (code, smoke test, push)              Windows (pull, GPU verify, run)
───────────────────────────────           ─────────────────────────────────
git checkout dev                          git checkout dev
edit / add tests                          git pull
pytest                                    pytest
git commit -m "..."                       python -m capture.vtrack_watcher
git push origin dev                       (real workflow)
```

Open a `dev → main` PR when a phase is done. Review on phone or laptop. Merge
manually after pulling and verifying on Windows.

## Windows install order (RTX 5080, Blackwell sm_120)

**Do these steps in order. Do not skip ahead.**

### 1. Pre-flight

- NVIDIA driver **570 or newer** installed (`nvidia-smi` shows it)
- Python **3.12** from python.org (NOT the Microsoft Store build — it has
  pathing quirks with venv on some setups)
- Git for Windows
- Optional: Visual Studio Build Tools (only needed if MMCV builds from source
  later)

### 2. Bootstrap the venv

```powershell
.\scripts\setup_windows.ps1
```

The script:
1. Creates `.venv`
2. Installs PyTorch 2.10 stable cu128
3. Runs `python -c "import torch; torch.cuda.is_available()"` and reports
4. Installs the rest of the production stack
5. Copies `.env.example` → `.env`

### 3. PyTorch fallback (only if step 2 fails)

If `torch.cuda.is_available()` returns `False`, or you hit
`CUDA error: no kernel image is available for execution on the device` when
running anything on the GPU, fall back to nightly:

```powershell
pip uninstall -y torch torchvision torchaudio
pip install --pre torch torchvision torchaudio `
    --index-url https://download.pytorch.org/whl/nightly/cu128
```

Then re-test:

```powershell
python -c "import torch; x=torch.randn(2,2,device='cuda'); print(x@x)"
```

If nightly also fails, you're hitting one of:
- Stale NVIDIA driver (< 570) — update it
- Wrong Python version — must be 3.12+
- A second Python install on PATH stealing the venv — `where python` to check

Record the resolved version in `pip freeze > windows-resolved.txt` so we
have a snapshot of the working combo.

### 4. MMPose stack (Phase 2 only — skip until then)

MMPose's MMCV / MMEngine version matrix is the **single most fragile thing** in
this project. When Phase 2 starts:

1. **First try** `requirements-prod-windows.txt` with the commented MMPose
   block uncommented and exact versions left as `>=`.
2. If MMCV fails to import or build against PyTorch 2.10:
   - Pin to `mmengine==0.10.5`, `mmcv==2.1.0`, `mmpose==1.3.2`
   - Use `pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.4/index.html`
     even on torch 2.10 cu128 — the pre-built wheel is forward-compatible in
     practice (verified by community).
3. If that still fails: drop to `requirements-prod-windows-fallback.txt` and
   pair PyTorch nightly with the same MMCV pin.

Every successful resolve gets snapshotted to `windows-resolved.txt`.

### 5. 4D-Humans (Phase 3 only)

Install from source per upstream README — not on PyPI:

```powershell
git clone https://github.com/shubham-goel/4D-Humans.git
cd 4D-Humans
pip install -e .[all]
```

Weights auto-download to `~/.cache/4DHumans` on first run. **No SMPL
registration needed** for joint-position output (we don't render meshes).
If you ever want `.obj` mesh output, you'll need to register at
smpl.is.tue.mpg.de and place the model files manually.

### 6. Ollama + LLM (Phase 4 only)

```powershell
winget install Ollama.Ollama
ollama pull qwen2.5:14b-instruct-q4_K_M
ollama run qwen2.5:14b-instruct-q4_K_M "Say hello"
```

`pip install ollama` is already in `requirements-prod-windows.txt`.

## LLM model sizing — why 14B over 32B

- **Qwen 2.5 14B Q4_K_M** ≈ 9 GB → fits fully in 16 GB VRAM with headroom for
  KV cache → **60-65 tok/s** on RTX 5080 (extrapolated from RTX 5070 Ti
  benchmarks, same Blackwell arch, similar bandwidth).
- **Qwen 2.5 32B Q4_K_M** ≈ 22-24 GB → exceeds 16 GB → spills to system RAM →
  **~5-10 tok/s**, plus latency hit from PCIe transfers.
- Coaching prompts are short structured-metric → natural-language tasks. 14B
  is more than smart enough; the bottleneck for "feels live" is tokens/sec, not
  reasoning depth.

If quality ever feels insufficient, swap for `qwen2.5:14b-instruct-q5_K_M`
(~10.5 GB, slightly smarter, still fits) before climbing to 32B.

## Common gotchas

| Symptom | Likely cause | Fix |
|---|---|---|
| `import torch` is fine, `torch.cuda.is_available()` is `False` | Wrong PyTorch wheel index | Reinstall from `--index-url .../cu128` |
| `no kernel image is available for execution on the device` | Stable cu128 not yet on sm_120 (rare in 2026) | Switch to nightly cu128 |
| MMCV import error mentioning `_ext` symbols | MMCV/PyTorch version skew | Use the OpenMMLab wheel index URL, not PyPI |
| Watcher doesn't pick up new shots | Antivirus locking the file briefly, or VTrack writes incrementally | Increase the 50ms grace in `_ShotEventHandler._maybe_ingest` |
| `db is locked` errors | Another process has the SQLite db open | Close `sqlite3` shell, or switch journal_mode (already WAL) |

## File-pin hygiene

After every successful Windows install:

```powershell
pip freeze > windows-resolved.txt
git add windows-resolved.txt
git commit -m "snapshot: working windows resolve $(Get-Date -Format yyyy-MM-dd)"
git push
```

This is the source of truth for "what actually worked," separate from
`requirements-prod-windows.txt` (which is "what we intend").
