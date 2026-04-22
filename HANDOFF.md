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

### 5. SwingNet + NLF (Phase 2)

Both live under `third_party/`, gitignored. Clone once, then download
weights.

```powershell
mkdir third_party
git clone --depth 1 https://github.com/wmcnally/golfdb.git   third_party/golfdb
git clone --depth 1 https://github.com/isarandi/nlf.git       third_party/nlf

# SwingNet pretrained weights (Google Drive, ~63 MB)
pip install gdown
cd third_party/golfdb/models
python -m gdown 1MBIDwHSM8OKRbxS8YfyRLnUBAdt0nupW -O swingnet_1800.pth.tar
cd ../../..

# NLF multi-person TorchScript weights (GitHub Release, ~493 MB)
mkdir third_party/nlf/weights
cd third_party/nlf/weights
gh release download v0.3.2 --repo isarandi/nlf --pattern "nlf_l_multi_0.3.2.torchscript"
```

Known quirks fixed in our wrappers (`inference/swing_events.py` and
`inference/pose_3d.py`):
- `golfdb/model.py:17` calls `torch.load('mobilenet_v2.pth.tar')` unconditionally
  before the `pretrain` guard. Our SwingNet wrapper monkey-patches `torch.load`
  to no-op on that path so we don't need the 14 MB backbone file.
- NLF's TorchScript model references `torchvision::nms`. `torchvision` MUST
  be imported before `torch.jit.load` so the op is registered. Our wrapper
  does this at module-load time.
- NLF steady-state is ~150 ms/frame on RTX 5080 at 492×354. First call takes
  ~15 s for JIT kernel compile — that's normal.

### 6. LM Studio + Qwen 3 14B (Phase 2)

LM Studio replaced Ollama as of Phase 2 — the NVIDIA-partnered CUDA 12.8
runtime gets ~27% more throughput on Blackwell, and Qwen 3 14B's IFEval /
MMLU-Pro scores beat Qwen 2.5 32B at a fraction of the VRAM.

```powershell
# install (silent, accepts license)
winget install --id ElementLabs.LMStudio --exact `
    --accept-source-agreements --accept-package-agreements --silent

# Run the GUI once to bootstrap ~/.lmstudio/ (required — lms CLI needs it).
# Click "Skip for now" on the first-model screen; leave Developer Mode ON.
& "$env:LOCALAPPDATA\Programs\LM Studio\LM Studio.exe"

# Download Qwen 3 14B Q5_K_M (~10.5 GB)
$lms = "$HOME\.lmstudio\bin\lms.exe"
& $lms get "https://huggingface.co/bartowski/Qwen_Qwen3-14B-GGUF@Q5_K_M" -y --gguf

# Start server + load model (full GPU offload, 16K context)
& $lms server start --port 1234
& $lms load qwen_qwen3-14b --gpu max --context-length 16384 --identifier qwen3-14b -y
```

Configure `.env`:

```
SWINGSAGE_LLM_MODEL=qwen3-14b
SWINGSAGE_LLM_API_BASE=http://127.0.0.1:1234/v1
```

`openai>=1.50` is installed by the coach module's one-shot install; add
it to `requirements-prod-windows.txt` on the next sweep.

## Phase 2 pipeline — run it

```powershell
.\.venv\Scripts\python.exe scripts\analyze_swing.py third_party\golfdb\test_video.mp4
```

Outputs:
1. 8 SwingNet event frames + confidences (~1 s)
2. NLF 3D pose per frame (~30-60 s for a 200-frame clip, 0.5-1 GB VRAM)
3. Biomechanics payload: spine tilt, shoulder/hip/x-factor rotations,
   lead-arm abduction + flex, kinematic-sequence peak-velocity ordering
4. Qwen 3 14B JSON coaching: faults, diagnosis, drills, confidence
   (~3 s at 45 tok/s, 10 GB VRAM for the LLM)

CV and LLM VRAM don't overlap in the current flow — CV runs to completion
then Qwen loads. Peak simultaneous usage is ~1 GB (CV) then ~10 GB (LLM).

## LLM model sizing — why Qwen 3 14B over 32B

- **Qwen 3 14B Q5_K_M** ≈ 10.5 GB → fits with 16K context, ~45 tok/s on RTX 5080.
- **Qwen 3 32B Q4_K_M** ≈ 19 GB → spills to RAM, slow, and Q3 quant damage is significant.
- **Qwen 3 30B-A3B MoE Q4_K_M** ≈ 17 GB → tight fit but only 3B params active per
  token, so still 60+ tok/s. Worth swapping to if recall ever feels thin.
- IFEval + MMLU-Pro both jumped materially 2.5 → 3; the coaching prompt's
  schema-strict JSON output stresses IFEval, which is exactly where Qwen 3
  improved the most.

## Blackwell sm_120 gotchas (hit and cataloged)

| Issue | Impact on us | Resolution |
|---|---|---|
| `torch.load('mobilenet_v2.pth.tar')` in golfdb model.py | SwingNet crashes at init | Monkey-patch torch.load in our wrapper |
| `torchvision::nms` not registered | NLF TorchScript fails to load | `import torchvision` before `torch.jit.load` |
| Pip download timeouts (hotspot) | PyTorch 2.10 cu128 fails mid-install | `--timeout 120 --retries 20` |
| LM Studio `response_format: json_object` rejected | OpenAI SDK 400 | Use `json_schema` instead |
| Windows cp1252 console crash on Qwen's Unicode arrows | smoke test dies | `sys.stdout.reconfigure(encoding='utf-8', errors='replace')` |
| NLF gravity estimate biased by bent-over Address | spine_tilt ~2° instead of ~30° | Hard-code up = -Y (NLF camera frame) |
| Setup script exits 0 on pip failures | Silent install breakage | Known. Worth fixing in `setup_windows.ps1` on next pass |

## VTrack hardware status on this Windows box (as of 2026-04-22)

**Step 7 of Phase 1 handoff (real-hardware integration test) is DEFERRED.**

- Installed: `C:\Program Files\LAON PEOPLE\VTrackToolKit\` — **Win32 app**, build
  dated 2025-10-20, not UWP. The `v2.0.13_release.zip` leftover in `%TEMP%`
  suggests this is a 2.0.x install.
- The expected UWP path
  `%LOCALAPPDATA%\Packages\02ce737d-b4f8-4bbb-92b2-1355681ff1e8_qbntr2denpnae\LocalState\LAON PEOPLE\LPGDLL\ShotData`
  does NOT exist.
- Binary-string analysis of `LPGAgent.exe` + `LPGDLL_x64.dll` confirms this
  version does not emit ShotData JSON to disk — all `ShotData` occurrences are
  C# class/method names (`AddShotDataEntry`, `OnShotDataChanged`, `SendShotData`,
  `IntegratedShotDataPacket`, etc.). Shot data is sent straight to GSPro via
  the in-process integration (`Settings.xml` → `SimulatorType="GSPro"`).

**To unblock real-hardware capture:** upgrade to VTrackToolKit v2.1.5+ UWP
(Microsoft Store build), then the AppData `ShotData` folder should start
appearing. Once it exists and has at least one JSON file, flip
`SWINGSAGE_USE_MOCK_VTRACK=false` in `.env` and re-run Step 7.

Until then `.env` has `SWINGSAGE_USE_MOCK_VTRACK=true` so the watcher uses
`tests/fixtures/vtrack_shots/` and Phase 2 dev can proceed without hardware.

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
