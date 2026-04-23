# SwingSage

**A local, GPU-accelerated AI golf coach for the home simulator.**

Drop a swing video onto a web page. 60 seconds later you get a biomechanical breakdown (shoulder turn, hip turn, X-factor, spine tilt, lead-arm geometry, kinematic sequence), a coaching verdict from a local large language model (faults, diagnosis, drills, confidence), and a conversational chat panel that can answer follow-up questions grounded in your actual numbers. Nothing leaves the machine.

SwingSage targets the core capture-to-feedback loop of [LAON SwingCraft's Swing EZ](https://laonsports.com/) as an interim system for daily practice. Personal use only. See [Licensing](#licensing) for upstream model constraints.

> **Status:** Phases 1 and 2 shipped. The full video-in, coaching-out pipeline is live as a web app. VTrack-hardware capture is deferred (the installed VTrackToolKit build does not emit ShotData JSON to disk; see [VTrack hardware status](#vtrack-hardware-status)). Phases 3 through 6 refine on the shipped pipeline.

---

## Table of contents

- [What it does](#what-it-does)
- [System architecture](#system-architecture)
- [The pipeline, stage by stage](#the-pipeline-stage-by-stage)
- [The web app](#the-web-app)
- [Hardware](#hardware)
- [Project layout](#project-layout)
- [Database schema](#database-schema)
- [Cross-platform development model](#cross-platform-development-model)
- [Quick start](#quick-start)
- [Configuration](#configuration)
- [Testing](#testing)
- [Build phases](#build-phases)
- [VTrack hardware status](#vtrack-hardware-status)
- [Upstream components](#upstream-components)
- [Blackwell gotchas, catalogued](#blackwell-gotchas-catalogued)
- [Engineering principles](#engineering-principles)
- [Known limits and assumptions](#known-limits-and-assumptions)
- [Licensing](#licensing)

---

## What it does

End to end on a single swing clip:

1. **Decode.** OpenCV reads every frame of the uploaded video (rotation-corrected from EXIF metadata where present).
2. **3D pose (NLF).** Neural Localizer Fields produces per-frame SMPL-style 3D joint positions for the golfer, running on the GPU as a cached TorchScript model.
3. **Auto-trim.** A lead-wrist velocity peak pinpoints impact; a motion-based window picks clean start and end bounds. The rest of the pipeline sees only this trimmed segment.
4. **SwingNet events.** GolfDB's 8-event detector (Address, Toe-up, Mid-backswing, Top, Mid-downswing, Impact, Mid-follow-through, Finish) runs on the person-cropped window as a diagnostic reference.
5. **Biomechanics.** The 3D joint series is smoothed (Savitzky-Golay), projected into a body-relative frame, and reduced to coaching-grade metrics: pelvis and chest rotation, X-factor, spine tilt (forward and side), lead-arm abduction and flex, and kinematic-sequence peak-velocity ordering.
6. **Coaching.** The metric bundle is handed to Qwen 3 14B (via LM Studio) with a schema-strict JSON prompt. Output: an array of faults, a plain-English diagnosis, one to three drill suggestions, and a low / medium / high confidence flag.
7. **Follow-up chat.** A second endpoint streams a conversational turn grounded in the same swing numbers, so the user can ask "what is X-factor?" or "why is my shoulder turn low?" and get answers that cite their actual values.

Target latency on an RTX 5080: about 60 seconds total (roughly 45 s for the NLF pose pass over a 200-frame clip, about 3 s for Qwen to stream the JSON verdict at 45 tok/s).

---

## System architecture

```
                                 ┌──────────────────────────┐
                                 │   Browser (Vite+React)   │
                                 │  upload / progress / UI  │
                                 └────────────┬─────────────┘
                                              │  HTTP + WebSocket
                                              ▼
                                 ┌──────────────────────────┐
                                 │    FastAPI (uvicorn)     │
                                 │  /api/upload  /api/ws    │
                                 │  /api/chat    /captures  │
                                 └────────────┬─────────────┘
                                              │  background thread
                                              ▼
┌────────────┐  decode   ┌────────────┐ 3D   ┌────────────┐  trim
│  MP4/MOV   │──────────▶│   OpenCV   │─────▶│    NLF     │─────▶ impact frame
│  upload    │           │   PyAV     │      │ TorchScript│       + window
└────────────┘           └────────────┘      └────────────┘
                                                                   │
                                    ┌──────────────────────────────┤
                                    ▼                              ▼
                           ┌────────────────┐             ┌────────────────┐
                           │   SwingNet     │             │  Biomechanics  │
                           │  (8 events)    │             │  shoulder/hip  │
                           │  diagnostic    │             │  X-factor      │
                           └────────────────┘             │  spine tilt    │
                                                          │  lead-arm      │
                                                          │  kinematic seq │
                                                          └───────┬────────┘
                                                                  │ JSON
                                                                  ▼
                                                          ┌────────────────┐
                                                          │   LM Studio    │
                                                          │  Qwen 3 14B    │
                                                          │  Q5_K_M        │
                                                          └───────┬────────┘
                                                                  │ faults,
                                                                  │ diagnosis,
                                                                  │ drills
                                                                  ▼
                                                          ┌────────────────┐
                                                          │  Chat stream   │
                                                          │  (follow-up)   │
                                                          └────────────────┘
```

Every stage writes structured progress events onto the job's WebSocket (`decode`, `pose`, `trim`, `swingnet`, `metrics`, `coaching`) so the browser shows a live progress indicator and can surface timing for each step. Trimmed video + keyframe JPGs + pose-overlay MP4 land under `captures/` and are served as static assets.

---

## The pipeline, stage by stage

### 1. Decode

`inference/video_io.py` wraps OpenCV's `VideoCapture` with an EXIF-aware rotation helper. Phone verticals come through the right way up without the decoder caring.

### 2. 3D pose (NLF)

[Neural Localizer Fields](https://github.com/isarandi/nlf) runs as a TorchScript model. Chosen over 4D-Humans because NLF produces single-pass multi-person 3D joint positions without the SMPL registration wall, and its temporal consistency on fast-motion clips (golf is the hard case) is notably better. Two real-world wrappers live in `inference/pose_3d.py`:

- `torchvision` is imported before `torch.jit.load` so `torchvision::nms` is registered (otherwise NLF's `jit.load` errors out).
- The gravity axis is hard-coded to `up = -Y` in the NLF camera frame instead of trusting NLF's own gravity estimate, which biases toward the spine line of a bent-over address and reports spine-tilt values around 2 degrees instead of the actual 30.

Steady-state cost on RTX 5080 at 492 by 354 input: about 150 ms per frame. First call pays about 15 seconds of JIT kernel compile, so the server preloads the model at startup.

### 3. Auto-trim

`analytics/auto_trim.py` finds the impact frame by picking the lead-wrist speed peak (the unambiguous physics signal), then sizes a symmetric window around it using integrated motion energy. If the confidence ratio is low (tripod shakes, partial swing), the trimmer falls back to the full clip and the UI surfaces the fallback so the user knows.

### 4. SwingNet events

`inference/swing_events.py` wraps [wmcnally/golfdb](https://github.com/wmcnally/golfdb)'s EventDetector. The GolfDB model file has a bug where `torch.load('mobilenet_v2.pth.tar')` is called unconditionally before the `pretrain` guard, so every init needs that file on disk. Our wrapper monkey-patches `torch.load` to no-op on that path, skipping a pointless 14 MB weights download for values we overwrite from the SwingNet checkpoint anyway.

In the current pipeline, SwingNet's 8 event frames are exposed for diagnostics (the Results view shows them in the timeline) but the biomechanical metrics no longer anchor to any single event frame. See the [Engineering principles](#engineering-principles) section on extrema-based metrics.

### 5. Biomechanics

`analytics/joint_angles.py` reduces the 3D joint series to the numbers a coach actually talks about:

| Metric | What it is |
|---|---|
| `shoulder_turn_deg` at top | magnitude of upper body rotation from address |
| `hip_turn_deg` at top | magnitude of pelvis rotation from address |
| `x_factor_deg` at top | chest rotation minus pelvis rotation (bigger = more potential power if controlled) |
| `spine_tilt_forward_deg` at address | pitch: positive = bent toward ball |
| `spine_tilt_side_deg` | roll: side bend relative to vertical |
| `lead_arm_abduction_deg` at top | shoulder elevation; about 90 = arm horizontal |
| `lead_arm_flex_deg` at top | elbow angle; 180 = fully straight arm |
| Kinematic sequence | peak-velocity ordering of pelvis, chest, lead arm, lead wrist |

All "at top" and "at address" metrics are extrema or medians over the appropriate swing phase, not single-frame samples. That means a few-frame SwingNet jitter cannot flip a metric into a different bin, which was the bug that killed the first metrics version.

### 6. Coaching (Qwen 3 14B via LM Studio)

`coach/llm.py` talks to LM Studio's OpenAI-compatible endpoint on `http://127.0.0.1:1234/v1`. Two flavors:

**Structured verdict.** A schema-strict JSON prompt yields:

```json
{
  "faults": ["early extension", "lead arm bent at top"],
  "diagnosis": "X-factor of 28° is short of the 35 to 50 range, and lead_arm_flex of 148° means the arm folds well before the top. Power loss is coming from both collapsed rotation and arm structure, not tempo.",
  "drills": [
    {"name": "Pump drill", "why": "Builds the feel of a straight lead arm into the top, directly targeting lead_arm_flex."},
    {"name": "Wall turn drill", "why": "Forces separation between hips and shoulders, raising x_factor_deg."}
  ],
  "confidence": "high"
}
```

**Follow-up chat.** `/api/chat/{job_id}` streams a conversational turn grounded in the same metrics and coaching, so "what is X-factor?" gets a definition followed by the user's specific value.

LM Studio replaced Ollama at the Phase 2 shift. The NVIDIA-partnered CUDA 12.8 runtime delivers about 27 percent more throughput on Blackwell, and Qwen 3 14B's IFEval and MMLU-Pro scores beat Qwen 2.5 32B at a fraction of the VRAM.

### Why Qwen 3 14B?

| Model | VRAM | Speed on RTX 5080 | Notes |
|---|---|---|---|
| **Qwen 3 14B Q5_K_M** | about 10.5 GB | 45 tok/s | Current pick. Fits with 16K context, leaves room for CV pipeline. |
| Qwen 3 32B Q4_K_M | about 19 GB | spills to RAM | Q3 quant damage is significant; avoid. |
| Qwen 3 30B-A3B MoE Q4_K_M | about 17 GB | 60+ tok/s | Only 3B params active per token. Worth swapping to if recall thin. |

The coaching prompt's schema-strict JSON stresses IFEval specifically, which is exactly where Qwen 3 improved most over 2.5.

---

## The web app

`web/` holds a Vite + React + TypeScript frontend; `server/` holds a FastAPI backend. They can run separately in development (`npm run dev` on 5173, uvicorn on 8000, dev proxy handles `/api` and `/captures`) or as a single uvicorn process that serves the built SPA alongside the API in production.

### UI flow

```
UploadView ── drag-drop swing video ──▶ POST /api/upload
        returns { job_id }
                 │
                 ▼
ProcessingView ── WS /api/ws/{job_id} ──▶ stage events
        decode → pose → trim → swingnet → metrics → coaching
                 │
                 ▼ on {type:"done", result}
ResultsView
  • <video> of the trimmed swing
  • pose-overlay toggle (raw vs pose-skeleton MP4)
  • 4 keyframe thumbnails (window start, top, impact, window end)
  • metric dashboard with color-coded in-range / out-of-range bands
  • coaching panel (faults, diagnosis, drills)
  • chat panel streaming answers from /api/chat/{job_id}
```

### Design system

Dark warm-black theme with champagne gold and fairway green accents. Tokens live in `web/tailwind.config.js`:

- `ink-{100..950}` for warm near-black neutrals
- `champagne-{50..600}` for gold accents
- `fairway-{400..600}` for in-range / success
- `ember-{400..600}` for out-of-range / warning

Fonts: Fraunces (display, italic-capable serif), Geist (sans), Geist Mono. Loaded from Google Fonts in `index.html`.

### One-click launcher (Windows)

`scripts/start_swingsage.ps1` and `scripts/install_desktop_shortcut.ps1` together install a double-clickable Desktop shortcut. The launcher:

1. Checks the venv, first-run-builds `web/dist/` if needed.
2. Starts uvicorn on `0.0.0.0:8000` (phone-on-same-wifi friendly).
3. Opens the default browser to `http://127.0.0.1:8000/` after a 2-second delay.

Close the terminal window to stop the server.

---

## Hardware

|  | Dev (Mac) | Prod (Windows) |
|---|---|---|
| **Role** | Code authoring, smoke tests, frontend dev | All real inference and coaching |
| **GPU** | None / Metal | RTX 5080 (16 GB, Blackwell sm_120) |
| **PyTorch** | CPU | CUDA 12.8 + cu128 wheels |
| **Camera** | Mocked (fixture video) | Phone or USB camera upload |
| **VTrack** | Fixture JSON files | See [VTrack hardware status](#vtrack-hardware-status) |
| **OS** | macOS (arm64) | Windows 11 |
| **Python** | 3.12 | 3.12 |

The Mac is for writing, committing, and pushing code. The Windows box is the only place real models run.

---

## Project layout

```
swingsage/
├── capture/                     # VTrack file watcher + platform config
│   ├── config.py                #   .env loader + platform-aware defaults
│   └── vtrack_watcher.py        #   Watchdog observer (Phase 1 path; deferred)
├── inference/                   # GPU model wrappers
│   ├── pose_3d.py               #   NLF TorchScript, 3D SMPL joints per frame
│   ├── swing_events.py          #   SwingNet 8-event detector (GolfDB)
│   ├── video_io.py              #   OpenCV + EXIF rotation helpers
│   └── visualization.py         #   Pose-overlay MP4 renderer
├── analytics/                   # Body-frame biomechanics
│   ├── auto_trim.py             #   Lead-wrist velocity-peak impact detector
│   └── joint_angles.py          #   Rotations, tilts, kinematic sequence
├── coach/                       # LM Studio client
│   └── llm.py                   #   Qwen 3 14B structured + streaming chat
├── server/                      # FastAPI backend
│   ├── main.py                  #   /api/upload, /api/ws, /api/chat, /captures
│   └── README.md
├── web/                         # Vite + React frontend
│   ├── src/App.tsx              #   Upload -> Processing -> Results state machine
│   ├── src/components/          #   Brand, UploadView, ProcessingView, ResultsView
│   ├── src/lib/api.ts           #   fetch + WebSocket helpers
│   └── README.md
├── data/
│   ├── schema.sql               # SQLite schema (sessions/shots/swings/events)
│   └── runtime/                 # Runtime db (gitignored)
├── scripts/
│   ├── setup_mac.sh             # Mac venv bootstrap
│   ├── setup_windows.ps1        # Windows venv bootstrap (PyTorch cu128)
│   ├── start_swingsage.ps1      # One-click launcher (web app)
│   ├── install_desktop_shortcut.ps1
│   ├── analyze_swing.py         # CLI entry point into the pipeline
│   └── generate_sample_video.py # Synthetic test video generator
├── tests/
│   ├── conftest.py
│   ├── test_imports.py
│   ├── test_schema.py
│   ├── test_vtrack_watcher.py
│   └── fixtures/
│       ├── vtrack_shots/        # 4 fake VTrack ShotData JSONs
│       └── videos/              # Synthetic sample mp4
├── docs/
│   └── phase1.md                # Phase 1 ship summary
├── third_party/                 # gitignored: golfdb + nlf checkouts + weights
├── captures/                    # gitignored: uploaded videos, trimmed MP4s, keyframes
├── HANDOFF.md                   # Cross-machine workflow + Blackwell gotchas
├── pyproject.toml               # ruff + pytest config
├── requirements-base.txt        # Shared deps (watchdog, dotenv)
├── requirements-dev-mac.txt     # CPU-only torch + lint
├── requirements-prod-windows.txt# CUDA 12.8 stack
└── requirements-prod-windows-fallback.txt  # Nightly cu128 fallback
```

---

## Database schema

Four tables, one purpose each. Defined in `data/schema.sql`, idempotent (`IF NOT EXISTS`), with `PRAGMA journal_mode = WAL` for concurrent reads while the watcher writes.

```
sessions ──┐
           │ 1:N
           ▼
        shots ──┐
                │ 1:N
                ▼
             swings ──┐
                      │ 1:N
                      ▼
                   events
```

| Table | Purpose | Key columns |
|---|---|---|
| `sessions` | One row per practice session | `started_at`, `ended_at`, `label`, `notes` |
| `shots` | One row per VTrack shot | `vtrack_shot_id` (UNIQUE, dedupe key), full ball-flight + club-delivery fields, `raw_json` |
| `swings` | One row per captured swing video | `video_path`, `pose_path` (2D keypoints), `mesh_path` (3D SMPL), `fps`, `frame_count` |
| `events` | One row per detected swing event | `event_type` in {`address`, `takeaway`, `mid_backswing`, `top`, `mid_downswing`, `impact`, `mid_followthrough`, `finish`}, `frame_number`, `confidence` |

**`raw_json` design choice:** even though every known VTrack field has a typed column, the entire raw JSON payload is stored alongside. Future fields, undocumented fields, and proprietary additions all survive. Disk is cheap; data loss is not.

---

## Cross-platform development model

```
Mac (code, smoke test, push)              Windows (pull, GPU verify, run)
.............................             ................................
git checkout dev                          git checkout dev
edit / add tests                          git pull
pytest                                    pytest
git commit -m "..."                       python scripts\analyze_swing.py clip.mp4
git push origin dev                         or double-click the desktop shortcut
```

Open a `dev` to `main` PR when a phase is done. Review on phone or laptop. Merge manually after pulling and verifying on Windows.

### Hard rules

1. **No hardcoded paths.** Everything via `pathlib.Path` and `.env`.
2. **All CUDA and camera code is `sys.platform`-guarded** so Mac smoke tests never crash on `torch.cuda` style failures.
3. **`requirements-base.txt` is shared.** `requirements-dev-mac.txt` and `requirements-prod-windows.txt` are environment-specific and must never be mixed.
4. **Every module has a CPU-only smoke test** that runs on Mac with fixture data.
5. **Mock VTrack is the default** on anything that is not Windows, unless explicitly disabled.

These five rules mean every commit can be safely written on the Mac and every pull on Windows is a simple `git pull && pytest`.

---

## Quick start

### Mac (development)

```bash
git clone https://github.com/TenaciousButterChicken/swingsage.git
cd swingsage
./scripts/setup_mac.sh                        # creates .venv, installs deps, copies .env
source .venv/bin/activate
pytest                                        # all green
python scripts/generate_sample_video.py       # creates the synthetic test video
python -m capture.vtrack_watcher              # watches the fixture folder; Ctrl-C to stop
```

Frontend dev:

```bash
cd web
npm install
npm run dev     # Vite on :5173, proxies /api and /captures to :8000
```

### Windows (production)

Full install guide including PyTorch cu128 wheels, SwingNet and NLF weight downloads, and the LM Studio + Qwen 3 setup is in [HANDOFF.md](HANDOFF.md).

TL;DR once installed:

```powershell
.\scripts\setup_windows.ps1     # first time
.\scripts\start_swingsage.ps1   # or double-click the desktop shortcut
```

Opens `http://127.0.0.1:8000/`. Drop a swing video on the upload card, hit Analyse. About 60 seconds later you have the full breakdown.

---

## Configuration

All configuration is environment-driven. `.env.example` is the source of truth. The important variables:

```bash
# ─ VTrack launch monitor ──────────────────────────────────────────────
# See "VTrack hardware status" below for the Win32 vs UWP situation.
VTRACK_SHOTDATA_PATH=

# Force mock VTrack mode even on Windows (useful while hardware deferred).
SWINGSAGE_USE_MOCK_VTRACK=true

# ─ Storage ────────────────────────────────────────────────────────────
SWINGSAGE_DATA_DIR=./data/runtime
SWINGSAGE_DB_PATH=./data/runtime/swingsage.db

# ─ Coaching LLM (LM Studio, OpenAI-compatible) ────────────────────────
SWINGSAGE_LLM_MODEL=qwen3-14b
SWINGSAGE_LLM_API_BASE=http://127.0.0.1:1234/v1

# ─ Logging ────────────────────────────────────────────────────────────
SWINGSAGE_LOG_LEVEL=INFO
```

`capture/config.py` loads the `.env`, applies platform-aware defaults (mock VTrack by default on anything that is not Windows), and returns a frozen `Config` dataclass that everything else reads from.

---

## Testing

```bash
pytest                  # everything
pytest -v               # verbose
pytest tests/test_vtrack_watcher.py::test_process_existing_is_idempotent
pytest --cov=capture    # coverage on the capture package
```

The core test suite covers:

- **`test_imports.py`** verifies every package and submodule imports cleanly. Catches accidental syntax errors and broken `__init__.py` files before any logic runs.
- **`test_schema.py`** verifies the SQLite schema applies cleanly to a fresh in-memory db.
- **`test_vtrack_watcher.py`**
  - All 4 fixture shots ingest in one pass.
  - Re-running `process_existing` on the same folder is a no-op (idempotency).
  - Unknown or future JSON fields survive in `raw_json`, so future VTrack additions do not silently drop data.

Tests run in seconds and require no GPU. They are the safety net for the cross-machine flow: Mac commits land green, Windows pulls land green.

The long-running filesystem observer is intentionally **not** tested in CI because filesystem-event semantics differ across macOS, Linux, and Windows enough to make tests flaky. The `process_existing` path covers the same parsing and dedupe logic without that risk.

---

## Build phases

| Phase | Scope | Status |
|---|---|---|
| **1** | Repo, schema, VTrack watcher, fixtures, smoke tests, dual-OS bootstrap, HANDOFF doc | ✅ |
| **2** | Video capture via browser upload, NLF 3D pose, auto-trim, SwingNet events, biomechanical metrics, LM Studio coaching, web app, follow-up chat, desktop shortcut | ✅ |
| **3** | Real-hardware VTrack integration (unblocked when VTrackToolKit reaches a UWP build that emits ShotData JSON; see below) | Deferred by upstream |
| **4** | Pose-overlay UX polish, multi-swing session view, metric-trend dashboard | Partially shipped (single-swing overlay live) |
| **5** | TTS playback of the coaching verdict | Pending |
| **6** | Latency tuning, on-disk swing library, retirement when real Swing EZ arrives | Pending |

Phase 1 ships a summary in `docs/phase1.md`. Future phases will mirror that format.

---

## VTrack hardware status

**Current blocker.** The Windows box runs VTrackToolKit v2.0.x (Win32 install at `C:\Program Files\LAON PEOPLE\VTrackToolKit\`, build dated 2025-10-20), not the UWP version that writes shots to disk. Binary-string analysis of `LPGAgent.exe` and `LPGDLL_x64.dll` confirms this build does not emit `ShotData` JSON anywhere on disk: every `ShotData` occurrence is a C# class or method name (`AddShotDataEntry`, `OnShotDataChanged`, `SendShotData`, `IntegratedShotDataPacket`, etc.), and shot data is sent straight to GSPro via the in-process integration.

**To unblock.** Install VTrackToolKit v2.1.5+ from the Microsoft Store (UWP build). The AppData `ShotData` folder should start appearing at
`%LOCALAPPDATA%\Packages\02ce737d-b4f8-4bbb-92b2-1355681ff1e8_qbntr2denpnae\LocalState\LAON PEOPLE\LPGDLL\ShotData`.
Once it exists and has at least one JSON file, flip `SWINGSAGE_USE_MOCK_VTRACK=false` in `.env` and the Phase 1 watcher picks up real shots.

Until then, Phase 2's video-upload flow is the main entry point; the watcher runs against fixture JSONs for structural testing.

---

## Upstream components

| Component | Repo | License | Notes |
|---|---|---|---|
| 3D pose | [isarandi/nlf](https://github.com/isarandi/nlf) | MIT | Multi-person TorchScript, about 150 ms/frame on RTX 5080 |
| Swing events | [wmcnally/golfdb](https://github.com/wmcnally/golfdb) (SwingNet) | **CC BY-NC 4.0** | Personal use only. See [Licensing](#licensing) |
| LLM runtime | [LM Studio](https://lmstudio.ai/) | Proprietary (free tier) | CUDA 12.8 partnered runtime, OpenAI-compatible API |
| LLM model | [Qwen 3 14B Instruct](https://huggingface.co/Qwen/Qwen3-14B) | Apache-2.0 | Q5_K_M GGUF quant |
| Web backend | [FastAPI](https://github.com/tiangolo/fastapi) + [uvicorn](https://github.com/encode/uvicorn) | MIT / BSD | WebSocket + static serving |
| Frontend | [Vite](https://github.com/vitejs/vite) + [React](https://github.com/facebook/react) + [Tailwind](https://github.com/tailwindlabs/tailwindcss) | MIT | |
| File watching | [gorakhargosh/watchdog](https://github.com/gorakhargosh/watchdog) | Apache-2.0 | Cross-platform `Observer` |
| Video I/O | [PyAV](https://github.com/PyAV-Org/PyAV) + [OpenCV](https://github.com/opencv/opencv) | BSD / Apache-2.0 | |

---

## Blackwell gotchas, catalogued

Every real-hardware issue hit so far on RTX 5080 sm_120, with resolutions. Full list in `HANDOFF.md`; the big ones:

| Issue | Impact | Resolution |
|---|---|---|
| `torch.load('mobilenet_v2.pth.tar')` in golfdb `model.py` | SwingNet crashes at init | Monkey-patch `torch.load` in our wrapper |
| `torchvision::nms` not registered | NLF TorchScript fails to load | `import torchvision` before `torch.jit.load` |
| LM Studio rejects `response_format: json_object` | OpenAI SDK returns 400 | Use `json_schema` instead |
| Windows cp1252 console crash on Qwen's Unicode arrows | CLI smoke test dies | `sys.stdout.reconfigure(encoding='utf-8', errors='replace')` |
| NLF gravity estimate biased by bent-over address | `spine_tilt` reported about 2 degrees instead of about 30 | Hard-code `up = -Y` in NLF camera frame |
| NLF first-call JIT compile takes about 15 s | Every upload feels slow | Preload NLF + SwingNet at server startup |
| Stable cu128 wheel did not have sm_120 kernels at first | "no kernel image is available for execution on the device" | Fall back to nightly cu128 |
| MMCV wheel version skew vs torch 2.10 | Import errors mentioning `_ext` symbols | Use the OpenMMLab wheel index URL, not PyPI |

---

## Engineering principles

A few non-obvious decisions that shape the codebase:

**Extrema over event frames.** All "at top" and "at address" metrics are extrema or medians over swing phases, not single-frame samples at SwingNet events. Event-frame localization can be off by a handful of frames, which used to flip metrics across in-range boundaries. Extrema are robust to this by construction. The only anchor the pipeline actually needs is the lead-wrist velocity peak (unambiguous physics), which drives the auto-trim.

**Cache at startup, not on demand.** NLF's first call pays about 15 seconds of TorchScript JIT compile. SwingNet loads weights from disk. Both are loaded at uvicorn startup so every upload starts on a warm pipeline.

**Stable seams via JSON payloads.** Each stage emits a structured event on the WebSocket. The browser renders progress off those events, and the final `{type: "done", result}` is a single self-contained payload with metrics, artifact URLs, and the coaching JSON. No stage needs to know about downstream shape.

**`raw_json` lives forever.** VTrack payloads are stored verbatim alongside the parsed columns. Field-name guesses can be wrong, VTrack can add fields in firmware updates, and forensic recovery is occasionally necessary.

**Mock-by-default on Mac.** The pipeline runs end-to-end with fixture data on a machine with no GPU and no launch monitor. This is not an afterthought; it is the only way the Mac-to-Windows commit flow stays sane.

**Pin the working stack, document the path to get there.** `requirements-prod-windows.txt` pins exact versions because the PyTorch + cu128 + sm_120 + MMCV matrix is fragile. `HANDOFF.md` walks through the resolution order step by step and asks for `pip freeze > windows-resolved.txt` snapshots after every successful install.

**Small, single-responsibility commits.** The git log reads as a project outline, not a blow-by-blow:

```
phase2: browser-playable video, trim-fallback UI, chat with Qwen
perf(server): cache NLF + SwingNet at startup
phase2: pose-overlay video + raw/pose toggle in results
phase2: one-click desktop shortcut (launcher + icon + installer)
phase2: web app: FastAPI server + Vite/React frontend
phase2: extrema-based metrics: remove event-frame anchoring entirely
phase2: pose-geometry override for Top and Impact anchors
phase2: biomechanical analytics + LM Studio coaching client
phase2: SwingNet event detection + NLF 3D pose wrappers
```

**Honest READMEs.** The current status is current status. Deferred work is marked deferred. The VTrack firmware issue is called out in the open, not hidden behind a TODO.

---

## Known limits and assumptions

1. **VTrack JSON field names.** The `_FIELD_MAP` in `capture/vtrack_watcher.py` is best-guess based on common launch-monitor conventions; the exact VTrack schema is not publicly documented. On the first real shot, dump the JSON and update the map.
2. **`shotID` uniqueness.** Dedupe relies on VTrack assigning a unique ID per shot. If it does not, switch the dedupe key to a hash of `raw_json`.
3. **Handedness.** The current auto-trim and metrics code is tuned for right-handed golfers. Left-handed support needs a sign flip in `analytics/auto_trim.py` and the body-frame construction in `analytics/joint_angles.py`.
4. **Single-golfer framing.** The NLF wrapper takes the largest detected person box. Multi-person frames (instructor on-screen) will grab whichever box is largest at each frame.
5. **Full ISB joint coordinate systems are not implemented.** The biomechanics code uses simple geometric operations in the NLF camera frame rather than per-joint 3-axis reference frames. This is deliberate for the Phase 2 MVP; accuracy is coaching-grade, not lab-grade.
6. **Setup script exits 0 on pip failures.** `setup_windows.ps1` does not bubble pip errors as exit codes. Known. Worth fixing on the next pass.

---

## Licensing

**SwingSage code: MIT.** Everything in this repo is MIT licensed. See [LICENSE](LICENSE).

**Upstream components retain their own licenses.** The full list is in [Upstream components](#upstream-components).

**SwingNet (CC BY-NC 4.0) carve-out.** GolfDB / SwingNet is non-commercial. SwingSage uses it for personal practice and is not distributed or sold. If this project is ever open-sourced publicly with SwingNet weights bundled, or used commercially in any form, **SwingNet must be replaced** with a permissively-licensed alternative or a trained-from-scratch model.

**This is an interim system.** SwingSage exists to bridge the gap until the real Swing EZ ships. When it does, Phase 6 is the polite retirement.

---

*Built for fewer 7-iron mishits. Powered by an RTX 5080, a warm FastAPI process, and a refusal to send swing data to anyone else's server.*
