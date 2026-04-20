# SwingSage

Local, GPU-accelerated AI golf coach. Replicates the core capture → analysis → feedback loop of LAON SwingCraft's **Swing EZ** for daily practice on a home simulator setup.

This is an **interim system** built to bridge the gap until the real Swing EZ ships. Personal use only — see licensing notes below.

## What it does

1. **Capture** — watches the VTrack launch monitor's `ShotData` JSON drop folder and (later) ingests synchronized camera footage of each swing.
2. **Inference** — runs 2D pose estimation (RTMPose), swing-event segmentation (SwingNet), and 3D human reconstruction (4D-Humans) on each shot.
3. **Analytics** — derives phase-wise kinematic metrics (X-factor, hip/shoulder rotation, sway, lateral shift, kinematic sequence timing) and fuses them with VTrack ball-flight data.
4. **Coach** — feeds the structured metrics to a local LLM (Qwen 2.5 14B via Ollama) that produces conversational, personalized feedback. TTS reads it back.
5. **UI** — PyQt6 dashboard with shot history, swing replay overlays, and the coaching transcript.

## Hardware

| | Dev (Mac) | Prod (Windows) |
|---|---|---|
| **Role** | Code authoring + smoke tests | All inference + coaching |
| **GPU** | None / Metal | RTX 5080 (16GB, Blackwell sm_120) |
| **Inference** | CPU only, mocked GPU calls | CUDA 12.8 + PyTorch cu128 |
| **Camera** | Mocked (loops sample video) | High-FPS USB camera (TBD Phase 2) |
| **VTrack** | Fixture JSON files | Real `ShotData` folder watch |

The Mac is for writing, committing, and pushing code. The Windows box is the only place real models run.

## Cross-platform rules

- No hardcoded paths — everything via `pathlib.Path` + `.env`.
- All CUDA/camera code is `sys.platform`-guarded so Mac smoke tests don't crash.
- `requirements-base.txt` is shared. `requirements-dev-mac.txt` and `requirements-prod-windows.txt` are environment-specific and must never be mixed.
- Every module has a CPU-only smoke test that runs on Mac with fixture data.

## Project layout

```
swingsage/
├── capture/      # VTrack file watcher + camera capture
├── inference/    # Pose, SwingNet, 3D lifting
├── analytics/    # Phase-wise metrics, VTrack fusion
├── coach/        # Local LLM + TTS
├── ui/           # PyQt6 dashboard
├── data/         # SQLite schema + runtime db (gitignored)
├── scripts/      # setup_mac.sh, setup_windows.ps1
├── tests/        # Smoke tests + fixtures
└── docs/         # Phase notes, HANDOFF.md
```

## Upstream components

| Component | Repo | License | Notes |
|---|---|---|---|
| 2D Pose | [open-mmlab/mmpose](https://github.com/open-mmlab/mmpose) (RTMPose) | Apache-2.0 | RTMPose-L 384×288 |
| Swing events | [wmcnally/golfdb](https://github.com/wmcnally/golfdb) (SwingNet) | **CC BY-NC 4.0** | Personal use only — see below |
| 3D human | [shubham-goel/4D-Humans](https://github.com/shubham-goel/4D-Humans) | MIT | Auto-downloads weights, no academic registration needed for joint extraction |
| LLM runtime | [ollama/ollama](https://github.com/ollama/ollama) | MIT | Qwen 2.5 14B-Instruct Q4_K_M |

**SwingNet license note**: GolfDB / SwingNet is CC BY-NC 4.0. SwingSage uses it for personal practice only and is not distributed or sold. If this ever changes (open-sourcing publicly with that model included, commercial use), SwingNet must be replaced.

## Build phases

| Phase | Scope | Status |
|---|---|---|
| **1** | Repo, schema, VTrack watcher, fixtures, smoke tests | ✅ |
| 2 | Camera capture + RTMPose 2D pose pipeline | — |
| 3 | SwingNet event segmentation + 4D-Humans 3D lifting | — |
| 4 | Phase-wise metrics + Ollama coaching loop | — |
| 5 | PyQt6 dashboard + TTS | — |
| 6 | Polish, latency tuning, retire when real Swing EZ arrives | — |

## Quick start (Mac dev)

```bash
./scripts/setup_mac.sh
source .venv/bin/activate
pytest
```

See `HANDOFF.md` for Windows install order, PyTorch fallback, and MMPose lockfile guidance.

## License

MIT for the SwingSage code itself. Upstream components retain their own licenses (see table above).
