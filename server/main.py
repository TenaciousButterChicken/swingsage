"""FastAPI server wrapping the SwingSage pipeline.

Exposes three endpoints:
  POST /api/upload            — accept a video file, start analysis in a
                                background task, return job_id.
  WS   /api/ws/{job_id}       — stream stage updates (decode / NLF / trim /
                                SwingNet / metrics / LLM) and the final
                                results payload.
  GET  /captures/*            — static serving of trimmed videos + keyframes
                                so the frontend can <video> and <img> them.

The pipeline runs on this machine's GPU (same venv as analyze_swing.py)
against the locally-loaded Qwen 3 14B in LM Studio.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
import traceback
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any

# Make the repo root importable so we can pull in our pipeline modules.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import cv2  # noqa: E402
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from analytics.auto_trim import TrimWindow, detect_swing_window, find_impact_frame  # noqa: E402
from analytics.joint_angles import compute_metrics, metrics_to_coach_dict  # noqa: E402
from coach.llm import CoachClient  # noqa: E402
from inference.pose_3d import load_model as load_nlf_model, predict_from_frames  # noqa: E402
from inference.swing_events import (  # noqa: E402
    detect_events_from_frames,
    load_model as load_swingnet_model,
)
from inference.video_io import apply_rotation, open_video  # noqa: E402

# Reuse the crop + save-trim helpers from analyze_swing.py
from scripts.analyze_swing import _crop_frames_to_person, _save_trim_artifacts  # noqa: E402


CAPTURES_DIR = _REPO_ROOT / "captures"
CAPTURES_DIR.mkdir(exist_ok=True)

app = FastAPI(title="SwingSage", version="0.1.0")

# CORS: Vite dev server runs on :5173 by default; the production build
# is served from the same origin so no CORS entry needed for it.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static: expose the captures directory so the frontend can reference
# trimmed videos and keyframe JPGs by URL.
app.mount("/captures", StaticFiles(directory=CAPTURES_DIR), name="captures")


# Serve the built frontend from the same origin so a single uvicorn process
# runs the whole app (no separate `npm run dev` required). Mounted later
# than the API routes so /api/* wins over the SPA catch-all.
_WEB_DIST = _REPO_ROOT / "web" / "dist"


# In-memory job registry. Each job has an asyncio.Queue for progress
# events and a final result dict (populated when done).
class Job:
    def __init__(self, job_id: str, video_path: Path):
        self.job_id = job_id
        self.video_path = video_path
        self.queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self.done = asyncio.Event()
        self.result: dict[str, Any] | None = None
        self.error: str | None = None


jobs: dict[str, Job] = {}


# Cached models — loaded once at server startup, reused for every analysis.
# Without this, each upload paid ~15 s of NLF TorchScript JIT warmup and
# a few seconds of SwingNet load. See @app.on_event("startup") below.
_nlf_model = None
_swingnet_model = None


@app.on_event("startup")
async def _preload_models() -> None:
    """Load NLF + SwingNet once so the first /api/upload request doesn't
    pay the JIT compilation penalty. Runs synchronously on the event loop
    thread pool to keep the server responsive during warmup."""
    global _nlf_model, _swingnet_model
    loop = asyncio.get_event_loop()

    def _load() -> tuple[Any, Any]:
        import time as _time
        t0 = _time.perf_counter()
        nlf = load_nlf_model()
        t1 = _time.perf_counter()
        swingnet = load_swingnet_model()
        t2 = _time.perf_counter()
        print(
            f"[startup] NLF loaded in {t1-t0:.1f}s, "
            f"SwingNet loaded in {t2-t1:.1f}s — server ready"
        )
        return nlf, swingnet

    _nlf_model, _swingnet_model = await loop.run_in_executor(None, _load)


async def _emit(job: Job, event: dict[str, Any]) -> None:
    """Push an event onto the job's queue. The WS handler drains it."""
    await job.queue.put(event)


def _emit_sync(job: Job, loop: asyncio.AbstractEventLoop, event: dict[str, Any]) -> None:
    """Thread-safe version for use from the synchronous pipeline worker."""
    asyncio.run_coroutine_threadsafe(_emit(job, event), loop)


def _run_pipeline(job: Job, loop: asyncio.AbstractEventLoop) -> None:
    """Run the SwingSage pipeline synchronously (on a thread) and emit
    progress events onto the job's queue as it progresses.

    Mirrors scripts/analyze_swing.py:run() but yields structured events
    instead of printing. Any exception bubbles into job.error."""

    def stage(name: str, label: str, **extra) -> None:
        _emit_sync(job, loop, {"type": "stage", "name": name, "label": label, **extra})

    def log(msg: str) -> None:
        _emit_sync(job, loop, {"type": "log", "message": msg})

    try:
        t_total = time.perf_counter()

        # ── Stage 1: Decode ─────────────────────────────────────
        stage("decode", "Decoding video")
        t0 = time.perf_counter()
        cap, rot = open_video(job.video_path, rotation=None)
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        all_frames: list = []
        try:
            while True:
                ok, bgr = cap.read()
                if not ok:
                    break
                all_frames.append(apply_rotation(bgr, rot))
        finally:
            cap.release()
        if not all_frames:
            raise RuntimeError("Video decoded to zero frames")
        log(f"Decoded {len(all_frames)} frames @ {fps:.1f} fps in {time.perf_counter()-t0:.1f}s")

        # ── Stage 2: NLF 3D pose ────────────────────────────────
        stage("pose", "NLF 3D pose", total_frames=len(all_frames))
        t0 = time.perf_counter()
        # Use the cached model loaded at startup — skips the ~15s JIT
        # warmup that otherwise fires on every single analysis.
        all_poses = predict_from_frames(all_frames, model=_nlf_model, verbose=False)
        detected = sum(1 for f in all_poses if f.detected)
        log(f"Pose: {detected}/{len(all_poses)} detected in {time.perf_counter()-t0:.1f}s")

        # ── Stage 3: Auto-trim ──────────────────────────────────
        stage("trim", "Auto-trimming swing")
        window: TrimWindow = detect_swing_window(all_poses, fps=fps, handedness="right")
        if window.used_fallback:
            log(f"Auto-trim fell back (confidence={window.confidence:.1f}x); using full clip")
            impact_rel, _ = find_impact_frame(all_poses, handedness="right", fps=fps)
            window_frames = all_frames
            window_poses = all_poses
            window_start = 0
        else:
            log(
                f"Impact at frame {window.impact}, window [{window.start}, {window.end}), "
                f"confidence={window.confidence:.1f}x"
            )
            window_frames = all_frames[window.start:window.end]
            window_poses = all_poses[window.start:window.end]
            impact_rel = window.impact - window.start
            window_start = window.start

        # ── Stage 4: SwingNet (diagnostic) ──────────────────────
        stage("swingnet", "SwingNet event detection")
        cropped_frames = _crop_frames_to_person(window_frames, window_poses)
        events_rel = detect_events_from_frames(cropped_frames, model=_swingnet_model)
        swingnet_summary = [
            {"name": name, "frame": int(f + window_start), "confidence": float(c)}
            for name, f, c in zip(events_rel.names, events_rel.frames, events_rel.confidences)
        ]

        # ── Stage 5: Metrics ────────────────────────────────────
        stage("metrics", "Computing biomechanical metrics")
        m = compute_metrics(window_poses, impact_frame=impact_rel, fps=fps, handedness="right")
        payload = metrics_to_coach_dict(m)
        # Remap window-relative frame numbers to original timeline
        phases = payload.get("phases", {})
        for k in ("impact_frame", "peak_shoulder_frame", "setup_end_frame"):
            if phases.get(k) is not None:
                phases[k] = int(phases[k]) + window_start
        peaks = payload.get("kinematic_sequence", {}).get("peak_velocity_frames", {})
        for k, v in list(peaks.items()):
            if v is not None and v >= 0:
                peaks[k] = int(v) + window_start

        # Save trim artifacts so the frontend can show them. The helper
        # writes under the repo's ./captures/ relative to the CWD, so we
        # resolve to absolute before deriving a URL path.
        trim_dir = Path(_save_trim_artifacts(
            video_path=job.video_path,
            cropped_frames=cropped_frames,
            fps=fps,
            window_start_orig=window_start,
            peak_shoulder_orig=m.peak_shoulder_frame + window_start,
            impact_orig=impact_rel + window_start,
            window_end_orig=window_start + len(window_frames),
            window_uncropped_frames=window_frames,
            window_poses=window_poses,
        )).resolve()
        # Build URL-relative paths the frontend can fetch via /captures/...
        trim_rel = trim_dir.relative_to(CAPTURES_DIR.resolve()).as_posix()
        artifacts = {
            "trimmed_mp4": f"/captures/{trim_rel}/trimmed.mp4",
            "pose_overlay_mp4": f"/captures/{trim_rel}/pose_overlay.mp4",
            "keyframes": {
                "window_start": f"/captures/{trim_rel}/keyframe_window_start_f{window_start:04d}.jpg",
                "peak_shoulder_top": f"/captures/{trim_rel}/keyframe_peak_shoulder_top_f{m.peak_shoulder_frame + window_start:04d}.jpg",
                "impact": f"/captures/{trim_rel}/keyframe_impact_f{impact_rel + window_start:04d}.jpg",
                "window_end": f"/captures/{trim_rel}/keyframe_window_end_f{window_start + len(window_frames) - 1:04d}.jpg",
            },
        }

        # ── Stage 6: Coaching ───────────────────────────────────
        stage("coaching", "Qwen 3 14B coaching")
        t0 = time.perf_counter()
        coaching: dict[str, Any] | None = None
        try:
            client = CoachClient()
            if client.is_alive():
                fb = client.coach(payload)
                coaching = {
                    "model": client.model,
                    "confidence": fb.confidence,
                    "faults": fb.faults,
                    "diagnosis": fb.diagnosis,
                    "drills": fb.drills,
                }
                log(f"Coaching done in {time.perf_counter()-t0:.1f}s")
            else:
                log(f"LM Studio unreachable at {client.base_url} — skipping coaching")
        except Exception as e:  # noqa: BLE001 — show message to the user
            log(f"Coaching step failed: {e}")

        total_seconds = time.perf_counter() - t_total
        job.result = {
            "elapsed_seconds": round(total_seconds, 1),
            "fps": fps,
            "total_frames": len(all_frames),
            "artifacts": artifacts,
            "swingnet_events": swingnet_summary,
            "metrics": payload,
            "coaching": coaching,
        }
        _emit_sync(job, loop, {"type": "done", "result": job.result})

    except Exception as e:  # noqa: BLE001
        job.error = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        _emit_sync(job, loop, {"type": "error", "message": job.error})
    finally:
        loop.call_soon_threadsafe(job.done.set)


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)) -> JSONResponse:
    """Save the uploaded video into captures/uploads/ with a job-id filename
    and kick off the pipeline in a background thread."""
    job_id = uuid.uuid4().hex[:12]
    ext = (Path(file.filename or "upload.mp4").suffix or ".mp4").lower()
    uploads_dir = CAPTURES_DIR / "uploads"
    uploads_dir.mkdir(exist_ok=True)
    video_path = uploads_dir / f"{job_id}{ext}"
    with video_path.open("wb") as f:
        f.write(await file.read())

    job = Job(job_id=job_id, video_path=video_path)
    jobs[job_id] = job

    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, _run_pipeline, job, loop)

    return JSONResponse({"job_id": job_id, "filename": video_path.name})


@app.websocket("/api/ws/{job_id}")
async def ws_progress(websocket: WebSocket, job_id: str) -> None:
    await websocket.accept()
    job = jobs.get(job_id)
    if job is None:
        await websocket.send_json({"type": "error", "message": f"Unknown job_id {job_id}"})
        await websocket.close()
        return
    try:
        while True:
            # Drain any already-queued events first.
            try:
                event = await asyncio.wait_for(job.queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                # No event this tick. If the job is done AND the queue is
                # truly empty, exit. Otherwise keep waiting.
                if job.done.is_set() and job.queue.empty():
                    break
                continue

            await websocket.send_json(event)
            if event.get("type") in ("done", "error"):
                break
    except WebSocketDisconnect:
        return
    finally:
        try:
            await websocket.close()
        except Exception:  # noqa: BLE001
            pass


@app.get("/api/health")
async def health() -> dict[str, Any]:
    return {"ok": True, "jobs": len(jobs)}


# SPA mount must be last so /api/* and /captures/* take precedence over the
# catch-all. html=True makes StaticFiles fall back to index.html for any
# path that doesn't resolve to a file — required for SPA routes.
if _WEB_DIST.exists():
    app.mount("/", StaticFiles(directory=_WEB_DIST, html=True), name="web")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server.main:app", host="0.0.0.0", port=8000, reload=False)
