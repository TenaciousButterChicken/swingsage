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
from fastapi.responses import JSONResponse, StreamingResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from analytics.auto_trim import TrimWindow, detect_swing_window, find_impact_frame  # noqa: E402
from analytics.joint_angles import compute_metrics, metrics_to_coach_dict  # noqa: E402
from capture.config import load_config  # noqa: E402
from capture.vtrack_openconnect import (  # noqa: E402
    latest_ball_flight,
    serve as serve_openconnect,
)
from capture.vtrack_watcher import _connect as _db_connect, _ensure_db  # noqa: E402
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

# Config is loaded once; the VTrack bridge and the ball-flight lookup both
# read from it. Env changes require a server restart to pick up.
_CFG = load_config()

# Bridge task handle — created in the startup hook when enabled, cancelled
# in the shutdown hook. Kept at module scope so both hooks can see it.
_openconnect_task: asyncio.Task | None = None


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


def _bridge_is_running() -> bool:
    return _openconnect_task is not None and not _openconnect_task.done()


async def _launch_bridge() -> bool:
    """Start the bridge as a background task if it's not already running.
    Returns True if we just started it, False if it was already up."""
    global _openconnect_task
    if _bridge_is_running():
        return False
    _openconnect_task = asyncio.create_task(serve_openconnect(_CFG))
    return True


async def _cancel_bridge() -> bool:
    """Stop the bridge. Returns True if it was running, False otherwise."""
    global _openconnect_task
    task = _openconnect_task
    if task is None or task.done():
        return False
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    _openconnect_task = None
    return True


@app.on_event("startup")
async def _start_openconnect_bridge() -> None:
    """Launch the VTrack → GSPro OpenConnect bridge as a background task so
    every shot gets captured the moment it hits. Disable via
    SWINGSAGE_OPENCONNECT_ENABLED=false to start the server with the bridge
    off — the UI toggle can flip it on at runtime either way."""
    if not _CFG.openconnect_enabled:
        print("[startup] OpenConnect bridge disabled (SWINGSAGE_OPENCONNECT_ENABLED=false)")
        return
    await _launch_bridge()
    print(
        f"[startup] OpenConnect bridge listening on "
        f"{_CFG.openconnect_host}:{_CFG.openconnect_port} "
        f"-> forwarding to GSPro at "
        f"{_CFG.openconnect_gspro_host}:{_CFG.openconnect_gspro_port}"
    )


@app.on_event("shutdown")
async def _stop_openconnect_bridge() -> None:
    await _cancel_bridge()


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
        trim_info = {
            "used_fallback": bool(window.used_fallback),
            "confidence": float(window.confidence),
            "window_start": int(window_start),
            "window_end": int(window_start + len(window_frames)),
            "window_seconds": round(len(window_frames) / fps, 2),
        }

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

        # ── Pair with the most recent VTrack shot ───────────────
        # Look up any shot the OpenConnect bridge captured in the last
        # ball_shot_max_age_sec seconds. If none, coaching runs on body
        # metrics only, same as before.
        ball_flight = latest_ball_flight(_CFG.db_path, _CFG.ball_shot_max_age_sec)
        if ball_flight is not None:
            log(
                f"Paired with VTrack shot captured at {ball_flight['captured_at']} "
                f"(ball_speed_mps={ball_flight.get('ball_speed_mps')}, "
                f"carry_m={ball_flight.get('carry_distance_m')})"
            )
        else:
            log(
                f"No VTrack shot in last {_CFG.ball_shot_max_age_sec}s — "
                f"coaching on body metrics only"
            )

        # ── Stage 6: Coaching ───────────────────────────────────
        stage("coaching", "Qwen 3 14B coaching")
        t0 = time.perf_counter()
        coaching: dict[str, Any] | None = None
        try:
            client = CoachClient()
            if client.is_alive():
                fb = client.coach(payload, ball_data=ball_flight)
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
            "trim": trim_info,
            "artifacts": artifacts,
            "swingnet_events": swingnet_summary,
            "metrics": payload,
            "coaching": coaching,
            "ball_flight": ball_flight,
        }

        # Persist the analysis so it survives uvicorn restarts and shows up
        # in Swing History. We look up the paired shot's row id (if any)
        # by the vtrack_shot_id we already captured, so deleting a shot
        # cleanly orphans its analysis via ON DELETE SET NULL.
        try:
            _persist_analysis(
                job_id=job.job_id,
                video_path=job.video_path,
                trim_dir=trim_dir,
                result=job.result,
                ball_flight=ball_flight,
            )
        except Exception as e:  # noqa: BLE001
            # Don't fail the user's analysis if history persistence breaks.
            log(f"Warning: failed to persist analysis to history: {e}")

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


class ChatMessage(BaseModel):
    role: str  # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]


@app.post("/api/chat/{job_id}")
async def chat(job_id: str, req: ChatRequest) -> StreamingResponse:
    """Stream a follow-up conversation turn grounded in job_id's analysis.

    Client sends the full user/assistant history; the server prepends the
    system prompt + this swing's numbers and streams Qwen's reply as
    text/plain chunks. Not SSE-formatted — the response body is just the
    raw answer tokens, which is trivially consumable by fetch + ReadableStream.

    Falls back to the analyses table when the job isn't in the in-memory
    registry so chat keeps working on history entries after a server
    restart.
    """
    result: dict[str, Any] | None = None
    job = jobs.get(job_id)
    if job is not None and job.result is not None:
        result = job.result
    else:
        _ensure_db(_CFG.db_path)
        with _db_connect(_CFG.db_path) as conn:
            row = conn.execute(
                "SELECT result_json FROM analyses WHERE job_id = ?",
                (job_id,),
            ).fetchone()
        if row is not None:
            try:
                result = json.loads(row["result_json"])
            except (json.JSONDecodeError, TypeError):
                result = None

    if result is None:
        raise HTTPException(status_code=404, detail=f"No completed job {job_id}")

    metrics = result.get("metrics", {})
    coaching = result.get("coaching")
    ball_flight = result.get("ball_flight")
    if ball_flight is not None:
        # Embed ball_flight as a nested key so stream_chat's context block
        # (which serializes metrics verbatim) shows the model both body
        # biomechanics and the paired shot numbers in one JSON blob.
        metrics = {**metrics, "ball_flight": ball_flight}
    history = [{"role": m.role, "content": m.content} for m in req.messages]

    client = CoachClient()
    if not client.is_alive():
        raise HTTPException(
            status_code=503,
            detail=f"LM Studio unreachable at {client.base_url}",
        )

    def _token_stream():
        try:
            for token in client.stream_chat(metrics, coaching, history):
                yield token
        except Exception as e:  # noqa: BLE001
            # Surface the failure inline so the client sees something rather
            # than silently truncated output. The chat UI renders whatever
            # arrives, so the error text lands in the message bubble.
            yield f"\n\n[error: {type(e).__name__}: {e}]"

    return StreamingResponse(_token_stream(), media_type="text/plain; charset=utf-8")


@app.get("/api/health")
async def health() -> dict[str, Any]:
    return {"ok": True, "jobs": len(jobs)}


@app.get("/api/history")
async def history() -> dict[str, Any]:
    """Lightweight listing of past analyses for the Swing History view.

    Each entry has just enough for a card: job_id, created_at, a thumbnail
    URL (reusing the window_start keyframe already on disk), a short
    metric summary, and whether a ball-flight shot was paired.
    """
    _ensure_db(_CFG.db_path)
    with _db_connect(_CFG.db_path) as conn:
        rows = conn.execute(
            "SELECT job_id, created_at, shot_id, result_json "
            "FROM analyses ORDER BY id DESC"
        ).fetchall()

    items: list[dict[str, Any]] = []
    for row in rows:
        try:
            result = json.loads(row["result_json"])
        except (json.JSONDecodeError, TypeError):
            # Stored row is corrupt — skip instead of 500'ing the whole list.
            continue
        metrics = result.get("metrics", {}) or {}
        at_top = metrics.get("at_top", {}) or {}
        items.append({
            "job_id": row["job_id"],
            "created_at": row["created_at"],
            "thumbnail": (result.get("artifacts") or {}).get("keyframes", {}).get("window_start"),
            "has_ball_flight": result.get("ball_flight") is not None,
            "shoulder_turn_deg": at_top.get("shoulder_turn_deg"),
            "x_factor_deg": at_top.get("x_factor_deg"),
            "elapsed_seconds": result.get("elapsed_seconds"),
            "faults": ((result.get("coaching") or {}).get("faults") or [])[:2],
        })
    return {"items": items}


def _safe_under_captures(p: Path) -> bool:
    """True if `p` resolves to a path inside the repo's captures/ folder.

    Paths stored in the DB come from us writing to captures/<video>_trim/,
    but we still verify before deleting — cheap insurance against a
    malformed DB row accidentally pointing at something important.
    """
    try:
        p.resolve().relative_to(CAPTURES_DIR.resolve())
        return True
    except ValueError:
        return False


@app.delete("/api/analysis/{job_id}")
async def delete_analysis(job_id: str) -> dict[str, Any]:
    """Remove one analysis from history. Deletes the DB row and, on a
    best-effort basis, the uploaded video and the trim-artifact directory
    so local disk doesn't keep growing forever."""
    import shutil

    _ensure_db(_CFG.db_path)
    with _db_connect(_CFG.db_path) as conn:
        row = conn.execute(
            "SELECT video_path, trim_dir FROM analyses WHERE job_id = ?",
            (job_id,),
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail=f"No saved analysis for {job_id}")

        video_rel = row["video_path"]
        trim_rel = row["trim_dir"]

        conn.execute("DELETE FROM analyses WHERE job_id = ?", (job_id,))
        conn.commit()

    # Drop the in-memory job too (if the just-deleted analysis is still in
    # the current session), so a stale pointer doesn't keep serving it.
    jobs.pop(job_id, None)

    # Best-effort artifact cleanup. Failures here don't undo the DB delete.
    artifacts_removed: list[str] = []
    for rel in filter(None, (video_rel, trim_rel)):
        p = (_REPO_ROOT / rel).resolve() if not Path(rel).is_absolute() else Path(rel).resolve()
        if not _safe_under_captures(p):
            continue
        try:
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
                artifacts_removed.append(str(p))
            elif p.is_file():
                p.unlink(missing_ok=True)
                artifacts_removed.append(str(p))
        except OSError:
            pass

    return {"job_id": job_id, "deleted": True, "artifacts_removed": artifacts_removed}


@app.get("/api/analysis/{job_id}")
async def analysis_by_id(job_id: str) -> dict[str, Any]:
    """Rehydrate a past analysis by job_id. Returns the full result payload
    the user saw when the analysis completed — same shape as the `done`
    WebSocket event's result, so the frontend can feed it straight into
    ResultsView."""
    _ensure_db(_CFG.db_path)
    with _db_connect(_CFG.db_path) as conn:
        row = conn.execute(
            "SELECT result_json FROM analyses WHERE job_id = ?",
            (job_id,),
        ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail=f"No saved analysis for {job_id}")
    try:
        return json.loads(row["result_json"])
    except (json.JSONDecodeError, TypeError) as e:
        raise HTTPException(
            status_code=500,
            detail=f"Saved analysis for {job_id} is corrupt: {e}",
        )


def _persist_analysis(
    *,
    job_id: str,
    video_path: Path,
    trim_dir: Path,
    result: dict[str, Any],
    ball_flight: dict[str, Any] | None,
) -> None:
    """Save one completed analysis to the analyses table for Swing History.

    Paths are stored relative to the repo root so the DB stays portable if
    the project ever moves. The whole result dict is stored as JSON so the
    history detail view can rehydrate the exact payload the user saw.
    """
    _ensure_db(_CFG.db_path)

    # Look up the paired shot's row id so the FK is meaningful. ball_flight
    # as returned by latest_ball_flight() is a subset of columns; the row
    # id has to come from a separate query.
    shot_id: int | None = None
    if ball_flight is not None:
        with _db_connect(_CFG.db_path) as conn:
            row = conn.execute(
                "SELECT id FROM shots WHERE captured_at = ? ORDER BY id DESC LIMIT 1",
                (ball_flight.get("captured_at"),),
            ).fetchone()
            if row is not None:
                shot_id = int(row[0])

    # Make the stored paths repo-root-relative for portability, but fall
    # back to absolute if the artifact sat somewhere unexpected.
    def _rel(p: Path) -> str:
        try:
            return str(p.resolve().relative_to(_REPO_ROOT.resolve()))
        except ValueError:
            return str(p.resolve())

    with _db_connect(_CFG.db_path) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO analyses
              (job_id, shot_id, video_path, trim_dir, result_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                job_id,
                shot_id,
                _rel(video_path),
                _rel(trim_dir),
                json.dumps(result),
            ),
        )
        conn.commit()


def _bridge_state_dict() -> dict[str, Any]:
    """Shape of GET /api/bridge/status — reused by the toggle response."""
    task = _openconnect_task
    error: str | None = None
    if task is not None and task.done():
        # If the task died (port collision, etc.) surface the exception to
        # the UI so the user isn't left wondering why the chip is grey.
        exc = task.exception()
        if exc is not None:
            error = f"{type(exc).__name__}: {exc}"
    return {
        "listening": _bridge_is_running(),
        "host": _CFG.openconnect_host,
        "port": _CFG.openconnect_port,
        "gspro_host": _CFG.openconnect_gspro_host,
        "gspro_port": _CFG.openconnect_gspro_port,
        "ball_shot_max_age_sec": _CFG.ball_shot_max_age_sec,
        "error": error,
    }


@app.get("/api/bridge/status")
async def bridge_status() -> dict[str, Any]:
    return _bridge_state_dict()


@app.post("/api/bridge/toggle")
async def bridge_toggle() -> dict[str, Any]:
    """Flip the bridge between listening and off. Lets the user turn capture
    on/off from the UI without touching .env or restarting the server."""
    if _bridge_is_running():
        await _cancel_bridge()
    else:
        await _launch_bridge()
        # Give start_server a tick to bind; surfaces an immediate error if
        # the port is already in use.
        await asyncio.sleep(0.1)
    return _bridge_state_dict()


# SPA mount must be last so /api/* and /captures/* take precedence over the
# catch-all. html=True makes StaticFiles fall back to index.html for any
# path that doesn't resolve to a file — required for SPA routes.
if _WEB_DIST.exists():
    app.mount("/", StaticFiles(directory=_WEB_DIST, html=True), name="web")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server.main:app", host="0.0.0.0", port=8000, reload=False)
