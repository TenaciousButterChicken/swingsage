"""End-to-end SwingSage pipeline: video → NLF → auto-trim → SwingNet → metrics → Qwen.

Usage:
    .venv/Scripts/python.exe scripts/analyze_swing.py path/to/swing.mp4

Auto-trim is ON by default. The pipeline runs NLF on the full clip, locates
impact via the lead-wrist velocity peak, then windows SwingNet + metrics to
[impact − 2s, impact + 1.5s]. Pass --no-auto-trim to analyze the full clip
verbatim. Skips the LLM step if LM Studio isn't reachable.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

# When invoked as `python scripts/analyze_swing.py ...` the repo root isn't on
# sys.path. Fix that before our imports so direct CLI invocation works.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import cv2  # noqa: E402

from analytics.auto_trim import TrimWindow, detect_swing_window  # noqa: E402
from analytics.joint_angles import compute_metrics, metrics_to_coach_dict  # noqa: E402
from coach.llm import CoachClient  # noqa: E402
from inference.pose_3d import FramePose, predict_from_frames  # noqa: E402
from inference.swing_events import SwingEvents, detect_events_from_frames  # noqa: E402
from inference.video_io import Rotation, apply_rotation, open_video  # noqa: E402


def _utf8_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except AttributeError:
            pass


def _decode_all_frames(video_path: Path, rotation: Rotation | None) -> tuple[list, float]:
    """Return (list of BGR uint8 frames, fps). Applies rotation during decode."""
    cap, rot = open_video(video_path, rotation=rotation)
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    frames = []
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            frames.append(apply_rotation(bgr, rot))
    finally:
        cap.release()
    return frames, fps


def _remap_events(events: SwingEvents, offset: int) -> SwingEvents:
    """Shift event frame indices from window-relative to original-video timeline."""
    if offset == 0:
        return events
    shifted = tuple(f + offset for f in events.frames)
    return replace(events, frames=shifted)


def _remap_payload_frames(payload: dict, offset: int) -> dict:
    """Shift all frame indices in the coach payload from window-relative to original."""
    if offset == 0:
        return payload
    ev = payload.get("events", {})
    for k in ("address_frame", "top_frame", "impact_frame"):
        if ev.get(k) is not None:
            ev[k] = int(ev[k]) + offset
    peaks = payload.get("kinematic_sequence", {}).get("peak_velocity_frames", {})
    for k, v in list(peaks.items()):
        if v is not None and v >= 0:
            peaks[k] = int(v) + offset
    return payload


def run(
    video: Path,
    handedness: str = "right",
    skip_llm: bool = False,
    rotation: int | None = None,
    auto_trim: bool = True,
) -> int:
    _utf8_stdout()
    rot = rotation  # None = auto-detect from metadata

    print(f"[1/5] Decoding {video}")
    if rot is not None:
        print(f"  (rotation override: {rot}°)")
    t0 = time.perf_counter()
    all_frames, fps = _decode_all_frames(video, rot)
    print(f"  done in {time.perf_counter() - t0:.1f}s: {len(all_frames)} frames @ {fps:.1f} fps")
    if not all_frames:
        print("  ERROR: video decoded to zero frames.")
        return 2

    print("\n[2/5] NLF 3D pose (full clip)")
    t0 = time.perf_counter()
    all_poses: list[FramePose] = predict_from_frames(all_frames, verbose=True)
    detected = sum(1 for f in all_poses if f.detected)
    print(f"  done in {time.perf_counter() - t0:.1f}s, {detected}/{len(all_poses)} detected")

    if auto_trim:
        print("\n[3/5] Auto-trim")
        window: TrimWindow = detect_swing_window(all_poses, fps=fps, handedness=handedness)
        if window.used_fallback:
            print(
                f"  confidence={window.confidence:.1f}x (< threshold) — using full clip. "
                f"Likely no real swing in this video, or pose detection noisy."
            )
        else:
            print(
                f"  impact={window.impact}  window=[{window.start}, {window.end})  "
                f"confidence={window.confidence:.1f}x  ({(window.end - window.start) / fps:.1f}s)"
            )
    else:
        print("\n[3/5] Auto-trim skipped (--no-auto-trim)")
        window = TrimWindow(
            start=0, end=len(all_frames), impact=0,
            confidence=0.0, fps=fps, used_fallback=True,
        )

    window_frames = all_frames[window.start:window.end]
    window_poses = all_poses[window.start:window.end]

    print(f"\n[4/5] SwingNet event detection on window ({len(window_frames)} frames)")
    t0 = time.perf_counter()
    events_rel = detect_events_from_frames(window_frames)
    print(f"  done in {time.perf_counter() - t0:.1f}s")
    events = _remap_events(events_rel, offset=window.start)
    for name, frame, conf in zip(events.names, events.frames, events.confidences):
        print(f"    {name:22s}  frame {frame:4d}  conf {conf:.3f}")

    if max(events_rel.confidences) < 0.1:
        print("  WARNING: all SwingNet confidences < 0.1. Even after auto-trim, the")
        print("  model can't find a clean swing. Try a longer recording, a face-on")
        print("  or down-the-line camera angle, or check rotation if not yet set.")

    print("\n[5/5] Biomechanical metrics + Qwen 3 14B coaching")
    # Metrics computed on the windowed pose slice with windowed events.
    m = compute_metrics(window_poses, events=events_rel, handedness=handedness)
    payload = metrics_to_coach_dict(m)
    payload = _remap_payload_frames(payload, offset=window.start)
    print(json.dumps(payload, indent=2))

    if skip_llm:
        print("\nLLM skipped (--no-llm)")
        return 0

    t0 = time.perf_counter()
    client = CoachClient()
    if not client.is_alive():
        print(f"\n  LM Studio unreachable at {client.base_url}; skipping coaching step.")
        print("  Start it with:")
        print("    lms server start --port 1234")
        print("    lms load qwen_qwen3-14b --gpu max --context-length 16384 --identifier qwen3-14b -y")
        return 2

    fb = client.coach(payload)
    print(f"\n  Qwen: done in {time.perf_counter() - t0:.1f}s")
    print(f"  model: {client.model}")
    print(f"  confidence: {fb.confidence}")
    print(f"  faults: {fb.faults}")
    print(f"  diagnosis: {fb.diagnosis}")
    print("  drills:")
    for d in fb.drills:
        print(f"    - {d.get('name', '?')}: {d.get('why', '')}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("video", type=Path)
    parser.add_argument("--handedness", choices=("right", "left"), default="right")
    parser.add_argument("--no-llm", action="store_true", help="Skip the coaching step")
    parser.add_argument(
        "--rotate",
        type=int,
        choices=(0, 90, 180, 270),
        default=None,
        help="Rotate frames clockwise by this many degrees. Omit to auto-detect from container metadata.",
    )
    parser.add_argument(
        "--no-auto-trim",
        dest="auto_trim",
        action="store_false",
        help="Analyze the full clip verbatim instead of auto-locating the swing window.",
    )
    args = parser.parse_args()
    sys.exit(run(
        args.video,
        handedness=args.handedness,
        skip_llm=args.no_llm,
        rotation=args.rotate,
        auto_trim=args.auto_trim,
    ))


if __name__ == "__main__":
    main()
