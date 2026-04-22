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


def _crop_frames_to_person(
    bgr_frames: list,
    poses: list[FramePose],
    pad_top_frac: float = 0.40,
    pad_side_frac: float = 0.20,
    pad_bottom_frac: float = 0.10,  # keep the ball in frame for Impact detection
) -> list:
    """Crop all frames to a single stable box that contains every detected person
    across the sequence, padded for the club's reach.

    SwingNet was trained on clips pre-cropped to the golfer (per GolfDB paper:
    "extracting the range of frames using the Bbox coordinates to place the
    golfer at the center"). Feeding it the full room drops the golfer to ~30%
    of a 160x160 input where keypoints become indistinguishable — confidences
    collapse below 0.1. A stable crop across the window restores the golfer
    to ~70% of input height, which is what the model expects.

    Padding defaults reflect golf-specific club reach: lots of headroom for
    the top-of-backswing position, modest side room for the swing arc.
    """
    boxes = [p.box_xyxy_conf[:4] for p in poses if p.detected and p.box_xyxy_conf is not None]
    if not boxes or not bgr_frames:
        return bgr_frames

    H, W = bgr_frames[0].shape[:2]
    xs1 = min(b[0] for b in boxes)
    ys1 = min(b[1] for b in boxes)
    xs2 = max(b[2] for b in boxes)
    ys2 = max(b[3] for b in boxes)

    box_w = xs2 - xs1
    box_h = ys2 - ys1
    pad_x = pad_side_frac * box_w
    pad_top = pad_top_frac * box_h
    pad_bot = pad_bottom_frac * box_h

    x1 = max(0, int(xs1 - pad_x))
    y1 = max(0, int(ys1 - pad_top))
    x2 = min(W, int(xs2 + pad_x))
    y2 = min(H, int(ys2 + pad_bot))

    return [f[y1:y2, x1:x2] for f in bgr_frames]


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
    cropped_frames = _crop_frames_to_person(window_frames, window_poses)
    orig_h, orig_w = window_frames[0].shape[:2]
    crop_h, crop_w = cropped_frames[0].shape[:2]
    crop_frac = crop_h / orig_h if orig_h else 1.0

    print(f"\n[4/5] SwingNet event detection on window ({len(cropped_frames)} frames)")
    print(f"  person-centered crop: {orig_w}x{orig_h} -> {crop_w}x{crop_h} ({crop_frac:.0%} of frame height)")
    t0 = time.perf_counter()
    events_rel = detect_events_from_frames(cropped_frames)
    print(f"  done in {time.perf_counter() - t0:.1f}s")
    events = _remap_events(events_rel, offset=window.start)
    for name, frame, conf in zip(events.names, events.frames, events.confidences):
        print(f"    {name:22s}  frame {frame:4d}  conf {conf:.3f}")

    if max(events_rel.confidences) < 0.1:
        print("  WARNING: all SwingNet confidences < 0.1. Even after auto-trim, the")
        print("  model can't find a clean swing. Try a longer recording, a face-on")
        print("  or down-the-line camera angle, or check rotation if not yet set.")

    print("\n[5/5] Biomechanical metrics + Qwen 3 14B coaching")
    # Use auto-trim's impact frame (wrist-velocity peak) as the Impact anchor;
    # it's geometric and reliable even when SwingNet's Impact is weak.
    # Window-relative impact index for compute_metrics:
    impact_hint_rel = window.impact - window.start if not window.used_fallback else None
    m = compute_metrics(
        window_poses,
        events=events_rel,
        handedness=handedness,
        impact_hint=impact_hint_rel,
    )
    payload = metrics_to_coach_dict(m)
    # Tell the user which events we're actually using vs SwingNet's raw output.
    if m.top_frame is not None and events_rel.frames[3] != m.top_frame:
        orig = events_rel.frames[3] + window.start
        ref = m.top_frame + window.start
        print(f"  [refined] Top: SwingNet said {orig}, pose geometry says {ref} ({abs(orig - ref)} frame shift)")
    if m.impact_frame is not None and events_rel.frames[5] != m.impact_frame:
        orig = events_rel.frames[5] + window.start
        ref = m.impact_frame + window.start
        print(f"  [refined] Impact: SwingNet said {orig}, wrist-velocity peak at {ref} ({abs(orig - ref)} frame shift)")
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
