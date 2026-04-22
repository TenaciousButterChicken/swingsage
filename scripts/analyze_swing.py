"""End-to-end SwingSage pipeline: video → SwingNet → NLF → metrics → Qwen 3 coaching.

Usage:
    .venv/Scripts/python.exe scripts/analyze_swing.py path/to/swing.mp4

Skips the LLM step if LM Studio isn't reachable at SWINGSAGE_LLM_API_BASE.
Dumps the full biomechanics payload as JSON so downstream tools (UI, export)
can consume it without re-running the CV pipeline.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# When invoked as `python scripts/analyze_swing.py ...` the repo root isn't on
# sys.path. Fix that before our imports so direct CLI invocation works.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from analytics.joint_angles import compute_metrics, metrics_to_coach_dict  # noqa: E402
from coach.llm import CoachClient  # noqa: E402
from inference.pose_3d import predict_video  # noqa: E402
from inference.swing_events import detect_events  # noqa: E402


def _utf8_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except AttributeError:
            pass


def run(
    video: Path,
    handedness: str = "right",
    skip_llm: bool = False,
    rotation: int | None = None,
) -> int:
    _utf8_stdout()
    rot = rotation  # forwarded to inference modules; None = auto-detect from metadata

    print(f"[1/4] SwingNet event detection on {video}")
    if rot is not None:
        print(f"  (rotation override: {rot}°)")
    t0 = time.perf_counter()
    events = detect_events(video, rotation=rot)
    print(f"  done in {time.perf_counter() - t0:.1f}s")
    for name, frame, conf in zip(events.names, events.frames, events.confidences):
        print(f"    {name:22s}  frame {frame:4d}  conf {conf:.3f}")

    if max(events.confidences) < 0.1:
        print("  WARNING: all event confidences < 0.1 — SwingNet did not")
        print("  recognize this clip. Common causes: sideways phone video")
        print("  (try --rotate 90), single-swing missing, golfer not framed.")

    print("\n[2/4] NLF 3D pose (per-frame)")
    t0 = time.perf_counter()
    frames = predict_video(video, verbose=True, rotation=rot)
    detected = sum(1 for f in frames if f.detected)
    print(f"  done in {time.perf_counter() - t0:.1f}s, {detected}/{len(frames)} detected")

    print("\n[3/4] Biomechanical metrics")
    m = compute_metrics(frames, events=events, handedness=handedness)
    payload = metrics_to_coach_dict(m)
    print(json.dumps(payload, indent=2))

    if skip_llm:
        print("\n[4/4] skipped (--no-llm)")
        return 0

    print("\n[4/4] Qwen 3 14B coaching")
    t0 = time.perf_counter()
    client = CoachClient()
    if not client.is_alive():
        print(f"  LM Studio unreachable at {client.base_url}; skipping coaching step.")
        print("  Start it with:")
        print("    lms server start --port 1234")
        print("    lms load qwen_qwen3-14b --gpu max --context-length 16384 --identifier qwen3-14b -y")
        return 2

    fb = client.coach(payload)
    print(f"  done in {time.perf_counter() - t0:.1f}s")
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
        help="Rotate frames by this many degrees (clockwise). Omit to auto-detect from container metadata.",
    )
    args = parser.parse_args()
    sys.exit(run(
        args.video,
        handedness=args.handedness,
        skip_llm=args.no_llm,
        rotation=args.rotate,
    ))


if __name__ == "__main__":
    main()
