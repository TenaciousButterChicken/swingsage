"""Generate a tiny synthetic test video for fixture use.

Writes tests/fixtures/videos/sample_synthetic.mp4 — 2 seconds, 30 fps, 320x240,
moving colored rectangle. Lets the video-decode pipeline be exercised on Mac
without committing real footage.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = REPO_ROOT / "tests" / "fixtures" / "videos" / "sample_synthetic.mp4"


def main() -> int:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    fps = 30
    duration_s = 2
    width, height = 320, 240
    n_frames = fps * duration_s

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(OUT_PATH), fourcc, fps, (width, height))
    if not writer.isOpened():
        print(f"ERROR: could not open VideoWriter for {OUT_PATH}", file=sys.stderr)
        return 1

    for i in range(n_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        x = int((i / n_frames) * (width - 40))
        cv2.rectangle(frame, (x, 100), (x + 40, 140), (0, 200, 255), -1)
        cv2.putText(frame, f"f{i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        writer.write(frame)

    writer.release()
    print(f"Wrote {OUT_PATH} ({n_frames} frames @ {fps} fps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
