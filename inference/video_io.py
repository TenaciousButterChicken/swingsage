"""Shared video-reading helpers.

Why this exists: phone video frequently ships rotated — iPhones stamp a
rotation flag in the .mov container that OpenCV doesn't always apply on
decode. An unrotated frame is a sideways human, which both SwingNet and
NLF were trained to not recognize. These helpers normalize orientation
before the models ever see the pixels.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Literal

import cv2
import numpy as np

Rotation = Literal[0, 90, 180, 270]

_ROTATION_CV2 = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


def detect_rotation(cap: cv2.VideoCapture) -> Rotation:
    """Best-effort rotation from container metadata. Returns one of 0/90/180/270.

    OpenCV 4.7+ exposes CAP_PROP_ORIENTATION_META. Values other than these
    four quadrants are clamped to the nearest multiple of 90. Older OpenCV
    builds or containers without metadata return 0.
    """
    try:
        raw = cap.get(cv2.CAP_PROP_ORIENTATION_META)
    except Exception:
        return 0
    if raw is None or raw != raw:  # NaN
        return 0
    # iOS stores rotations in 90° increments; round defensively.
    deg = int(round(float(raw) / 90.0)) * 90 % 360
    return deg if deg in (0, 90, 180, 270) else 0  # type: ignore[return-value]


def apply_rotation(frame: np.ndarray, rotation: Rotation) -> np.ndarray:
    if rotation == 0:
        return frame
    return cv2.rotate(frame, _ROTATION_CV2[rotation])


def open_video(
    path: str | Path,
    rotation: Rotation | None = None,
) -> tuple[cv2.VideoCapture, Rotation]:
    """Open a video and return (capture, effective_rotation).

    If `rotation` is None, reads container metadata; an explicit override
    (e.g. `--rotate 90` on the CLI) always wins. Raises FileNotFoundError
    when OpenCV can't open the file.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")
    effective: Rotation = rotation if rotation is not None else detect_rotation(cap)
    return cap, effective


def iter_frames(
    path: str | Path,
    rotation: Rotation | None = None,
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (frame_idx, bgr_frame_uint8) for every frame, rotation-corrected."""
    cap, rot = open_video(path, rotation=rotation)
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield idx, apply_rotation(frame, rot)
            idx += 1
    finally:
        cap.release()
