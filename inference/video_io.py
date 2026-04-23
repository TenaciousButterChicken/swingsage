"""Shared video-reading helpers.

Why this exists: phone video frequently ships rotated — iPhones stamp a
rotation flag in the .mov container that OpenCV doesn't always apply on
decode. An unrotated frame is a sideways human, which both SwingNet and
NLF were trained to not recognize. These helpers normalize orientation
before the models ever see the pixels.

Also owns the browser-compatible H.264 writer. OpenCV's built-in
``cv2.VideoWriter`` with fourcc ``mp4v`` produces MPEG-4 Part 2 files
that Chrome/Edge/Safari will not decode natively — playback in a <video>
tag shows a blank frame. We pipe BGR frames to the bundled imageio-ffmpeg
binary instead, producing H.264/yuv420p streams that every browser plays.
"""

from __future__ import annotations

import subprocess
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


def _ffmpeg_exe() -> str:
    """Path to the bundled ffmpeg binary. Raises if imageio-ffmpeg is missing."""
    import imageio_ffmpeg  # noqa: WPS433 — runtime import so video_io stays light
    return imageio_ffmpeg.get_ffmpeg_exe()


def write_browser_mp4(
    bgr_frames: list[np.ndarray],
    out_path: str | Path,
    fps: float,
    crf: int = 20,
) -> Path:
    """Write BGR frames to a browser-playable H.264 MP4 via bundled ffmpeg.

    Uses libx264 + yuv420p (universal browser support), even frame
    dimensions (H.264 requires them), and ``+faststart`` so the moov
    atom sits at the head of the file — lets <video> start streaming
    before the whole file downloads.

    Returns the output path. Raises RuntimeError if ffmpeg exits nonzero.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not bgr_frames:
        return out_path

    h, w = bgr_frames[0].shape[:2]
    # libx264 requires even dimensions when using yuv420p chroma subsampling.
    pad_w, pad_h = w + (w & 1), h + (h & 1)

    cmd = [
        _ffmpeg_exe(),
        "-y",
        "-loglevel", "error",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{w}x{h}",
        "-r", f"{fps:.3f}",
        "-i", "-",
        "-an",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        "-vf", f"pad={pad_w}:{pad_h}:0:0:color=black",
        "-movflags", "+faststart",
        str(out_path),
    ]

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdin is not None
    try:
        for frame in bgr_frames:
            if frame.shape[:2] != (h, w):
                # Defensive: ffmpeg's rawvideo input is fixed-dimension. Resize
                # any stragglers (shouldn't happen in normal pipeline flow).
                frame = cv2.resize(frame, (w, h))
            proc.stdin.write(np.ascontiguousarray(frame).tobytes())
    finally:
        proc.stdin.close()
        _, stderr = proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed ({proc.returncode}): {stderr.decode(errors='replace')}"
        )
    return out_path
