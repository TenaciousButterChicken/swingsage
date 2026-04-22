"""SwingNet (GolfDB) wrapper — 8-event swing sequencing from video.

Wraps `wmcnally/golfdb`'s EventDetector model so the rest of SwingSage doesn't
need to know about the golfdb code layout or its PyTorch 1.x-era quirks.

Public API:
    detect_events(video_path) -> SwingEvents  # 8 frame indices + confidences
    load_model(device=None) -> EventDetector  # reusable if you want to batch
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from capture.config import load_config
from inference.video_io import Rotation, apply_rotation, open_video

EVENT_NAMES: tuple[str, ...] = (
    "Address",
    "Toe-up",
    "Mid-backswing",
    "Top",
    "Mid-downswing",
    "Impact",
    "Mid-follow-through",
    "Finish",
)

_INPUT_SIZE = 160
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_BORDER_BGR = (0.406 * 255, 0.456 * 255, 0.485 * 255)


@dataclass(frozen=True)
class SwingEvents:
    """Frame indices for each of the 8 SwingNet events, plus per-event confidence."""

    frames: tuple[int, ...]
    confidences: tuple[float, ...]
    names: tuple[str, ...] = EVENT_NAMES

    def as_dict(self) -> dict[str, dict[str, float | int]]:
        return {
            name: {"frame": int(f), "confidence": float(c)}
            for name, f, c in zip(self.names, self.frames, self.confidences)
        }


def _ensure_golfdb_on_path(golfdb_dir: Path) -> None:
    s = str(golfdb_dir)
    if s not in sys.path:
        sys.path.insert(0, s)


def load_model(device: torch.device | str | None = None) -> torch.nn.Module:
    """Build EventDetector and load swingnet_1800 weights onto the target device."""
    cfg = load_config()
    _ensure_golfdb_on_path(cfg.golfdb_dir)

    from model import EventDetector  # noqa: E402 — path injected above

    # golfdb/model.py:17 calls `torch.load('mobilenet_v2.pth.tar')` unconditionally
    # before the `if pretrain:` guard, so it ALWAYS needs that file on disk —
    # even when we don't use the result. We patch torch.load to no-op for that
    # exact call, avoiding a pointless 14MB download for weights we overwrite.
    original_load = torch.load

    def _patched_load(f, *a, **kw):
        if isinstance(f, str) and f.endswith("mobilenet_v2.pth.tar"):
            return {}
        return original_load(f, *a, **kw)

    torch.load = _patched_load
    try:
        net = EventDetector(
            pretrain=False,
            width_mult=1.0,
            lstm_layers=1,
            lstm_hidden=256,
            bidirectional=True,
            dropout=False,
        )
    finally:
        torch.load = original_load

    if not cfg.swingnet_weights.exists():
        raise FileNotFoundError(
            f"SwingNet weights not found at {cfg.swingnet_weights}. "
            "Download swingnet_1800.pth.tar from the wmcnally/golfdb README "
            "and place it there, or set SWINGSAGE_SWINGNET_WEIGHTS."
        )

    ckpt = torch.load(cfg.swingnet_weights, map_location="cpu", weights_only=False)
    net.load_state_dict(ckpt["model_state_dict"])

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(device).eval()
    return net


def _preprocess_bgr_frames(bgr_frames: list[np.ndarray]) -> np.ndarray:
    """Letterbox a list of BGR uint8 frames into (T, 3, 160, 160) float32 tensor.

    Input frames must already be rotation-corrected. Uses the first frame's
    dimensions to size the letterbox, so all frames in the list must share
    shape (they will if they came from the same video).
    """
    if not bgr_frames:
        raise ValueError("bgr_frames is empty — nothing to preprocess")

    h, w = bgr_frames[0].shape[:2]
    ratio = _INPUT_SIZE / max(h, w)
    new_h, new_w = int(h * ratio), int(w * ratio)
    pad_w = _INPUT_SIZE - new_w
    pad_h = _INPUT_SIZE - new_h
    top, bottom = pad_h // 2, pad_h - pad_h // 2
    left, right = pad_w // 2, pad_w - pad_w // 2

    out: list[np.ndarray] = []
    for img in bgr_frames:
        resized = cv2.resize(img, (new_w, new_h))
        bordered = cv2.copyMakeBorder(
            resized, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=_BORDER_BGR,
        )
        rgb = cv2.cvtColor(bordered, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        out.append(rgb)

    arr = np.stack(out, axis=0)                         # (T, H, W, 3)
    arr = (arr - _IMAGENET_MEAN) / _IMAGENET_STD
    arr = arr.transpose(0, 3, 1, 2).astype(np.float32)  # (T, 3, H, W)
    return arr


def _read_and_preprocess(
    video_path: Path,
    rotation: Rotation | None = None,
) -> np.ndarray:
    """Read video → letterboxed 160x160 RGB frames. Thin wrapper over _preprocess_bgr_frames."""
    cap, rot = open_video(video_path, rotation=rotation)
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not total:
            raise ValueError(f"Video has zero frames: {video_path}")
        bgr_frames: list[np.ndarray] = []
        for _ in range(total):
            ok, img = cap.read()
            if not ok:
                break
            bgr_frames.append(apply_rotation(img, rot))
    finally:
        cap.release()
    return _preprocess_bgr_frames(bgr_frames)


def _run_swingnet(
    preprocessed: np.ndarray,
    model: torch.nn.Module,
    seq_length: int,
) -> SwingEvents:
    """Run the SwingNet model on a pre-letterboxed (T, 3, 160, 160) tensor."""
    device_t = next(model.parameters()).device
    t = torch.from_numpy(preprocessed).unsqueeze(0)  # (1, T, 3, 160, 160)
    total_frames = t.shape[1]

    probs_chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, total_frames, seq_length):
            end = min(start + seq_length, total_frames)
            chunk = t[:, start:end].to(device_t)
            logits = model(chunk)
            probs_chunks.append(F.softmax(logits, dim=1).cpu().numpy())

    probs = np.concatenate(probs_chunks, axis=0)  # (T, 9)
    event_frames = np.argmax(probs, axis=0)[:-1]   # drop the "no event" class
    confidences = [float(probs[f, i]) for i, f in enumerate(event_frames)]

    return SwingEvents(
        frames=tuple(int(f) for f in event_frames),
        confidences=tuple(confidences),
    )


def detect_events_from_frames(
    bgr_frames: list[np.ndarray],
    model: torch.nn.Module | None = None,
    seq_length: int = 64,
    device: torch.device | str | None = None,
) -> SwingEvents:
    """Run SwingNet on already-decoded, already-rotated BGR frames.

    Use this path when the pipeline has already loaded the video for NLF — it
    lets SwingNet and NLF share one decode. Frame indices in the returned
    `SwingEvents` are relative to the input list.
    """
    if model is None:
        model = load_model(device=device)
    preprocessed = _preprocess_bgr_frames(bgr_frames)
    return _run_swingnet(preprocessed, model, seq_length)


def detect_events(
    video_path: str | Path,
    model: torch.nn.Module | None = None,
    seq_length: int = 64,
    device: torch.device | str | None = None,
    rotation: Rotation | None = None,
) -> SwingEvents:
    """Run SwingNet over a full video and return 8 event-frame indices + confidences.

    The model predicts per-frame probabilities over 9 classes (8 events + "no event").
    Each event's frame is the argmax of its probability column — this is the
    standard GolfDB evaluation protocol.
    """
    video_path = Path(video_path)
    if model is None:
        model = load_model(device=device)
    preprocessed = _read_and_preprocess(video_path, rotation=rotation)  # (T, 3, 160, 160)
    return _run_swingnet(preprocessed, model, seq_length)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SwingNet event detection smoke test")
    parser.add_argument("video", type=Path, help="Path to a swing video")
    args = parser.parse_args()

    events = detect_events(args.video)
    for name, frame, conf in zip(events.names, events.frames, events.confidences):
        print(f"  {name:22s}  frame {frame:4d}  conf {conf:.3f}")
