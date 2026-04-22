"""NLF (Neural Localizer Fields) wrapper — 3D SMPL joints + mesh per frame.

Wraps `isarandi/nlf`'s multi-person TorchScript model so the rest of SwingSage
doesn't need NLF's training-time dependencies (cameralib, posepile, smplfitter,
etc). Runtime is pure torch + torchvision.

The TorchScript model exposes `detect_smpl_batched(frames_uint8_BCHW)` which
bundles detection + pose estimation into one call and returns SMPL parametric
+ nonparametric predictions plus per-joint uncertainties.

Observed perf on RTX 5080 (Blackwell, cu128) at 492×354 input:
  - first call: ~15s (JIT kernel compile / lazy graph realization)
  - steady state: ~150 ms/frame, ~1 GB peak VRAM
  - batching is SLOWER per-frame than sequential (dynamic detection shapes
    prevent batch kernel fusion), so sequential is the sweet spot.

Public API:
    load_model(device=None) -> torch.jit.ScriptModule
    predict_frame(model, rgb_uint8_hwc) -> FramePose
    predict_video(video_path, model=None) -> list[FramePose]
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import torch
import torchvision  # noqa: F401 — MUST be imported before torch.jit.load to register torchvision::nms

from capture.config import load_config
from inference.video_io import Rotation, iter_frames as _iter_bgr_frames

# SMPL joint order used by NLF (24 joints).
SMPL_JOINT_NAMES: tuple[str, ...] = (
    "pelvis",
    "left_hip", "right_hip",
    "spine1",
    "left_knee", "right_knee",
    "spine2",
    "left_ankle", "right_ankle",
    "spine3",
    "left_foot", "right_foot",
    "neck",
    "left_collar", "right_collar",
    "head",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hand", "right_hand",
)


@dataclass(frozen=True)
class FramePose:
    """Single-frame pose prediction for the single best-scored person.

    All positions are in the NLF camera coordinate system (millimeters, Z forward).
    Set `person_idx = -1` if no person was detected.
    """

    frame_idx: int
    person_idx: int
    box_xyxy_conf: tuple[float, float, float, float, float] | None
    joints3d: np.ndarray      # (24, 3) float32 — primary output for biomechanics
    joints2d: np.ndarray      # (24, 2) float32 — projected to image space
    joint_uncertainties: np.ndarray  # (24,) float32
    pose: np.ndarray          # (72,) float32 — SMPL axis-angle, optional downstream use
    betas: np.ndarray         # (10,) float32
    trans: np.ndarray         # (3,) float32

    @property
    def detected(self) -> bool:
        return self.person_idx >= 0


EMPTY_24x3 = np.zeros((24, 3), dtype=np.float32)
EMPTY_24x2 = np.zeros((24, 2), dtype=np.float32)
EMPTY_24 = np.zeros((24,), dtype=np.float32)
EMPTY_72 = np.zeros((72,), dtype=np.float32)
EMPTY_10 = np.zeros((10,), dtype=np.float32)
EMPTY_3 = np.zeros((3,), dtype=np.float32)


def load_model(device: torch.device | str | None = None) -> torch.jit.ScriptModule:
    """Load NLF multi-person TorchScript model onto the target device."""
    cfg = load_config()
    if not cfg.nlf_weights.exists():
        raise FileNotFoundError(
            f"NLF TorchScript weights not found at {cfg.nlf_weights}. "
            f"Download from https://github.com/isarandi/nlf/releases or set "
            f"SWINGSAGE_NLF_WEIGHTS."
        )

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = torch.jit.load(cfg.nlf_weights, map_location=device).eval()
    return model


def _pick_best_person(boxes) -> int:
    """Among detected persons in a frame, return the index of the highest-score box.

    Accepts a tensor of shape (num_people, 5) in (x1, y1, x2, y2, score) format.
    Returns -1 if boxes is None, empty, or otherwise unusable.
    """
    if boxes is None:
        return -1
    try:
        if boxes.numel() == 0:
            return -1
        scores = boxes[:, 4]
        return int(scores.argmax().item())
    except (AttributeError, IndexError):
        return -1


def predict_frame(
    model: torch.jit.ScriptModule,
    rgb_hwc_uint8: np.ndarray,
    frame_idx: int = 0,
) -> FramePose:
    """Run NLF on a single RGB uint8 HWC frame. Returns the best-scored person."""
    device = next(model.parameters()).device
    tensor = (
        torch.from_numpy(rgb_hwc_uint8)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device)
    )
    with torch.inference_mode(), torch.device(str(device)):
        pred = model.detect_smpl_batched(tensor)

    def _empty() -> FramePose:
        return FramePose(
            frame_idx=frame_idx, person_idx=-1, box_xyxy_conf=None,
            joints3d=EMPTY_24x3.copy(), joints2d=EMPTY_24x2.copy(),
            joint_uncertainties=EMPTY_24.copy(), pose=EMPTY_72.copy(),
            betas=EMPTY_10.copy(), trans=EMPTY_3.copy(),
        )

    # NLF only includes "boxes" when ≥1 person was detected in the batch.
    # Graceful fallback keeps the pipeline running over a full video even
    # when some frames (e.g. occluded Top) temporarily lose the golfer.
    boxes_list = pred.get("boxes")
    if not boxes_list:
        return _empty()
    boxes = boxes_list[0]
    idx = _pick_best_person(boxes)
    if idx < 0:
        return _empty()

    box = boxes[idx].detach().cpu().numpy()
    joints3d = pred["joints3d"][0][idx].detach().cpu().numpy().astype(np.float32)
    joints2d = pred["joints2d"][0][idx].detach().cpu().numpy().astype(np.float32)
    uncert = pred["joint_uncertainties"][0][idx].detach().cpu().numpy().astype(np.float32)
    pose = pred["pose"][0][idx].detach().cpu().numpy().astype(np.float32)
    betas = pred["betas"][0][idx].detach().cpu().numpy().astype(np.float32)
    trans = pred["trans"][0][idx].detach().cpu().numpy().astype(np.float32)

    return FramePose(
        frame_idx=frame_idx,
        person_idx=idx,
        box_xyxy_conf=tuple(float(x) for x in box),
        joints3d=joints3d,
        joints2d=joints2d,
        joint_uncertainties=uncert,
        pose=pose,
        betas=betas,
        trans=trans,
    )


def iter_video_frames(
    video_path: str | Path,
    rotation: Rotation | None = None,
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (frame_idx, rgb_uint8_hwc) for every frame, rotation-corrected.

    `rotation` is None → auto-detect from container metadata (iPhone portrait
    videos ship a 90° flag OpenCV otherwise ignores). Pass an explicit value
    (0/90/180/270) to override when metadata is wrong or absent.
    """
    for idx, bgr in _iter_bgr_frames(video_path, rotation=rotation):
        yield idx, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def predict_video(
    video_path: str | Path,
    model: torch.jit.ScriptModule | None = None,
    verbose: bool = False,
    rotation: Rotation | None = None,
) -> list[FramePose]:
    """Run NLF per-frame over a full video. Sequential — batching is slower for this model."""
    if model is None:
        model = load_model()

    out: list[FramePose] = []
    t_start = time.perf_counter()
    for idx, rgb in iter_video_frames(video_path, rotation=rotation):
        out.append(predict_frame(model, rgb, frame_idx=idx))
        if verbose and idx and idx % 25 == 0:
            elapsed = time.perf_counter() - t_start
            print(f"  {idx} frames  ({elapsed/idx*1000:.0f} ms/frame avg)")

    if verbose:
        elapsed = time.perf_counter() - t_start
        print(f"Done: {len(out)} frames in {elapsed:.1f}s ({elapsed/max(len(out),1)*1000:.0f} ms/frame avg)")
    return out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NLF 3D pose smoke test")
    parser.add_argument("video", type=Path)
    parser.add_argument("--max-frames", type=int, default=None, help="Limit frames for quick checks")
    args = parser.parse_args()

    model = load_model()
    print(f"NLF loaded on {next(model.parameters()).device}")

    frames: list[FramePose] = []
    t0 = time.perf_counter()
    for idx, rgb in iter_video_frames(args.video):
        if args.max_frames is not None and idx >= args.max_frames:
            break
        fp = predict_frame(model, rgb, frame_idx=idx)
        frames.append(fp)
        if idx % 25 == 0:
            print(f"  frame {idx}: detected={fp.detected} box_conf={fp.box_xyxy_conf[4] if fp.box_xyxy_conf else 0:.3f}")
    elapsed = time.perf_counter() - t0

    detected = sum(1 for f in frames if f.detected)
    print()
    print(f"Processed {len(frames)} frames in {elapsed:.2f}s ({elapsed/len(frames)*1000:.0f} ms/frame)")
    print(f"Person detected in {detected}/{len(frames)} frames")
    if detected:
        mid = frames[len(frames)//2]
        print(f"Mid-frame ({mid.frame_idx}) sample joints3d (pelvis=0, neck=12, l_wrist=20):")
        for j in (0, 12, 20):
            x, y, z = mid.joints3d[j]
            print(f"  {SMPL_JOINT_NAMES[j]:15s}  ({x:+.2f}, {y:+.2f}, {z:+.2f})  uncert={mid.joint_uncertainties[j]:.3f}")
    print(f"Peak VRAM: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")
