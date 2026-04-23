"""Render the NLF skeleton overlay on top of video frames.

NLF's FramePose.joints2d gives us (24, 2) image-space joint coordinates in
the same coordinate system as the input frame, which makes drawing the
SMPL skeleton trivial with OpenCV. The output is a cv2-written .mp4 at
the same dimensions and fps as the input — meant to be played back next
to the raw trimmed clip so the user can visually verify pose tracking
before they trust the derived metrics.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from inference.pose_3d import FramePose

# SMPL 24-joint skeleton edges — the body parts of interest for golf. Uses
# the same joint-index convention as pose_3d.SMPL_JOINT_NAMES:
#   0 pelvis / 1 lhip / 2 rhip / 3 spine1 / 4 lknee / 5 rknee / 6 spine2 /
#   7 lankle / 8 rankle / 9 spine3 / 10 lfoot / 11 rfoot / 12 neck /
#   13 lcollar / 14 rcollar / 15 head / 16 lshoulder / 17 rshoulder /
#   18 lelbow / 19 relbow / 20 lwrist / 21 rwrist / 22 lhand / 23 rhand
SMPL_EDGES: tuple[tuple[int, int], ...] = (
    # pelvis connections
    (0, 1), (0, 2), (0, 3),
    # spine chain
    (3, 6), (6, 9), (9, 12), (12, 15),
    # spine3 to collars to shoulders
    (9, 13), (9, 14), (13, 16), (14, 17),
    # arms
    (16, 18), (17, 19), (18, 20), (19, 21), (20, 22), (21, 23),
    # legs
    (1, 4), (2, 5), (4, 7), (5, 8), (7, 10), (8, 11),
)

# BGR colors (OpenCV) chosen to match the UI palette:
#   champagne gold for bones, fairway green for joints, ember red for the
#   root pelvis so it's easy to see the body's center.
_BONE_COLOR = (136, 196, 230)    # BGR of #E6C488 champagne gold
_JOINT_COLOR = (161, 231, 110)   # BGR of #6EE7A1 fairway green
_ROOT_COLOR = (92, 92, 239)      # BGR of #EF5C5C ember red


def _draw_skeleton(frame_bgr: np.ndarray, pose: FramePose) -> None:
    """Draw the SMPL skeleton onto frame_bgr IN-PLACE."""
    if not pose.detected:
        return
    h, w = frame_bgr.shape[:2]
    j2d = pose.joints2d
    # Skip any joint landing outside the frame — NLF occasionally projects
    # occluded joints wildly; clipping them avoids weird long lines.
    inside = (
        (j2d[:, 0] >= 0) & (j2d[:, 0] < w) &
        (j2d[:, 1] >= 0) & (j2d[:, 1] < h)
    )
    pts = j2d.astype(int)

    for a, b in SMPL_EDGES:
        if not (inside[a] and inside[b]):
            continue
        cv2.line(
            frame_bgr,
            (int(pts[a, 0]), int(pts[a, 1])),
            (int(pts[b, 0]), int(pts[b, 1])),
            _BONE_COLOR, 3, cv2.LINE_AA,
        )

    for j, (x, y) in enumerate(pts):
        if not inside[j]:
            continue
        color = _ROOT_COLOR if j == 0 else _JOINT_COLOR
        cv2.circle(frame_bgr, (int(x), int(y)), 5, color, -1, cv2.LINE_AA)


def render_pose_overlay_video(
    bgr_frames: list[np.ndarray],
    poses: list[FramePose],
    out_path: Path,
    fps: float,
) -> Path | None:
    """Write an MP4 with the SMPL skeleton drawn on every frame.

    Both inputs must already be aligned (same length, same frame ordering).
    Returns the output path, or None if the input was empty.
    """
    if not bgr_frames:
        return None
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    h, w = bgr_frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    try:
        for frame, pose in zip(bgr_frames, poses):
            canvas = frame.copy()
            _draw_skeleton(canvas, pose)
            writer.write(canvas)
    finally:
        writer.release()
    return out_path
