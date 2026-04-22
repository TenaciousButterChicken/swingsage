"""Auto-trim: locate the swing window in a longer video via NLF wrist velocity.

The user-facing problem we're solving:
SwingNet was trained on clips manually trimmed to include a stationary address
hold, one swing, and a stationary finish hold. If the user records a longer
clip or skips the stationary bookends, SwingNet's event sequencer produces
nonsense (see PR #1 discussion and captures/swing1.mov for a real example).

Signal: the lead wrist is by far the fastest body landmark at impact (~100 mph),
so `argmax` of its smoothed velocity is a reliable impact detector that doesn't
depend on SwingNet being right. Once impact is located, we window to
[impact - pre, impact + post] and feed only that window to SwingNet.

Prior art:
- Swing Profile (commercial app) advertises a patented "2-second vital swing
  motion" auto-trim. Algorithm undisclosed, but the 2-second number is a
  sanity check on our window size.
- Wrist-IMU papers (e.g. arxiv:2506.17505) converge on peak wrist acceleration
  as the impact signal. We use velocity instead because NLF gives us position,
  and integrating once (position → velocity) is lower-noise than twice
  (position → acceleration).
- Savitzky-Golay smoothing (scipy.signal.savgol_filter) preserves narrow peaks
  better than Gaussian. Impact is a narrow peak; Savitzky-Golay is the right
  tool.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.signal import savgol_filter

from inference.pose_3d import SMPL_JOINT_NAMES, FramePose

Handedness = Literal["right", "left"]

_J: dict[str, int] = {name: i for i, name in enumerate(SMPL_JOINT_NAMES)}

# Window defaults: 2.0 s before impact + 1.5 s after. At 30 fps that's 105
# frames, similar to the shorter GolfDB training clips and wide enough to
# cover address + backswing + impact + follow-through + finish hold.
DEFAULT_PRE_SEC: float = 2.0
DEFAULT_POST_SEC: float = 1.5

# Confidence threshold: if the peak wrist speed is less than this multiple of
# the median, the clip probably doesn't contain a real swing. Falls back to
# full-video analysis with a warning rather than trimming to noise.
MIN_PEAK_RATIO: float = 1.8


@dataclass(frozen=True)
class TrimWindow:
    """The detected swing window in the original video's frame indices.

    `start` is inclusive, `end` is exclusive, matching Python slice semantics.
    `impact` is the detected impact frame, also in the original timeline.
    `confidence` is peak/median of the smoothed wrist speed; ≥ MIN_PEAK_RATIO
    means we trusted the detection and trimmed, otherwise `start=0, end=total`
    and the caller should treat this as "auto-trim didn't find a swing".
    """

    start: int
    end: int
    impact: int
    confidence: float
    fps: float
    used_fallback: bool  # True when we couldn't detect and kept the full video


def _lead_wrist_joint_index(handedness: Handedness) -> int:
    """SMPL joint index for the lead wrist (left for RH, right for LH)."""
    return _J["left_wrist" if handedness == "right" else "right_wrist"]


def _lead_wrist_speed_series(
    frames: list[FramePose],
    handedness: Handedness,
) -> np.ndarray:
    """Return the per-frame lead-wrist velocity magnitude (mm/frame)."""
    j = _lead_wrist_joint_index(handedness)
    positions = np.full((len(frames), 3), np.nan, dtype=np.float32)
    for i, f in enumerate(frames):
        if f.detected:
            positions[i] = f.joints3d[j]
    if len(positions) < 2:
        return np.zeros((len(positions),), dtype=np.float32)

    # Forward-fill NaN positions so the first-difference doesn't blow up on
    # single-frame detection drops. Any isolated gaps get their last known
    # position; consecutive gaps keep the last-known value until a new
    # detection lands.
    for i in range(1, len(positions)):
        if np.isnan(positions[i]).any():
            positions[i] = positions[i - 1]
    if np.isnan(positions[0]).any():
        # Nothing detected at all — return zeros.
        return np.zeros((len(positions),), dtype=np.float32)

    deltas = np.diff(positions, axis=0)
    speeds = np.linalg.norm(deltas, axis=-1).astype(np.float32)
    # Pad tail so series length matches frame count.
    return np.concatenate([speeds, speeds[-1:]], axis=0)


def _smooth(series: np.ndarray) -> np.ndarray:
    """Savitzky-Golay smoothing, auto-sizing for short clips."""
    n = len(series)
    if n < 5:
        return series
    # window_length must be odd and ≤ len(series); polyorder must be < window_length.
    target = min(7, max(5, n // 3))
    if target % 2 == 0:
        target -= 1
    if target < 5:
        return series
    return savgol_filter(series, window_length=target, polyorder=2).astype(np.float32)


def find_impact_frame(
    frames: list[FramePose],
    handedness: Handedness = "right",
) -> tuple[int, float]:
    """Return (impact_frame_index, peak_to_median_ratio).

    The ratio is a confidence proxy: for a real swing in a stationary clip,
    peak speed dwarfs the median (>5x is typical). For a clip with no swing
    or with lots of non-swing motion, the ratio collapses toward 1.
    """
    speeds = _lead_wrist_speed_series(frames, handedness)
    if np.all(speeds == 0):
        return 0, 0.0
    smoothed = _smooth(speeds)
    impact = int(np.argmax(smoothed))
    median = float(np.median(smoothed))
    peak = float(smoothed[impact])
    ratio = peak / median if median > 1e-6 else float("inf")
    return impact, ratio


def compute_window(
    impact_frame: int,
    total_frames: int,
    fps: float,
    pre_sec: float = DEFAULT_PRE_SEC,
    post_sec: float = DEFAULT_POST_SEC,
) -> tuple[int, int]:
    """Return (start, end) clamped to [0, total_frames). End is exclusive."""
    pre = int(round(pre_sec * fps))
    post = int(round(post_sec * fps))
    start = max(0, impact_frame - pre)
    end = min(total_frames, impact_frame + post + 1)  # +1 to include impact+post
    return start, end


def detect_swing_window(
    frames: list[FramePose],
    fps: float,
    handedness: Handedness = "right",
    pre_sec: float = DEFAULT_PRE_SEC,
    post_sec: float = DEFAULT_POST_SEC,
    min_peak_ratio: float = MIN_PEAK_RATIO,
) -> TrimWindow:
    """One-call helper: find impact, check confidence, compute window, return TrimWindow."""
    total = len(frames)
    if total == 0:
        return TrimWindow(0, 0, 0, 0.0, fps, used_fallback=True)

    impact, ratio = find_impact_frame(frames, handedness=handedness)

    if ratio < min_peak_ratio:
        # Not confident — keep the whole clip and flag the fallback.
        return TrimWindow(
            start=0, end=total, impact=impact, confidence=ratio,
            fps=fps, used_fallback=True,
        )

    start, end = compute_window(impact, total, fps, pre_sec=pre_sec, post_sec=post_sec)
    return TrimWindow(
        start=start, end=end, impact=impact, confidence=ratio,
        fps=fps, used_fallback=False,
    )
