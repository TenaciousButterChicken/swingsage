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
    """Return the per-frame lead-wrist velocity magnitude (mm/frame).

    Handles undetected frames by backfilling from the first real detection
    and then forward-filling — so a user who takes a few seconds to step
    into frame doesn't poison the whole series with zero-velocity artifacts.
    If no frame is ever detected, returns zeros.
    """
    j = _lead_wrist_joint_index(handedness)
    positions = np.full((len(frames), 3), np.nan, dtype=np.float32)
    for i, f in enumerate(frames):
        if f.detected:
            positions[i] = f.joints3d[j]
    if len(positions) < 2:
        return np.zeros((len(positions),), dtype=np.float32)

    # Find the first and last detected frames. Without at least one real
    # detection there's nothing to integrate, so return zeros.
    detected_rows = np.where(~np.isnan(positions[:, 0]))[0]
    if len(detected_rows) == 0:
        return np.zeros((len(positions),), dtype=np.float32)
    first, last = int(detected_rows[0]), int(detected_rows[-1])

    # Backfill: frames before the first detection get the first detection's
    # position. This makes their speed = 0 (no motion yet) which is correct —
    # the golfer hasn't moved their wrist because they weren't in frame yet.
    positions[:first] = positions[first]
    # Forward-fill: frames after the last detection get the last known.
    positions[last + 1:] = positions[last]
    # Fill interior gaps by carrying forward.
    for i in range(first + 1, last + 1):
        if np.isnan(positions[i]).any():
            positions[i] = positions[i - 1]

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


# ~300 ms at 30 fps = 9 frames. Matches the typical downswing duration (Top→Impact
# is ~250 ms for a full swing), so a box convolution over this width rewards
# sustained motion and penalizes 1-2 frame jitter spikes from NLF pose noise.
_SWING_INTEGRATION_FRAMES = 9


def _integrated_speed(series: np.ndarray, fps: float) -> np.ndarray:
    """Sum of |speed| over a ~300 ms rolling window, centered on each frame.

    This is the key trick: a real swing has sustained high wrist motion over
    ~10 frames, while pose-estimation jitter and phone-handling artifacts
    produce 1-2 frame velocity spikes. Summing over a ~300 ms window preserves
    the swing peak while attenuating narrow artifacts.
    """
    width = max(3, int(round(0.3 * fps)))  # ~300 ms worth of frames
    if width % 2 == 0:
        width += 1  # keep odd so the window is symmetric around each frame
    if width >= len(series):
        return series
    # Convolve with a box kernel (uniform weights), 'same' output length.
    kernel = np.ones(width, dtype=np.float32)
    return np.convolve(series, kernel, mode="same")


def find_impact_frame(
    frames: list[FramePose],
    handedness: Handedness = "right",
    fps: float = 30.0,
) -> tuple[int, float]:
    """Return (impact_frame_index, peak_to_median_ratio).

    Uses integrated wrist speed over a ~300 ms window as the scoring signal,
    which filters out the 1-2 frame NLF-jitter spikes that would otherwise
    hijack argmax of raw speed (observed on a 22 s clip where phone-handling
    at t≈1 s produced a higher instantaneous spike than the actual swing).

    The ratio is peak/median of the integrated series — for a real swing,
    typically 5-20x; for a clip with no swing, closer to 1-2x.
    """
    speeds = _lead_wrist_speed_series(frames, handedness)
    if np.all(speeds == 0):
        return 0, 0.0
    smoothed = _smooth(speeds)
    integrated = _integrated_speed(smoothed, fps)
    impact = int(np.argmax(integrated))
    median = float(np.median(integrated))
    peak = float(integrated[impact])
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

    impact, ratio = find_impact_frame(frames, handedness=handedness, fps=fps)

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
