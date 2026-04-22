"""Biomechanical analytics from NLF 3D joint positions.

Computes the golf-specific metrics a coach actually talks about:
  - Pelvis rotation, chest rotation, and X-factor (chest minus pelvis)
  - Spine tilt (forward lean and side bend) from vertical
  - Lead-arm abduction (angle between trunk and lead arm)
  - Lead-arm flex (elbow angle)
  - Kinematic sequence timing (peak angular-velocity order across pelvis →
    chest → lead arm → lead wrist — the single most important biomechanical
    signature in golf)

Architecture (updated): all "at_top" and "at_address" values are computed as
extrema or medians over swing phases, NOT sampled at single event frames.
This is the pattern established by the MIT thesis on MeTRAbs-based golf
biomechanics (Taylor 2025) and multiple MediaPipe-based open-source
analyzers — extrema are insensitive to event-localization error. The only
anchor we need is the auto-trim impact frame (from lead-wrist velocity peak,
unambiguous physics).

Design note: we do NOT implement full ISB joint coordinate systems (too heavy
for Phase 2 MVP — needs per-joint reference frames with 3 axes each). Instead
we use simple geometric operations on 3D joint positions in the NLF camera
frame, which is sufficient for coaching-grade feedback.

Inputs: list[FramePose] from inference.pose_3d, plus the impact frame index
(window-relative) and fps from analytics.auto_trim.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.signal import savgol_filter

from inference.pose_3d import SMPL_JOINT_NAMES, FramePose

_J = {name: idx for idx, name in enumerate(SMPL_JOINT_NAMES)}

# NLF outputs in mm, Y-down camera frame. For angle math we use the body
# frame reconstructed from the golfer's own hip/shoulder geometry, so the
# camera's exact Y convention only affects the sign of the gravity axis.
_MM_PER_M = 1000.0
Handedness = Literal["right", "left"]

# Backswing takes ~0.8-1.2 s at normal tempo, so "more than 1.5 s before
# impact" is safely pre-takeaway even for slow backswings. Used to define
# the Setup phase over which address-pose medians are computed.
_SETUP_LEAD_SECONDS = 1.5

# Savitzky-Golay smoothing window for NLF joint time-series. 7 frames is
# ~0.23 s at 30 fps — short enough to preserve impact spikes but long
# enough to kill per-frame NLF jitter (~5-20 mm range).
_SMOOTH_WINDOW = 7


@dataclass(frozen=True)
class SwingMetrics:
    """Per-frame biomechanical series + extrema/median summary values.

    The summary values use extrema over swing phases (Setup / Backswing /
    Downswing) rather than single-frame samples, so they're robust to the
    event-localization errors that plagued the old frame-anchored version.
    """

    frame_count: int
    fps: float

    # Per-frame series (shape: (frame_count,))
    pelvis_rotation_deg: np.ndarray       # yaw of hip line vs. baseline; signed (backswing negative for RH golfer)
    chest_rotation_deg: np.ndarray        # yaw of shoulder line vs. baseline; signed
    x_factor_deg: np.ndarray              # chest minus pelvis, signed
    spine_tilt_forward_deg: np.ndarray    # pitch: positive = bent forward (toward ball)
    spine_tilt_side_deg: np.ndarray       # roll: side bend relative to vertical
    lead_arm_abduction_deg: np.ndarray    # angle at shoulder; ~90° = arm horizontal
    lead_arm_flex_deg: np.ndarray         # angle at elbow; 180 = straight arm
    lead_wrist_speed: np.ndarray          # magnitude of per-frame velocity, mm/frame

    # Phase boundaries (window-relative frame indices)
    impact_frame: int
    peak_shoulder_frame: int      # frame of max |chest_rot| during backswing — coupling frame for "at top" metrics
    setup_end_frame: int          # last frame of the Setup phase (inclusive)

    # Summary values (all computed as extrema/medians, never single-frame samples)
    x_factor_at_top_deg: float | None
    shoulder_turn_at_top_deg: float | None
    hip_turn_at_top_deg: float | None
    spine_tilt_forward_at_address_deg: float | None
    spine_tilt_side_at_address_deg: float | None
    lead_arm_abduction_at_top_deg: float | None
    lead_arm_flex_at_top_deg: float | None

    # Kinematic sequence: frame index of peak angular velocity, per segment.
    # A mechanically sound swing peaks in order: pelvis → chest → lead_arm → lead_wrist.
    peak_velocity_frame: dict[str, int]
    kinematic_sequence_correct: bool


# ─── Geometry helpers ────────────────────────────────────────────────


def _unit(v: np.ndarray, axis: int = -1, eps: float = 1e-8) -> np.ndarray:
    norm = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / np.maximum(norm, eps)


def _signed_angle_between(
    v: np.ndarray, ref: np.ndarray, normal: np.ndarray
) -> float:
    """Signed angle (degrees) from `ref` to `v`, measured around `normal`."""
    v_u = _unit(v)
    r_u = _unit(ref)
    cos = float(np.clip(np.dot(v_u, r_u), -1.0, 1.0))
    sin = float(np.dot(normal, np.cross(r_u, v_u)))
    return float(np.degrees(np.arctan2(sin, cos)))


def _unsigned_angle_between(v: np.ndarray, ref: np.ndarray) -> float:
    cos = float(np.clip(np.dot(_unit(v), _unit(ref)), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos)))


def _joints_array(frames: list[FramePose]) -> np.ndarray:
    """Stack per-frame joints into (T, 24, 3). Undetected frames get NaN."""
    out = np.full((len(frames), 24, 3), np.nan, dtype=np.float32)
    for i, f in enumerate(frames):
        if f.detected:
            out[i] = f.joints3d
    return out


def _smooth_joints(joints: np.ndarray) -> np.ndarray:
    """Savitzky-Golay-smooth the (T, 24, 3) joint time-series along the time axis.

    NaN-filled undetected frames get forward/backward-filled before smoothing
    so the filter doesn't spread NaN. The filled-in values are then restored
    to NaN in the output to preserve detected/undetected information for the
    downstream angle math.
    """
    T = joints.shape[0]
    if T < _SMOOTH_WINDOW:
        return joints  # Too few frames to smooth meaningfully.

    nan_mask = np.isnan(joints[:, 0, 0])  # (T,) — NaN-ness of any joint implies all are NaN
    detected = np.where(~nan_mask)[0]
    if len(detected) < _SMOOTH_WINDOW:
        return joints

    first, last = int(detected[0]), int(detected[-1])
    filled = joints.copy()

    # Back-fill leading undetected frames with first detection.
    filled[:first] = joints[first]
    # Forward-fill trailing undetected frames.
    filled[last + 1:] = joints[last]
    # Carry-forward interior gaps.
    for i in range(first + 1, last + 1):
        if nan_mask[i]:
            filled[i] = filled[i - 1]

    # Smooth each joint axis separately along time (axis=0).
    smoothed = savgol_filter(filled, window_length=_SMOOTH_WINDOW, polyorder=2, axis=0)

    # Re-mask originally-undetected frames as NaN so downstream code treats
    # them as missing (smoothed edge values there aren't meaningful anyway).
    smoothed[nan_mask] = np.nan
    return smoothed.astype(np.float32)


def _project_onto_horizontal_plane(v: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Remove the component of `v` along `up` — projects into the horizontal plane."""
    return v - np.dot(v, up) * up


def _velocity_magnitude(series: np.ndarray) -> np.ndarray:
    """First-difference velocity magnitude per frame. Pads the last value."""
    if len(series) < 2:
        return np.zeros((len(series),), dtype=np.float32)
    d = np.diff(series, axis=0)
    mag = np.linalg.norm(d, axis=-1)
    return np.concatenate([mag, mag[-1:]], axis=0).astype(np.float32)


def _angular_speed(angle_series_deg: np.ndarray) -> np.ndarray:
    """|dθ/dt| in deg/frame. Pads the last value to keep length."""
    if len(angle_series_deg) < 2:
        return np.zeros_like(angle_series_deg)
    d = np.abs(np.diff(angle_series_deg))
    return np.concatenate([d, d[-1:]], axis=0).astype(np.float32)


def _peak_frame(series: np.ndarray) -> int:
    """Index of the max value; returns -1 if all NaN."""
    if np.all(np.isnan(series)):
        return -1
    s = np.nan_to_num(series, nan=-np.inf)
    return int(np.argmax(s))


def _gravity_axis() -> np.ndarray:
    """World vertical in NLF's camera frame. NLF outputs X=right, Y=down, Z=forward,
    so world-up is always -Y regardless of golfer posture."""
    return np.array([0.0, -1.0, 0.0], dtype=np.float32)


# ─── Core computation ───────────────────────────────────────────────


def compute_metrics(
    frames: list[FramePose],
    impact_frame: int,
    fps: float,
    handedness: Handedness = "right",
) -> SwingMetrics:
    """Run the full biomechanical analysis. See module docstring for scope.

    Args:
        frames: list of per-frame NLF pose predictions (window-relative).
        impact_frame: window-relative frame index of ball impact, from
            auto-trim's wrist-velocity peak. Used as the upper bound of
            the Backswing phase and the anchor for the Setup phase.
        fps: source video frame rate. Used to convert _SETUP_LEAD_SECONDS
            into a frame count for the Setup phase boundary.
        handedness: "right" (lead side = left) or "left" (lead side = right).
    """
    if not frames:
        raise ValueError("No frames supplied")
    if impact_frame < 0 or impact_frame >= len(frames):
        raise ValueError(f"impact_frame {impact_frame} out of range [0, {len(frames)})")

    joints_raw = _joints_array(frames)              # (T, 24, 3), NaN where undetected
    joints = _smooth_joints(joints_raw)             # Savitzky-Golay-smoothed along time
    T = joints.shape[0]

    # Lead side depends on handedness. Right-handed golfer's "lead" side is LEFT.
    lead_is_left = handedness == "right"
    j_lead_hip = _J["left_hip" if lead_is_left else "right_hip"]
    j_trail_hip = _J["right_hip" if lead_is_left else "left_hip"]
    j_lead_sh = _J["left_shoulder" if lead_is_left else "right_shoulder"]
    j_trail_sh = _J["right_shoulder" if lead_is_left else "left_shoulder"]
    j_lead_elbow = _J["left_elbow" if lead_is_left else "right_elbow"]
    j_lead_wrist = _J["left_wrist" if lead_is_left else "right_wrist"]
    j_neck = _J["neck"]
    j_pelvis = _J["pelvis"]

    up = _gravity_axis()

    # Body-frame axes: use the first reliably-detected frame as the hip-line
    # reference. All rotations are measured relative to this baseline.
    detected_idx = np.where(~np.isnan(joints[:, j_pelvis, 0]))[0]
    if len(detected_idx) == 0:
        raise ValueError("No detected frames — cannot compute metrics")
    anchor = int(detected_idx[0])

    def _plane_vec(v):
        return _project_onto_horizontal_plane(v, up)

    hip_vec0 = _plane_vec(joints[anchor, j_lead_hip] - joints[anchor, j_trail_hip])
    hip_ref = _unit(hip_vec0)
    target_normal = _unit(np.cross(up, hip_ref))

    # ── Per-frame series ─────────────────────────────────────────────
    pelvis_rot = np.full((T,), np.nan, dtype=np.float32)
    chest_rot = np.full((T,), np.nan, dtype=np.float32)
    spine_fwd = np.full((T,), np.nan, dtype=np.float32)
    spine_side = np.full((T,), np.nan, dtype=np.float32)
    lead_abd = np.full((T,), np.nan, dtype=np.float32)
    lead_flex = np.full((T,), np.nan, dtype=np.float32)

    for t in range(T):
        if np.isnan(joints[t, j_pelvis, 0]):
            continue
        hip_vec = _plane_vec(joints[t, j_lead_hip] - joints[t, j_trail_hip])
        sh_vec = _plane_vec(joints[t, j_lead_sh] - joints[t, j_trail_sh])
        if np.linalg.norm(hip_vec) > 1e-3:
            pelvis_rot[t] = _signed_angle_between(hip_vec, hip_ref, up)
        if np.linalg.norm(sh_vec) > 1e-3:
            chest_rot[t] = _signed_angle_between(sh_vec, hip_ref, up)

        spine = joints[t, j_neck] - joints[t, j_pelvis]
        spine_u = _unit(spine)
        spine_sag = spine_u - np.dot(spine_u, hip_ref) * hip_ref
        if np.linalg.norm(spine_sag) > 1e-3:
            spine_fwd[t] = _signed_angle_between(_unit(spine_sag), up, hip_ref)
        spine_fr = spine_u - np.dot(spine_u, target_normal) * target_normal
        if np.linalg.norm(spine_fr) > 1e-3:
            spine_side[t] = _signed_angle_between(_unit(spine_fr), up, target_normal)

        trunk_down = -spine_u
        upper_arm = joints[t, j_lead_elbow] - joints[t, j_lead_sh]
        if np.linalg.norm(upper_arm) > 1e-3:
            lead_abd[t] = _unsigned_angle_between(upper_arm, trunk_down)

        forearm = joints[t, j_lead_wrist] - joints[t, j_lead_elbow]
        if np.linalg.norm(forearm) > 1e-3:
            lead_flex[t] = 180.0 - _unsigned_angle_between(upper_arm, forearm)

    x_factor = chest_rot - pelvis_rot
    lead_wrist_pos = joints[:, j_lead_wrist]
    wrist_speed = _velocity_magnitude(lead_wrist_pos)

    # ── Phase boundaries ─────────────────────────────────────────────
    # Backswing phase = [0, impact_frame]. The peak of |chest_rot| in this
    # range is the geometric top-of-backswing — the frame where the at-top
    # metrics are coupled (shoulder/hip/x-factor/abduction all use this frame).
    backswing_end = int(impact_frame) + 1
    chest_abs_backswing = np.abs(np.where(np.isnan(chest_rot[:backswing_end]),
                                          -np.inf, chest_rot[:backswing_end]))
    if np.all(np.isneginf(chest_abs_backswing)):
        peak_shoulder_frame = anchor  # fallback — no valid rotation data
    else:
        peak_shoulder_frame = int(np.argmax(chest_abs_backswing))

    # Setup phase = everything more than 1.5 s before impact. Medians over
    # this phase are stable even when the golfer waggles throughout setup.
    setup_end = max(1, int(impact_frame) - int(round(_SETUP_LEAD_SECONDS * fps)))
    setup_end = min(setup_end, T)

    # ── Extrema / medians ───────────────────────────────────────────
    def _val_at(series: np.ndarray, idx: int) -> float | None:
        if idx < 0 or idx >= len(series):
            return None
        v = series[idx]
        return None if np.isnan(v) else float(v)

    def _nanmedian(series: np.ndarray) -> float | None:
        if len(series) == 0 or np.all(np.isnan(series)):
            return None
        return float(np.nanmedian(series))

    def _nanmax(series: np.ndarray) -> float | None:
        if len(series) == 0 or np.all(np.isnan(series)):
            return None
        return float(np.nanmax(series))

    shoulder_turn_at_top = (
        abs(chest_rot[peak_shoulder_frame])
        if not np.isnan(chest_rot[peak_shoulder_frame])
        else None
    )
    hip_turn_at_top = (
        abs(pelvis_rot[peak_shoulder_frame])
        if not np.isnan(pelvis_rot[peak_shoulder_frame])
        else None
    )
    x_factor_at_top = (
        shoulder_turn_at_top - hip_turn_at_top
        if shoulder_turn_at_top is not None and hip_turn_at_top is not None
        else None
    )
    lead_arm_abduction_at_top = _val_at(lead_abd, peak_shoulder_frame)
    lead_arm_flex_at_top = _nanmax(lead_flex[:backswing_end])

    spine_tilt_forward_at_address = _nanmedian(spine_fwd[:setup_end])
    spine_tilt_side_at_address = _nanmedian(spine_side[:setup_end])

    # ── Kinematic sequence (uses peak_shoulder_frame as Top) ────────
    if peak_shoulder_frame < int(impact_frame):
        lo, hi = peak_shoulder_frame, int(impact_frame) + 1
    else:
        lo, hi = 0, T

    def _peak_in_window(series: np.ndarray) -> int:
        sub = series[lo:hi]
        rel = _peak_frame(sub)
        return -1 if rel < 0 else lo + rel

    pelvis_omega = _angular_speed(pelvis_rot)
    chest_omega = _angular_speed(chest_rot)
    lead_arm_linear = _velocity_magnitude(joints[:, j_lead_sh])
    lead_wrist_linear = wrist_speed

    peak_frames = {
        "pelvis": _peak_in_window(pelvis_omega),
        "chest": _peak_in_window(chest_omega),
        "lead_arm": _peak_in_window(lead_arm_linear),
        "lead_wrist": _peak_in_window(lead_wrist_linear),
    }
    order = ["pelvis", "chest", "lead_arm", "lead_wrist"]
    peaks_in_sequence = [peak_frames[k] for k in order]
    sequence_correct = (
        all(f >= 0 for f in peaks_in_sequence)
        and peaks_in_sequence == sorted(peaks_in_sequence)
    )

    return SwingMetrics(
        frame_count=T,
        fps=float(fps),
        pelvis_rotation_deg=pelvis_rot,
        chest_rotation_deg=chest_rot,
        x_factor_deg=x_factor,
        spine_tilt_forward_deg=spine_fwd,
        spine_tilt_side_deg=spine_side,
        lead_arm_abduction_deg=lead_abd,
        lead_arm_flex_deg=lead_flex,
        lead_wrist_speed=wrist_speed,
        impact_frame=int(impact_frame),
        peak_shoulder_frame=peak_shoulder_frame,
        setup_end_frame=setup_end,
        x_factor_at_top_deg=x_factor_at_top,
        shoulder_turn_at_top_deg=shoulder_turn_at_top,
        hip_turn_at_top_deg=hip_turn_at_top,
        spine_tilt_forward_at_address_deg=spine_tilt_forward_at_address,
        spine_tilt_side_at_address_deg=spine_tilt_side_at_address,
        lead_arm_abduction_at_top_deg=lead_arm_abduction_at_top,
        lead_arm_flex_at_top_deg=lead_arm_flex_at_top,
        peak_velocity_frame=peak_frames,
        kinematic_sequence_correct=sequence_correct,
    )


def metrics_to_coach_dict(m: SwingMetrics) -> dict:
    """Flatten SwingMetrics to a JSON-safe dict for the LLM prompt template.

    All "at_top" values are extrema over the Backswing phase (or coupled to
    the peak-shoulder-rotation frame). All "at_address" values are medians
    over the Setup phase (everything more than 1.5 s before impact). Humans
    talk about "90° of shoulder turn", not "-90°", so backswing rotations
    are reported as unsigned magnitudes.
    """

    def _round(x):
        return None if x is None else round(float(x), 1)

    return {
        "phases": {
            "impact_frame": m.impact_frame,
            "peak_shoulder_frame": m.peak_shoulder_frame,
            "setup_end_frame": m.setup_end_frame,
        },
        "at_address": {
            "spine_tilt_forward_deg": _round(m.spine_tilt_forward_at_address_deg),
            "spine_tilt_side_deg": _round(m.spine_tilt_side_at_address_deg),
        },
        "at_top": {
            "shoulder_turn_deg": _round(m.shoulder_turn_at_top_deg),
            "hip_turn_deg": _round(m.hip_turn_at_top_deg),
            "x_factor_deg": _round(m.x_factor_at_top_deg),
            "lead_arm_abduction_deg": _round(m.lead_arm_abduction_at_top_deg),
            "lead_arm_flex_deg": _round(m.lead_arm_flex_at_top_deg),
        },
        "kinematic_sequence": {
            "peak_velocity_frames": m.peak_velocity_frame,
            "order_correct": m.kinematic_sequence_correct,
            "ideal_order": ["pelvis", "chest", "lead_arm", "lead_wrist"],
        },
    }
