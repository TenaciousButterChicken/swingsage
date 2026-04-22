"""Biomechanical analytics from NLF 3D joint positions.

Computes the golf-specific metrics a coach actually talks about:
  - Pelvis rotation, chest rotation, and X-factor (chest minus pelvis)
  - Spine tilt (forward lean and side bend) from vertical
  - Lead-arm abduction (angle between trunk and lead arm)
  - Kinematic sequence timing (peak angular-velocity order across pelvis →
    chest → lead arm → lead wrist — the single most important biomechanical
    signature in golf)

Design note: we do NOT implement full ISB joint coordinate systems (too heavy
for Phase 2 MVP — needs per-joint reference frames with 3 axes each). Instead
we use simple geometric operations on 3D joint positions in the NLF camera
frame, which is sufficient for coaching-grade feedback.

Input: list[FramePose] from inference.pose_3d, plus optional SwingEvents from
inference.swing_events to anchor metrics at Address / Top / Impact.

Output: SwingMetrics dataclass with per-frame series and event-anchored values.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from inference.pose_3d import SMPL_JOINT_NAMES, FramePose
from inference.swing_events import SwingEvents

_J = {name: idx for idx, name in enumerate(SMPL_JOINT_NAMES)}

# NLF outputs in mm, Y-down camera frame. For angle math we use the body
# frame reconstructed from the golfer's own hip/shoulder geometry, so the
# camera's exact Y convention only affects the sign of the gravity axis.
_MM_PER_M = 1000.0
Handedness = Literal["right", "left"]


@dataclass(frozen=True)
class SwingMetrics:
    """Per-frame biomechanical series + event-anchored summary values."""

    frame_count: int

    # Per-frame series (shape: (frame_count,))
    pelvis_rotation_deg: np.ndarray       # yaw of hip line vs. baseline
    chest_rotation_deg: np.ndarray        # yaw of shoulder line vs. baseline
    x_factor_deg: np.ndarray              # chest minus pelvis
    spine_tilt_forward_deg: np.ndarray    # pitch: mid-shoulder-pelvis vector vs vertical
    spine_tilt_side_deg: np.ndarray       # roll: side bend in body frame
    lead_arm_abduction_deg: np.ndarray    # 180 = straight out from torso
    lead_wrist_speed: np.ndarray          # magnitude of per-frame velocity, mm/frame

    # Event-anchored scalar metrics (None if the event couldn't be located)
    address_frame: int | None
    top_frame: int | None
    impact_frame: int | None
    x_factor_at_top_deg: float | None
    shoulder_turn_at_top_deg: float | None
    hip_turn_at_top_deg: float | None
    spine_tilt_forward_at_address_deg: float | None
    spine_tilt_side_at_address_deg: float | None
    lead_arm_abduction_at_top_deg: float | None

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
    """Signed angle (degrees) from `ref` to `v`, measured around `normal`.

    All three inputs are 3-vectors. Positive when the rotation follows the
    right-hand rule around `normal`.
    """
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


# ─── Core computation ───────────────────────────────────────────────


def _gravity_axis_from_frames(joints: np.ndarray) -> np.ndarray:
    """Estimate the world vertical direction (unit vector in NLF camera frame).

    Uses the median pelvis→neck vector across all detected frames as the
    long axis of the body. Golfer's spine approximates vertical through
    Address and Finish even if bent at Top, and the median is robust to that.
    """
    pelvis = joints[:, _J["pelvis"]]
    neck = joints[:, _J["neck"]]
    up = neck - pelvis  # (T, 3), points from pelvis up toward head
    # ignore NaN rows
    ok = ~np.isnan(up).any(axis=1)
    if not ok.any():
        # fallback: NLF outputs Y-down, so world-up is -Y
        return np.array([0.0, -1.0, 0.0], dtype=np.float32)
    return _unit(np.median(up[ok], axis=0))


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


def compute_metrics(
    frames: list[FramePose],
    events: SwingEvents | None = None,
    handedness: Handedness = "right",
) -> SwingMetrics:
    """Run the full biomechanical analysis. See module docstring for scope."""
    if not frames:
        raise ValueError("No frames supplied")

    joints = _joints_array(frames)  # (T, 24, 3), NaN where undetected
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

    up = _gravity_axis_from_frames(joints)

    # Body-frame axes established from the first reliably-detected frame (Address).
    detected_idx = np.where(~np.isnan(joints[:, j_pelvis, 0]))[0]
    if len(detected_idx) == 0:
        raise ValueError("No detected frames — cannot compute metrics")
    anchor = int(detected_idx[0])

    def _plane_vec(v):
        return _project_onto_horizontal_plane(v, up)

    hip_vec0 = _plane_vec(joints[anchor, j_lead_hip] - joints[anchor, j_trail_hip])
    hip_ref = _unit(hip_vec0)
    # Compute a target-line normal pointing "forward" for the golfer — cross of up x hip_ref.
    target_normal = _unit(np.cross(up, hip_ref))

    # Per-frame series ───────────────────────────────────────────────
    pelvis_rot = np.full((T,), np.nan, dtype=np.float32)
    chest_rot = np.full((T,), np.nan, dtype=np.float32)
    spine_fwd = np.full((T,), np.nan, dtype=np.float32)
    spine_side = np.full((T,), np.nan, dtype=np.float32)
    lead_abd = np.full((T,), np.nan, dtype=np.float32)

    for t in range(T):
        if np.isnan(joints[t, j_pelvis, 0]):
            continue
        hip_vec = _plane_vec(joints[t, j_lead_hip] - joints[t, j_trail_hip])
        sh_vec = _plane_vec(joints[t, j_lead_sh] - joints[t, j_trail_sh])
        if np.linalg.norm(hip_vec) > 1e-3:
            pelvis_rot[t] = _signed_angle_between(hip_vec, hip_ref, up)
        if np.linalg.norm(sh_vec) > 1e-3:
            chest_rot[t] = _signed_angle_between(sh_vec, hip_ref, up)

        # Spine tilt: vector from mid-hip to neck, decomposed in the body frame.
        spine = joints[t, j_neck] - joints[t, j_pelvis]
        spine_u = _unit(spine)
        # Forward pitch: angle between spine and up, measured in the plane
        # spanned by up and target_normal (the sagittal plane).
        spine_sag = spine_u - np.dot(spine_u, hip_ref) * hip_ref
        if np.linalg.norm(spine_sag) > 1e-3:
            spine_sag_u = _unit(spine_sag)
            spine_fwd[t] = _signed_angle_between(spine_sag_u, up, hip_ref)
        # Side bend: angle between spine and up, measured in the frontal plane
        # spanned by up and hip_ref.
        spine_fr = spine_u - np.dot(spine_u, target_normal) * target_normal
        if np.linalg.norm(spine_fr) > 1e-3:
            spine_fr_u = _unit(spine_fr)
            spine_side[t] = _signed_angle_between(spine_fr_u, up, target_normal)

        # Lead-arm abduction: angle between trunk-down axis and lead upper arm.
        trunk_down = -spine_u  # from neck toward pelvis
        upper_arm = joints[t, j_lead_elbow] - joints[t, j_lead_sh]
        if np.linalg.norm(upper_arm) > 1e-3:
            lead_abd[t] = _unsigned_angle_between(upper_arm, trunk_down)

    x_factor = chest_rot - pelvis_rot

    # Lead wrist linear speed (proxy for club-head motion; the most responsive
    # velocity channel for kinematic-sequence timing since we don't see the club).
    lead_wrist_pos = joints[:, j_lead_wrist]
    wrist_speed = _velocity_magnitude(lead_wrist_pos)

    # ── Event-anchored single values ─────────────────────────────────
    address_frame = events.frames[0] if events else None
    top_frame = events.frames[3] if events else None
    impact_frame = events.frames[5] if events else None

    def _at(frame: int | None, series: np.ndarray) -> float | None:
        if frame is None or frame < 0 or frame >= T:
            return None
        v = series[frame]
        return None if np.isnan(v) else float(v)

    # ── Kinematic sequence ──────────────────────────────────────────
    # Restrict analysis to the downswing window (Top → Impact) if we have
    # events — that's where peak angular velocities actually occur.
    if top_frame is not None and impact_frame is not None and impact_frame > top_frame:
        lo, hi = int(top_frame), int(impact_frame) + 1
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
        pelvis_rotation_deg=pelvis_rot,
        chest_rotation_deg=chest_rot,
        x_factor_deg=x_factor,
        spine_tilt_forward_deg=spine_fwd,
        spine_tilt_side_deg=spine_side,
        lead_arm_abduction_deg=lead_abd,
        lead_wrist_speed=wrist_speed,
        address_frame=address_frame,
        top_frame=top_frame,
        impact_frame=impact_frame,
        x_factor_at_top_deg=_at(top_frame, x_factor),
        shoulder_turn_at_top_deg=_at(top_frame, chest_rot),
        hip_turn_at_top_deg=_at(top_frame, pelvis_rot),
        spine_tilt_forward_at_address_deg=_at(address_frame, spine_fwd),
        spine_tilt_side_at_address_deg=_at(address_frame, spine_side),
        lead_arm_abduction_at_top_deg=_at(top_frame, lead_abd),
        peak_velocity_frame=peak_frames,
        kinematic_sequence_correct=sequence_correct,
    )


def metrics_to_coach_dict(m: SwingMetrics) -> dict:
    """Flatten SwingMetrics to a JSON-safe dict for the LLM prompt template."""

    def _round(x):
        return None if x is None else round(float(x), 1)

    return {
        "events": {
            "address_frame": m.address_frame,
            "top_frame": m.top_frame,
            "impact_frame": m.impact_frame,
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
        },
        "kinematic_sequence": {
            "peak_velocity_frames": m.peak_velocity_frame,
            "order_correct": m.kinematic_sequence_correct,
            "ideal_order": ["pelvis", "chest", "lead_arm", "lead_wrist"],
        },
    }
