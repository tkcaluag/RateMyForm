"""SYNTHETIC squat landmark sequences for automated tests and demos.

!!! Everything produced here is SIMULATED. It is not a real person, not a
!!! real video, and not evidence of real-world accuracy.

Why it exists: unit tests need inputs where we KNOW the right answer (how
many reps, what the knee angle was at the bottom). Real videos do not come
with ground truth, so we generate a stick-figure squat from a simple 2D side-
view body model, project it to "image" and "world" coordinates the same way
MediaPipe reports them, and optionally add noise, spikes, missing frames,
occlusion, or a non-side camera angle.

Body model (side view, x forward, y up, ankle at the origin):
    knee     = ankle + shin  * (sin(shin_lean),  cos(shin_lean))
    hip      = knee  + femur * (-sin(thigh),     cos(thigh))
    shoulder = hip   + torso * (sin(torso_lean), cos(torso_lean))
The knee angle is then 180 - thigh - shin_lean, so for a requested bottom
knee angle we solve thigh = 180 - knee - shin_lean.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .landmarks import NUM_LANDMARKS, LandmarkSequence

# Lateral half-widths (m) by body part; the near side is closer to the camera.
# Hip, knee, ankle, and heel share one width so each simulated leg moves in a
# single plane; then its 3D knee angle equals the 2D side-view ground truth.
_HALF_WIDTH = {
    "shoulder": 0.19, "elbow": 0.21, "wrist": 0.20, "hand": 0.20, "hip": 0.10,
    "knee": 0.10, "ankle": 0.10, "heel": 0.10, "foot": 0.11, "ear": 0.08, "eye": 0.035, "mouth": 0.025,
}
# (left_index, right_index, part) for paired landmarks
_PAIRS = [
    (2, 5, "eye"), (1, 4, "eye"), (3, 6, "eye"), (7, 8, "ear"), (9, 10, "mouth"),
    (11, 12, "shoulder"), (13, 14, "elbow"), (15, 16, "wrist"), (17, 18, "hand"), (19, 20, "hand"),
    (21, 22, "hand"), (23, 24, "hip"), (25, 26, "knee"), (27, 28, "ankle"), (29, 30, "heel"), (31, 32, "foot"),
]


def _per_rep(value, n: int) -> list[float]:
    if isinstance(value, (list, tuple, np.ndarray)):
        if len(value) != n:
            raise ValueError(f"expected {n} per-rep values, got {len(value)}")
        return [float(v) for v in value]
    return [float(value)] * n


@dataclass
class SyntheticSquatSpec:
    """Parameters for one synthetic squat set (all times in seconds)."""

    fps: float = 30.0
    n_reps: int = 5
    lead_in_s: float = 1.0
    descent_s: float = 1.2
    pause_s: float = 0.3
    ascent_s: float = 1.0
    rest_s: float = 0.8
    tail_s: float = 1.0
    bottom_knee_angle_deg: float | Sequence[float] = 80.0
    bottom_torso_lean_deg: float | Sequence[float] = 35.0
    bottom_shin_lean_deg: float = 30.0
    chest_drop_deg: float | Sequence[float] = 0.0
    torso_cm: float = 50.0
    femur_cm: float = 43.0
    lower_leg_cm: float = 43.0
    upper_arm_cm: float = 30.0
    forearm_cm: float = 27.0
    facing: int = 1
    near_side: str = "left"
    camera_yaw_deg: float = 0.0  # 0 = true side view, 90 = facing camera
    width: int = 720
    height: int = 1280
    pixel_noise_px: float = 0.0
    world_noise_m: float = 0.0
    spike_probability: float = 0.0
    spike_px: float = 60.0
    dropout_probability: float = 0.0
    dropout_windows_s: Sequence[tuple[float, float]] = ()
    # (start_s, end_s, parts): hides near-side parts, e.g. ("shoulder",) or ("knee", "ankle")
    occlusion_windows_s: Sequence[tuple[float, float, tuple[str, ...]]] = ()
    end_mid_rep: bool = False
    partial_dip_knee_angles_deg: Sequence[float] = ()  # shallow dips added before the first rep
    bottom_bounce_deg: float = 0.0  # oscillation amplitude while at the bottom
    seed: int = 0


@dataclass
class SyntheticSquat:
    sequence: LandmarkSequence
    truth: dict = field(default_factory=dict)
    true_knee_angle: np.ndarray | None = None
    true_torso_lean: np.ndarray | None = None


def _ease(u: np.ndarray) -> np.ndarray:
    """Smooth 0->1 transition (cosine ease)."""
    return 0.5 - 0.5 * np.cos(np.pi * np.clip(u, 0.0, 1.0))


def _build_timeline(spec: SyntheticSquatSpec):
    """Return per-frame (time, progress p, knee_total_bottom, torso_bottom, chest_extra, rep_id)."""
    n = spec.n_reps
    knees = _per_rep(spec.bottom_knee_angle_deg, n)
    leans = _per_rep(spec.bottom_torso_lean_deg, n)
    drops = _per_rep(spec.chest_drop_deg, n)
    events: list[tuple] = [("stand", spec.lead_in_s)]
    for dip in spec.partial_dip_knee_angles_deg:
        events += [("dip", float(dip)), ("stand", spec.rest_s)]
    for k in range(n):
        last = k == n - 1
        events.append(("rep", k, knees[k], leans[k], drops[k], last and spec.end_mid_rep))
        if not (last and spec.end_mid_rep):
            events.append(("stand", spec.rest_s if not last else spec.tail_s))

    dt = 1.0 / spec.fps
    # (start, end, kind, params)
    spans = []
    t = 0.0
    for ev in events:
        if ev[0] == "stand":
            spans.append((t, t + ev[1], "stand", None))
            t += ev[1]
        elif ev[0] == "dip":
            d = 0.6 + 0.6
            spans.append((t, t + d, "dip", ev[1]))
            t += d
        else:
            _, k, knee, lean, drop, cut = ev
            d = spec.descent_s + spec.pause_s + spec.ascent_s
            if cut:
                d = spec.descent_s + spec.pause_s * 0.5
            spans.append((t, t + d, "rep", (k, knee, lean, drop, cut)))
            t += d
    times = np.arange(0.0, t, dt)
    p = np.zeros(len(times))
    p_torso = np.zeros(len(times))  # torso keeps its bottom lean for the first half of the ascent
    knee_bottom = np.full(len(times), 180.0)
    lean_bottom = np.zeros(len(times))
    chest = np.zeros(len(times))
    rep_id = np.full(len(times), -1)
    bounce = np.zeros(len(times))
    for start, end, kind, params in spans:
        m = (times >= start) & (times < end)
        if kind == "stand" or not m.any():
            continue
        local = times[m] - start
        if kind == "dip":
            p[m] = np.where(local < 0.6, _ease(local / 0.6), _ease(1 - (local - 0.6) / 0.6))
            p_torso[m] = p[m]
            knee_bottom[m] = params
            lean_bottom[m] = 10.0
            continue
        k, knee, lean, drop, _cut = params
        d1, d2 = spec.descent_s, spec.descent_s + spec.pause_s
        prog = np.where(local < d1, _ease(local / d1), np.where(local < d2, 1.0, _ease(1 - (local - d2) / spec.ascent_s)))
        p[m] = prog
        knee_bottom[m] = knee
        lean_bottom[m] = lean
        rep_id[m] = k
        asc = local >= d2
        u = np.clip((local - d2) / spec.ascent_s, 0, 1)
        p_torso[m] = np.where(asc, np.where(u < 0.5, 1.0, _ease(1 - (u - 0.5) / 0.5)), prog)
        chest[m] = np.where(asc, drop * np.sin(np.pi * u), 0.0)
        at_bottom = (local >= d1) & (local < d2)
        bounce[m] = np.where(at_bottom, spec.bottom_bounce_deg * np.sin(2 * np.pi * 4 * (local - d1)), 0.0)
    return times, p, p_torso, knee_bottom, lean_bottom, chest, rep_id, bounce, spans


def generate_squat(spec: SyntheticSquatSpec | None = None) -> SyntheticSquat:
    spec = spec or SyntheticSquatSpec()
    rng = np.random.default_rng(spec.seed)
    times, p, p_torso, knee_bottom, lean_bottom, chest, rep_id, bounce, spans = _build_timeline(spec)
    n = len(times)

    shin_lean = np.radians(spec.bottom_shin_lean_deg) * p
    knee_total = (180.0 - knee_bottom) * p  # thigh + shin angles (deg)
    thigh = np.radians(knee_total) - shin_lean
    thigh = np.maximum(thigh, 0.0)
    # Bottom bounce: oscillate the knee angle by moving the thigh.
    thigh = thigh + np.radians(bounce)
    torso = np.radians(lean_bottom * p_torso + chest)

    Ls, Lf, Lt = spec.lower_leg_cm / 100, spec.femur_cm / 100, spec.torso_cm / 100
    Lua, Lfa = spec.upper_arm_cm / 100, spec.forearm_cm / 100

    ankle = np.stack([np.zeros(n), np.full(n, 0.08)], axis=1)
    knee = ankle + Ls * np.stack([np.sin(shin_lean), np.cos(shin_lean)], axis=1)
    hip = knee + Lf * np.stack([-np.sin(thigh), np.cos(thigh)], axis=1)
    tdir = np.stack([np.sin(torso), np.cos(torso)], axis=1)
    shoulder = hip + Lt * tdir
    ear = shoulder + 0.20 * tdir
    nose = ear + np.array([0.10, -0.02])
    eye = ear + np.array([0.07, 0.02])
    mouth = nose + np.array([-0.01, -0.04])
    elbow = shoulder + np.array([Lua, 0.0])
    wrist = elbow + np.array([Lfa, 0.0])
    hand = wrist + np.array([0.07, 0.0])
    heel = ankle + np.array([-0.07, -0.06])
    foot = ankle + np.array([0.17, -0.07])
    parts = {
        "eye": eye, "ear": ear, "mouth": mouth, "shoulder": shoulder, "elbow": elbow, "wrist": wrist,
        "hand": hand, "hip": hip, "knee": knee, "ankle": ankle, "heel": heel, "foot": foot,
    }

    # Sagittal (x, y) and lateral z for all 33 landmarks.
    sag = np.zeros((n, NUM_LANDMARKS, 2))
    lat = np.zeros(NUM_LANDMARKS)
    sag[:, 0] = nose
    near_sign = -1.0  # near side is closer to the camera (negative z)
    for left, right, part in _PAIRS:
        sag[:, left] = parts[part]
        sag[:, right] = parts[part]
        w = _HALF_WIDTH[part]
        lat[left] = near_sign * w if spec.near_side == "left" else -near_sign * w
        lat[right] = -lat[left]

    yaw = np.radians(spec.camera_yaw_deg)
    x_rot = sag[..., 0] * np.cos(yaw) + lat[None, :] * np.sin(yaw)
    z_rot = -sag[..., 0] * np.sin(yaw) + lat[None, :] * np.cos(yaw)

    # World coordinates: meters, origin at the hip midpoint, y pointing down.
    hip_mid = sag[:, 23:24, :]  # both hips share the sagittal hip point
    hip_x_rot = hip_mid[..., 0] * np.cos(yaw)
    world = np.stack([x_rot - hip_x_rot, -(sag[..., 1] - hip_mid[..., 1]), z_rot], axis=-1)

    # Image coordinates: pixels, y pointing down, person centered.
    ppm = 0.75 * spec.height / 1.8
    cx = spec.width / 2 - spec.facing * ppm * 0.25  # leave room for arms reaching forward
    floor_y = 0.92 * spec.height
    image = np.stack([cx + spec.facing * ppm * x_rot, floor_y - ppm * sag[..., 1]], axis=-1)

    # Visibility: near side clearly visible, far side partly occluded.
    vis = np.full((n, NUM_LANDMARKS), 0.98)
    far_vis = min(0.95, 0.55 + 0.4 * abs(np.sin(yaw)))
    for left, right, _ in _PAIRS:
        far = right if spec.near_side == "left" else left
        vis[:, far] = far_vis
    vis += rng.uniform(-0.02, 0.02, size=vis.shape)
    vis = np.clip(vis, 0.0, 1.0)

    # Noise and artifacts ---------------------------------------------------
    if spec.pixel_noise_px > 0:
        image = image + rng.normal(0.0, spec.pixel_noise_px, size=image.shape)
    if spec.world_noise_m > 0:
        world = world + rng.normal(0.0, spec.world_noise_m, size=world.shape)
    near_leg = [23, 25, 27] if spec.near_side == "left" else [24, 26, 28]
    if spec.spike_probability > 0:
        for i in np.flatnonzero(rng.random(n) < spec.spike_probability):
            j = rng.choice(near_leg)
            direction = rng.normal(size=2)
            direction /= np.linalg.norm(direction)
            image[i, j] += spec.spike_px * direction
            world[i, j, :2] += (spec.spike_px / ppm) * direction
    detected = np.ones(n, dtype=bool)
    if spec.dropout_probability > 0:
        detected &= rng.random(n) >= spec.dropout_probability
    for start, end in spec.dropout_windows_s:
        detected &= ~((times >= start) & (times < end))
    near_index = {part: (left if spec.near_side == "left" else right) for left, right, part in _PAIRS}
    for start, end, occluded_parts in spec.occlusion_windows_s:
        m = (times >= start) & (times < end)
        for part in occluded_parts:
            j = near_index[part]
            vis[m, j] = 0.2
            image[m, j] += rng.normal(0, 15, size=(m.sum(), 2))

    image[~detected] = np.nan
    world[~detected] = np.nan
    vis[~detected] = 0.0

    seq = LandmarkSequence(
        timestamps_s=times,
        frame_indices=np.arange(n),
        detected=detected,
        image_xy=image,
        visibility=vis,
        world_xyz=world,
        width=spec.width,
        height=spec.height,
        fps=spec.fps,
        source="synthetic",
        metadata={"model_variant": "synthetic", "label": "SYNTHETIC - not a real person"},
    )

    knees = _per_rep(spec.bottom_knee_angle_deg, spec.n_reps)
    leans = _per_rep(spec.bottom_torso_lean_deg, spec.n_reps)
    drops = _per_rep(spec.chest_drop_deg, spec.n_reps)
    complete = spec.n_reps - (1 if spec.end_mid_rep else 0)
    truth = {
        "label": "SYNTHETIC",
        "expected_rep_count": complete,
        "bottom_knee_angles_deg": knees[:complete],
        "bottom_torso_leans_deg": leans[:complete],
        "bottom_shin_lean_deg": spec.bottom_shin_lean_deg,
        "chest_drops_deg": drops[:complete],
        "partial_dips": len(spec.partial_dip_knee_angles_deg),
        "incomplete_last_rep": spec.end_mid_rep,
        "duration_s": float(times[-1]) if n else 0.0,
    }
    true_knee = 180.0 - np.degrees(thigh) - np.degrees(shin_lean)
    true_lean = np.degrees(torso)
    return SyntheticSquat(sequence=seq, truth=truth, true_knee_angle=true_knee, true_torso_lean=true_lean)


def default_measurements_for(spec: SyntheticSquatSpec):
    """Body measurements consistent with the synthetic skeleton."""
    from .personalization import BodyMeasurements

    return BodyMeasurements(
        torso_cm=spec.torso_cm,
        femur_cm=spec.femur_cm,
        lower_leg_cm=spec.lower_leg_cm,
        arm_cm=spec.upper_arm_cm + spec.forearm_cm,
    )


def scenarios() -> dict[str, tuple[SyntheticSquatSpec, str]]:
    """Named SYNTHETIC scenarios used by tests, the evaluation script, and the demo."""
    return {
        "clean_5_reps": (SyntheticSquatSpec(), "5 clean reps, knee 80 deg, torso lean 35 deg, no noise"),
        "shallow_reps": (
            SyntheticSquatSpec(bottom_knee_angle_deg=[115, 120, 112, 118, 125]),
            "5 reps that stop above the 100 deg depth target",
        ),
        "forward_lean_long_femur": (
            SyntheticSquatSpec(bottom_torso_lean_deg=50.0, femur_cm=50.0, torso_cm=48.0, lower_leg_cm=42.0),
            "Long femur relative to torso, 50 deg torso lean (personalization should relax the limit)",
        ),
        "forward_lean_short_femur": (
            SyntheticSquatSpec(bottom_torso_lean_deg=50.0, femur_cm=38.0, torso_cm=55.0, lower_leg_cm=42.0),
            "Short femur relative to torso, 50 deg torso lean (personalization should not relax the limit)",
        ),
        "chest_drop": (
            SyntheticSquatSpec(chest_drop_deg=[0, 20, 25, 0, 22]),
            "Torso tips forward while standing up in reps 2, 3, 5",
        ),
        "noisy_jitter": (
            SyntheticSquatSpec(pixel_noise_px=4.0, world_noise_m=0.012, spike_probability=0.05, seed=7),
            "5 reps with Gaussian landmark noise and random one-frame spikes",
        ),
        "bounce_at_bottom": (
            SyntheticSquatSpec(bottom_knee_angle_deg=128.0, pause_s=1.5, bottom_bounce_deg=10.0),
            "Knee angle oscillates 118-138 deg at the bottom (jitter around the 140 deg threshold)",
        ),
        "random_dropouts": (
            SyntheticSquatSpec(dropout_probability=0.15, seed=3),
            "15% of frames randomly have no detected person",
        ),
        # Rep k (0-based) starts at 1.0 + 3.3*k s; its bottom is 1.2-1.5 s later.
        "occluded_shoulder": (
            SyntheticSquatSpec(bottom_torso_lean_deg=[35, 60, 35, 35, 35], occlusion_windows_s=[(5.2, 6.1, ("shoulder",))], seed=5),
            "Near-side shoulder hidden around the bottom of rep 2 (which also leans 60 deg)",
        ),
        "tracking_gap": (
            SyntheticSquatSpec(bottom_knee_angle_deg=[80, 80, 80, 120, 80], dropout_windows_s=[(11.8, 12.6)], seed=11),
            "0.8 s with no detected person around the bottom of rep 4",
        ),
        "incomplete_last_rep": (
            SyntheticSquatSpec(n_reps=4, end_mid_rep=True),
            "Video ends at the bottom of the 4th rep (3 complete reps)",
        ),
        "partial_dips": (
            SyntheticSquatSpec(n_reps=3, partial_dip_knee_angles_deg=[150.0, 147.0]),
            "Two shallow dips (150 and 147 deg) before 3 real reps",
        ),
        "front_view": (
            SyntheticSquatSpec(camera_yaw_deg=80.0),
            "Camera almost facing the person (not a side view)",
        ),
        "fps_24": (SyntheticSquatSpec(fps=24.0), "Same 5 reps sampled at 24 fps"),
        "fps_60": (SyntheticSquatSpec(fps=60.0), "Same 5 reps sampled at 60 fps"),
        "fps_120": (SyntheticSquatSpec(fps=120.0), "Same 5 reps sampled at 120 fps"),
    }
