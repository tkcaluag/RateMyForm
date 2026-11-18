"""Turn raw landmarks into squat measurements for every frame.

Measurements (degrees unless noted):
    knee_angle   hip-knee-ankle angle. 180 = straight leg.
    hip_angle    shoulder-hip-knee angle. 180 = standing tall.
    torso_lean   hip->shoulder line vs. vertical. 0 = upright, + = forward.
    shin_lean    ankle->knee line vs. vertical. + = knees forward.
    depth_ratio  (hip height below knee) / thigh length, in the image.
                 0 = hip level with knee ("parallel"), > 0 = below parallel.

Joint angles (knee, hip) are computed from MediaPipe's 3D world landmarks by
default, matching our poster methodology, because an angle between two body
segments does not depend on where the camera is. Lean angles and depth need a
"vertical" direction, which we take from the 2D image, so they assume the
phone is upright and level.

Each measurement only uses one side of the body (the side facing the camera)
and is set to NaN in frames where any landmark it needs is not visible
enough. Missing data is never treated as bad form.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import AnalysisConfig
from .geometry import angle_at, lean_from_vertical, segment_length
from .landmarks import SIDE_LANDMARKS, LandmarkSequence
from .smoothing import smooth_series

METRIC_NAMES = ("knee_angle", "hip_angle", "torso_lean", "shin_lean", "depth_ratio")

# Landmarks each metric needs to be "seen" in a frame.
REQUIRED_POINTS = {
    "knee_angle": ("hip", "knee", "ankle"),
    "hip_angle": ("shoulder", "hip", "knee"),
    "torso_lean": ("shoulder", "hip"),
    "shin_lean": ("knee", "ankle"),
    "depth_ratio": ("hip", "knee"),
}


@dataclass
class OrientationCheck:
    """Is the camera looking at the person from the side?"""

    index: float | None  # ~0 = side view; larger = facing the camera
    label: str  # "side", "angled", "front_or_back", "unknown"
    message: str


@dataclass
class FrameMetrics:
    """Per-frame measurements. Each array has one value per analyzed frame."""

    times_s: np.ndarray
    side: str
    facing: int  # +1 facing right in the image, -1 facing left, 0 unknown
    raw: dict[str, np.ndarray]  # before smoothing (NaN where not visible)
    smoothed: dict[str, np.ndarray]  # after gap filling + smoothing
    valid: dict[str, np.ndarray]  # True where the raw value was measured
    interpolated: dict[str, np.ndarray]  # True where a short gap was filled
    knee_angle_2d: np.ndarray  # cross-check: knee angle from 2D pixels
    knee_angle_3d: np.ndarray  # cross-check: knee angle from 3D world
    side_visibility: dict[str, float] = field(default_factory=dict)


def select_side(seq: LandmarkSequence) -> tuple[str, dict[str, float]]:
    """Pick the body side the camera sees best (higher mean visibility)."""
    scores: dict[str, float] = {}
    for side, idx in SIDE_LANDMARKS.items():
        points = [idx[name] for name in ("shoulder", "hip", "knee", "ankle")]
        vis = seq.visibility[seq.detected][:, points] if seq.detected.any() else np.zeros((0, len(points)))
        scores[side] = float(np.mean(vis)) if vis.size else 0.0
    side = "left" if scores["left"] >= scores["right"] else "right"
    return side, scores


def estimate_facing(seq: LandmarkSequence, side: str, min_vis: float) -> int:
    """+1 if the person faces the right edge of the image, -1 if left.

    Uses toe (foot index) vs. heel: toes point the way the person faces.
    Falls back to nose vs. ear if the feet are not visible.
    """
    idx = SIDE_LANDMARKS[side]
    for front, back in ((idx["foot_index"], idx["heel"]), (0, idx["ear"])):
        ok = seq.detected & (seq.visibility[:, front] >= min_vis) & (seq.visibility[:, back] >= min_vis)
        if ok.sum() >= 3:
            dx = seq.image_xy[ok, front, 0] - seq.image_xy[ok, back, 0]
            median = float(np.median(dx))
            if abs(median) > 1e-6:
                return 1 if median > 0 else -1
    return 0


def check_orientation(seq: LandmarkSequence, standing_mask: np.ndarray, cfg: AnalysisConfig) -> OrientationCheck:
    """Estimate whether the video is a side view.

    In a side view the left and right shoulders (and hips) overlap
    horizontally, so their horizontal separation is small compared with the
    torso length. Facing the camera, they are far apart.
    """
    ocfg = cfg.orientation
    frames = seq.detected & standing_mask
    if frames.sum() < 3:
        frames = seq.detected
    if frames.sum() < 3:
        return OrientationCheck(None, "unknown", "Not enough frames with a detected person to check the camera angle.")
    xy = seq.image_xy[frames]
    l, r = SIDE_LANDMARKS["left"], SIDE_LANDMARKS["right"]
    shoulder_sep = np.abs(xy[:, l["shoulder"], 0] - xy[:, r["shoulder"], 0])
    hip_sep = np.abs(xy[:, l["hip"], 0] - xy[:, r["hip"], 0])
    mid_sh = (xy[:, l["shoulder"]] + xy[:, r["shoulder"]]) / 2
    mid_hip = (xy[:, l["hip"]] + xy[:, r["hip"]]) / 2
    torso = segment_length(mid_sh, mid_hip)
    with np.errstate(invalid="ignore", divide="ignore"):
        index_series = (shoulder_sep + hip_sep) / (2 * torso)
    index = float(np.nanmedian(index_series)) if np.isfinite(index_series).any() else None
    if index is None:
        return OrientationCheck(None, "unknown", "Could not estimate the camera angle.")
    if index <= ocfg.side_view_max_index:
        return OrientationCheck(index, "side", "Side view detected. Lean and depth measurements are usable.")
    if index <= ocfg.angled_view_max_index:
        return OrientationCheck(
            index,
            "angled",
            "The camera looks partly from the front or back. Torso lean, shin lean, and depth "
            "may be less accurate; a true side view gives the most reliable results.",
        )
    return OrientationCheck(
        index,
        "front_or_back",
        "The camera appears to face the person from the front or back. Torso lean, shin lean, and "
        "depth cannot be measured reliably from this angle, so they were not scored. Please record "
        "from the side.",
    )


def _visible(seq: LandmarkSequence, side: str, names: tuple[str, ...], min_vis: float) -> np.ndarray:
    idx = SIDE_LANDMARKS[side]
    mask = seq.detected.copy()
    for name in names:
        mask &= seq.visibility[:, idx[name]] >= min_vis
    return mask


def compute_frame_metrics(seq: LandmarkSequence, cfg: AnalysisConfig, side: str | None = None) -> FrameMetrics:
    """Compute every squat measurement for every frame, then smooth them."""
    min_vis = cfg.pose.min_landmark_visibility
    side_scores: dict[str, float] = {}
    if side is None:
        side, side_scores = select_side(seq)
    idx = SIDE_LANDMARKS[side]
    facing = estimate_facing(seq, side, min_vis)

    img = {name: seq.image_xy[:, i] for name, i in idx.items()}
    world = {name: seq.world_xyz[:, i] for name, i in idx.items()}

    knee_2d = angle_at(img["hip"], img["knee"], img["ankle"])
    knee_3d = angle_at(world["hip"], world["knee"], world["ankle"])
    hip_2d = angle_at(img["shoulder"], img["hip"], img["knee"])
    hip_3d = angle_at(world["shoulder"], world["hip"], world["knee"])
    use_world = cfg.joint_angle_source == "world" and np.isfinite(knee_3d).any()

    thigh_px = segment_length(img["hip"], img["knee"])
    with np.errstate(invalid="ignore", divide="ignore"):
        depth = (img["hip"][:, 1] - img["knee"][:, 1]) / thigh_px

    raw = {
        "knee_angle": knee_3d if use_world else knee_2d,
        "hip_angle": hip_3d if use_world else hip_2d,
        "torso_lean": lean_from_vertical(img["hip"], img["shoulder"], facing),
        "shin_lean": lean_from_vertical(img["ankle"], img["knee"], facing),
        "depth_ratio": depth,
    }

    valid: dict[str, np.ndarray] = {}
    for name in METRIC_NAMES:
        mask = _visible(seq, side, REQUIRED_POINTS[name], min_vis) & np.isfinite(raw[name])
        raw[name] = np.where(mask, raw[name], np.nan)
        valid[name] = mask

    smoothed: dict[str, np.ndarray] = {}
    interpolated: dict[str, np.ndarray] = {}
    sc = cfg.smoothing
    for name in METRIC_NAMES:
        smoothed[name], interpolated[name] = smooth_series(
            raw[name], seq.timestamps_s, seq.fps, sc.max_interpolation_gap_s, sc.median_window_s, sc.mean_window_s
        )

    knee_mask = valid["knee_angle"]
    return FrameMetrics(
        times_s=seq.timestamps_s,
        side=side,
        facing=facing,
        raw=raw,
        smoothed=smoothed,
        valid=valid,
        interpolated=interpolated,
        knee_angle_2d=np.where(knee_mask, knee_2d, np.nan),
        knee_angle_3d=np.where(knee_mask, knee_3d, np.nan),
        side_visibility=side_scores,
    )


def estimate_video_ratios(seq: LandmarkSequence, side: str, standing_mask: np.ndarray, min_vis: float) -> dict[str, float | None]:
    """Estimate body-proportion ratios from 3D world landmarks while standing.

    Returns median femur/torso, lower-leg/femur, and arm/torso ratios. These
    are rough estimates: MediaPipe places joint centers, which do not match
    exactly where a person measures with a tape.
    """
    idx = SIDE_LANDMARKS[side]
    names = ("shoulder", "elbow", "wrist", "hip", "knee", "ankle")
    ok = standing_mask & _visible(seq, side, names, min_vis)
    if ok.sum() < 3:
        return {"femur_to_torso": None, "lower_leg_to_femur": None, "arm_to_torso": None, "frames_used": int(ok.sum())}
    w = {name: seq.world_xyz[ok, idx[name]] for name in names}
    torso = segment_length(w["shoulder"], w["hip"])
    femur = segment_length(w["hip"], w["knee"])
    shin = segment_length(w["knee"], w["ankle"])
    arm = segment_length(w["shoulder"], w["elbow"]) + segment_length(w["elbow"], w["wrist"])
    return {
        "femur_to_torso": float(np.median(femur / torso)),
        "lower_leg_to_femur": float(np.median(shin / femur)),
        "arm_to_torso": float(np.median(arm / torso)),
        "frames_used": int(ok.sum()),
    }
