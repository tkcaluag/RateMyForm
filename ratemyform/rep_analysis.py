"""Measure each counted repetition at its lowest point.

For every rep we read the smoothed measurements at the bottom frame (the
frame with the smallest knee angle). We also decide, metric by metric,
whether the measurement is RELIABLE enough to score. A metric is unreliable
when its landmarks were hidden near the bottom, when tracking was lost for a
long time during the rep, or when the camera angle makes it unmeasurable.
Unreliable metrics are reported, but never deducted.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import AnalysisConfig
from .rep_counter import RepWindow
from .squat_metrics import FrameMetrics, OrientationCheck

# Metrics measured from the 2D image (need a true side view).
IMAGE_PLANE_METRICS = ("torso_lean", "shin_lean", "depth_ratio")


@dataclass
class RepMeasurements:
    rep_number: int
    start_time_s: float
    bottom_time_s: float
    end_time_s: float
    duration_s: float
    knee_angle_deg: float | None
    hip_angle_deg: float | None
    torso_lean_deg: float | None
    shin_lean_deg: float | None
    depth_ratio: float | None
    depth_class: str
    chest_drop_deg: float | None  # extra forward lean while standing up
    knee_angle_2d_deg: float | None  # cross-check values
    knee_angle_3d_deg: float | None
    reliable: dict[str, bool] = field(default_factory=dict)
    reliability_notes: list[str] = field(default_factory=list)
    low_confidence: bool = False

    def to_dict(self) -> dict:
        def r(x, nd=1):
            return None if x is None else round(float(x), nd)

        return {
            "rep_number": self.rep_number,
            "start_time_s": r(self.start_time_s, 2),
            "bottom_time_s": r(self.bottom_time_s, 2),
            "end_time_s": r(self.end_time_s, 2),
            "duration_s": r(self.duration_s, 2),
            "knee_angle_deg": r(self.knee_angle_deg),
            "hip_angle_deg": r(self.hip_angle_deg),
            "torso_lean_deg": r(self.torso_lean_deg),
            "shin_lean_deg": r(self.shin_lean_deg),
            "depth_ratio": r(self.depth_ratio, 2),
            "depth_class": self.depth_class,
            "chest_drop_deg": r(self.chest_drop_deg),
            "knee_angle_2d_deg": r(self.knee_angle_2d_deg),
            "knee_angle_3d_deg": r(self.knee_angle_3d_deg),
            "reliable": dict(self.reliable),
            "reliability_notes": list(self.reliability_notes),
            "low_confidence": self.low_confidence,
        }


def classify_depth(depth_ratio: float | None) -> str:
    """Hip-joint height relative to knee-joint height in the image.

    Note: MediaPipe's hip landmark is the hip JOINT, not the hip crease that
    powerlifting judges use, so "parallel" here is approximate.
    """
    if depth_ratio is None or not np.isfinite(depth_ratio):
        return "unknown"
    if depth_ratio > 0.05:
        return "below parallel"
    if depth_ratio >= -0.05:
        return "around parallel"
    return "above parallel"


def _value(series: np.ndarray, i: int) -> float | None:
    v = series[i]
    return float(v) if np.isfinite(v) else None


def measure_rep(rep: RepWindow, fm: FrameMetrics, orientation: OrientationCheck, cfg: AnalysisConfig) -> RepMeasurements:
    sc = cfg.scoring
    t = fm.times_s
    b = rep.bottom_index
    near_bottom = np.abs(t - t[b]) <= sc.bottom_window_s
    in_rep = np.zeros(len(t), dtype=bool)
    in_rep[rep.start_index : rep.end_index + 1] = True

    reliable: dict[str, bool] = {}
    notes: list[str] = []
    labels = {
        "knee_angle": "knee angle",
        "hip_angle": "hip angle",
        "torso_lean": "torso lean",
        "shin_lean": "shin lean",
        "depth_ratio": "depth",
    }
    for name, label in labels.items():
        frac_bottom = float(np.mean(fm.valid[name][near_bottom])) if near_bottom.any() else 0.0
        ok = frac_bottom >= sc.min_valid_fraction_near_bottom and np.isfinite(fm.smoothed[name][b])
        if not ok:
            notes.append(
                f"The {label} could not be measured reliably near the bottom of this rep "
                f"(landmarks visible in {frac_bottom:.0%} of nearby frames), so it was not scored."
            )
        reliable[name] = bool(ok)

    knee_frac_rep = float(np.mean(fm.valid["knee_angle"][in_rep]))
    if knee_frac_rep < sc.min_valid_fraction_in_rep:
        notes.append(f"The knee was clearly visible in only {knee_frac_rep:.0%} of this rep's frames.")
        reliable["knee_angle"] = False

    if rep.low_confidence:
        notes.append(
            f"Tracking was lost for {rep.longest_gap_s:.2f} s during this rep, so the true lowest point "
            "may have been missed. Bottom-position measurements were not scored."
        )
        for name in reliable:
            reliable[name] = False

    if orientation.label == "front_or_back":
        for name in IMAGE_PLANE_METRICS:
            if reliable.get(name):
                reliable[name] = False
        notes.append("The camera is not side-on, so torso lean, shin lean, and depth were not scored.")

    # Chest drop: did the torso tip further forward while standing back up?
    ascent = np.zeros(len(t), dtype=bool)
    ascent[b : rep.end_index + 1] = True
    lean = fm.smoothed["torso_lean"]
    chest_drop = None
    ascent_valid = float(np.mean(fm.valid["torso_lean"][ascent])) if ascent.any() else 0.0
    if reliable.get("torso_lean") and ascent_valid >= sc.min_valid_fraction_in_rep and np.isfinite(lean[ascent]).any():
        chest_drop = float(max(0.0, np.nanmax(lean[ascent]) - lean[b]))
        reliable["chest_drop"] = True
    else:
        reliable["chest_drop"] = False

    depth_ratio = _value(fm.smoothed["depth_ratio"], b)
    return RepMeasurements(
        rep_number=rep.rep_number,
        start_time_s=rep.start_time_s,
        bottom_time_s=rep.bottom_time_s,
        end_time_s=rep.end_time_s,
        duration_s=rep.duration_s,
        knee_angle_deg=_value(fm.smoothed["knee_angle"], b),
        hip_angle_deg=_value(fm.smoothed["hip_angle"], b),
        torso_lean_deg=_value(fm.smoothed["torso_lean"], b),
        shin_lean_deg=_value(fm.smoothed["shin_lean"], b),
        depth_ratio=depth_ratio,
        depth_class=classify_depth(depth_ratio) if reliable.get("depth_ratio") else "unknown",
        chest_drop_deg=chest_drop,
        knee_angle_2d_deg=_value(fm.knee_angle_2d, b),
        knee_angle_3d_deg=_value(fm.knee_angle_3d, b),
        reliable=reliable,
        reliability_notes=notes,
        low_confidence=rep.low_confidence or not all(reliable.get(k, False) for k in labels),
    )
