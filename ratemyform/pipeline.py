"""End-to-end analysis pipeline: the one entry point the frontend calls.

    video file + body measurements
        -> pose_detection.extract_landmarks()   (MediaPipe + OpenCV)
        -> squat_metrics.compute_frame_metrics() (angles, lean, depth, smoothing)
        -> rep_counter.detect_reps()            (hysteresis state machine)
        -> rep_analysis.measure_rep()           (bottom-of-rep values + reliability)
        -> scoring.score_rep() twice            (baseline vs. personalized)
        -> feedback.summary_feedback()
        -> AnalysisResult (rendered by app.py, exported as JSON/CSV)

`analyze_sequence` works on landmarks from ANY source (a real video or the
synthetic generator), which is what makes the logic unit-testable without
video files.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from .config import DEFAULT_CONFIG, DISCLAIMER, SCHEMA_VERSION, AnalysisConfig
from .feedback import summary_feedback
from .landmarks import LandmarkSequence
from .personalization import (
    BodyMeasurements,
    RatioCheck,
    Thresholds,
    baseline_thresholds,
    compare_ratios,
    personalized_thresholds,
)
from .rep_analysis import RepMeasurements, measure_rep
from .rep_counter import RepDetectionResult, detect_reps
from .scoring import RepScore, compare_scores, overall_score, score_rep
from .squat_metrics import (
    FrameMetrics,
    OrientationCheck,
    check_orientation,
    compute_frame_metrics,
    estimate_video_ratios,
)

ProgressCallback = Callable[[float, str], None]


@dataclass
class AnalysisResult:
    exercise: str
    source: str
    video_info: dict
    processing: dict
    measurements: BodyMeasurements
    video_ratios: dict
    ratio_checks: list[RatioCheck]
    orientation: OrientationCheck
    side: str
    baseline: Thresholds
    personalized: Thresholds
    rep_detection: RepDetectionResult
    rep_measurements: list[RepMeasurements]
    baseline_scores: list[RepScore]
    personalized_scores: list[RepScore]
    comparison: list[dict]
    overall_baseline: float | None
    overall_personalized: float | None
    feedback: list[str]
    warnings: list[str]
    frame_metrics: FrameMetrics
    config: AnalysisConfig = DEFAULT_CONFIG
    key_frames: dict[int, bytes] = field(default_factory=dict)
    overlay_video: bytes | None = None
    overlay_mime: str = "video/webm"

    @property
    def rep_count(self) -> int:
        return self.rep_detection.rep_count

    def to_dict(self) -> dict:
        """JSON-serializable report (see docs/DATA_SCHEMA.md)."""
        return {
            "schema_version": SCHEMA_VERSION,
            "exercise": self.exercise,
            "source": self.source,
            "disclaimer": DISCLAIMER,
            "input": {
                "video": self.video_info,
                "measurements_cm": self.measurements.to_dict(),
                "entered_ratios": self.measurements.ratios().to_dict(),
            },
            "processing": self.processing,
            "camera": {
                "side_analyzed": self.side,
                "orientation": {
                    "label": self.orientation.label,
                    "index": None if self.orientation.index is None else round(self.orientation.index, 3),
                    "message": self.orientation.message,
                },
            },
            "ratio_checks": [vars(c) for c in self.ratio_checks],
            "video_ratio_estimates": {
                k: (round(v, 3) if isinstance(v, float) else v) for k, v in self.video_ratios.items()
            },
            "thresholds": {"baseline": self.baseline.to_dict(), "personalized": self.personalized.to_dict()},
            "summary": {
                "rep_count": self.rep_count,
                "overall_score_baseline": self.overall_baseline,
                "overall_score_personalized": self.overall_personalized,
                "feedback": list(self.feedback),
                "warnings": list(self.warnings),
            },
            "reps": [
                {
                    "measurements": m.to_dict(),
                    "baseline": b.to_dict(),
                    "personalized": p.to_dict(),
                }
                for m, b, p in zip(self.rep_measurements, self.baseline_scores, self.personalized_scores)
            ],
            "baseline_vs_personalized": self.comparison,
            "uncounted_movements": [
                {
                    "kind": u.kind,
                    "start_time_s": round(u.start_time_s, 2),
                    "end_time_s": None if u.end_time_s is None else round(u.end_time_s, 2),
                    "min_knee_angle_deg": round(u.min_knee_angle_deg, 1),
                    "reason": u.reason,
                }
                for u in self.rep_detection.uncounted
            ],
        }

    def reps_table(self) -> list[dict]:
        """One row per rep for the results table / CSV export."""
        rows = []
        for m, b, p in zip(self.rep_measurements, self.baseline_scores, self.personalized_scores):
            d = m.to_dict()
            rows.append(
                {
                    "rep": m.rep_number,
                    "bottom_time_s": d["bottom_time_s"],
                    "knee_angle_deg": d["knee_angle_deg"],
                    "hip_angle_deg": d["hip_angle_deg"],
                    "torso_lean_deg": d["torso_lean_deg"],
                    "shin_lean_deg": d["shin_lean_deg"],
                    "depth": d["depth_class"],
                    "chest_drop_deg": d["chest_drop_deg"],
                    "baseline_score": b.score,
                    "personalized_score": p.score,
                    "low_confidence": m.low_confidence,
                }
            )
        return rows

    def timeseries_rows(self) -> list[dict]:
        """Per-frame smoothed measurements (for CSV export and charts)."""
        fm = self.frame_metrics
        rows = []
        for i, t in enumerate(fm.times_s):
            row = {"time_s": round(float(t), 3), "state": self.rep_detection.states[i] if i < len(self.rep_detection.states) else ""}
            for name, series in fm.smoothed.items():
                v = series[i]
                row[name] = None if not np.isfinite(v) else round(float(v), 2)
            rows.append(row)
        return rows


def analyze_sequence(
    seq: LandmarkSequence,
    measurements: BodyMeasurements,
    cfg: AnalysisConfig = DEFAULT_CONFIG,
    video_info: dict | None = None,
    pose_seconds: float | None = None,
) -> AnalysisResult:
    """Run every analysis step after pose detection."""
    started = time.perf_counter()
    warnings: list[str] = []

    fm = compute_frame_metrics(seq, cfg)
    reps = detect_reps(fm.times_s, fm.smoothed["knee_angle"], cfg.reps)
    standing = np.nan_to_num(fm.smoothed["knee_angle"], nan=0.0) > cfg.reps.up_threshold_deg
    orientation = check_orientation(seq, standing, cfg)
    video_ratios = estimate_video_ratios(seq, fm.side, standing, cfg.pose.min_landmark_visibility)
    ratio_checks = compare_ratios(measurements.ratios(), video_ratios, cfg.personalization.ratio_mismatch_tolerance)

    base_th = baseline_thresholds(cfg.scoring)
    pers_th = personalized_thresholds(measurements, cfg.scoring, cfg.personalization)

    rep_meas = [measure_rep(r, fm, orientation, cfg) for r in reps.reps]
    base_scores = [score_rep(m, base_th, cfg.scoring) for m in rep_meas]
    pers_scores = [score_rep(m, pers_th, cfg.scoring) for m in rep_meas]

    # ---- warnings about data quality (shown, never deducted) -------------
    if seq.detection_rate < 0.5:
        warnings.append(
            f"A person was detected in only {seq.detection_rate:.0%} of frames. Make sure your whole body is "
            "visible, well lit, and the only person in view."
        )
    if orientation.label in ("angled", "front_or_back"):
        warnings.append(orientation.message)
    for c in ratio_checks:
        if c.consistent is False:
            warnings.append(c.message)
    if not standing.any():
        warnings.append(
            f"The knee angle never went above {cfg.reps.up_threshold_deg:.0f} deg, so a standing position was "
            "never detected and no reps could be counted."
        )
    both = np.isfinite(fm.knee_angle_2d) & np.isfinite(fm.knee_angle_3d)
    if both.sum() >= 10:
        disagreement = float(np.median(np.abs(fm.knee_angle_2d[both] - fm.knee_angle_3d[both])))
        if disagreement > 15:
            warnings.append(
                f"The 2D and 3D knee-angle estimates differ by a median of {disagreement:.0f} deg, which usually "
                "means the camera is not side-on or the pose estimate is unstable. Treat the angles with caution."
            )
    else:
        disagreement = None
    for u in reps.uncounted:
        warnings.append(f"Not counted ({u.kind.replace('_', ' ')}) at {u.start_time_s:.1f} s: {u.reason}")

    processing = {
        "frames_analyzed": seq.num_frames,
        "frames_with_person": int(seq.detected.sum()),
        "detection_rate": round(seq.detection_rate, 3),
        "analysis_fps": round(seq.fps, 2),
        "joint_angle_source": cfg.joint_angle_source,
        "facing": {1: "right", -1: "left", 0: "unknown"}[fm.facing],
        "side_visibility": {k: round(v, 3) for k, v in fm.side_visibility.items()},
        "median_knee_2d_vs_3d_difference_deg": None if disagreement is None else round(disagreement, 1),
        "pose_seconds": pose_seconds,
        "analysis_seconds": round(time.perf_counter() - started, 3),
        "model_variant": seq.metadata.get("model_variant"),
    }

    return AnalysisResult(
        exercise="squat",
        source=seq.source,
        video_info=video_info or {},
        processing=processing,
        measurements=measurements,
        video_ratios=video_ratios,
        ratio_checks=ratio_checks,
        orientation=orientation,
        side=fm.side,
        baseline=base_th,
        personalized=pers_th,
        rep_detection=reps,
        rep_measurements=rep_meas,
        baseline_scores=base_scores,
        personalized_scores=pers_scores,
        comparison=compare_scores(base_scores, pers_scores),
        overall_baseline=overall_score(base_scores),
        overall_personalized=overall_score(pers_scores),
        feedback=summary_feedback(rep_meas, pers_scores, pers_th),
        warnings=warnings,
        frame_metrics=fm,
        config=cfg,
    )


def analyze_video(
    video_path: str | Path,
    measurements: BodyMeasurements,
    cfg: AnalysisConfig = DEFAULT_CONFIG,
    progress: ProgressCallback | None = None,
    make_overlay: bool = True,
) -> AnalysisResult:
    """Full pipeline for a video file on disk (stays on this computer)."""
    from .overlay import render_overlay
    from .pose_detection import extract_landmarks, probe_video

    def stage(lo: float, hi: float) -> ProgressCallback | None:
        if progress is None:
            return None
        return lambda f, msg: progress(lo + (hi - lo) * f, msg)

    info = probe_video(video_path)
    t0 = time.perf_counter()
    seq = extract_landmarks(video_path, cfg.pose, stage(0.0, 0.8 if make_overlay else 0.95))
    pose_seconds = round(time.perf_counter() - t0, 3)
    result = analyze_sequence(seq, measurements, cfg, video_info=info.to_dict(), pose_seconds=pose_seconds)
    if make_overlay:
        t1 = time.perf_counter()
        video, frames = render_overlay(video_path, seq, result.frame_metrics, result.rep_detection, cfg, stage(0.8, 1.0))
        result.overlay_video = video
        result.key_frames = frames
        result.processing["overlay_seconds"] = round(time.perf_counter() - t1, 3)
        if video is None:
            result.warnings.append("The skeleton overlay video could not be encoded on this computer.")
    result.processing["total_seconds"] = round(time.perf_counter() - t0, 3)
    if progress:
        progress(1.0, "Analysis complete")
    return result
