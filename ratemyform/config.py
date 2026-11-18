"""Central configuration for the RateMyForm squat analysis.

Every threshold used by the analysis lives in this file so the whole team can
review it in one place. Each value is tagged with where it comes from:

    TEAM_METHOD  Stated in our CPSC 490 midterm methodology (poster).
    MODEL_CARD   From the MediaPipe BlazePose GHUM 3D model card.
    PROTOTYPE    An engineering choice made for this prototype. It is NOT a
                 validated exercise-science guideline. These values should be
                 reviewed against the guideline research assigned in Status
                 Report 2 ("Research exercise guidelines and specifications").

Changing a value here changes the behavior of the app, the CLI scripts, and
the expected results of some automated tests, so re-run `pytest` after edits.
"""
from __future__ import annotations

from dataclasses import dataclass, field

SCHEMA_VERSION = "1.0"

SUPPORTED_EXERCISES = ("squat",)
PLANNED_EXERCISES = ("deadlift", "bench press", "push-up", "lunge")


@dataclass(frozen=True)
class PoseConfig:
    """MediaPipe Pose Landmarker settings."""

    # lite / full / heavy. The model card reports higher keypoint accuracy for
    # larger models (avg PDJ lite 87.0, full 91.8, heavy 94.2) at higher CPU
    # cost. "full" is a balance for CPU-only laptops. MODEL_CARD / PROTOTYPE.
    model_variant: str = "full"
    min_pose_detection_confidence: float = 0.5  # MediaPipe default
    min_pose_presence_confidence: float = 0.5  # MediaPipe default
    min_tracking_confidence: float = 0.5  # MediaPipe default
    # A landmark below this visibility is treated as "not reliably seen".
    # Visibility = probability the point is in frame and not occluded
    # (MODEL_CARD definition). 0.5 is a PROTOTYPE choice.
    min_landmark_visibility: float = 0.5
    # Videos recorded above this rate are subsampled (e.g., 60 fps -> every
    # 2nd frame). Rep timing uses timestamps, so results do not depend on it.
    max_analysis_fps: float = 30.0
    # Frames are downscaled so the longer side is at most this many pixels.
    # MediaPipe resizes internally (256x256 crop per the model card), so this
    # mainly speeds up decoding and overlay rendering.
    max_frame_long_side_px: int = 960


@dataclass(frozen=True)
class SmoothingConfig:
    """Signal smoothing applied to per-frame angles (all values in seconds,
    so behavior is the same at 24, 30, 60, or 120 fps)."""

    max_interpolation_gap_s: float = 0.25  # PROTOTYPE: fill short dropouts only
    median_window_s: float = 0.15  # PROTOTYPE: removes single-frame spikes
    mean_window_s: float = 0.15  # PROTOTYPE: reduces jitter


@dataclass(frozen=True)
class RepCounterConfig:
    """Hysteresis state machine for counting squat repetitions."""

    down_threshold_deg: float = 140.0  # TEAM_METHOD: knee bends past 140 deg
    up_threshold_deg: float = 160.0  # TEAM_METHOD: straightens past 160 deg
    # A threshold must stay crossed this long before the state changes.
    min_state_duration_s: float = 0.10  # PROTOTYPE debounce
    min_rep_duration_s: float = 0.6  # PROTOTYPE: faster "reps" are jitter
    max_rep_duration_s: float = 15.0  # PROTOTYPE: longer is not one rep
    # A tracking gap longer than this inside a rep lowers its confidence.
    max_tracking_gap_s: float = 0.5  # PROTOTYPE
    # Dips below this (but never below down_threshold) are reported as
    # partial movements that were not counted.
    partial_threshold_deg: float = 155.0  # PROTOTYPE


@dataclass(frozen=True)
class OrientationConfig:
    """Side-view check. Index = horizontal shoulder+hip separation divided by
    torso length in the image. ~0 for a true side view, larger when facing
    the camera."""

    side_view_max_index: float = 0.25  # PROTOTYPE
    angled_view_max_index: float = 0.45  # PROTOTYPE


@dataclass(frozen=True)
class ScoringConfig:
    """Explainable deduction rules. Score starts at 10 (TEAM_METHOD)."""

    start_score: float = 10.0  # TEAM_METHOD
    # Depth: the poster's example tip is "aim for a knee angle under 100 deg
    # at the bottom". TEAM_METHOD. Deduction rate/cap are PROTOTYPE.
    depth_target_knee_angle_deg: float = 100.0
    depth_points_per_deg: float = 0.1  # 1 point per 10 deg above target
    depth_max_points: float = 3.0
    # Torso lean at the bottom, measured from vertical. Baseline reference
    # limit is a PROTOTYPE value (to be replaced by researched guidelines).
    torso_lean_limit_deg: float = 45.0
    torso_lean_points_per_deg: float = 0.2  # 1 point per 5 deg over limit
    torso_lean_max_points: float = 3.0
    # "Chest drop": torso tips further forward while standing up (hips rise
    # faster than the chest). PROTOTYPE threshold.
    chest_drop_threshold_deg: float = 10.0
    chest_drop_points_per_deg: float = 0.1
    chest_drop_max_points: float = 2.0
    # Reliability gates. A metric is only scored when it was measured
    # directly (not interpolated) in enough frames near the bottom.
    bottom_window_s: float = 0.25
    min_valid_fraction_near_bottom: float = 0.5
    min_valid_fraction_in_rep: float = 0.6


@dataclass(frozen=True)
class PersonalizationConfig:
    """Body-proportion personalization (see docs/PERSONALIZATION.md).

    The reference profile is an approximate "average adult" proportion used
    only as a comparison point. PROTOTYPE assumption: femur/torso = 0.85 and
    lower-leg/femur = 1.00. These should be confirmed against an
    anthropometric source (e.g., segment-length tables attributed to
    Drillis & Contini, 1966) before being presented as population norms.
    """

    reference_femur_to_torso: float = 0.85  # PROTOTYPE
    reference_lower_leg_to_femur: float = 1.00  # PROTOTYPE
    # Shin angle (from vertical) assumed by the geometric balance model.
    reference_shin_lean_deg: float = 30.0  # PROTOTYPE
    # Personalized adjustments are clamped to +/- this many degrees so that
    # one bad measurement cannot change the evaluation dramatically.
    max_torso_lean_adjustment_deg: float = 10.0  # PROTOTYPE bound
    # Entered vs. video-estimated ratios that differ by more than this
    # fraction trigger a "check your measurements / camera" warning.
    ratio_mismatch_tolerance: float = 0.25  # PROTOTYPE


@dataclass(frozen=True)
class InputLimits:
    """Upload and measurement validation limits."""

    allowed_video_extensions: tuple[str, ...] = (".mp4", ".mov", ".avi", ".m4v")
    max_upload_mb: float = 200.0
    min_duration_s: float = 1.5
    max_duration_s: float = 60.0
    min_fps: float = 10.0
    max_fps: float = 240.0
    min_short_side_px: int = 240
    # Broad sanity bounds for adult measurements in centimeters. These are
    # input-error guards, not anthropometric norms. PROTOTYPE.
    measurement_bounds_cm: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "torso": (25.0, 90.0),
            "femur": (20.0, 75.0),
            "lower_leg": (20.0, 75.0),
            "arm": (35.0, 110.0),
        }
    )
    # Ratios outside these ranges produce a warning (not an error).
    plausible_femur_to_torso: tuple[float, float] = (0.5, 1.5)
    plausible_lower_leg_to_femur: tuple[float, float] = (0.6, 1.5)


@dataclass(frozen=True)
class AnalysisConfig:
    """Bundle of every configuration section used by the pipeline."""

    pose: PoseConfig = field(default_factory=PoseConfig)
    smoothing: SmoothingConfig = field(default_factory=SmoothingConfig)
    reps: RepCounterConfig = field(default_factory=RepCounterConfig)
    orientation: OrientationConfig = field(default_factory=OrientationConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    personalization: PersonalizationConfig = field(default_factory=PersonalizationConfig)
    limits: InputLimits = field(default_factory=InputLimits)
    # "world" = MediaPipe 3D world landmarks (poster: "We use the 3D
    # landmarks"); "image" = 2D pixel coordinates. Joint angles use this;
    # lean angles and depth always use the 2D image (they need "vertical").
    joint_angle_source: str = "world"


DEFAULT_CONFIG = AnalysisConfig()

DISCLAIMER = (
    "RateMyForm scores are prototype estimates from a single-camera video. "
    "They are not a validated injury-risk, medical, or clinical assessment, "
    "and they do not replace a certified trainer, physical therapist, or "
    "medical professional. Stop if you feel pain."
)
