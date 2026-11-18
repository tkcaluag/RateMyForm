"""Input validation for body measurements and uploaded videos.

Errors block the analysis. Warnings let it run but are shown to the user.
Validation happens here (server side, in Python) rather than only in the
Streamlit widgets, so the CLI scripts and tests get the same protection.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path

from .config import InputLimits
from .personalization import BodyMeasurements

CM_PER_INCH = 2.54
UNITS = ("cm", "in")
MEASUREMENT_FIELDS = ("torso", "femur", "lower_leg", "arm")
FIELD_LABELS = {"torso": "Torso", "femur": "Femur (thigh)", "lower_leg": "Lower leg", "arm": "Arm"}


@dataclass
class ValidationResult:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


@dataclass
class MeasurementValidation(ValidationResult):
    measurements: BodyMeasurements | None = None


def to_cm(value: float, unit: str) -> float:
    if unit == "cm":
        return float(value)
    if unit == "in":
        return float(value) * CM_PER_INCH
    raise ValueError(f"Unsupported unit {unit!r}; use 'cm' or 'in'.")


def validate_measurements(values: dict, unit: str, limits: InputLimits | None = None) -> MeasurementValidation:
    """Check that all four measurements are present, numeric, and plausible."""
    limits = limits or InputLimits()
    result = MeasurementValidation()
    if unit not in UNITS:
        result.errors.append(f"Unit must be one of {', '.join(UNITS)}.")
        return result

    cm: dict[str, float] = {}
    for name in MEASUREMENT_FIELDS:
        label = FIELD_LABELS[name]
        raw = values.get(name)
        if raw is None or (isinstance(raw, str) and not raw.strip()):
            result.errors.append(f"{label} length is required.")
            continue
        if isinstance(raw, bool):
            result.errors.append(f"{label} length must be a number.")
            continue
        try:
            number = float(raw)
        except (TypeError, ValueError):
            result.errors.append(f"{label} length must be a number.")
            continue
        if not math.isfinite(number) or number <= 0:
            result.errors.append(f"{label} length must be a positive number.")
            continue
        value_cm = to_cm(number, unit)
        lo, hi = limits.measurement_bounds_cm[name]
        if not lo <= value_cm <= hi:
            lo_u, hi_u = (lo, hi) if unit == "cm" else (lo / CM_PER_INCH, hi / CM_PER_INCH)
            result.errors.append(
                f"{label} length {number:g} {unit} is outside the accepted range "
                f"{lo_u:.0f}-{hi_u:.0f} {unit}. Check the value and the unit."
            )
            continue
        cm[name] = value_cm

    if result.errors:
        return result

    m = BodyMeasurements(torso_cm=cm["torso"], femur_cm=cm["femur"], lower_leg_cm=cm["lower_leg"], arm_cm=cm["arm"])
    r = m.ratios()
    lo, hi = limits.plausible_femur_to_torso
    if not lo <= r.femur_to_torso <= hi:
        result.warnings.append(
            f"Your femur/torso ratio is {r.femur_to_torso:.2f}, which is unusual (expected roughly {lo}-{hi}). "
            "Please double-check the torso and femur measurements."
        )
    lo, hi = limits.plausible_lower_leg_to_femur
    if not lo <= r.lower_leg_to_femur <= hi:
        result.warnings.append(
            f"Your lower-leg/femur ratio is {r.lower_leg_to_femur:.2f}, which is unusual (expected roughly "
            f"{lo}-{hi}). Please double-check the lower-leg and femur measurements."
        )
    result.measurements = m
    return result


def validate_upload(filename: str, size_bytes: int, limits: InputLimits | None = None) -> ValidationResult:
    """Check the file extension and size before saving anything to disk."""
    limits = limits or InputLimits()
    result = ValidationResult()
    ext = os.path.splitext(filename or "")[1].lower()
    if ext not in limits.allowed_video_extensions:
        allowed = ", ".join(limits.allowed_video_extensions)
        result.errors.append(f"Unsupported file type {ext or '(none)'}. Please upload one of: {allowed}.")
    if size_bytes <= 0:
        result.errors.append("The uploaded file is empty.")
    elif size_bytes > limits.max_upload_mb * 1024 * 1024:
        result.errors.append(f"The file is {size_bytes / 1e6:.0f} MB; the limit is {limits.max_upload_mb:.0f} MB.")
    return result


def validate_video_properties(fps: float, frame_count: int, width: int, height: int, limits: InputLimits | None = None) -> ValidationResult:
    """Check frame rate, duration, and resolution after OpenCV opens the file."""
    limits = limits or InputLimits()
    result = ValidationResult()
    if not (limits.min_fps <= fps <= limits.max_fps):
        result.errors.append(
            f"Frame rate {fps:.1f} fps is outside the supported range {limits.min_fps:.0f}-{limits.max_fps:.0f} fps."
        )
    duration = frame_count / fps if fps > 0 and frame_count > 0 else 0.0
    if frame_count <= 0:
        result.warnings.append("The video does not report its length; it will be checked while processing.")
    elif duration < limits.min_duration_s:
        result.errors.append(f"The video is {duration:.1f} s long; please upload at least {limits.min_duration_s:.1f} s.")
    elif duration > limits.max_duration_s:
        result.errors.append(
            f"The video is {duration:.0f} s long; the prototype processes up to {limits.max_duration_s:.0f} s. "
            "Please trim it to the squat set."
        )
    if min(width, height) < limits.min_short_side_px:
        result.errors.append(f"The video resolution {width}x{height} is too low (need at least {limits.min_short_side_px}p).")
    return result


def safe_temp_video_path(directory: str | Path, original_filename: str) -> Path:
    """Build a temp path that never uses the user's filename (only its extension).

    Prevents path-traversal tricks like '../../evil.mp4' in uploaded names.
    """
    ext = os.path.splitext(original_filename or "")[1].lower()
    if ext not in InputLimits().allowed_video_extensions:
        ext = ".mp4"
    return Path(directory) / f"upload{ext}"
