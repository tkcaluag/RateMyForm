"""Vector math for joint angles and segment lean.

All functions are vectorized: pass single points with shape (D,) or whole
time series with shape (N, D), where D is 2 (image pixels) or 3 (world
meters). Invalid inputs (NaN or zero-length segments) return NaN instead of
raising, so one bad frame never crashes the analysis.
"""
from __future__ import annotations

import numpy as np

_EPS = 1e-9


def angle_at(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Interior angle ABC in degrees, with the vertex at point B.

    Uses the dot-product formula  cos(theta) = (BA . BC) / (|BA| |BC|).
    Example: knee angle = angle_at(hip, knee, ankle). A straight leg gives
    180 degrees; a deep squat gives a small angle.
    """
    a, b, c = (np.asarray(p, dtype=float) for p in (a, b, c))
    ba = a - b
    bc = c - b
    norm_ba = np.linalg.norm(ba, axis=-1)
    norm_bc = np.linalg.norm(bc, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos_theta = np.sum(ba * bc, axis=-1) / (norm_ba * norm_bc)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    angle = np.degrees(np.arccos(cos_theta))
    degenerate = (norm_ba < _EPS) | (norm_bc < _EPS)
    return np.where(degenerate, np.nan, angle)


def lean_from_vertical(lower: np.ndarray, upper: np.ndarray, facing: int = 0) -> np.ndarray:
    """Signed angle (degrees) between segment lower->upper and image vertical.

    Image coordinates have y pointing DOWN, so "straight up" is (0, -1).
    0 deg = segment is vertical. Positive = leaning in the direction the
    person faces (forward). `facing` is +1 if the person faces the +x
    direction (right side of the image), -1 if facing left, 0 if unknown (in
    which case the unsigned magnitude is returned).

    Examples:
        torso lean = lean_from_vertical(hip, shoulder, facing)
        shin lean  = lean_from_vertical(ankle, knee, facing)
    """
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    v = upper - lower
    dx = v[..., 0]
    up = -v[..., 1]  # flip because image y points down
    length = np.hypot(dx, up)
    direction = facing if facing in (1, -1) else 1
    signed = np.degrees(np.arctan2(dx * direction, up))
    result = signed if facing in (1, -1) else np.abs(signed)
    return np.where(length < _EPS, np.nan, result)


def segment_length(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Euclidean distance between two points (or two series of points)."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    return np.linalg.norm(p - q, axis=-1)
