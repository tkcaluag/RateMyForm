"""Turn scores and measurements into short, actionable tips.

Rules-based on purpose: every sentence can be traced to a measured number
and a configured reference value. Language stays non-diagnostic ("may",
"reference range"), following our Project Considerations: the app reports
what it measured and never says a movement is definitely safe or unsafe.
"""
from __future__ import annotations

import numpy as np

from .personalization import Thresholds
from .rep_analysis import RepMeasurements
from .scoring import RepScore


def _reps_with(scores: list[RepScore], rule: str) -> list[int]:
    return [s.rep_number for s in scores if any(d.rule == rule for d in s.deductions)]


def _fmt_reps(reps: list[int]) -> str:
    if len(reps) == 1:
        return f"rep {reps[0]}"
    return "reps " + ", ".join(str(r) for r in reps[:-1]) + f" and {reps[-1]}"


def summary_feedback(
    measurements: list[RepMeasurements],
    scores: list[RepScore],
    thresholds: Thresholds,
) -> list[str]:
    """Overall tips for the personalized evaluation."""
    tips: list[str] = []
    if not measurements:
        return [
            "No complete squat reps were detected. Make sure your whole body is in the frame, you start "
            "and finish each rep standing tall, and the camera is placed to your side."
        ]
    total = len(measurements)

    depth_reps = _reps_with(scores, "depth")
    if depth_reps:
        angles = [m.knee_angle_deg for m in measurements if m.rep_number in depth_reps and m.knee_angle_deg is not None]
        tips.append(
            f"Depth: in {_fmt_reps(depth_reps)} ({len(depth_reps)} of {total}) your knees bent to about "
            f"{np.mean(angles):.0f} deg at the bottom, above the {thresholds.depth_target_knee_angle_deg:.0f} deg "
            "reference. If it is comfortable and pain-free, try sitting a little lower."
        )

    lean_reps = _reps_with(scores, "torso_lean")
    if lean_reps:
        worst = max(
            (m for m in measurements if m.rep_number in lean_reps and m.torso_lean_deg is not None),
            key=lambda m: m.torso_lean_deg,
        )
        tips.append(
            f"Torso lean: your torso leaned farther forward than the {thresholds.torso_lean_limit_deg:.0f} deg "
            f"reference range during {_fmt_reps(lean_reps)} (up to {worst.torso_lean_deg:.0f} deg in rep "
            f"{worst.rep_number}). Try keeping your chest up and bracing your core."
        )

    drop_reps = _reps_with(scores, "chest_drop")
    if drop_reps:
        tips.append(
            f"Standing up: during {_fmt_reps(drop_reps)} your torso tipped forward as you rose. Try driving "
            "your chest and hips up together."
        )

    knee = [m.knee_angle_deg for m in measurements if m.reliable.get("knee_angle") and m.knee_angle_deg is not None]
    if len(knee) >= 3:
        spread = max(knee) - min(knee)
        if spread >= 15:
            tips.append(
                f"Consistency: your bottom knee angle varied by {spread:.0f} deg between reps. "
                "Aiming for the same depth each rep makes progress easier to track."
            )

    unscored = sum(1 for s in scores if s.not_scored)
    if unscored:
        tips.append(
            f"{unscored} of {total} reps had at least one measurement that could not be scored reliably "
            "(see each rep for details). Better lighting, fitted clothing, and a side-on camera usually help."
        )

    if all(s.score is None for s in scores):
        tips.insert(
            0,
            "None of the counted reps could be scored reliably, so no form score is given. "
            "This is a data-quality problem, not a judgment of your form.",
        )
    elif not any(s.deductions for s in scores):
        tips.insert(
            0,
            "Every measurement that could be scored stayed within the reference ranges this prototype "
            "checks (depth, torso lean, and chest drop).",
        )
    return tips
