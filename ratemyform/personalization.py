"""Body-proportion personalization.

THE IDEA (from our proposal): two people can squat differently and both be
moving reasonably, because their bones are different lengths. A person with
long thighs (femurs) relative to their torso usually has to lean their torso
further forward to stay balanced at the same depth.

WHAT WE ACTUALLY PERSONALIZE (and what we do not):
  * Personalized: the reference LIMIT for torso lean at the bottom.
  * Not personalized: the depth target and the chest-drop rule. We did not
    find a defensible, documented way to adjust those from body ratios yet.

HOW (a simple geometric balance model, a PROTOTYPE heuristic):
  Look at the body from the side at parallel depth (thigh horizontal).
  Put the ankle at x = 0 and let x point forward. With torso length = 1:

      knee_x     =  L_shin * sin(shin_lean)
      hip_x      =  knee_x - L_femur            (thigh is horizontal)
      shoulder_x =  hip_x + sin(torso_lean)

  If we assume the shoulders stay roughly over the feet for balance
  (shoulder_x = 0), then

      sin(torso_lean) = L_femur - L_shin * sin(shin_lean)
                      = R1 * (1 - R2 * sin(shin_lean))

  where R1 = femur / torso and R2 = lower_leg / femur.

  Longer femur (bigger R1) -> more lean predicted.
  Longer shin (bigger R2) -> knees travel further forward -> less lean.

  We do NOT use the model's absolute prediction as a rule. We only use the
  DIFFERENCE between the user's prediction and a reference body's
  prediction, and we clamp that difference to +/- 10 degrees. So the model
  can nudge the reference range, but never replace it.

LIMITATIONS: the model ignores mass distribution, arm position, bar
position, ankle mobility differences, and flexibility. It has not been
validated against measured squats. It is a transparent starting point for
the team's guideline research, not a scientific rule.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

from .config import PersonalizationConfig, ScoringConfig


@dataclass(frozen=True)
class BodyMeasurements:
    """User-entered lengths, always stored in centimeters."""

    torso_cm: float
    femur_cm: float
    lower_leg_cm: float
    arm_cm: float

    def ratios(self) -> "BodyRatios":
        return BodyRatios(
            femur_to_torso=self.femur_cm / self.torso_cm,
            lower_leg_to_femur=self.lower_leg_cm / self.femur_cm,
            arm_to_torso=self.arm_cm / self.torso_cm,
        )

    def to_dict(self) -> dict:
        return {k: round(v, 2) for k, v in asdict(self).items()}


@dataclass(frozen=True)
class BodyRatios:
    femur_to_torso: float
    lower_leg_to_femur: float
    arm_to_torso: float

    def to_dict(self) -> dict:
        return {k: round(v, 3) for k, v in asdict(self).items()}


@dataclass
class Thresholds:
    """The reference values one evaluation uses."""

    label: str  # "baseline" or "personalized"
    depth_target_knee_angle_deg: float
    torso_lean_limit_deg: float
    chest_drop_threshold_deg: float
    torso_lean_adjustment_deg: float = 0.0
    rationale: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        for key in ("depth_target_knee_angle_deg", "torso_lean_limit_deg", "chest_drop_threshold_deg", "torso_lean_adjustment_deg"):
            d[key] = round(d[key], 1)
        return d


@dataclass
class RatioCheck:
    """Entered ratio vs. the ratio estimated from the video."""

    name: str
    entered: float
    video_estimate: float | None
    relative_difference: float | None
    consistent: bool | None  # None = could not check
    message: str


def predicted_torso_lean_deg(femur_to_torso: float, lower_leg_to_femur: float, shin_lean_deg: float) -> float:
    """Torso lean (deg from vertical) the balance model predicts at parallel."""
    s = femur_to_torso * (1.0 - lower_leg_to_femur * math.sin(math.radians(shin_lean_deg)))
    s = max(-1.0, min(1.0, s))
    return math.degrees(math.asin(s))


def baseline_thresholds(scoring: ScoringConfig) -> Thresholds:
    """The same reference values for everyone (no personalization)."""
    return Thresholds(
        label="baseline",
        depth_target_knee_angle_deg=scoring.depth_target_knee_angle_deg,
        torso_lean_limit_deg=scoring.torso_lean_limit_deg,
        chest_drop_threshold_deg=scoring.chest_drop_threshold_deg,
        rationale=["Baseline: identical reference values for every user (the 'one correct form' approach)."],
    )


def personalized_thresholds(
    measurements: BodyMeasurements,
    scoring: ScoringConfig,
    pcfg: PersonalizationConfig,
) -> Thresholds:
    """Adjust the torso-lean limit from body ratios, within +/- the bound."""
    r = measurements.ratios()
    user_pred = predicted_torso_lean_deg(r.femur_to_torso, r.lower_leg_to_femur, pcfg.reference_shin_lean_deg)
    ref_pred = predicted_torso_lean_deg(
        pcfg.reference_femur_to_torso, pcfg.reference_lower_leg_to_femur, pcfg.reference_shin_lean_deg
    )
    raw_adjust = user_pred - ref_pred
    bound = pcfg.max_torso_lean_adjustment_deg
    adjust = round(max(-bound, min(bound, raw_adjust)), 1)
    clamped = abs(raw_adjust) > bound

    direction = "more" if adjust > 0 else "less" if adjust < 0 else "the same"
    rationale = [
        (
            f"Your femur/torso ratio is {r.femur_to_torso:.2f} (reference {pcfg.reference_femur_to_torso:.2f}) "
            f"and your lower-leg/femur ratio is {r.lower_leg_to_femur:.2f} "
            f"(reference {pcfg.reference_lower_leg_to_femur:.2f})."
        ),
        (
            f"A simple balance model predicts about {user_pred:.1f} deg of torso lean at parallel for your "
            f"proportions vs. {ref_pred:.1f} deg for the reference proportions, so your torso-lean limit "
            f"allows {abs(adjust):.1f} deg {direction} lean than the baseline."
        ),
    ]
    if clamped:
        rationale.append(
            f"The model suggested {raw_adjust:+.1f} deg, which was capped at {bound:.0f} deg so a single "
            "measurement cannot change the evaluation too much."
        )
    rationale.append(
        "Depth and chest-drop reference values are not personalized in this prototype. "
        "This adjustment is a prototype heuristic, not a validated guideline."
    )
    return Thresholds(
        label="personalized",
        depth_target_knee_angle_deg=scoring.depth_target_knee_angle_deg,
        torso_lean_limit_deg=scoring.torso_lean_limit_deg + adjust,
        chest_drop_threshold_deg=scoring.chest_drop_threshold_deg,
        torso_lean_adjustment_deg=adjust,
        rationale=rationale,
    )


_RATIO_LABELS = {
    "femur_to_torso": "femur/torso",
    "lower_leg_to_femur": "lower-leg/femur",
    "arm_to_torso": "arm/torso",
}


def compare_ratios(entered: BodyRatios, video: dict, tolerance: float) -> list[RatioCheck]:
    """Flag entered ratios that disagree with the video estimate.

    A mismatch can mean a typo, a different measuring method, a camera that
    is not side-on, or an inaccurate pose estimate. It is a prompt to
    double-check, not proof that either number is wrong.
    """
    checks: list[RatioCheck] = []
    for name, label in _RATIO_LABELS.items():
        e = getattr(entered, name)
        v = video.get(name)
        if v is None or not math.isfinite(v):
            checks.append(RatioCheck(name, e, None, None, None, f"Could not estimate the {label} ratio from the video."))
            continue
        diff = abs(e - v) / e
        ok = diff <= tolerance
        msg = (
            f"Entered {label} ratio {e:.2f} vs. video estimate {v:.2f} ({diff:.0%} difference)."
            + ("" if ok else " Please double-check this measurement and that the camera is side-on.")
        )
        checks.append(RatioCheck(name, round(e, 3), round(v, 3), round(diff, 3), ok, msg))
    return checks
