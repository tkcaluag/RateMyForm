"""Explainable 0-10 scoring.

Every rep starts at 10 points (our methodology). Each rule below can deduct
points, and every deduction records the measured value, the reference value,
and a plain-language explanation. The same inputs always produce the same
score because there is no randomness and every number is rounded the same
way.

Rules (rates and caps are in config.ScoringConfig):
    depth       knee angle at the bottom above the depth target
    torso_lean  torso lean at the bottom beyond the (personalized) limit
    chest_drop  torso tips further forward while standing up

Metrics marked unreliable for a rep are skipped and reported as
observations instead. Missing landmarks therefore never lower a score.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .config import ScoringConfig
from .personalization import Thresholds
from .rep_analysis import RepMeasurements


@dataclass
class Deduction:
    rule: str
    points: float
    measured_value: float
    reference_value: float
    unit: str
    message: str
    personalized_reference: bool  # True if personalization changed the reference

    def to_dict(self) -> dict:
        d = asdict(self)
        d["measured_value"] = round(self.measured_value, 1)
        d["reference_value"] = round(self.reference_value, 1)
        return d


RULES = ("depth", "torso_lean", "chest_drop")


@dataclass
class RepScore:
    rep_number: int
    label: str  # "baseline" or "personalized"
    score: float | None  # None = no rule could be scored reliably for this rep
    deductions: list[Deduction] = field(default_factory=list)
    not_scored: list[str] = field(default_factory=list)  # rules skipped (unreliable data)

    @property
    def fully_scored(self) -> bool:
        return not self.not_scored

    def to_dict(self) -> dict:
        return {
            "rep_number": self.rep_number,
            "label": self.label,
            "score": self.score,
            "deductions": [d.to_dict() for d in self.deductions],
            "not_scored": list(self.not_scored),
        }


def _points(excess: float, rate: float, cap: float) -> float:
    return round(min(cap, max(0.0, excess) * rate), 1)


def score_rep(m: RepMeasurements, th: Thresholds, cfg: ScoringConfig) -> RepScore:
    deductions: list[Deduction] = []
    not_scored: list[str] = []
    n = m.rep_number

    # Depth --------------------------------------------------------------
    if m.reliable.get("knee_angle") and m.knee_angle_deg is not None:
        excess = m.knee_angle_deg - th.depth_target_knee_angle_deg
        pts = _points(excess, cfg.depth_points_per_deg, cfg.depth_max_points)
        if pts > 0:
            deductions.append(
                Deduction(
                    rule="depth",
                    points=pts,
                    measured_value=m.knee_angle_deg,
                    reference_value=th.depth_target_knee_angle_deg,
                    unit="deg",
                    message=(
                        f"Rep {n}: your knee angle at the bottom was about {m.knee_angle_deg:.0f} deg, "
                        f"above the {th.depth_target_knee_angle_deg:.0f} deg reference depth "
                        f"(-{pts:.1f})."
                    ),
                    personalized_reference=False,
                )
            )
    else:
        not_scored.append("depth")

    # Torso lean ----------------------------------------------------------
    if m.reliable.get("torso_lean") and m.torso_lean_deg is not None:
        excess = m.torso_lean_deg - th.torso_lean_limit_deg
        pts = _points(excess, cfg.torso_lean_points_per_deg, cfg.torso_lean_max_points)
        if pts > 0:
            ref_note = " for your body proportions" if th.label == "personalized" else ""
            deductions.append(
                Deduction(
                    rule="torso_lean",
                    points=pts,
                    measured_value=m.torso_lean_deg,
                    reference_value=th.torso_lean_limit_deg,
                    unit="deg",
                    message=(
                        f"Rep {n}: your torso leaned about {m.torso_lean_deg:.0f} deg from vertical at the "
                        f"bottom, farther forward than the {th.torso_lean_limit_deg:.0f} deg reference"
                        f"{ref_note} (-{pts:.1f})."
                    ),
                    personalized_reference=th.torso_lean_adjustment_deg != 0.0,
                )
            )
    else:
        not_scored.append("torso_lean")

    # Chest drop ----------------------------------------------------------
    if m.reliable.get("chest_drop") and m.chest_drop_deg is not None:
        excess = m.chest_drop_deg - th.chest_drop_threshold_deg
        pts = _points(excess, cfg.chest_drop_points_per_deg, cfg.chest_drop_max_points)
        if pts > 0:
            deductions.append(
                Deduction(
                    rule="chest_drop",
                    points=pts,
                    measured_value=m.chest_drop_deg,
                    reference_value=th.chest_drop_threshold_deg,
                    unit="deg",
                    message=(
                        f"Rep {n}: your torso tipped about {m.chest_drop_deg:.0f} deg further forward while "
                        f"standing up (reference: under {th.chest_drop_threshold_deg:.0f} deg), which can mean "
                        f"the hips rose faster than the chest (-{pts:.1f})."
                    ),
                    personalized_reference=False,
                )
            )
    else:
        not_scored.append("chest_drop")

    if len(not_scored) == len(RULES):
        # Nothing could be measured reliably: report "not scored" rather than
        # a misleading perfect 10.
        return RepScore(rep_number=n, label=th.label, score=None, deductions=[], not_scored=not_scored)
    total = round(sum(d.points for d in deductions), 1)
    score = round(max(0.0, min(cfg.start_score, cfg.start_score - total)), 1)
    return RepScore(rep_number=n, label=th.label, score=score, deductions=deductions, not_scored=not_scored)


def overall_score(rep_scores: list[RepScore]) -> float | None:
    """Average of the scored reps, or None if no rep could be scored."""
    scored = [r.score for r in rep_scores if r.score is not None]
    if not scored:
        return None
    return round(sum(scored) / len(scored), 1)


def compare_scores(baseline: list[RepScore], personalized: list[RepScore]) -> list[dict]:
    """Rep-by-rep difference between the two evaluations."""
    rows = []
    for b, p in zip(baseline, personalized):
        b_rules = {d.rule: d.points for d in b.deductions}
        p_rules = {d.rule: d.points for d in p.deductions}
        changed = sorted(r for r in set(b_rules) | set(p_rules) if b_rules.get(r, 0.0) != p_rules.get(r, 0.0))
        diff = None if b.score is None or p.score is None else round(p.score - b.score, 1)
        rows.append(
            {
                "rep_number": b.rep_number,
                "baseline_score": b.score,
                "personalized_score": p.score,
                "difference": diff,
                "rules_changed": changed,
            }
        )
    return rows
