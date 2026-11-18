"""Explainable 0-10 scoring: exact deductions, reliability gating, determinism."""
import pytest

from ratemyform.config import PersonalizationConfig, ScoringConfig
from ratemyform.personalization import BodyMeasurements, baseline_thresholds, personalized_thresholds
from ratemyform.rep_analysis import RepMeasurements
from ratemyform.scoring import compare_scores, overall_score, score_rep

S = ScoringConfig()
BASE = baseline_thresholds(S)
ALL_OK = {"knee_angle": True, "hip_angle": True, "torso_lean": True, "shin_lean": True, "depth_ratio": True, "chest_drop": True}


def rep(knee=85.0, lean=35.0, drop=0.0, reliable=None, n=1):
    return RepMeasurements(
        rep_number=n, start_time_s=0, bottom_time_s=1, end_time_s=2, duration_s=2,
        knee_angle_deg=knee, hip_angle_deg=70, torso_lean_deg=lean, shin_lean_deg=30,
        depth_ratio=0.0, depth_class="around parallel", chest_drop_deg=drop,
        knee_angle_2d_deg=knee, knee_angle_3d_deg=knee, reliable=dict(reliable or ALL_OK),
    )


def test_good_rep_scores_ten():
    s = score_rep(rep(), BASE, S)
    assert s.score == 10.0 and s.deductions == [] and s.not_scored == []


def test_depth_deduction_math():
    s = score_rep(rep(knee=120.0), BASE, S)  # 20 deg over 100 -> 20 * 0.1 = 2.0
    assert s.score == 8.0
    (d,) = s.deductions
    assert d.rule == "depth" and d.points == 2.0
    assert d.measured_value == 120.0 and d.reference_value == 100.0
    assert "120" in d.message and "100" in d.message


def test_depth_deduction_is_capped():
    assert score_rep(rep(knee=170.0), BASE, S).score == 10.0 - S.depth_max_points


def test_torso_lean_deduction_math():
    s = score_rep(rep(lean=55.0), BASE, S)  # 10 deg over 45 -> 10 * 0.2 = 2.0
    assert s.score == 8.0
    assert s.deductions[0].rule == "torso_lean"
    assert "farther forward than the 45 deg reference" in s.deductions[0].message


def test_chest_drop_deduction_math():
    s = score_rep(rep(drop=25.0), BASE, S)  # 15 deg over 10 -> 1.5
    assert s.score == 8.5 and s.deductions[0].rule == "chest_drop"


def test_deductions_add_up_and_never_go_below_zero():
    s = score_rep(rep(knee=180.0, lean=90.0, drop=90.0), BASE, S)
    assert s.score == pytest.approx(10 - 3 - 3 - 2)
    assert s.score >= 0


def test_unreliable_metric_is_not_deducted():
    unreliable = dict(ALL_OK, torso_lean=False, chest_drop=False)
    s = score_rep(rep(lean=80.0, drop=40.0, reliable=unreliable), BASE, S)
    assert s.score == 10.0
    assert set(s.not_scored) == {"torso_lean", "chest_drop"}


def test_nothing_reliable_means_no_score_not_a_perfect_ten():
    none_ok = {k: False for k in ALL_OK}
    s = score_rep(rep(knee=160.0, reliable=none_ok), BASE, S)
    assert s.score is None and s.deductions == []


def test_overall_score_ignores_unscored_reps():
    none_ok = {k: False for k in ALL_OK}
    scores = [score_rep(rep(knee=120.0, n=1), BASE, S), score_rep(rep(n=2, reliable=none_ok), BASE, S), score_rep(rep(n=3), BASE, S)]
    assert overall_score(scores) == 9.0  # mean of 8.0 and 10.0
    assert overall_score([]) is None
    assert overall_score([scores[1]]) is None


def test_scoring_is_deterministic():
    a = [score_rep(rep(knee=113.3, lean=51.7, drop=13.9), BASE, S).to_dict() for _ in range(5)]
    assert all(x == a[0] for x in a)


def test_personalized_vs_baseline_for_long_femur():
    long_femur = BodyMeasurements(torso_cm=46, femur_cm=48, lower_leg_cm=42, arm_cm=60)
    pers = personalized_thresholds(long_femur, S, PersonalizationConfig())
    m = rep(lean=52.0)
    b, p = score_rep(m, BASE, S), score_rep(m, pers, S)
    assert b.score == pytest.approx(10 - 7 * 0.2)  # 52 vs 45
    assert p.score == 10.0  # 52 vs 55 (limit raised by the bounded +10 adjustment)
    rows = compare_scores([b], [p])
    assert rows[0]["difference"] == pytest.approx(1.4)
    assert rows[0]["rules_changed"] == ["torso_lean"]


def test_personalized_message_mentions_proportions():
    pers = personalized_thresholds(BodyMeasurements(56, 38, 42, 60), S, PersonalizationConfig())
    s = score_rep(rep(lean=50.0), pers, S)
    assert "for your body proportions" in s.deductions[0].message
    assert s.deductions[0].personalized_reference is True
