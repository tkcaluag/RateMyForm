"""Body-ratio personalization: bounded, explainable, distinguishable from baseline."""
import pytest

from ratemyform.config import PersonalizationConfig, ScoringConfig
from ratemyform.personalization import (
    BodyMeasurements,
    BodyRatios,
    baseline_thresholds,
    compare_ratios,
    personalized_thresholds,
    predicted_torso_lean_deg,
)


def test_ratios():
    r = BodyMeasurements(torso_cm=50, femur_cm=45, lower_leg_cm=40, arm_cm=60).ratios()
    assert r.femur_to_torso == pytest.approx(0.9)
    assert r.lower_leg_to_femur == pytest.approx(40 / 45)
    assert r.arm_to_torso == pytest.approx(1.2)


def test_model_longer_femur_predicts_more_lean():
    assert predicted_torso_lean_deg(1.0, 1.0, 30) > predicted_torso_lean_deg(0.8, 1.0, 30)


def test_model_longer_shin_predicts_less_lean():
    assert predicted_torso_lean_deg(0.9, 1.2, 30) < predicted_torso_lean_deg(0.9, 0.9, 30)


def test_model_hand_calculation():
    # sin(lean) = 0.85 * (1 - 1.0 * sin 30) = 0.425 -> 25.15 deg
    assert predicted_torso_lean_deg(0.85, 1.0, 30) == pytest.approx(25.15, abs=0.01)


def test_reference_proportions_get_no_adjustment(average_body):
    th = personalized_thresholds(average_body, ScoringConfig(), PersonalizationConfig())
    assert th.torso_lean_adjustment_deg == 0.0
    assert th.torso_lean_limit_deg == ScoringConfig().torso_lean_limit_deg


def test_long_femur_relaxes_limit_and_short_femur_tightens_it():
    s, p = ScoringConfig(), PersonalizationConfig()
    long_f = personalized_thresholds(BodyMeasurements(46, 48, 42, 60), s, p)
    short_f = personalized_thresholds(BodyMeasurements(56, 38, 42, 60), s, p)
    assert long_f.torso_lean_limit_deg > s.torso_lean_limit_deg
    assert short_f.torso_lean_limit_deg < s.torso_lean_limit_deg


def test_adjustment_is_bounded():
    s, p = ScoringConfig(), PersonalizationConfig()
    extreme = personalized_thresholds(BodyMeasurements(30, 70, 25, 60), s, p)
    assert extreme.torso_lean_adjustment_deg == pytest.approx(p.max_torso_lean_adjustment_deg)
    assert any("capped" in line for line in extreme.rationale)
    extreme_low = personalized_thresholds(BodyMeasurements(85, 25, 70, 60), s, p)
    assert extreme_low.torso_lean_adjustment_deg == pytest.approx(-p.max_torso_lean_adjustment_deg)


def test_only_torso_lean_is_personalized():
    s, p = ScoringConfig(), PersonalizationConfig()
    th = personalized_thresholds(BodyMeasurements(46, 48, 42, 60), s, p)
    base = baseline_thresholds(s)
    assert th.depth_target_knee_angle_deg == base.depth_target_knee_angle_deg
    assert th.chest_drop_threshold_deg == base.chest_drop_threshold_deg
    assert th.label == "personalized" and base.label == "baseline"


def test_rationale_labels_heuristic():
    th = personalized_thresholds(BodyMeasurements(46, 48, 42, 60), ScoringConfig(), PersonalizationConfig())
    text = " ".join(th.rationale)
    assert "not a validated guideline" in text
    assert "femur/torso" in text


def test_compare_ratios_flags_mismatch():
    entered = BodyRatios(femur_to_torso=0.9, lower_leg_to_femur=1.0, arm_to_torso=1.2)
    checks = compare_ratios(entered, {"femur_to_torso": 0.6, "lower_leg_to_femur": 1.02, "arm_to_torso": None}, 0.25)
    by_name = {c.name: c for c in checks}
    assert by_name["femur_to_torso"].consistent is False
    assert by_name["lower_leg_to_femur"].consistent is True
    assert by_name["arm_to_torso"].consistent is None
