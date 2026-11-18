"""End-to-end analysis on SYNTHETIC scenarios with known ground truth.

These run the real analysis code (metrics -> rep counter -> scoring ->
feedback) on generated landmarks. They are simulated data: passing them shows
the logic does what it is designed to do, not that it is accurate on real
people.
"""
import json

import numpy as np
import pytest

from ratemyform.pipeline import analyze_sequence
from ratemyform.synthetic import default_measurements_for, generate_squat, scenarios

SCENARIOS = scenarios()


def run(name):
    spec, _ = SCENARIOS[name]
    syn = generate_squat(spec)
    return syn, analyze_sequence(syn.sequence, default_measurements_for(spec))


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_rep_count_matches_ground_truth(name):
    syn, res = run(name)
    assert res.rep_count == syn.truth["expected_rep_count"]


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_report_is_json_serializable(name):
    _, res = run(name)
    report = json.loads(json.dumps(res.to_dict()))
    assert report["source"] == "synthetic"
    assert report["summary"]["rep_count"] == res.rep_count
    assert len(report["reps"]) == res.rep_count


def test_bottom_angles_match_ground_truth():
    syn, res = run("shallow_reps")
    measured = [m.knee_angle_deg for m in res.rep_measurements]
    np.testing.assert_allclose(measured, syn.truth["bottom_knee_angles_deg"], atol=1.0)


def test_clean_reps_score_ten_with_no_warnings():
    _, res = run("clean_5_reps")
    assert res.overall_baseline == 10.0 and res.overall_personalized == 10.0
    assert res.warnings == []
    assert res.orientation.label == "side"


def test_shallow_reps_get_depth_deductions():
    _, res = run("shallow_reps")
    for s in res.personalized_scores:
        assert [d.rule for d in s.deductions] == ["depth"]
    assert any("Depth" in tip for tip in res.feedback)


def test_personalization_changes_feedback_for_long_femur():
    _, res = run("forward_lean_long_femur")
    assert res.personalized.torso_lean_adjustment_deg > 0
    assert res.overall_personalized > res.overall_baseline
    assert all(row["rules_changed"] == ["torso_lean"] for row in res.comparison)


def test_personalization_is_stricter_for_short_femur():
    _, res = run("forward_lean_short_femur")
    assert res.personalized.torso_lean_adjustment_deg < 0
    assert res.overall_personalized < res.overall_baseline


def test_chest_drop_detected_only_in_affected_reps():
    syn, res = run("chest_drop")
    flagged = [s.rep_number for s in res.personalized_scores if any(d.rule == "chest_drop" for d in s.deductions)]
    assert flagged == [2, 3, 5]
    measured = [m.chest_drop_deg for m in res.rep_measurements]
    np.testing.assert_allclose(measured, syn.truth["chest_drops_deg"], atol=2.0)


def test_hidden_shoulder_is_not_scored_as_bad_form():
    # Rep 2 leans 60 deg, but the shoulder is hidden: lean must NOT be deducted.
    _, res = run("occluded_shoulder")
    rep2 = res.personalized_scores[1]
    assert "torso_lean" in rep2.not_scored
    assert all(d.rule != "torso_lean" for d in rep2.deductions)
    assert res.rep_measurements[1].reliable["torso_lean"] is False
    assert res.rep_measurements[1].reliable["knee_angle"] is True


def test_tracking_gap_rep_is_low_confidence_and_unscored():
    _, res = run("tracking_gap")
    rep4 = res.rep_measurements[3]
    assert rep4.low_confidence
    assert res.personalized_scores[3].score is None
    # The other reps are still scored normally.
    assert all(s.score == 10.0 for i, s in enumerate(res.personalized_scores) if i != 3)


def test_front_view_is_flagged_and_lean_not_scored():
    _, res = run("front_view")
    assert res.orientation.label == "front_or_back"
    assert any("side" in w.lower() for w in res.warnings)
    for s in res.personalized_scores:
        assert "torso_lean" in s.not_scored


def test_incomplete_and_partial_movements_are_reported():
    _, res = run("incomplete_last_rep")
    assert [u.kind for u in res.rep_detection.uncounted] == ["incomplete"]
    _, res = run("partial_dips")
    assert [u.kind for u in res.rep_detection.uncounted] == ["partial", "partial"]


def test_frame_rate_does_not_change_results():
    results = [run(n)[1] for n in ("fps_24", "clean_5_reps", "fps_60", "fps_120")]
    counts = {r.rep_count for r in results}
    assert counts == {5}
    for r in results:
        np.testing.assert_allclose([m.knee_angle_deg for m in r.rep_measurements], 80.0, atol=1.0)


def test_noisy_input_still_counts_correctly_and_scores_cleanly():
    _, res = run("noisy_jitter")
    assert res.rep_count == 5
    assert res.overall_personalized >= 9.5


def test_identical_inputs_give_identical_reports():
    a = run("noisy_jitter")[1].to_dict()
    b = run("noisy_jitter")[1].to_dict()
    a["processing"].pop("analysis_seconds")
    b["processing"].pop("analysis_seconds")
    assert a == b


def test_no_detections_gives_no_score_and_clear_warning(cfg, average_body):
    from ratemyform.landmarks import LandmarkSequence

    seq = LandmarkSequence.empty(90, 640, 480, fps=30.0)
    res = analyze_sequence(seq, average_body, cfg)
    assert res.rep_count == 0
    assert res.overall_personalized is None
    assert any("detected in only 0%" in w for w in res.warnings)
    assert "No complete squat reps" in res.feedback[0]


def test_ratio_mismatch_warning():
    spec, _ = SCENARIOS["clean_5_reps"]
    syn = generate_squat(spec)
    from ratemyform.personalization import BodyMeasurements

    wrong = BodyMeasurements(torso_cm=50, femur_cm=60, lower_leg_cm=43, arm_cm=57)  # femur typo
    res = analyze_sequence(syn.sequence, wrong)
    assert any("femur/torso" in w for w in res.warnings)
