"""Per-frame squat measurements on SYNTHETIC landmark sequences."""
import numpy as np
import pytest

from ratemyform.landmarks import LandmarkSequence
from ratemyform.squat_metrics import (
    check_orientation,
    compute_frame_metrics,
    estimate_facing,
    estimate_video_ratios,
    select_side,
)
from ratemyform.synthetic import SyntheticSquatSpec, generate_squat


def test_clean_angles_match_ground_truth(cfg):
    syn = generate_squat(SyntheticSquatSpec(n_reps=2))
    fm = compute_frame_metrics(syn.sequence, cfg)
    raw = fm.raw["knee_angle"]
    ok = np.isfinite(raw)
    assert ok.mean() > 0.99
    np.testing.assert_allclose(raw[ok], syn.true_knee_angle[ok], atol=0.01)
    np.testing.assert_allclose(fm.raw["torso_lean"][ok], syn.true_torso_lean[ok], atol=0.01)


def test_2d_and_3d_knee_angles_agree_in_side_view(cfg):
    syn = generate_squat(SyntheticSquatSpec(n_reps=1))
    fm = compute_frame_metrics(syn.sequence, cfg)
    both = np.isfinite(fm.knee_angle_2d) & np.isfinite(fm.knee_angle_3d)
    np.testing.assert_allclose(fm.knee_angle_2d[both], fm.knee_angle_3d[both], atol=0.01)


@pytest.mark.parametrize("near", ["left", "right"])
def test_selects_the_side_facing_the_camera(near):
    syn = generate_squat(SyntheticSquatSpec(n_reps=1, near_side=near))
    side, scores = select_side(syn.sequence)
    assert side == near
    assert scores[near] > scores["right" if near == "left" else "left"]


@pytest.mark.parametrize("facing", [1, -1])
def test_facing_direction_and_lean_sign(cfg, facing):
    syn = generate_squat(SyntheticSquatSpec(n_reps=1, facing=facing))
    assert estimate_facing(syn.sequence, "left", 0.5) == facing
    fm = compute_frame_metrics(syn.sequence, cfg)
    # Torso leans FORWARD (positive) whichever way the person faces.
    assert np.nanmax(fm.smoothed["torso_lean"]) == pytest.approx(35.0, abs=0.5)


def test_missing_landmarks_become_nan_not_zero(cfg):
    syn = generate_squat(SyntheticSquatSpec(n_reps=1, dropout_windows_s=[(1.5, 1.8)]))
    fm = compute_frame_metrics(syn.sequence, cfg)
    t = fm.times_s
    window = (t >= 1.5) & (t < 1.8)
    assert np.isnan(fm.raw["knee_angle"][window]).all()
    assert not fm.valid["knee_angle"][window].any()


def test_low_visibility_landmark_invalidates_only_dependent_metrics(cfg):
    syn = generate_squat(SyntheticSquatSpec(n_reps=1, occlusion_windows_s=[(1.5, 2.5, ("shoulder",))]))
    fm = compute_frame_metrics(syn.sequence, cfg)
    window = (fm.times_s >= 1.5) & (fm.times_s < 2.5)
    assert not fm.valid["torso_lean"][window].any()  # needs the shoulder
    assert not fm.valid["hip_angle"][window].any()  # needs the shoulder
    assert fm.valid["knee_angle"][window].all()  # does not


def test_orientation_side_vs_front(cfg):
    side = generate_squat(SyntheticSquatSpec(n_reps=1)).sequence
    front = generate_squat(SyntheticSquatSpec(n_reps=1, camera_yaw_deg=85.0)).sequence
    angled = generate_squat(SyntheticSquatSpec(n_reps=1, camera_yaw_deg=40.0)).sequence
    standing = np.ones(side.num_frames, dtype=bool)
    assert check_orientation(side, standing, cfg).label == "side"
    assert check_orientation(front, np.ones(front.num_frames, dtype=bool), cfg).label == "front_or_back"
    assert check_orientation(angled, np.ones(angled.num_frames, dtype=bool), cfg).label in ("angled", "front_or_back")


def test_orientation_unknown_without_detections(cfg):
    seq = LandmarkSequence.empty(30, 640, 480, fps=30.0)
    assert check_orientation(seq, np.ones(30, dtype=bool), cfg).label == "unknown"


def test_video_ratio_estimates_match_skeleton():
    spec = SyntheticSquatSpec(n_reps=1, torso_cm=50, femur_cm=45, lower_leg_cm=40)
    seq = generate_squat(spec).sequence
    ratios = estimate_video_ratios(seq, "left", np.ones(seq.num_frames, dtype=bool), 0.5)
    # The estimate uses 3D joint-to-joint distance. The synthetic shoulder sits
    # 0.09 m further out to the side than the hip, so the 3D torso is
    # sqrt(0.50^2 + 0.09^2) = 0.508 m, slightly longer than the 0.50 m side-view
    # length a tape measure would give. (Same effect exists for real people.)
    torso_3d = np.hypot(0.50, 0.19 - 0.10)
    assert ratios["femur_to_torso"] == pytest.approx(0.45 / torso_3d, abs=0.005)
    assert ratios["lower_leg_to_femur"] == pytest.approx(40 / 45, abs=0.005)


def test_noisy_angles_are_close_after_smoothing(cfg):
    syn = generate_squat(SyntheticSquatSpec(pixel_noise_px=4.0, world_noise_m=0.012, spike_probability=0.05, seed=7))
    fm = compute_frame_metrics(syn.sequence, cfg)
    ok = np.isfinite(fm.smoothed["knee_angle"])
    mae = float(np.mean(np.abs(fm.smoothed["knee_angle"][ok] - syn.true_knee_angle[ok])))
    # SYNTHETIC noise model only; this is not a real-world accuracy claim.
    assert mae < 4.0
