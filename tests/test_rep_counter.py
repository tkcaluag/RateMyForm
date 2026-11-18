"""Hysteresis state machine for counting reps (SYNTHETIC hand-built signals)."""
import numpy as np
import pytest

from ratemyform.config import RepCounterConfig
from ratemyform.rep_counter import RepCounter, detect_reps


def signal(points, fps=30.0):
    """Linear interpolation through (time_s, knee_angle) points, sampled at fps."""
    times = np.arange(0.0, points[-1][0] + 1e-9, 1.0 / fps)
    angles = np.interp(times, [p[0] for p in points], [p[1] for p in points])
    return times, angles


def squats(n, bottom=80.0, top=175.0, start=1.0, down=1.0, hold=0.2, up=1.0, rest=0.8):
    pts = [(0.0, top), (start, top)]
    t = start
    for _ in range(n):
        pts += [(t + down, bottom), (t + down + hold, bottom), (t + down + hold + up, top), (t + down + hold + up + rest, top)]
        t += down + hold + up + rest
    return pts


def test_counts_clean_reps():
    t, a = signal(squats(3))
    result = detect_reps(t, a)
    assert result.rep_count == 3
    assert [r.rep_number for r in result.reps] == [1, 2, 3]
    assert result.uncounted == []


def test_bottom_is_lowest_point():
    t, a = signal(squats(1, bottom=70.0))
    rep = detect_reps(t, a).reps[0]
    assert rep.min_knee_angle_deg == pytest.approx(70.0, abs=0.5)
    assert a[rep.bottom_index] == pytest.approx(rep.min_knee_angle_deg)
    assert rep.start_time_s < rep.bottom_time_s < rep.end_time_s


def test_jitter_inside_hysteresis_band_counts_once():
    # At the bottom the angle oscillates between 130 and 155: crosses 140
    # several times but never exceeds 160, so it must stay ONE rep.
    pts = [(0, 175), (1, 175), (2, 130)]
    t = 2.0
    for _ in range(6):
        pts += [(t + 0.15, 155), (t + 0.3, 130)]
        t += 0.3
    pts += [(t + 1, 175), (t + 2, 175)]
    times, a = signal(pts)
    assert detect_reps(times, a).rep_count == 1


def test_standing_noise_around_160_counts_nothing():
    rng = np.random.default_rng(0)
    t = np.arange(0, 5, 1 / 30)
    a = 163 + rng.normal(0, 3, len(t))  # wobbles across 160, never below 140
    assert detect_reps(t, a).rep_count == 0


def test_single_frame_spike_is_debounced():
    t = np.arange(0, 4, 1 / 30)
    a = np.full(len(t), 175.0)
    a[60] = 100.0  # one bad frame
    result = detect_reps(t, a)
    assert result.rep_count == 0


def test_too_fast_movement_is_not_counted():
    t, a = signal([(0, 175), (1, 175), (1.15, 120), (1.3, 175), (2.5, 175)])
    result = detect_reps(t, a)
    assert result.rep_count == 0
    assert [u.kind for u in result.uncounted] == ["too_fast"]


def test_dip_shorter_than_debounce_never_enters_squat_state():
    # Below 140 deg for only ~0.07 s (2 frames): rejected by the debounce,
    # reported as a partial movement.
    t, a = signal([(0, 175), (1, 175), (1.1, 120), (1.2, 175), (2.5, 175)])
    result = detect_reps(t, a)
    assert result.rep_count == 0
    assert "squatting" not in result.states
    assert [u.kind for u in result.uncounted] == ["partial"]


def test_incomplete_final_rep_is_reported_not_counted():
    t, a = signal(squats(2) + [(8.0, 175), (9.0, 90), (9.5, 90)])
    result = detect_reps(t, a)
    assert result.rep_count == 2
    assert result.uncounted[-1].kind == "incomplete"


def test_video_starting_mid_squat():
    t, a = signal([(0, 90), (0.5, 90), (1.5, 175), (2.0, 175)] + [(p[0] + 2.0, p[1]) for p in squats(2)[1:]])
    result = detect_reps(t, a)
    assert result.rep_count == 2
    assert result.uncounted[0].kind == "started_mid_rep"


def test_partial_dip_is_reported_not_counted():
    t, a = signal([(0, 175), (1, 175), (1.6, 148), (2.2, 175), (3.0, 175)])
    result = detect_reps(t, a)
    assert result.rep_count == 0
    assert result.uncounted[0].kind == "partial"
    assert result.uncounted[0].min_knee_angle_deg == pytest.approx(148, abs=1)


def test_tracking_gap_marks_rep_low_confidence():
    t, a = signal(squats(1))
    a = a.copy()
    gap = (t > 1.7) & (t < 2.5)  # ~0.8 s missing around the bottom
    a[gap] = np.nan
    result = detect_reps(t, a)
    assert result.rep_count == 1
    assert result.reps[0].low_confidence
    assert result.reps[0].longest_gap_s > 0.5


@pytest.mark.parametrize("fps", [15.0, 24.0, 30.0, 60.0, 120.0, 240.0])
def test_frame_rate_independence(fps):
    t, a = signal(squats(4), fps=fps)
    result = detect_reps(t, a)
    assert result.rep_count == 4
    # bottoms are at the same times regardless of sampling rate (within one frame + hold)
    expected = [1.0 + 1.0 + k * 3.0 for k in range(4)]
    for rep, e in zip(result.reps, expected):
        assert e - 1 / fps <= rep.bottom_time_s <= e + 0.2 + 1 / fps


def test_state_trace_has_one_entry_per_frame():
    t, a = signal(squats(2))
    result = detect_reps(t, a)
    assert len(result.states) == len(t)
    assert set(result.states) <= {"waiting", "standing", "squatting"}


def test_streaming_matches_batch():
    t, a = signal(squats(3))
    counter = RepCounter()
    for ti, ai in zip(t, a):
        counter.update(ti, ai)
    assert counter.finish().rep_count == detect_reps(t, a).rep_count


def test_invalid_thresholds_rejected():
    with pytest.raises(ValueError):
        RepCounter(RepCounterConfig(down_threshold_deg=160, up_threshold_deg=140))


def test_shape_mismatch_rejected():
    with pytest.raises(ValueError):
        detect_reps(np.arange(5), np.arange(4))


def test_configurable_thresholds():
    t, a = signal(squats(2, bottom=125.0))
    assert detect_reps(t, a).rep_count == 2  # 125 < 140
    strict = RepCounterConfig(down_threshold_deg=110.0)
    result = detect_reps(t, a, strict)
    assert result.rep_count == 0
    # With a 110 deg rep threshold, the 125 deg squats become reported partial movements.
    assert [u.kind for u in result.uncounted] == ["partial", "partial"]
