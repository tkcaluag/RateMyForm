"""Gap filling and smoothing of per-frame angle series."""
import numpy as np
import pytest

from ratemyform.smoothing import (
    interpolate_short_gaps,
    rolling_median,
    seconds_to_window,
    smooth_series,
)


def test_window_is_odd_and_scales_with_fps():
    assert seconds_to_window(0.15, 30) == 5
    assert seconds_to_window(0.15, 60) == 9
    assert seconds_to_window(0.15, 120) == 19
    assert seconds_to_window(0.0, 30) == 1
    for fps in (24, 25, 30, 50, 60, 120):
        assert seconds_to_window(0.15, fps) % 2 == 1


def test_short_gap_is_interpolated():
    t = np.arange(10) / 30
    v = np.arange(10, dtype=float)
    v[4:6] = np.nan  # 2 missing frames (~0.1 s)
    filled, interp = interpolate_short_gaps(v, t, max_gap_s=0.25)
    np.testing.assert_allclose(filled, np.arange(10, dtype=float))
    assert interp.tolist() == [False] * 4 + [True, True] + [False] * 4


def test_long_gap_stays_missing():
    t = np.arange(30) / 30
    v = np.ones(30)
    v[5:20] = np.nan  # 0.5 s gap
    filled, interp = interpolate_short_gaps(v, t, max_gap_s=0.25)
    assert np.isnan(filled[5:20]).all()
    assert not interp.any()


def test_edge_gaps_are_not_filled():
    t = np.arange(6) / 30
    v = np.array([np.nan, 1.0, 2.0, 3.0, 4.0, np.nan])
    filled, _ = interpolate_short_gaps(v, t, max_gap_s=1.0)
    assert np.isnan(filled[0]) and np.isnan(filled[-1])


def test_median_removes_single_frame_spike():
    v = np.full(15, 170.0)
    v[7] = 90.0  # one bad frame
    out = rolling_median(v, 5)
    assert out[7] == pytest.approx(170.0)


def test_smooth_series_keeps_long_gap_nan():
    t = np.arange(60) / 30
    v = np.linspace(180, 90, 60)
    v[20:40] = np.nan
    out, _ = smooth_series(v, t, 30, 0.25, 0.15, 0.15)
    assert np.isnan(out[20:40]).all()
    assert np.isfinite(out[:20]).all() and np.isfinite(out[40:]).all()
