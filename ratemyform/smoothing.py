"""Time-series cleanup for per-frame angles.

Pose estimates jitter from frame to frame. We clean each angle series in
three explainable steps:

1. Fill SHORT gaps (missing frames) by linear interpolation. Long gaps stay
   missing so we never invent data for a large part of a movement.
2. Rolling MEDIAN: removes one-frame spikes (a landmark jumping away).
3. Rolling MEAN: smooths the remaining small jitter.

Window sizes are given in seconds and converted to frames using the video's
frame rate, so a 60 fps video is smoothed over the same time span as a
30 fps video.
"""
from __future__ import annotations

import numpy as np


def seconds_to_window(seconds: float, fps: float) -> int:
    """Convert a window length in seconds to an odd number of frames (>= 1)."""
    if fps <= 0 or seconds <= 0:
        return 1
    frames = int(round(seconds * fps))
    if frames < 1:
        frames = 1
    if frames % 2 == 0:
        frames += 1
    return frames


def interpolate_short_gaps(values: np.ndarray, times: np.ndarray, max_gap_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Linearly fill NaN gaps no longer than `max_gap_s` seconds.

    Returns (filled_values, was_interpolated_mask). Gaps at the very start or
    end of the series are never filled (there is no value on both sides).
    """
    values = np.asarray(values, dtype=float).copy()
    times = np.asarray(times, dtype=float)
    interpolated = np.zeros(len(values), dtype=bool)
    missing = np.isnan(values)
    if not missing.any() or missing.all():
        return values, interpolated

    n = len(values)
    i = 0
    while i < n:
        if not missing[i]:
            i += 1
            continue
        start = i
        while i < n and missing[i]:
            i += 1
        end = i  # first valid index after the gap (or n)
        if start == 0 or end == n:
            continue  # gap touches an edge: leave it missing
        left, right = start - 1, end
        gap_duration = times[right] - times[left]
        if gap_duration <= max_gap_s:
            t0, t1 = times[left], times[right]
            v0, v1 = values[left], values[right]
            frac = (times[start:end] - t0) / (t1 - t0)
            values[start:end] = v0 + frac * (v1 - v0)
            interpolated[start:end] = True
    return values, interpolated


def _rolling(values: np.ndarray, window: int, func) -> np.ndarray:
    """Centered rolling window that ignores NaNs. NaN stays NaN."""
    values = np.asarray(values, dtype=float)
    if window <= 1 or len(values) == 0:
        return values.copy()
    half = window // 2
    out = np.full(len(values), np.nan)
    for i in range(len(values)):
        if np.isnan(values[i]):
            continue
        lo = max(0, i - half)
        hi = min(len(values), i + half + 1)
        chunk = values[lo:hi]
        chunk = chunk[~np.isnan(chunk)]
        out[i] = func(chunk)
    return out


def rolling_median(values: np.ndarray, window: int) -> np.ndarray:
    return _rolling(values, window, np.median)


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    return _rolling(values, window, np.mean)


def smooth_series(
    values: np.ndarray,
    times: np.ndarray,
    fps: float,
    max_gap_s: float,
    median_window_s: float,
    mean_window_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply gap filling, rolling median, then rolling mean.

    Returns (smoothed_values, was_interpolated_mask).
    """
    filled, interpolated = interpolate_short_gaps(values, times, max_gap_s)
    smoothed = rolling_median(filled, seconds_to_window(median_window_s, fps))
    smoothed = rolling_mean(smoothed, seconds_to_window(mean_window_s, fps))
    return smoothed, interpolated
