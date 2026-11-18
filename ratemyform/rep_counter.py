"""Squat repetition counter: a state machine with hysteresis.

Why not just "count every time the knee angle drops below 140 deg"? Because
pose estimates jitter. If the angle wobbles 139 -> 141 -> 139, a single
threshold would count several reps. Our methodology uses TWO thresholds:

    STANDING --(knee < 140 deg)--> SQUATTING --(knee > 160 deg)--> STANDING
                                                   (one rep counted)

The 20-degree gap between the thresholds is the "hysteresis band". Jitter
inside the band cannot change the state. On top of that, a threshold must
stay crossed for `min_state_duration_s` (debounce), and a counted rep must
last at least `min_rep_duration_s`. All timing uses timestamps in seconds, so
the counter behaves the same at any frame rate.

States:
    WAITING   - have not yet seen the person standing (video may start
                mid-squat); nothing is counted until they stand up.
    STANDING  - knee angle above the "up" threshold.
    SQUATTING - knee angle went below the "down" threshold.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from .config import RepCounterConfig

_TIME_TOLERANCE_S = 1e-6


class RepState(str, Enum):
    WAITING = "waiting"
    STANDING = "standing"
    SQUATTING = "squatting"


@dataclass
class RepWindow:
    """One counted repetition (indices refer to the analyzed frames)."""

    rep_number: int
    start_index: int
    bottom_index: int
    end_index: int
    start_time_s: float
    bottom_time_s: float
    end_time_s: float
    min_knee_angle_deg: float
    longest_gap_s: float
    low_confidence: bool

    @property
    def duration_s(self) -> float:
        return self.end_time_s - self.start_time_s


@dataclass
class UncountedMovement:
    """A movement that was detected but deliberately NOT counted as a rep."""

    kind: str  # "partial", "incomplete", "too_fast", "too_slow", "started_mid_rep"
    start_time_s: float
    end_time_s: float | None
    min_knee_angle_deg: float
    reason: str


@dataclass
class RepDetectionResult:
    reps: list[RepWindow] = field(default_factory=list)
    uncounted: list[UncountedMovement] = field(default_factory=list)
    states: list[str] = field(default_factory=list)  # state at every frame

    @property
    def rep_count(self) -> int:
        return len(self.reps)


class RepCounter:
    """Streaming rep counter. Call `update()` once per frame, then `finish()`.

    Streaming means the same class could later power a live-camera mode.
    """

    def __init__(self, config: RepCounterConfig | None = None) -> None:
        self.cfg = config or RepCounterConfig()
        if self.cfg.down_threshold_deg >= self.cfg.up_threshold_deg:
            raise ValueError("down_threshold_deg must be lower than up_threshold_deg")
        self.state = RepState.WAITING
        self._times: list[float] = []
        self._angles: list[float] = []
        self._result = RepDetectionResult()
        self._cond_since: int | None = None  # index where a pending transition began
        self._last_above_up: int | None = None
        self._rep_start: int | None = None
        self._partial_start: int | None = None
        self._partial_min = np.inf
        self._mid_rep_start: int | None = None

    # ------------------------------------------------------------------ helpers
    def _held_long_enough(self, i: int) -> bool:
        """True once the pending condition has held for the debounce time.

        A tiny tolerance absorbs floating-point error: at 30 fps, 3 frame
        steps are exactly 0.1 s but can compute as 0.09999999999999987.
        """
        if self._cond_since is None:
            self._cond_since = i
        held = self._times[i] - self._times[self._cond_since]
        return held >= self.cfg.min_state_duration_s - _TIME_TOLERANCE_S

    def _valid_window(self, start: int, end: int) -> tuple[np.ndarray, np.ndarray]:
        t = np.asarray(self._times[start : end + 1])
        a = np.asarray(self._angles[start : end + 1])
        ok = ~np.isnan(a)
        return t[ok], a[ok]

    # --------------------------------------------------------------- main loop
    def update(self, time_s: float, knee_angle_deg: float) -> RepState:
        i = len(self._times)
        self._times.append(float(time_s))
        self._angles.append(float(knee_angle_deg))
        a = knee_angle_deg

        if np.isnan(a):
            # Missing data never changes the state; gaps are measured later.
            self._result.states.append(self.state.value)
            return self.state

        cfg = self.cfg
        if self.state == RepState.WAITING:
            if a < cfg.down_threshold_deg and self._mid_rep_start is None:
                self._mid_rep_start = i
            if a > cfg.up_threshold_deg:
                if self._held_long_enough(i):
                    self._enter_standing(self._cond_since)
            else:
                self._cond_since = None

        elif self.state == RepState.STANDING:
            if a > cfg.up_threshold_deg:
                self._last_above_up = i
                self._close_partial(i)
            if a < cfg.partial_threshold_deg:
                if self._partial_start is None:
                    self._partial_start = self._last_above_up if self._last_above_up is not None else i
                self._partial_min = min(self._partial_min, a)
            if a < cfg.down_threshold_deg:
                if self._held_long_enough(i):
                    self.state = RepState.SQUATTING
                    self._rep_start = self._last_above_up if self._last_above_up is not None else self._cond_since
                    self._partial_start, self._partial_min = None, np.inf
                    self._cond_since = None
            else:
                self._cond_since = None

        elif self.state == RepState.SQUATTING:
            if a > cfg.up_threshold_deg:
                if self._held_long_enough(i):
                    self._finish_rep(end=self._cond_since)
                    self._enter_standing(self._cond_since)
                    self._last_above_up = i
            else:
                self._cond_since = None

        self._result.states.append(self.state.value)
        return self.state

    def _enter_standing(self, index: int) -> None:
        if self.state == RepState.WAITING and self._mid_rep_start is not None:
            t0 = self._times[self._mid_rep_start]
            seg = np.asarray(self._angles[self._mid_rep_start : index + 1])
            self._result.uncounted.append(
                UncountedMovement(
                    kind="started_mid_rep",
                    start_time_s=t0,
                    end_time_s=self._times[index],
                    min_knee_angle_deg=float(np.nanmin(seg)),
                    reason="The video started while already squatting, so this movement was not counted.",
                )
            )
        self.state = RepState.STANDING
        self._last_above_up = index
        self._cond_since = None

    def _close_partial(self, i: int) -> None:
        if self._partial_start is not None:
            self._result.uncounted.append(
                UncountedMovement(
                    kind="partial",
                    start_time_s=self._times[self._partial_start],
                    end_time_s=self._times[i],
                    min_knee_angle_deg=float(self._partial_min),
                    reason=(
                        f"Knee bent to {self._partial_min:.0f} deg but never passed the "
                        f"{self.cfg.down_threshold_deg:.0f} deg rep threshold."
                    ),
                )
            )
        self._partial_start, self._partial_min = None, np.inf

    def _finish_rep(self, end: int) -> None:
        start = self._rep_start if self._rep_start is not None else end
        t_valid, a_valid = self._valid_window(start, end)
        seg = np.asarray(self._angles[start : end + 1])
        bottom = start + int(np.nanargmin(seg))
        duration = self._times[end] - self._times[start]
        min_angle = float(np.nanmin(seg))
        longest_gap = float(np.max(np.diff(t_valid))) if len(t_valid) > 1 else 0.0
        # Gaps are measured between valid samples, so a normal frame step is
        # ~1/fps. Subtract nothing: the threshold is far larger than 1/fps.
        self._rep_start = None

        if duration < self.cfg.min_rep_duration_s:
            self._result.uncounted.append(
                UncountedMovement(
                    kind="too_fast",
                    start_time_s=self._times[start],
                    end_time_s=self._times[end],
                    min_knee_angle_deg=min_angle,
                    reason=(
                        f"Lasted {duration:.2f} s, shorter than the {self.cfg.min_rep_duration_s:.1f} s "
                        "minimum, so it was treated as tracking jitter."
                    ),
                )
            )
            return
        if duration > self.cfg.max_rep_duration_s:
            self._result.uncounted.append(
                UncountedMovement(
                    kind="too_slow",
                    start_time_s=self._times[start],
                    end_time_s=self._times[end],
                    min_knee_angle_deg=min_angle,
                    reason=f"Lasted {duration:.1f} s, longer than the {self.cfg.max_rep_duration_s:.0f} s maximum.",
                )
            )
            return

        self._result.reps.append(
            RepWindow(
                rep_number=len(self._result.reps) + 1,
                start_index=start,
                bottom_index=bottom,
                end_index=end,
                start_time_s=self._times[start],
                bottom_time_s=self._times[bottom],
                end_time_s=self._times[end],
                min_knee_angle_deg=min_angle,
                longest_gap_s=longest_gap,
                low_confidence=longest_gap > self.cfg.max_tracking_gap_s,
            )
        )

    def finish(self) -> RepDetectionResult:
        """Close the sequence and report any unfinished movement."""
        if self.state == RepState.SQUATTING and self._rep_start is not None:
            seg = np.asarray(self._angles[self._rep_start :])
            self._result.uncounted.append(
                UncountedMovement(
                    kind="incomplete",
                    start_time_s=self._times[self._rep_start],
                    end_time_s=None,
                    min_knee_angle_deg=float(np.nanmin(seg)),
                    reason="The video ended before returning to standing, so this rep was not counted.",
                )
            )
        elif self.state == RepState.STANDING and self._partial_start is not None:
            self._result.uncounted.append(
                UncountedMovement(
                    kind="partial",
                    start_time_s=self._times[self._partial_start],
                    end_time_s=None,
                    min_knee_angle_deg=float(self._partial_min),
                    reason="A shallow dip at the end of the video did not reach the rep threshold.",
                )
            )
        elif self.state == RepState.WAITING and self._mid_rep_start is not None:
            seg = np.asarray(self._angles[self._mid_rep_start :])
            self._result.uncounted.append(
                UncountedMovement(
                    kind="started_mid_rep",
                    start_time_s=self._times[self._mid_rep_start],
                    end_time_s=None,
                    min_knee_angle_deg=float(np.nanmin(seg)),
                    reason="The person was never seen standing, so no reps could be counted.",
                )
            )
        return self._result


def detect_reps(times_s: np.ndarray, knee_angles_deg: np.ndarray, config: RepCounterConfig | None = None) -> RepDetectionResult:
    """Run the state machine over a whole series (NaN = missing frame)."""
    times_s = np.asarray(times_s, dtype=float)
    knee_angles_deg = np.asarray(knee_angles_deg, dtype=float)
    if times_s.shape != knee_angles_deg.shape:
        raise ValueError("times_s and knee_angles_deg must have the same shape")
    counter = RepCounter(config)
    for t, a in zip(times_s, knee_angles_deg):
        counter.update(t, a)
    return counter.finish()
