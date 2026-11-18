"""Input validation: measurements, units, uploads, and video properties."""
import math
from pathlib import Path

import pytest

from ratemyform.validation import (
    safe_temp_video_path,
    to_cm,
    validate_measurements,
    validate_upload,
    validate_video_properties,
)

GOOD_CM = {"torso": 50, "femur": 43, "lower_leg": 42, "arm": 58}


def test_valid_measurements_in_cm():
    r = validate_measurements(GOOD_CM, "cm")
    assert r.ok and r.warnings == []
    assert r.measurements.femur_cm == 43


def test_inches_are_converted():
    r = validate_measurements({k: v / 2.54 for k, v in GOOD_CM.items()}, "in")
    assert r.ok
    assert r.measurements.torso_cm == pytest.approx(50.0)
    assert to_cm(10, "in") == pytest.approx(25.4)


def test_cm_values_typed_with_inch_unit_are_rejected():
    # 50 "inches" of torso is implausible -> caught by the bounds
    r = validate_measurements(GOOD_CM, "in")
    assert not r.ok


@pytest.mark.parametrize(
    "bad, fragment",
    [
        (None, "required"),
        ("", "required"),
        ("abc", "must be a number"),
        (True, "must be a number"),
        (-5, "positive"),
        (0, "positive"),
        (math.nan, "positive"),
        (math.inf, "positive"),
        (5, "outside the accepted range"),
        (400, "outside the accepted range"),
    ],
)
def test_bad_femur_values(bad, fragment):
    r = validate_measurements(dict(GOOD_CM, femur=bad), "cm")
    assert not r.ok
    assert any(fragment in e for e in r.errors)


def test_unknown_unit():
    assert not validate_measurements(GOOD_CM, "furlongs").ok


def test_unusual_ratio_is_a_warning_not_an_error():
    r = validate_measurements(dict(GOOD_CM, torso=30, femur=60), "cm")  # femur/torso = 2.0
    assert r.ok
    assert any("femur/torso" in w for w in r.warnings)


@pytest.mark.parametrize("name", ["squat.mp4", "SQUAT.MOV", "a.avi", "clip.m4v"])
def test_allowed_extensions(name):
    assert validate_upload(name, 1_000_000).ok


@pytest.mark.parametrize("name", ["squat.exe", "squat.gif", "noext", "squat.mp4.exe", ""])
def test_rejected_extensions(name):
    assert not validate_upload(name, 1_000_000).ok


def test_upload_size_limits():
    assert not validate_upload("a.mp4", 0).ok
    assert not validate_upload("a.mp4", 201 * 1024 * 1024).ok
    assert validate_upload("a.mp4", 199 * 1024 * 1024).ok


def test_video_property_checks():
    assert validate_video_properties(30, 300, 1080, 1920).ok
    assert not validate_video_properties(30, 30, 1080, 1920).ok  # 1 s: too short
    assert not validate_video_properties(30, 30 * 120, 1080, 1920).ok  # 2 min: too long
    assert not validate_video_properties(5, 100, 1080, 1920).ok  # fps too low
    assert not validate_video_properties(30, 300, 320, 180).ok  # resolution too low
    unknown_length = validate_video_properties(30, 0, 1080, 1920)
    assert unknown_length.ok and unknown_length.warnings


def test_temp_path_ignores_user_filename(tmp_path):
    p = safe_temp_video_path(tmp_path, "../../../Windows/evil.mp4")
    assert p.parent == Path(tmp_path)
    assert p.name == "upload.mp4"
    assert safe_temp_video_path(tmp_path, "x.exe").name == "upload.mp4"
