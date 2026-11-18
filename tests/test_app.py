"""Streamlit UI flow tests (streamlit.testing.v1.AppTest).

AppTest runs app.py headlessly. It cannot drive the file uploader, so the
video path is covered by test_pose_detection.py and the manual check in
docs/TESTING.md; the full results page is exercised through the SYNTHETIC
demo button.
"""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

pytestmark = pytest.mark.ui
APP = str(Path(__file__).resolve().parent.parent / "app.py")


def start() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    assert not at.exception
    return at


def button(at: AppTest, label: str):
    matches = [b for b in at.button if b.label == label]
    assert matches, f"button {label!r} not found; have {[b.label for b in at.button]}"
    return matches[0]


def to_stage_2(at: AppTest) -> AppTest:
    button(at, "Continue").click().run()
    assert not at.exception
    return at


def test_stage_1_shows_branding_and_squat_only():
    at = start()
    assert at.title[0].value == "RateMyForm"
    assert at.radio[0].options == ["Squat"]
    assert any("Planned, not yet supported" in c.value for c in at.caption)


def test_stage_2_requires_valid_inputs():
    at = to_stage_2(start())
    analyze = button(at, "Analyze my squats")
    assert analyze.disabled  # no video, no measurements yet
    at.number_input(key="m_torso").set_value(50.0)
    at.number_input(key="m_femur").set_value(5.0)  # implausible
    at.number_input(key="m_lower_leg").set_value(42.0)
    at.number_input(key="m_arm").set_value(58.0)
    at.run()
    assert any("outside the accepted range" in e.value for e in at.error)
    assert button(at, "Analyze my squats").disabled


def test_unit_switch_converts_values():
    at = to_stage_2(start())
    at.number_input(key="m_torso").set_value(50.8).run()
    at.radio(key="unit_choice").set_value("in").run()
    assert at.number_input(key="m_torso").value == pytest.approx(20.0, abs=0.05)


def test_synthetic_demo_reaches_results_and_is_labeled():
    at = to_stage_2(start())
    at.selectbox[0].set_value("forward_lean_long_femur")
    button(at, "Run synthetic demo").click().run()
    assert not at.exception
    assert at.title[0].value == "Your results"
    assert any("SYNTHETIC DEMO DATA" in w.value for w in at.warning)
    labels = {m.label: m.value for m in at.metric}
    assert labels["Reps counted"] == "5"
    assert labels["Personalized score"] == "10.0 / 10"
    assert labels["Baseline score"] == "9.0 / 10"
    assert len(at.tabs) == 5


def test_start_over_returns_to_stage_1():
    at = to_stage_2(start())
    button(at, "Run synthetic demo").click().run()
    button(at, "Start over").click().run()
    assert at.title[0].value == "RateMyForm"
