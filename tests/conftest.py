"""Shared test setup.

Most tests use SYNTHETIC landmark sequences (ratemyform.synthetic) because
they come with known ground truth. Tests marked `integration` exercise the
real MediaPipe model and OpenCV on generated videos.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ratemyform.config import DEFAULT_CONFIG  # noqa: E402
from ratemyform.personalization import BodyMeasurements  # noqa: E402


@pytest.fixture
def cfg():
    return DEFAULT_CONFIG


@pytest.fixture
def average_body():
    """Proportions equal to the prototype reference profile (femur/torso 0.85, lower-leg/femur 1.0)."""
    return BodyMeasurements(torso_cm=50.0, femur_cm=42.5, lower_leg_cm=42.5, arm_cm=57.0)
