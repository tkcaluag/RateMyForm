"""Angle and lean calculations (pure math, no video)."""
import numpy as np
import pytest

from ratemyform.geometry import angle_at, lean_from_vertical, segment_length


def test_right_angle_2d():
    assert angle_at([0, 1], [0, 0], [1, 0]) == pytest.approx(90.0)


def test_straight_line_is_180():
    # hip above knee above ankle: a straight leg
    assert angle_at([0, 0], [0, 1], [0, 2]) == pytest.approx(180.0)


def test_45_degrees():
    assert angle_at([1, 0], [0, 0], [1, 1]) == pytest.approx(45.0)


def test_3d_angle():
    assert angle_at([1, 0, 0], [0, 0, 0], [0, 0, 1]) == pytest.approx(90.0)


def test_angle_is_symmetric():
    a, b, c = np.array([3.0, 1.0]), np.array([0.5, 0.2]), np.array([-1.0, 4.0])
    assert angle_at(a, b, c) == pytest.approx(angle_at(c, b, a))


def test_vectorized_series():
    a = np.array([[0, 1], [1, 0], [0, 1]], dtype=float)
    b = np.zeros((3, 2))
    c = np.array([[1, 0], [0, 1], [0, -1]], dtype=float)
    np.testing.assert_allclose(angle_at(a, b, c), [90.0, 90.0, 180.0])


def test_degenerate_and_nan_inputs_return_nan():
    assert np.isnan(angle_at([0, 0], [0, 0], [1, 0]))  # zero-length segment
    assert np.isnan(angle_at([np.nan, 0], [0, 0], [1, 0]))


def test_lean_vertical_is_zero():
    # image coords: y points down, so the upper point has a smaller y
    assert lean_from_vertical([100, 200], [100, 100], facing=1) == pytest.approx(0.0)


def test_forward_lean_sign_depends_on_facing():
    hip, shoulder = [100.0, 200.0], [150.0, 150.0]  # shoulder up-and-right of hip
    assert lean_from_vertical(hip, shoulder, facing=1) == pytest.approx(45.0)
    assert lean_from_vertical(hip, shoulder, facing=-1) == pytest.approx(-45.0)
    assert lean_from_vertical(hip, shoulder, facing=0) == pytest.approx(45.0)  # unknown -> magnitude


def test_lean_zero_length_is_nan():
    assert np.isnan(lean_from_vertical([5, 5], [5, 5], facing=1))


def test_segment_length():
    assert segment_length([0, 0], [3, 4]) == pytest.approx(5.0)
