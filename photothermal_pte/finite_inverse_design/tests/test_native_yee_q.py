import numpy as np
import pytest

from photothermal_pte.finite_inverse_design.native_yee_q import (
    frequency_slice,
    integrate_xyz,
)


def test_frequency_slice_selects_declared_axis() -> None:
    array = np.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5)
    selected = frequency_slice(array, (2, 3, 4), 2, 5, "E")
    assert np.array_equal(selected, array[..., 2])


def test_frequency_slice_fails_on_unexpected_shape() -> None:
    with pytest.raises(ValueError, match="shape"):
        frequency_slice(np.zeros((2, 3, 4, 2, 2)), (2, 3, 4), 0, 2, "E")


def test_trapezoid_integrates_constant_volume() -> None:
    x = np.linspace(-1.0, 1.0, 5)
    y = np.linspace(0.0, 3.0, 7)
    z = np.linspace(2.0, 6.0, 9)
    assert integrate_xyz(np.ones((5, 7, 9)), x, y, z) == pytest.approx(24.0)
