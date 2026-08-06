import numpy as np
import pytest

from photothermal_pte.finite_inverse_design.native_yee_q import (
    frequency_slice,
    integrate_xyz,
)
from photothermal_pte.finite_inverse_design.run_v261_large_background_mixed_optical_adfd import (
    invert_fieldregion_linear_collocation,
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


def test_inverse_fieldregion_collocation_retains_outer_native_source() -> None:
    grid = {
        "x": np.array([-1.0, 0.0, 1.0]),
        "y": np.array([-1.0, 0.0, 1.0]),
        "z": np.array([-1.0, 0.0, 1.0]),
        "f": np.array([1.0]),
        "delta_x": np.full(3, 0.5),
        "delta_y": np.full(3, 0.5),
        "delta_z": np.full(3, 0.5),
    }
    profile = np.zeros((3, 3, 3, 1, 3), np.complex128)
    profile[..., 0, 0] = np.array([1.0, 2.0, 3.0])[:, None, None]
    profile[..., 0, 1] = np.array([2.0, 3.0, 4.0])[None, :, None]
    profile[..., 0, 2] = np.array([3.0, 4.0, 5.0])[None, None, :]
    common, extended, metadata = invert_fieldregion_linear_collocation(
        grid, profile
    )
    assert common.shape == (4, 4, 4, 1, 3)
    assert all(extended[axis].size == 4 for axis in "xyz")
    for component in "xyz":
        assert (
            metadata["components"][component][
                "reconstruction_max_abs_error"
            ]
            < 1.0e-14
        )
    assert metadata["empirical_normalization"] is False
    assert metadata["gradient_rescaling"] is False


def test_inverse_fieldregion_collocation_accepts_exact_solver_mesh_extension() -> None:
    grid = {
        "x": np.array([-1.0, 0.0, 1.0]),
        "y": np.array([-1.0, 0.0, 1.0]),
        "z": np.array([-1.0, 0.0, 1.0]),
        "f": np.array([1.0]),
        "delta_x": np.full(3, 0.25),
        "delta_y": np.full(3, 0.25),
        "delta_z": np.full(3, 0.25),
    }
    profile = np.ones((3, 3, 3, 1, 3), np.complex128)
    extension = {"x": 1.5, "y": 1.75, "z": 2.0}
    _, extended, metadata = invert_fieldregion_linear_collocation(
        grid,
        profile,
        positive_extension_coordinate_m=extension,
    )
    for axis in "xyz":
        assert extended[axis][-1] == extension[axis]
        assert metadata["components"][axis]["reconstruction_max_abs_error"] < 1e-14
    assert metadata["positive_extension_coordinates_from_solver_mesh"] is True
