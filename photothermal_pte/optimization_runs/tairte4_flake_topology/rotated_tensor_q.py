"""Absorption extraction for a fixed crystal tensor in device coordinates."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.interpolate import RegularGridInterpolator, interp1d

from photothermal_pte.finite_inverse_design.native_yee_q import (
    EPS0,
    frequency_slice,
    integrate_xyz,
)
from photothermal_pte.optimization_runs.tairte4_flake_topology.contract import (
    CONTRACT,
)
from photothermal_pte.optimization_runs.tairte4_flake_topology import optical


def principal_fields(
    electric_xyz: np.ndarray,
) -> dict[str, np.ndarray]:
    """Rotate device-frame E into the fixed global b/a/c crystal frame."""

    electric = np.asarray(electric_xyz, dtype=np.complex128)
    if electric.shape[-1] != 3:
        raise ValueError("electric field must have a final xyz component axis")
    scale = 1.0 / np.sqrt(2.0)
    return {
        "b": scale * (electric[..., 0] - electric[..., 1]),
        "a": scale * (electric[..., 0] + electric[..., 1]),
        "c": electric[..., 2],
    }


def principal_absorption_density(
    electric_xyz: np.ndarray,
    density: np.ndarray,
    *,
    omega_rad_s: float,
    epsilon_abc: dict[str, complex],
) -> dict[str, np.ndarray]:
    """Return nonnegative b/a/c loss terms for the rotated Hermitian form."""

    electric = np.asarray(electric_xyz, dtype=np.complex128)
    rho = np.asarray(density, dtype=np.float64)
    if electric.shape[:-1] != rho.shape:
        raise ValueError("electric and density spatial shapes differ")
    fields = principal_fields(electric)
    result = {}
    for axis in "bac":
        epsilon = 1.0 + rho * (complex(epsilon_abc[axis]) - 1.0)
        values = 0.5 * EPS0 * omega_rad_s * np.imag(epsilon) * np.abs(fields[axis]) ** 2
        if not np.all(np.isfinite(values)) or np.min(values) < -1.0e-18:
            raise RuntimeError(f"invalid rotated-tensor Q_{axis}")
        result[axis] = np.asarray(values, dtype=np.float64)
    return result


def _common_grid_electric(
    fdtd: Any,
    field_monitor: str,
    frequency_index: int,
    frequency_count: int,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    coordinates = {
        axis: np.asarray(fdtd.getdata(field_monitor, axis, 1), float).reshape(-1)
        for axis in "xyz"
    }
    shape = tuple(coordinates[axis].size for axis in "xyz")
    fields = []
    for component_index, component in enumerate("xyz"):
        raw = np.asarray(fdtd.getdata(field_monitor, f"E{component}", 1))
        values = frequency_slice(
            raw,
            shape,
            frequency_index,
            frequency_count,
            f"E{component}",
        )
        delta = np.asarray(
            fdtd.getdata(field_monitor, f"delta_{component}", 1), float
        ).reshape(-1)
        native = coordinates[component] + delta
        values = interp1d(
            native,
            values,
            axis=component_index,
            bounds_error=False,
            fill_value=0.0,
            assume_sorted=True,
        )(coordinates[component])
        fields.append(values)
    return coordinates, np.stack(fields, axis=-1)


def extract_rotated_tensor_flake_q(
    fdtd: Any,
    *,
    field_monitor: str,
    wavelength_m: float,
    rho_xy: np.ndarray,
) -> dict[str, object]:
    """Extract TaIrTe4-only Q on one common local device-coordinate grid."""

    if CONTRACT.rotated_optical_mode != "rotated_tensor_local_grid":
        raise RuntimeError("rotated-tensor Q extraction requires its optical mode")
    frequencies = np.asarray(fdtd.getdata(field_monitor, "f", 1), float).reshape(-1)
    target = 299792458.0 / wavelength_m
    frequency_index = int(np.argmin(np.abs(frequencies - target)))
    realized = 299792458.0 / frequencies[frequency_index]
    if abs(realized - wavelength_m) / wavelength_m > 1.0e-9:
        raise RuntimeError("requested rotated-tensor Q wavelength is unavailable")
    coordinates, electric = _common_grid_electric(
        fdtd, field_monitor, frequency_index, frequencies.size
    )
    design_x, design_y, _ = optical.design_nodes()
    density_nodes = CONTRACT.apply_fixed_contact_density(rho_xy)
    interpolator = RegularGridInterpolator(
        (design_x, design_y),
        density_nodes,
        method="linear",
        bounds_error=False,
        fill_value=0.0,
    )
    xx, yy = np.meshgrid(coordinates["x"], coordinates["y"], indexing="ij")
    density_xy = interpolator(np.column_stack((xx.ravel(), yy.ravel()))).reshape(xx.shape)
    z_support = (
        (coordinates["z"] >= -CONTRACT.flake_thickness_m - 1.0e-18)
        & (coordinates["z"] <= 1.0e-18)
    )
    density = density_xy[:, :, None] * z_support[None, None, :]
    endpoint = optical.material_epsilon()
    components = principal_absorption_density(
        electric,
        density,
        omega_rad_s=2.0 * np.pi * frequencies[frequency_index],
        epsilon_abc={"a": endpoint["y"], "b": endpoint["x"], "c": endpoint["z"]},
    )
    power = {
        axis: integrate_xyz(
            values, coordinates["x"], coordinates["y"], coordinates["z"]
        )
        for axis, values in components.items()
    }
    return {
        "frequency_hz": float(frequencies[frequency_index]),
        "wavelength_m": float(realized),
        "coordinates": coordinates,
        "Q_principal_components": components,
        "component_power_W": power,
        "P_Q_flake_W": float(sum(power.values())),
        "density_range": [float(np.min(density)), float(np.max(density))],
        "coordinate_frame": "axis-aligned local device u/v; rotate scalar Q +45 degrees before global thermal deposition",
        "interpolation": "native staggered E components linearly interpolated to the common monitor grid",
        "clipping_smoothing_gain_or_rescaling": False,
    }
