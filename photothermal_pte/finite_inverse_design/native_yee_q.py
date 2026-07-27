"""Native staggered-grid absorption extraction for v261 pabs_adv monitors."""

from __future__ import annotations

import numpy as np


EPS0 = 8.8541878128e-12


def trapezoid_weights(values: np.ndarray) -> np.ndarray:
    coordinate = np.asarray(values, float).reshape(-1)
    if coordinate.size < 2 or not np.all(np.diff(coordinate) > 0.0):
        raise ValueError("integration coordinate must be strictly increasing")
    weights = np.empty_like(coordinate)
    weights[0] = 0.5 * (coordinate[1] - coordinate[0])
    weights[-1] = 0.5 * (coordinate[-1] - coordinate[-2])
    weights[1:-1] = 0.5 * (coordinate[2:] - coordinate[:-2])
    return weights


def integrate_xyz(
    values: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
) -> float:
    cube = np.asarray(values, float)
    expected = (np.size(x), np.size(y), np.size(z))
    if cube.shape != expected:
        raise ValueError(f"cube shape {cube.shape} != coordinates {expected}")
    return float(
        np.einsum(
            "i,j,k,ijk->",
            trapezoid_weights(x),
            trapezoid_weights(y),
            trapezoid_weights(z),
            cube,
            optimize=True,
        )
    )


def frequency_slice(
    values: np.ndarray,
    spatial_shape: tuple[int, int, int],
    frequency_index: int,
    frequency_count: int,
    label: str,
) -> np.ndarray:
    """Return one frequency cube without silently squeezing useful axes."""

    array = np.asarray(values)
    while array.ndim > 3 and 1 in array.shape[4:]:
        array = np.squeeze(array, axis=tuple(
            axis for axis in range(4, array.ndim) if array.shape[axis] == 1
        ))
    if array.shape == spatial_shape:
        if frequency_count != 1 or frequency_index != 0:
            raise ValueError(
                f"{label} omitted frequency axis for {frequency_count} points"
            )
        return array
    expected = (*spatial_shape, frequency_count)
    if array.shape != expected:
        raise ValueError(f"{label} shape {array.shape} != {expected}")
    return array[..., frequency_index]


def extract_native_yee_q(
    fdtd: object,
    *,
    field_monitor: str,
    index_monitor: str,
    wavelength_m: float,
) -> dict[str, object]:
    """Extract Qx/Qy/Qz on their native Yee coordinates.

    The returned total power is the sum of the three independently integrated
    staggered components.  No interpolation, clipping, smoothing, gain, or
    post-solve rescaling is used for this power certificate.
    """

    x = np.asarray(fdtd.getdata(field_monitor, "x", 1), float).reshape(-1)
    y = np.asarray(fdtd.getdata(field_monitor, "y", 1), float).reshape(-1)
    z = np.asarray(fdtd.getdata(field_monitor, "z", 1), float).reshape(-1)
    spatial_shape = (x.size, y.size, z.size)
    frequency_hz = np.asarray(
        fdtd.getdata(field_monitor, "f", 1), float
    ).reshape(-1)
    target_frequency_hz = 299792458.0 / wavelength_m
    frequency_index = int(
        np.argmin(np.abs(frequency_hz - target_frequency_hz))
    )
    wavelength_realized_m = 299792458.0 / frequency_hz[frequency_index]
    if abs(wavelength_realized_m - wavelength_m) / wavelength_m > 1.0e-9:
        raise RuntimeError(
            f"analysis wavelength unavailable: {wavelength_realized_m}"
        )
    deltas = {
        axis: np.asarray(
            fdtd.getdata(field_monitor, f"delta_{axis}", 1), float
        ).reshape(-1)
        for axis in "xyz"
    }
    for axis, coordinate in zip("xyz", (x, y, z)):
        if deltas[axis].shape != coordinate.shape:
            raise ValueError(
                f"delta_{axis} shape {deltas[axis].shape} "
                f"!= coordinate {coordinate.shape}"
            )

    omega = 2.0 * np.pi * frequency_hz[frequency_index]
    components: dict[str, np.ndarray] = {}
    coordinates: dict[str, dict[str, np.ndarray]] = {}
    power: dict[str, float] = {}
    hotspot: dict[str, dict[str, float]] = {}
    raw_shapes: dict[str, object] = {}
    for component in "xyz":
        raw_electric = np.asarray(
            fdtd.getdata(field_monitor, f"E{component}", 1)
        )
        raw_index = np.asarray(
            fdtd.getdata(index_monitor, f"index_{component}", 1)
        )
        raw_shapes[component] = {
            "E": list(raw_electric.shape),
            "index": list(raw_index.shape),
        }
        electric = frequency_slice(
            raw_electric,
            spatial_shape,
            frequency_index,
            frequency_hz.size,
            f"E{component}",
        )
        refractive_index = frequency_slice(
            raw_index,
            spatial_shape,
            frequency_index,
            frequency_hz.size,
            f"index_{component}",
        )
        epsilon = refractive_index**2
        q_component = (
            0.5
            * EPS0
            * omega
            * np.imag(epsilon)
            * np.abs(electric) ** 2
        )
        if not np.all(np.isfinite(q_component)):
            raise RuntimeError(f"Q{component} contains NaN or Inf")
        native_coordinates = {
            "x": x.copy(),
            "y": y.copy(),
            "z": z.copy(),
        }
        native_coordinates[component] = (
            native_coordinates[component] + deltas[component]
        )
        components[component] = np.asarray(q_component, float)
        coordinates[component] = native_coordinates
        power[component] = integrate_xyz(
            q_component,
            native_coordinates["x"],
            native_coordinates["y"],
            native_coordinates["z"],
        )
        maximum_index = np.unravel_index(
            int(np.argmax(q_component)), q_component.shape
        )
        hotspot[component] = {
            "x_m": float(native_coordinates["x"][maximum_index[0]]),
            "y_m": float(native_coordinates["y"][maximum_index[1]]),
            "z_m": float(native_coordinates["z"][maximum_index[2]]),
            "Q_W_m3": float(q_component[maximum_index]),
        }

    return {
        "frequency_hz": float(frequency_hz[frequency_index]),
        "wavelength_m": float(wavelength_realized_m),
        "frequency_index_zero_based": frequency_index,
        "frequency_count": int(frequency_hz.size),
        "base_coordinates": {"x": x, "y": y, "z": z},
        "delta_coordinates": deltas,
        "native_coordinates": coordinates,
        "Q_components": components,
        "component_power_W": power,
        "P_Q_W": float(sum(power.values())),
        "component_hotspots": hotspot,
        "raw_array_shapes": raw_shapes,
        "operations_forbidden_and_absent": [
            "clipping",
            "smoothing",
            "gain",
            "global_rescaling",
            "tiling",
            "old_artifact_crop",
        ],
    }
