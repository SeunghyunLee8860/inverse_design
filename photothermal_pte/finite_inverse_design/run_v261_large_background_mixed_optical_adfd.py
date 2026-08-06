#!/usr/bin/env python3
"""Mixed CPU-forward/GPU-adjoint scalar optical AD--FD certificate.

The certificate deliberately starts with one uniform physical-density degree
of freedom.  It validates the complete solver-level chain

    rho -> epsilon_Yee -> E -> native-Yee absorbed power

before a spatial density grid, filter, projection, thermal operator, or PTE
functional is allowed to reuse the path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np
from scipy.interpolate import interp1d

from .audit_v261_large_background_tfsf_geometry import (
    add_large_background_geometry,
)
from .large_background_contract import baseline_contract
from .native_yee_q import (
    EPS0,
    extract_native_yee_q,
    frequency_slice,
    trapezoid_weights,
)
from .probe_v261_cpu_tfsf_device import (
    FREQUENCY_HZ,
    PABS_FIELD,
    PABS_GROUP,
    PABS_INDEX,
    SOURCE_NAME,
)
from .probe_v261_cpu_tfsf_roi import (
    add_single_frequency_monitor_settings,
    run_on_cpu,
)
from .probe_v261_gpu_plane_wave_roi import (
    APPROVED_API,
    APPROVED_ROOT,
    json_default,
    load_lumapi,
    resource_snapshot,
    run_on_gpu,
    scalar,
)
from .run_v261_large_background_tfsf_forward import (
    add_monitors,
    configure_design,
)


WAVELENGTH_M = 4.0e-6
FIELD_REGION = "large_background_q_fieldregion"
DESIGN_FIELD = "large_background_design_field"
DESIGN_INDEX = "large_background_design_index"
DESIGN_OBJECT = "finite_design_gray"
SIO2_EPSILON = 1.38**2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rho", type=float, default=0.5)
    parser.add_argument(
        "--fd-step",
        type=float,
        action="append",
        dest="fd_steps",
        default=None,
        help="Centered scalar-density FD step; repeat for a sweep.",
    )
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--gpu-device", default="GPU 1")
    parser.add_argument("--simulation-time-ps", type=float, default=4.0)
    parser.add_argument("--domain-um", type=float, default=6.4)
    parser.add_argument("--z-min-um", type=float, default=-3.2)
    parser.add_argument("--z-max-um", type=float, default=3.0)
    parser.add_argument("--pml-layers", type=int, default=24)
    parser.add_argument(
        "--pml-profile",
        choices=("standard", "stabilized-xy"),
        default="standard",
    )
    args = parser.parse_args()
    if not 0.0 < args.rho < 1.0:
        parser.error("rho must be strictly inside (0,1)")
    args.fd_steps = args.fd_steps or [0.02, 0.01]
    if any(step <= 0.0 for step in args.fd_steps):
        parser.error("FD steps must be positive")
    if any(
        args.rho - step <= 0.0 or args.rho + step >= 1.0
        for step in args.fd_steps
    ):
        parser.error("centered FD steps must stay strictly inside (0,1)")
    return args


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def as_e5(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, np.complex128)
    if array.ndim == 4 and array.shape[-1] == 3:
        array = array[..., None, :]
    if array.ndim != 5 or array.shape[-1] != 3:
        raise ValueError(f"expected E (...,nf,3), received {array.shape}")
    return array


def set_bounds(obj: object, bounds: object) -> None:
    for axis in "xyz":
        lo, hi = getattr(bounds, axis)
        obj[f"{axis} min"] = lo
        obj[f"{axis} max"] = hi


def set_single_frequency(obj: object) -> None:
    for prop, value in (
        ("override global monitor settings", True),
        ("use source limits", False),
        ("use wavelength spacing", True),
        ("wavelength center", WAVELENGTH_M),
        ("wavelength span", 0.0),
        ("frequency points", 1),
    ):
        try:
            obj[prop] = value
        except Exception:
            pass


def add_adjoint_monitors(fdtd: object, contract: object) -> None:
    region = fdtd.addfieldregion()
    region["name"] = FIELD_REGION
    region["monitor type"] = "3D"
    set_bounds(region, contract.omega_q)
    set_single_frequency(region)
    region["source mode"] = False
    try:
        region["nuttall window pulse"] = False
    except Exception:
        pass

    design = fdtd.addpower()
    design["name"] = DESIGN_FIELD
    design["monitor type"] = "3D"
    set_bounds(design, contract.design)
    set_single_frequency(design)
    try:
        design["spatial interpolation"] = "none"
    except Exception:
        pass

    design_index = fdtd.addindex()
    design_index["name"] = DESIGN_INDEX
    design_index["monitor type"] = "3D"
    set_bounds(design_index, contract.design)
    set_single_frequency(design_index)
    try:
        design_index["record conformal mesh when possible"] = True
    except Exception:
        pass


def configure_cpu(fdtd: object, threads: int) -> dict[str, object]:
    fdtd.setresource("FDTD", 1, "active", 1)
    fdtd.setresource("FDTD", 1, "processes", "1")
    fdtd.setresource("FDTD", 1, "threads", str(threads))
    fdtd.setresource("FDTD", 2, "active", 0)
    return resource_snapshot(fdtd)


def configure_gpu(
    fdtd: object, threads: int, gpu_device: str
) -> dict[str, object]:
    fdtd.setresource("FDTD", 1, "active", 0)
    fdtd.setresource("FDTD", 2, "active", 1)
    fdtd.setresource("FDTD", 2, "processes", "1")
    fdtd.setresource("FDTD", 2, "threads", str(threads))
    fdtd.setresource("FDTD", 2, "device type", gpu_device)
    fdtd.setresource(
        "FDTD", 2, "solver extra command line options", "-gpu"
    )
    return resource_snapshot(fdtd)


def configure_pml_profile(fdtd: object, profile: str) -> list:
    if profile == "standard":
        fdtd.setnamed("FDTD", "same settings on all boundaries", 1)
        fdtd.setnamed("FDTD", "pml profile", 1)
    elif profile == "stabilized-xy":
        fdtd.setnamed("FDTD", "same settings on all boundaries", 0)
        # v261 enum: standard=1, stabilized=2.  Layered dispersive material
        # interfaces cross only the lateral PML; retain standard z PML.
        fdtd.setnamed(
            "FDTD",
            "pml profile",
            np.asarray([[2], [2], [2], [2], [1], [1]], float),
        )
    else:
        raise ValueError(profile)
    return np.asarray(fdtd.getnamed("FDTD", "pml profile")).tolist()


def monitor_grid(fdtd: object, monitor: str) -> dict[str, np.ndarray]:
    if monitor == FIELD_REGION:
        dataset = fdtd.getresult(monitor, "E")
        result = {
            axis: np.asarray(dataset[axis], float).reshape(-1)
            for axis in "xyzf"
        }
        result.update(
            {
                f"delta_{axis}": np.asarray(
                    fdtd.getresult(monitor, f"delta_{axis}"), float
                ).reshape(-1)
                for axis in "xyz"
            }
        )
        return result
    result = {
        axis: np.asarray(fdtd.getdata(monitor, axis, 1), float).reshape(-1)
        for axis in "xyzf"
    }
    result.update(
        {
            f"delta_{axis}": np.asarray(
                fdtd.getdata(monitor, f"delta_{axis}", 1), float
            ).reshape(-1)
            for axis in "xyz"
        }
    )
    return result


def monitor_electric(fdtd: object, monitor: str) -> tuple[np.ndarray, dict]:
    if monitor == FIELD_REGION:
        dataset = fdtd.getresult(monitor, "E")
        grid = monitor_grid(fdtd, monitor)
        electric = as_e5(np.asarray(dataset["E"], np.complex128))
        expected = tuple(grid[axis].size for axis in "xyz")
        if electric.shape[:3] != expected:
            raise RuntimeError(
                f"{monitor} E grid {electric.shape[:3]} != {expected}"
            )
        return electric, grid
    grid = monitor_grid(fdtd, monitor)
    spatial_shape = tuple(grid[axis].size for axis in "xyz")
    frequency_index = int(
        np.argmin(np.abs(grid["f"] - FREQUENCY_HZ))
    )
    fields = []
    for component in "xyz":
        fields.append(
            frequency_slice(
                np.asarray(fdtd.getdata(monitor, f"E{component}", 1)),
                spatial_shape,
                frequency_index,
                grid["f"].size,
                f"{monitor}.E{component}",
            )
        )
    electric = np.stack(fields, axis=-1)[..., None, :]
    grid["f"] = np.asarray([grid["f"][frequency_index]], float)
    return as_e5(electric), grid


def monitor_epsilon(fdtd: object, monitor: str) -> tuple[np.ndarray, dict]:
    grid = {
        axis: np.asarray(fdtd.getdata(monitor, axis, 1), float).reshape(-1)
        for axis in "xyzf"
    }
    spatial_shape = tuple(grid[axis].size for axis in "xyz")
    frequency_index = int(
        np.argmin(np.abs(grid["f"] - FREQUENCY_HZ))
    )
    components = []
    for component in "xyz":
        index = frequency_slice(
            np.asarray(fdtd.getdata(monitor, f"index_{component}", 1)),
            spatial_shape,
            frequency_index,
            grid["f"].size,
            f"{monitor}.index_{component}",
        )
        components.append(index**2)
    epsilon = np.stack(components, axis=-1)[..., None, :]
    grid["f"] = np.asarray([grid["f"][frequency_index]], float)
    return np.asarray(epsilon, np.complex128), grid


def component_volumes(grid: dict[str, np.ndarray]) -> dict[int, np.ndarray]:
    base = [np.asarray(grid[axis], float) for axis in "xyz"]
    delta = [
        np.asarray(grid[f"delta_{axis}"], float) for axis in "xyz"
    ]
    result = {}
    for component in range(3):
        coordinates = [np.array(axis, copy=True) for axis in base]
        coordinates[component] += delta[component]
        weights = [trapezoid_weights(axis) for axis in coordinates]
        result[component] = (
            weights[0][:, None, None]
            * weights[1][None, :, None]
            * weights[2][None, None, :]
        )
    return result


def absorption_objective_and_source(
    electric: np.ndarray,
    epsilon: np.ndarray,
    grid: dict[str, np.ndarray],
) -> tuple[float, np.ndarray, dict[str, float]]:
    if electric.shape != epsilon.shape:
        raise ValueError(
            f"objective E/index mismatch {electric.shape} != {epsilon.shape}"
        )
    frequency_index = int(
        np.argmin(np.abs(grid["f"] - FREQUENCY_HZ))
    )
    frequency = float(grid["f"][frequency_index])
    omega = 2.0 * np.pi * frequency
    volumes = component_volumes(grid)
    q_source = np.zeros_like(electric)
    powers: dict[str, float] = {}
    for component in range(3):
        loss = np.imag(epsilon[..., 0, component])
        field = electric[..., 0, component]
        volume = volumes[component]
        density = 0.5 * EPS0 * omega * loss * np.abs(field) ** 2
        powers["xyz"[component]] = float(np.sum(volume * density))
        q_source[..., 0, component] = (
            0.5 * EPS0 * omega * loss * volume * field
        )
    return float(sum(powers.values())), q_source, powers


def fieldregion_profile(
    q_d_f_d_e_conjugate: np.ndarray,
) -> tuple[np.ndarray, float]:
    profile = np.conj(as_e5(q_d_f_d_e_conjugate))
    scale = float(np.max(np.abs(profile)))
    if not np.isfinite(scale) or scale <= 0.0:
        raise RuntimeError(f"invalid adjoint source scale {scale}")
    return np.ascontiguousarray(profile / scale), scale


def invert_fieldregion_linear_collocation(
    grid: dict[str, np.ndarray],
    profile: np.ndarray,
    *,
    positive_extension_coordinate_m: dict[str, float] | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, object]]:
    """Solve the component-axis common-grid interpolation exactly.

    For positive Yee offsets, native sample i lies between common samples
    i and i+1.  v261's linear placement is therefore

        native_i = alpha_i * common_i + beta_i * common_{i+1}.

    A one-cell positive-axis extension supplies the final common-grid degree
    of freedom, so the last native target is retained.  The new exterior
    common sample is fixed to zero and the bidiagonal system is solved
    backwards.  This is a coordinate-operator inverse, not empirical
    gradient normalization.
    """

    value = as_e5(profile)
    expected = (
        grid["x"].size,
        grid["y"].size,
        grid["z"].size,
        1,
        3,
    )
    if value.shape != expected:
        raise ValueError(f"profile {value.shape} != {expected}")
    extended_grid = {
        key: np.array(item, copy=True) for key, item in grid.items()
    }
    for axis in "xyz":
        base = np.asarray(grid[axis], float)
        extension = (
            float(positive_extension_coordinate_m[axis])
            if positive_extension_coordinate_m is not None
            else float(base[-1] + (base[-1] - base[-2]))
        )
        if not extension > base[-1]:
            raise RuntimeError(
                f"{axis} positive extension {extension} does not exceed {base[-1]}"
            )
        extended_grid[axis] = np.append(
            base, extension
        )
    extended_shape = tuple(
        extended_grid[axis].size for axis in "xyz"
    ) + (1, 3)
    result = np.zeros(extended_shape, dtype=value.dtype)
    records: dict[str, object] = {}
    for axis_index, component in enumerate("xyz"):
        base = np.asarray(grid[component], float)
        common_coordinate = extended_grid[component]
        native = base + np.asarray(
            grid[f"delta_{component}"], float
        )
        spacing = np.diff(common_coordinate)
        alpha = (
            common_coordinate[1:] - native
        ) / spacing
        beta = 1.0 - alpha
        if np.any(alpha <= 0.0) or np.any(beta < 0.0):
            raise RuntimeError(
                f"{component} Yee points do not bracket on the base grid"
            )
        target = np.asarray(value[..., axis_index])
        padded_shape = tuple(
            extended_grid[axis].size for axis in "xyz"
        ) + (1,)
        padded_target = np.zeros(padded_shape, dtype=target.dtype)
        original = tuple(
            slice(0, grid[axis].size) for axis in "xyz"
        ) + (slice(None),)
        padded_target[original] = target
        common = np.zeros_like(padded_target)
        common_view = np.moveaxis(common, axis_index, 0)
        target_view = np.moveaxis(padded_target, axis_index, 0)
        for i in range(base.size - 1, -1, -1):
            common_view[i] = (
                target_view[i] - beta[i] * common_view[i + 1]
            ) / alpha[i]
        reconstructed = (
            alpha.reshape((-1,) + (1,) * (common_view.ndim - 1))
            * common_view[: base.size]
            + beta.reshape((-1,) + (1,) * (common_view.ndim - 1))
            * common_view[1 : base.size + 1]
        )
        reconstruction_error = float(
            np.max(np.abs(reconstructed - target_view[: base.size]))
        )
        result[..., axis_index] = common
        target_scale = float(np.max(np.abs(target)))
        records[component] = {
            "array_shape": list(target.shape),
            "native_coordinate_bounds_m": [
                float(native[0]),
                float(native[-1]),
            ],
            "extended_common_coordinate_bounds_m": [
                float(common_coordinate[0]),
                float(common_coordinate[-1]),
            ],
            "maximum_staggering_offset_m": float(
                np.max(np.abs(native - base))
            ),
            "alpha_bounds": [
                float(np.min(alpha)),
                float(np.max(alpha)),
            ],
            "beta_bounds": [
                float(np.min(beta)),
                float(np.max(beta)),
            ],
            "reconstruction_max_abs_error": reconstruction_error,
            "common_to_native_max_amplitude_ratio": float(
                np.max(np.abs(common))
                / max(target_scale, np.finfo(float).tiny)
            ),
            "boundary_condition": (
                "one-cell +axis common-grid extension with outer "
                "common sample zero; no native target deletion"
            ),
        }
    return np.ascontiguousarray(result), extended_grid, {
        "method": (
            "exact backward solve of component-wise linear "
            "common-FieldRegion-to-native-Yee collocation operator "
            "on one-cell +axis-extended source grid"
        ),
        "original_shape": list(value.shape),
        "extended_shape": list(result.shape),
        "components": records,
        "empirical_normalization": False,
        "gradient_rescaling": False,
        "positive_extension_coordinates_from_solver_mesh": (
            positive_extension_coordinate_m is not None
        ),
    }


def weighted_fieldregion_source_from_native_multiplier(
    *,
    electric: np.ndarray,
    epsilon: np.ndarray,
    coefficient: np.ndarray,
    grid: dict[str, np.ndarray],
    frequency_hz: float,
) -> tuple[np.ndarray, dict[str, object]]:
    """Keep the official FieldRegion E profile and collocate only dF/d|E|2.

    ``getresult(FieldRegion, "E")`` is the supported lumopt source dataset.
    Its complex E values must not be deconvolved.  The scalar absorption
    multiplier, however, is defined at each component's native Yee
    coordinate.  Interpolate that slowly varying scalar onto the common
    FieldRegion labels before multiplying the unchanged forward E dataset.
    """

    field = as_e5(electric)
    permittivity = as_e5(epsilon)
    if field.shape != permittivity.shape:
        raise ValueError("E/epsilon shapes differ")
    if coefficient.shape != (*field.shape[:3], 3):
        raise ValueError("coefficient shape differs from E")
    omega = 2.0 * np.pi * float(frequency_hz)
    source = np.zeros_like(field)
    records = {}
    for axis_index, component in enumerate("xyz"):
        base = np.asarray(grid[component], float)
        native = base + np.asarray(
            grid[f"delta_{component}"], float
        )
        multiplier_native = (
            0.5
            * EPS0
            * omega
            * np.imag(permittivity[..., 0, axis_index])
            * coefficient[..., axis_index]
        )
        multiplier_common = interp1d(
            native,
            multiplier_native,
            axis=axis_index,
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",
            assume_sorted=True,
        )(base)
        source[..., 0, axis_index] = (
            multiplier_common * field[..., 0, axis_index]
        )
        records[component] = {
            "native_coordinate_bounds_m": [
                float(native[0]),
                float(native[-1]),
            ],
            "common_coordinate_bounds_m": [
                float(base[0]),
                float(base[-1]),
            ],
            "maximum_staggering_offset_m": float(
                np.max(np.abs(native - base))
            ),
            "multiplier_native_L2": float(
                np.linalg.norm(multiplier_native)
            ),
            "multiplier_common_L2": float(
                np.linalg.norm(multiplier_common)
            ),
            "interpolation": (
                "linear native-Yee scalar multiplier to common "
                "FieldRegion coordinates"
            ),
            "boundary_policy": "linear extrapolation; no source deletion",
        }
    return source, {
        "method": (
            "unchanged official FieldRegion forward-E dataset multiplied "
            "by coordinate-collocated native scalar absorption multiplier"
        ),
        "components": records,
        "forward_E_deconvolution": False,
        "empirical_normalization": False,
        "gradient_rescaling": False,
        "source_deletion": False,
    }


def import_fieldregion_profile(
    fdtd: object, grid: dict[str, np.ndarray], profile: np.ndarray
) -> float:
    return import_named_fieldregion_profile(
        fdtd, FIELD_REGION, grid, profile
    )


def import_named_fieldregion_profile(
    fdtd: object,
    object_name: str,
    grid: dict[str, np.ndarray],
    profile: np.ndarray,
) -> float:
    value = as_e5(profile)
    expected = (
        grid["x"].size,
        grid["y"].size,
        grid["z"].size,
        1,
        3,
    )
    if value.shape != expected:
        raise ValueError(f"profile {value.shape} != {expected}")
    fdtd.putv("ad_x", np.asarray(grid["x"], float))
    fdtd.putv("ad_y", np.asarray(grid["y"], float))
    fdtd.putv("ad_z", np.asarray(grid["z"], float))
    fdtd.putv("ad_f", float(np.asarray(grid["f"]).reshape(-1)[0]))
    fdtd.putv("ad_Ex", np.ascontiguousarray(value[..., 0]))
    fdtd.putv("ad_Ey", np.ascontiguousarray(value[..., 1]))
    fdtd.putv("ad_Ez", np.ascontiguousarray(value[..., 2]))
    fdtd.eval(
        'ad_ds=rectilineardataset("EM fields",ad_x,ad_y,ad_z);'
        'ad_ds.addparameter("lambda",c/ad_f,"f",ad_f);'
        'ad_ds.addattribute("E",ad_Ex,ad_Ey,ad_Ez);'
        f'select("{object_name}");importdataset(ad_ds);'
    )
    imported = as_e5(
        fdtd.getresult(object_name, "source profile")["E"]
    )
    error = float(np.max(np.abs(imported - value)))
    if error != 0.0:
        raise RuntimeError(
            f"FieldRegion source round-trip error {error:.3e}"
        )
    return error


def prepare_component_yee_adjoint_layout(
    fdtd: object,
    *,
    grid: dict[str, np.ndarray],
    native_source: np.ndarray,
    template: Path,
) -> tuple[float, dict[str, object]]:
    """Create three sources on the exact Ex/Ey/Ez Yee coordinate sets.

    A single FieldRegion vector source has one common rectilinear coordinate
    set.  Native Yee Ex, Ey, and Ez samples do not share that set.  Three
    simultaneously active FieldRegion sources avoid any component-to-common
    interpolation: each source contains one nonzero vector component and is
    placed directly on that component's physical Yee coordinates.
    """

    value = as_e5(native_source)
    expected = (
        grid["x"].size,
        grid["y"].size,
        grid["z"].size,
        1,
        3,
    )
    if value.shape != expected:
        raise ValueError(f"native source {value.shape} != {expected}")
    profile, profile_scale = fieldregion_profile(value)
    fdtd.switchtolayout()
    source_count_before = int(fdtd.getnamednumber(SOURCE_NAME))
    if source_count_before != 1:
        raise RuntimeError(
            f"expected exactly one TFSF source, found {source_count_before}"
        )
    fdtd.select(SOURCE_NAME)
    fdtd.delete()
    if int(fdtd.getnamednumber(SOURCE_NAME)) != 0:
        raise RuntimeError("TFSF source was not removed")

    original_name = FIELD_REGION
    records: dict[str, object] = {}
    base_amplitudes = []
    for index, component in enumerate("xyz"):
        name = f"{FIELD_REGION}_adjoint_{component}"
        if index == 0:
            if int(fdtd.getnamednumber(original_name)) != 1:
                raise RuntimeError("original FieldRegion is missing")
            fdtd.setnamed(original_name, "name", name)
        else:
            region = fdtd.addfieldregion()
            region["name"] = name
            region["monitor type"] = "3D"
            set_single_frequency(region)
        component_grid = {
            key: np.array(item, copy=True) for key, item in grid.items()
        }
        component_grid[component] = (
            np.asarray(grid[component], float)
            + np.asarray(grid[f"delta_{component}"], float)
        )
        for axis in "xyz":
            fdtd.setnamed(
                name, f"{axis} min", float(component_grid[axis][0])
            )
            fdtd.setnamed(
                name, f"{axis} max", float(component_grid[axis][-1])
            )
        fdtd.setnamed(name, "source mode", True)
        try:
            fdtd.setnamed(name, "nuttall window pulse", False)
        except Exception:
            pass
        component_profile = np.zeros_like(profile)
        component_profile[..., index] = profile[..., index]
        roundtrip = import_named_fieldregion_profile(
            fdtd, name, component_grid, component_profile
        )
        base_amplitude = float(fdtd.getnamed(name, "base amplitude"))
        base_amplitudes.append(base_amplitude)
        records[component] = {
            "object_name": name,
            "array_shape": list(component_profile.shape),
            "nonzero_vector_component": component,
            "physical_coordinate_bounds_m": {
                axis: [
                    float(component_grid[axis][0]),
                    float(component_grid[axis][-1]),
                ]
                for axis in "xyz"
            },
            "staggering_offset_bounds_m": [
                float(np.min(grid[f"delta_{component}"])),
                float(np.max(grid[f"delta_{component}"])),
            ],
            "source_profile_roundtrip_max_abs_error": roundtrip,
            "base_amplitude": base_amplitude,
        }
    reference_amplitude = base_amplitudes[0]
    amplitude_error = max(
        abs(item - reference_amplitude)
        / max(abs(reference_amplitude), np.finfo(float).tiny)
        for item in base_amplitudes
    )
    if amplitude_error > 1.0e-12:
        raise RuntimeError(
            "component FieldRegion base amplitudes differ: "
            f"{amplitude_error:.3e}"
        )
    fdtd.save(str(template))
    return profile_scale, {
        "method": (
            "three simultaneous one-component FieldRegion sources, each "
            "defined directly on its component-specific native Yee grid"
        ),
        "component_sources": records,
        "number_of_adjoint_sources": 3,
        "single_adjoint_Maxwell_solve": True,
        "common_grid_interpolation": False,
        "source_profile_global_scale": profile_scale,
        "base_amplitude": reference_amplitude,
        "base_amplitude_max_relative_difference": amplitude_error,
        "empirical_normalization": False,
        "gradient_rescaling": False,
        "source_deletion": False,
        "template": {
            "path": str(template),
            "byte_size": template.stat().st_size,
            "sha256": sha256(template),
        },
    }


def prepare_single_component_yee_adjoint_layout(
    fdtd: object,
    *,
    grid: dict[str, np.ndarray],
    native_source: np.ndarray,
    component: str,
    template: Path,
) -> tuple[float, dict[str, object]]:
    """Create one FieldRegion source on one exact Yee component grid."""

    if component not in "xyz":
        raise ValueError(component)
    index = "xyz".index(component)
    value = as_e5(native_source)
    component_value = np.zeros_like(value)
    component_value[..., index] = value[..., index]
    profile, profile_scale = fieldregion_profile(component_value)
    fdtd.switchtolayout()
    if int(fdtd.getnamednumber(SOURCE_NAME)) != 1:
        raise RuntimeError("expected exactly one TFSF source")
    fdtd.select(SOURCE_NAME)
    fdtd.delete()
    if int(fdtd.getnamednumber(SOURCE_NAME)) != 0:
        raise RuntimeError("TFSF source was not removed")
    if int(fdtd.getnamednumber(FIELD_REGION)) != 1:
        raise RuntimeError("expected exactly one original FieldRegion")
    component_grid = {
        key: np.array(item, copy=True) for key, item in grid.items()
    }
    component_grid[component] = (
        np.asarray(grid[component], float)
        + np.asarray(grid[f"delta_{component}"], float)
    )
    for axis in "xyz":
        fdtd.setnamed(
            FIELD_REGION,
            f"{axis} min",
            float(component_grid[axis][0]),
        )
        fdtd.setnamed(
            FIELD_REGION,
            f"{axis} max",
            float(component_grid[axis][-1]),
        )
    fdtd.setnamed(FIELD_REGION, "source mode", True)
    try:
        fdtd.setnamed(FIELD_REGION, "nuttall window pulse", False)
    except Exception:
        pass
    roundtrip = import_named_fieldregion_profile(
        fdtd, FIELD_REGION, component_grid, profile
    )
    base_amplitude = float(
        fdtd.getnamed(FIELD_REGION, "base amplitude")
    )
    fdtd.save(str(template))
    return profile_scale, {
        "method": (
            "one single-component FieldRegion source directly on the "
            f"native E{component} Yee grid"
        ),
        "component": component,
        "physical_coordinate_bounds_m": {
            axis: [
                float(component_grid[axis][0]),
                float(component_grid[axis][-1]),
            ]
            for axis in "xyz"
        },
        "staggering_offset_bounds_m": [
            float(np.min(grid[f"delta_{component}"])),
            float(np.max(grid[f"delta_{component}"])),
        ],
        "source_profile_roundtrip_max_abs_error": roundtrip,
        "source_profile_scale": profile_scale,
        "base_amplitude": base_amplitude,
        "common_grid_interpolation": False,
        "empirical_normalization": False,
        "gradient_rescaling": False,
        "template": {
            "path": str(template),
            "byte_size": template.stat().st_size,
            "sha256": sha256(template),
        },
    }


def set_gray_density(fdtd: object, rho: float) -> None:
    epsilon = 1.0 + float(rho) * (SIO2_EPSILON - 1.0)
    fdtd.setnamed(DESIGN_OBJECT, "index", float(np.sqrt(epsilon)))


def run_forward(
    fdtd: object,
    *,
    rho: float,
    project: Path,
    threads: int,
    flux_signs: dict[str, float],
) -> dict[str, object]:
    fdtd.switchtolayout()
    set_gray_density(fdtd, rho)
    fdtd.setnamed(FIELD_REGION, "source mode", False)
    fdtd.setnamed(SOURCE_NAME, "enabled", True)
    resources = configure_cpu(fdtd, threads)
    fdtd.save(str(project))
    started = time.monotonic()
    resource_name = run_on_cpu(fdtd)
    wall = time.monotonic() - started
    # Persist completed results before any post-processing assertion so a
    # failed extraction remains auditable rather than leaving a layout-only
    # project at the recorded path.
    fdtd.save(str(project))
    # The pabs field and pabs index d-cards are the already certified native
    # Yee pair.  v261 FieldRegion monitor coordinates are not identical to a
    # separately added index monitor even when their nominal bounds match.
    # Build both the scalar objective and dP/dE* on this one pabs grid, then
    # import that exact rectilinear dataset into FieldRegion source mode.
    electric, grid = monitor_electric(fdtd, PABS_FIELD)
    epsilon, index_grid = monitor_epsilon(fdtd, PABS_INDEX)
    for axis in "xyzf":
        if not np.array_equal(grid[axis], index_grid[axis]):
            raise RuntimeError(f"Q E/index {axis} grids differ")
    objective, q_source, components = absorption_objective_and_source(
        electric, epsilon, grid
    )
    native_pabs = extract_native_yee_q(
        fdtd,
        field_monitor=PABS_FIELD,
        index_monitor=PABS_INDEX,
        wavelength_m=WAVELENGTH_M,
    )
    source_power = scalar(
        fdtd.sourcepower(FREQUENCY_HZ, 2, SOURCE_NAME), "source power"
    )
    face_power: dict[str, object] = {}
    net_outward = 0.0
    for name, sign in flux_signs.items():
        signed_axis_power = (
            scalar(fdtd.transmission(name), name) * source_power
        )
        outward_power = sign * signed_axis_power
        face_power[name] = {
            "signed_axis_power_W": signed_axis_power,
            "outward_power_W": outward_power,
        }
        net_outward += outward_power
    p_six = -net_outward
    closure = abs(objective - p_six) / max(
        abs(p_six), np.finfo(float).tiny
    )
    fdtd.runanalysis(PABS_GROUP)
    fdtd.save(str(project))
    return {
        "rho": float(rho),
        "objective_W": objective,
        "component_power_W": components,
        "q_source": q_source,
        "electric": electric,
        "epsilon": epsilon,
        "grid": grid,
        "pabs_P_Q_W": float(native_pabs["P_Q_W"]),
        "P_six_W": p_six,
        "six_face_closure_relative_error": closure,
        "six_face_power": face_power,
        "fieldregion_vs_pabs_relative_error": abs(
            objective - native_pabs["P_Q_W"]
        )
        / max(abs(native_pabs["P_Q_W"]), np.finfo(float).tiny),
        "source_intensity_W_m2": scalar(
            fdtd.sourceintensity(FREQUENCY_HZ, 2, SOURCE_NAME),
            "source intensity",
        ),
        "run_wall_s": wall,
        "resource_name": resource_name,
        "resources": resources,
        "project": {
            "path": str(project),
            "byte_size": project.stat().st_size,
            "sha256": sha256(project),
        },
    }


def prepare_adjoint_layout(
    fdtd: object,
    *,
    grid: dict[str, np.ndarray],
    profile: np.ndarray,
    template: Path,
) -> dict[str, object]:
    fdtd.switchtolayout()
    source_count_before = int(fdtd.getnamednumber(SOURCE_NAME))
    if source_count_before != 1:
        raise RuntimeError(
            f"expected exactly one TFSF source, found {source_count_before}"
        )
    fdtd.select(SOURCE_NAME)
    fdtd.delete()
    source_count_after = int(fdtd.getnamednumber(SOURCE_NAME))
    if source_count_after != 0:
        raise RuntimeError("TFSF source was not removed from adjoint layout")
    fdtd.setnamed(FIELD_REGION, "source mode", True)
    try:
        fdtd.setnamed(FIELD_REGION, "nuttall window pulse", False)
    except Exception:
        pass
    roundtrip = import_fieldregion_profile(fdtd, grid, profile)
    base_amplitude = float(
        fdtd.getnamed(FIELD_REGION, "base amplitude")
    )
    fdtd.save(str(template))
    return {
        "tfsf_count_before": source_count_before,
        "tfsf_count_after": source_count_after,
        "source_profile_roundtrip_max_abs_error": roundtrip,
        "fieldregion_base_amplitude": base_amplitude,
        "template": {
            "path": str(template),
            "byte_size": template.stat().st_size,
            "sha256": sha256(template),
        },
    }


def run_adjoint(
    fdtd: object,
    *,
    template: Path,
    project: Path,
    engine: str,
    threads: int,
    gpu_device: str,
) -> dict[str, object]:
    fdtd.switchtolayout()
    fdtd.load(str(template))
    if int(fdtd.getnamednumber(SOURCE_NAME)) != 0:
        raise RuntimeError("adjoint template unexpectedly contains TFSF")
    if engine == "CPU":
        resources = configure_cpu(fdtd, threads)
        runner = run_on_cpu
    elif engine == "GPU":
        resources = configure_gpu(fdtd, threads, gpu_device)
        runner = run_on_gpu
    else:
        raise ValueError(engine)
    fdtd.save(str(project))
    started = time.monotonic()
    resource_name = runner(fdtd)
    wall = time.monotonic() - started
    electric, grid = monitor_electric(fdtd, DESIGN_FIELD)
    fdtd.save(str(project))
    return {
        "engine": engine,
        "electric": electric,
        "grid": grid,
        "run_wall_s": wall,
        "resource_name": resource_name,
        "resources": resources,
        "project": {
            "path": str(project),
            "byte_size": project.stat().st_size,
            "sha256": sha256(project),
        },
    }


def layout_density_derivative(
    fdtd: object,
    *,
    template: Path,
    rho: float,
    step: float,
) -> tuple[np.ndarray, dict[str, object]]:
    fdtd.switchtolayout()
    fdtd.load(str(template))
    pair = []
    axes = None
    for sign in (1.0, -1.0):
        set_gray_density(fdtd, rho + sign * step)
        epsilon, grid = monitor_epsilon(fdtd, DESIGN_INDEX)
        if axes is None:
            axes = grid
        else:
            for axis in "xyzf":
                if not np.array_equal(axes[axis], grid[axis]):
                    raise RuntimeError(
                        f"design index {axis} grid changed in layout probe"
                    )
        pair.append(epsilon)
    derivative = (pair[0] - pair[1]) / (2.0 * step)
    set_gray_density(fdtd, rho)
    return derivative, {
        "step": step,
        "method": "centered layout-only conformal epsilon_Yee derivative",
        "electromagnetic_solves": 0,
        "max_abs_imaginary_derivative": float(
            np.max(np.abs(np.imag(derivative)))
        ),
    }


def gradient_from_adjoint(
    *,
    forward_electric: np.ndarray,
    adjoint_electric: np.ndarray,
    design_grid: dict[str, np.ndarray],
    d_epsilon_d_rho: np.ndarray,
    profile_scale: float,
    base_amplitude: float,
    design_bounds_m: dict[str, tuple[float, float]],
) -> tuple[float, list[float]]:
    if not (
        forward_electric.shape
        == adjoint_electric.shape
        == d_epsilon_d_rho.shape
    ):
        raise ValueError(
            "forward/adjoint/d-epsilon design arrays do not match: "
            f"{forward_electric.shape}, {adjoint_electric.shape}, "
            f"{d_epsilon_d_rho.shape}"
        )
    # J=d(epsilon_Yee)/d(rho) already contains the exact conformal fill and
    # design-support intersection. The Maxwell bilinear form is integrated
    # over the complete Yee dual cell. Clipping dV to the nominal design box
    # here would apply the support fraction a second time.
    del design_bounds_m
    volumes = component_volumes(design_grid)
    components = []
    for component in range(3):
        value = np.sum(
            (2.0 * EPS0 / base_amplitude)
            * volumes[component][..., None]
            * forward_electric[..., component]
            * (adjoint_electric[..., component] * profile_scale)
            * d_epsilon_d_rho[..., component]
        )
        components.append(float(np.real(value)))
    return float(sum(components)), components


def strip_arrays(value: dict[str, object]) -> dict[str, object]:
    return {
        key: item
        for key, item in value.items()
        if key not in {"q_source", "electric", "epsilon", "grid"}
    }


def main() -> int:
    args = parse_args()
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "mixed_optical_adfd_result.json"
    result: dict[str, object] = {
        "status": "BLOCKED_MIXED_OPTICAL_ADFD_NOT_RUN",
        "passed": False,
        "purpose": (
            "solver-level scalar rho optical AD-FD; no thermal, PTE, "
            "filter, projection, adjoint optimization, or geometry update"
        ),
        "rho": args.rho,
        "fd_steps": args.fd_steps,
        "design_interpolation": (
            "epsilon(rho)=1+rho*(1.38^2-1), uniform scalar certificate"
        ),
        "forward_engine": "CPU TFSF",
        "adjoint_engines": ["CPU FieldRegion", "GPU FieldRegion"],
        "periodic_or_bloch": False,
        "tfsf_removed_from_adjoint": None,
        "solver_root": str(APPROVED_ROOT),
        "lumapi_path": str(APPROVED_API),
    }
    fdtd = None
    started = time.monotonic()
    try:
        contract = baseline_contract(
            lateral_domain_um=args.domain_um,
            z_min_um=args.z_min_um,
            z_max_um=args.z_max_um,
            pml_layers=args.pml_layers,
        )
        lumapi = load_lumapi()
        fdtd = lumapi.FDTD(hide=True, serverArgs={"platform": "offscreen"})
        add_large_background_geometry(fdtd, contract)
        pml_readback = configure_pml_profile(fdtd, args.pml_profile)
        result["pml"] = {
            "profile": args.pml_profile,
            "profile_matrix_readback": pml_readback,
            "layers": args.pml_layers,
            "outer_bounds_m": contract.fdtd_outer.to_dict(),
        }
        configure_design(fdtd, "gray", args.rho)
        fdtd.setnamed(
            "FDTD", "simulation time", args.simulation_time_ps * 1e-12
        )
        flux_signs = add_monitors(fdtd, contract)
        add_adjoint_monitors(fdtd, contract)

        base_project = output / "mixed_base_forward_cpu_tfsf.fsp"
        base = run_forward(
            fdtd,
            rho=args.rho,
            project=base_project,
            threads=args.threads,
            flux_signs=flux_signs,
        )
        result["base_forward"] = strip_arrays(base)

        profile, profile_scale = fieldregion_profile(base["q_source"])
        template = output / "mixed_adjoint_source_template.fsp"
        template_meta = prepare_adjoint_layout(
            fdtd,
            grid=base["grid"],
            profile=profile,
            template=template,
        )
        result["adjoint_source"] = {
            **template_meta,
            "definition": (
                "profile=conj(dP_Q/dE*)/max|dP_Q/dE*|; "
                "dP_Q/dE*=0.5*omega*epsilon0*Im(epsilon)*dV*E"
            ),
            "profile_scale": profile_scale,
        }
        result["tfsf_removed_from_adjoint"] = bool(
            template_meta["tfsf_count_after"] == 0
        )

        # Run the production-candidate GPU adjoint first so an unsupported or
        # unstable configuration fails before the slower CPU equivalence
        # control consumes time.
        gpu_adjoint = run_adjoint(
            fdtd,
            template=template,
            project=output / "mixed_adjoint_gpu.fsp",
            engine="GPU",
            threads=args.threads,
            gpu_device=args.gpu_device,
        )
        cpu_adjoint = run_adjoint(
            fdtd,
            template=template,
            project=output / "mixed_adjoint_cpu.fsp",
            engine="CPU",
            threads=args.threads,
            gpu_device=args.gpu_device,
        )
        for axis in "xyzf":
            if not np.array_equal(
                cpu_adjoint["grid"][axis], gpu_adjoint["grid"][axis]
            ):
                raise RuntimeError(
                    f"CPU/GPU adjoint {axis} grids are not identical"
                )
        cpu_field = cpu_adjoint["electric"]
        gpu_field = gpu_adjoint["electric"]
        field_nrmse = float(
            np.linalg.norm(gpu_field - cpu_field)
            / max(np.linalg.norm(cpu_field), np.finfo(float).tiny)
        )
        result["cpu_gpu_adjoint_equivalence"] = {
            "CPU": strip_arrays(cpu_adjoint),
            "GPU": strip_arrays(gpu_adjoint),
            "complex_field_NRMSE": field_nrmse,
            "limit": 5.0e-3,
        }

        fdtd.switchtolayout()
        fdtd.load(str(base_project))
        forward_design_e, design_grid = monitor_electric(
            fdtd, DESIGN_FIELD
        )
        layout_step = min(args.fd_steps) / 4.0
        derivative, derivative_meta = layout_density_derivative(
            fdtd,
            template=template,
            rho=args.rho,
            step=layout_step,
        )
        cpu_gradient, cpu_components = gradient_from_adjoint(
            forward_electric=forward_design_e,
            adjoint_electric=cpu_field,
            design_grid=design_grid,
            d_epsilon_d_rho=derivative,
            profile_scale=profile_scale,
            base_amplitude=template_meta["fieldregion_base_amplitude"],
            design_bounds_m={
                axis: getattr(contract.design, axis) for axis in "xyz"
            },
        )
        gpu_gradient, gpu_components = gradient_from_adjoint(
            forward_electric=forward_design_e,
            adjoint_electric=gpu_field,
            design_grid=design_grid,
            d_epsilon_d_rho=derivative,
            profile_scale=profile_scale,
            base_amplitude=template_meta["fieldregion_base_amplitude"],
            design_bounds_m={
                axis: getattr(contract.design, axis) for axis in "xyz"
            },
        )
        result["adjoint_gradient_dP_Q_d_rho_W"] = {
            "CPU": cpu_gradient,
            "GPU": gpu_gradient,
            "CPU_components_xyz": cpu_components,
            "GPU_components_xyz": gpu_components,
            "CPU_GPU_relative_difference": abs(
                gpu_gradient - cpu_gradient
            )
            / max(abs(cpu_gradient), np.finfo(float).tiny),
            "layout_derivative": derivative_meta,
            "explicit_design_loss_term": (
                "zero: the air/SiO2 design is lossless and TaIrTe4 is fixed"
            ),
        }

        fd_rows = []
        for step in args.fd_steps:
            pair = []
            pair_meta = []
            for sign in (1.0, -1.0):
                rho = args.rho + sign * step
                label = f"fd_rho_{rho:.8f}".replace(".", "p")
                # Every independent FD forward starts from the exact saved
                # CPU-TFSF project.  The immediately preceding adjoint layout
                # has intentionally deleted the TFSF object.
                fdtd.switchtolayout()
                fdtd.load(str(base_project))
                forward = run_forward(
                    fdtd,
                    rho=rho,
                    project=output / f"{label}_cpu_tfsf.fsp",
                    threads=args.threads,
                    flux_signs=flux_signs,
                )
                pair.append(forward["objective_W"])
                pair_meta.append(strip_arrays(forward))
            centered = (pair[0] - pair[1]) / (2.0 * step)
            error = abs(gpu_gradient - centered) / max(
                abs(centered), np.finfo(float).tiny
            )
            fd_rows.append(
                {
                    "step": step,
                    "rho_plus": args.rho + step,
                    "rho_minus": args.rho - step,
                    "objective_plus_W": pair[0],
                    "objective_minus_W": pair[1],
                    "centered_FD_dP_Q_d_rho_W": centered,
                    "GPU_adjoint_dP_Q_d_rho_W": gpu_gradient,
                    "relative_error": error,
                    "forward_cases": pair_meta,
                }
            )
        result["centered_FD_step_sweep"] = fd_rows
        best_error = min(row["relative_error"] for row in fd_rows)
        step_change = (
            abs(
                fd_rows[-1]["centered_FD_dP_Q_d_rho_W"]
                - fd_rows[-2]["centered_FD_dP_Q_d_rho_W"]
            )
            / max(
                abs(fd_rows[-1]["centered_FD_dP_Q_d_rho_W"]),
                np.finfo(float).tiny,
            )
            if len(fd_rows) >= 2
            else None
        )
        result["gates"] = {
            "fieldregion_vs_pabs_Q_error_lt_0p5pct": (
                base["fieldregion_vs_pabs_relative_error"] < 5.0e-3
            ),
            "base_Q_six_face_closure_lt_0p5pct": (
                base["six_face_closure_relative_error"] < 5.0e-3
            ),
            "source_intensity_error_lt_0p5pct": (
                abs(base["source_intensity_W_m2"] - 1.0) < 5.0e-3
            ),
            "tfsf_absent_from_adjoint": result[
                "tfsf_removed_from_adjoint"
            ],
            "source_profile_roundtrip_exact": (
                template_meta[
                    "source_profile_roundtrip_max_abs_error"
                ]
                == 0.0
            ),
            "cpu_gpu_adjoint_field_NRMSE_lt_0p5pct": (
                field_nrmse < 5.0e-3
            ),
            "cpu_gpu_gradient_difference_lt_1pct": (
                result["adjoint_gradient_dP_Q_d_rho_W"][
                    "CPU_GPU_relative_difference"
                ]
                < 1.0e-2
            ),
            "best_AD_FD_error_lt_1pct": best_error < 1.0e-2,
            "FD_step_change_lt_1pct": (
                step_change is not None and step_change < 1.0e-2
            ),
            "all_FD_Q_six_face_closures_lt_0p5pct": all(
                case["six_face_closure_relative_error"] < 5.0e-3
                for row in fd_rows
                for case in row["forward_cases"]
            ),
        }
        result["best_AD_FD_relative_error"] = best_error
        result["FD_step_change_relative"] = step_change
        result["passed"] = bool(all(result["gates"].values()))
        result["status"] = (
            "VALIDATED_MIXED_CPU_TFSF_GPU_FIELDREGION_OPTICAL_ADFD"
            if result["passed"]
            else "FAILED_MIXED_CPU_TFSF_GPU_FIELDREGION_OPTICAL_ADFD"
        )
    except Exception as exc:
        result["status"] = "BLOCKED_MIXED_OPTICAL_ADFD_EXECUTION"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
        result["passed"] = False
    finally:
        result["total_wall_s"] = time.monotonic() - started
        if fdtd is not None:
            try:
                fdtd.close()
            except Exception:
                pass
        result_path.write_text(
            json.dumps(result, indent=2, default=json_default) + "\n"
        )
    print(json.dumps(result, indent=2, default=json_default), flush=True)
    return 0 if result.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
