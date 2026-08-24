"""Exact-Au geometry and dispersive-material builder for the 4 um route.

This module is intentionally usable without opening Lumerical.  Geometry and
material tables can therefore be audited on a CPU-only host before a B200 run.
The functions that accept ``fdtd`` only create layout objects; they never run
the Maxwell engine.
"""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys
from typing import Any

import numpy as np

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.contract import (
    CONTRACT,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.lumerical_maxwell_contract import (
    canonical_exact_au_geometry,
    exact_au_geometry_audit,
)


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]
C0_M_S = 299_792_458.0
AU_TABLE = (
    REPOSITORY
    / "photothermal_pte"
    / "optimization_runs"
    / "au_on_fixed_tairte4_validation"
    / "data"
    / "au_ordal_1987_nk.csv"
)
TAIRTE4_TABLE = REPOSITORY / "photothermal_pte" / "bundle" / "perm_data.txt"
KITAMURA_IMPLEMENTATION = (
    REPOSITORY
    / "photothermal_pte"
    / "validation"
    / "paper_ir_sanity"
    / "run_lumerical_device_a_ir_q.py"
)

SOURCE_WAVELENGTH_BAND_M = (3.60e-6, 4.40e-6)
MATERIAL_FIT_WAVELENGTH_BAND_M = (3.20e-6, 4.80e-6)
MATERIAL_SAMPLE_COUNT = 161
# Keep the high ceiling for the smoother non-Au fits, but do not reuse it for
# Ordal Au.  On v261 the Au 4-um fit has a stable 4--16 coefficient plateau;
# allowing 20 selects a different overfit branch that passes pointwise n-k
# readback yet fails closed-surface energy balance by about 29%.
MATERIAL_MAX_COEFFICIENTS = 20
AU_MATERIAL_MAX_COEFFICIENTS = 6
MATERIAL_FIT_TOLERANCE = 0.0

AU_MATERIAL = "Au_Ordal_4um_sampled_dispersive"
TAIRTE4_MATERIAL = "TaIrTe4_4um_anisotropic_sampled"
SIO2_MATERIAL = "SiO2_Kitamura_4um_sampled"
SI_MATERIAL = "Si (Silicon) - Palik"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_interpolation_domain(
    coordinates: np.ndarray, lower: float, upper: float, label: str
) -> np.ndarray:
    values = np.asarray(coordinates, dtype=np.float64)
    if (
        values.ndim != 1
        or values.size < 2
        or np.any(~np.isfinite(values))
        or np.any(np.diff(values) <= 0.0)
    ):
        raise RuntimeError(f"{label} coordinates are not finite/strictly increasing")
    if not values[0] <= lower < upper <= values[-1]:
        raise RuntimeError(
            f"{label} fit band [{lower:.9g}, {upper:.9g}] is outside "
            f"the tabulated domain [{values[0]:.9g}, {values[-1]:.9g}]"
        )
    return values


def design_edges() -> tuple[np.ndarray, np.ndarray]:
    """Return exact 100-nm physical cell edges for ``mask[ix, iy]``."""

    nx, ny = CONTRACT.design_shape
    x = np.linspace(
        -0.5 * CONTRACT.design_span_x_m,
        0.5 * CONTRACT.design_span_x_m,
        nx + 1,
        dtype=np.float64,
    )
    y = np.linspace(
        -0.5 * CONTRACT.design_span_y_m,
        0.5 * CONTRACT.design_span_y_m,
        ny + 1,
        dtype=np.float64,
    )
    if not np.allclose(np.diff(x), CONTRACT.design_pitch_m, rtol=0.0, atol=1e-18):
        raise RuntimeError("x design pitch does not match the physical contract")
    if not np.allclose(np.diff(y), CONTRACT.design_pitch_m, rtol=0.0, atol=1e-18):
        raise RuntimeError("y design pitch does not match the physical contract")
    return x, y


def exact_control_masks() -> dict[str, np.ndarray]:
    """Return deterministic empty, full, and 500-nm-safe L-shaped controls."""

    x_edges, y_edges = design_edges()
    x = 0.5 * (x_edges[:-1] + x_edges[1:])
    y = 0.5 * (y_edges[:-1] + y_edges[1:])
    empty = np.zeros(CONTRACT.design_shape, dtype=np.uint8)
    full = np.ones(CONTRACT.design_shape, dtype=np.uint8)
    # A 600-nm-wide asymmetric L exercises both in-plane axes and creates a
    # nontrivial temperature/current map while remaining above the 500-nm
    # solid-feature target on the exact 100-nm geometry grid.
    horizontal = (
        (x[:, None] >= -2.0e-6)
        & (x[:, None] < 2.0e-6)
        & (y[None, :] >= -1.5e-6)
        & (y[None, :] < -0.9e-6)
    )
    vertical = (
        (x[:, None] >= 1.4e-6)
        & (x[:, None] < 2.0e-6)
        & (y[None, :] >= -1.5e-6)
        & (y[None, :] < 2.0e-6)
    )
    simple_l = np.asarray(horizontal | vertical, dtype=np.uint8)
    return {"empty": empty, "full": full, "simple_L": simple_l}


def control_geometry_audits() -> dict[str, dict[str, Any]]:
    x_edges, y_edges = design_edges()
    z_bounds = np.asarray([0.0, CONTRACT.design_thickness_m])
    return {
        name: exact_au_geometry_audit(
            mask,
            x_edges_m=x_edges,
            y_edges_m=y_edges,
            z_bounds_m=z_bounds,
            axis_x=CONTRACT.axis_x,
            axis_y=CONTRACT.axis_y,
        )
        for name, mask in exact_control_masks().items()
    }


def mask_rectangles(
    mask: np.ndarray,
    *,
    x_edges_m: np.ndarray,
    y_edges_m: np.ndarray,
    z_bounds_m: np.ndarray,
) -> list[dict[str, float]]:
    """Coalesce an exact grid mask into deterministic non-overlapping prisms."""

    geometry = canonical_exact_au_geometry(
        mask,
        x_edges_m=x_edges_m,
        y_edges_m=y_edges_m,
        z_bounds_m=z_bounds_m,
        axis_x=CONTRACT.axis_x,
        axis_y=CONTRACT.axis_y,
    )
    value = geometry["mask"]
    x_edges = geometry["x_edges_m"]
    y_edges = geometry["y_edges_m"]
    z_bounds = geometry["z_bounds_m"]

    def runs(column: np.ndarray) -> list[tuple[int, int]]:
        padded = np.pad(np.asarray(column, dtype=np.int8), (1, 1))
        changes = np.diff(padded)
        starts = np.flatnonzero(changes == 1)
        stops = np.flatnonzero(changes == -1)
        return list(zip(starts.tolist(), stops.tolist(), strict=True))

    active: dict[tuple[int, int], int] = {}
    rectangles: list[dict[str, float]] = []

    def close(run: tuple[int, int], y_start: int, y_stop: int) -> None:
        ix0, ix1 = run
        rectangles.append(
            {
                "x_min_m": float(x_edges[ix0]),
                "x_max_m": float(x_edges[ix1]),
                "y_min_m": float(y_edges[y_start]),
                "y_max_m": float(y_edges[y_stop]),
                "z_min_m": float(z_bounds[0]),
                "z_max_m": float(z_bounds[1]),
            }
        )

    for iy in range(value.shape[1]):
        present = set(runs(value[:, iy]))
        for run in sorted(set(active) - present):
            close(run, active.pop(run), iy)
        for run in sorted(present - set(active)):
            active[run] = iy
    for run in sorted(active):
        close(run, active[run], value.shape[1])
    rectangles.sort(
        key=lambda item: (
            item["y_min_m"],
            item["x_min_m"],
            item["y_max_m"],
            item["x_max_m"],
        )
    )
    return rectangles


def _load_kitamura_function():
    spec = importlib.util.spec_from_file_location(
        "au_dualpol_4um_kitamura", KITAMURA_IMPLEMENTATION
    )
    if spec is None or spec.loader is None:
        raise ImportError(KITAMURA_IMPLEMENTATION)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.kitamura_2007_sio2_epsilon


def sampled_material_data() -> dict[str, np.ndarray]:
    """Return frequency-ascending complex epsilon tables for Lumerical fits."""

    wavelengths = np.linspace(
        MATERIAL_FIT_WAVELENGTH_BAND_M[0],
        MATERIAL_FIT_WAVELENGTH_BAND_M[1],
        MATERIAL_SAMPLE_COUNT,
        dtype=np.float64,
    )
    wavelength_um = wavelengths * 1.0e6
    wavelength_nm = wavelengths * 1.0e9

    au = np.genfromtxt(AU_TABLE, delimiter=",", names=True)
    au_names = au.dtype.names
    if au_names is None or len(au_names) != 3:
        raise RuntimeError("unexpected Ordal Au table schema")
    au_wavelength_um = _require_interpolation_domain(
        np.asarray(au[au_names[0]], dtype=np.float64),
        MATERIAL_FIT_WAVELENGTH_BAND_M[0] * 1.0e6,
        MATERIAL_FIT_WAVELENGTH_BAND_M[1] * 1.0e6,
        "Ordal Au wavelength_um",
    )
    au_n = np.interp(wavelength_um, au_wavelength_um, au[au_names[1]])
    au_k = np.interp(wavelength_um, au_wavelength_um, au[au_names[2]])
    epsilon_au = (au_n + 1j * au_k) ** 2

    ta = np.loadtxt(TAIRTE4_TABLE)
    ta = ta[np.argsort(ta[:, 0])]
    _require_interpolation_domain(
        ta[:, 0],
        MATERIAL_FIT_WAVELENGTH_BAND_M[0] * 1.0e9,
        MATERIAL_FIT_WAVELENGTH_BAND_M[1] * 1.0e9,
        "TaIrTe4 wavelength_nm",
    )
    epsilon_ta = {
        axis: np.interp(wavelength_nm, ta[:, 0], ta[:, real_column])
        + 1j * np.interp(wavelength_nm, ta[:, 0], ta[:, real_column + 1])
        for axis, real_column in (("a", 1), ("b", 3), ("c", 5))
    }
    if not np.array_equal(epsilon_ta["b"], epsilon_ta["c"]):
        raise RuntimeError("TaIrTe4 epsilon_c=epsilon_b closure changed")
    epsilon_sio2 = np.asarray(_load_kitamura_function()(wavelengths), complex)

    frequencies = C0_M_S / wavelengths
    order = np.argsort(frequencies)
    result = {
        "frequency_hz": np.ascontiguousarray(frequencies[order]),
        "wavelength_m": np.ascontiguousarray(wavelengths[order]),
        "epsilon_au": np.ascontiguousarray(epsilon_au[order]),
        "epsilon_ta_x_b": np.ascontiguousarray(epsilon_ta["b"][order]),
        "epsilon_ta_y_a": np.ascontiguousarray(epsilon_ta["a"][order]),
        "epsilon_ta_z_c": np.ascontiguousarray(epsilon_ta["c"][order]),
        "epsilon_sio2": np.ascontiguousarray(epsilon_sio2[order]),
    }
    if np.any(np.diff(result["frequency_hz"]) <= 0.0):
        raise RuntimeError("material frequencies are not strictly increasing")
    for key in (
        "epsilon_au",
        "epsilon_ta_x_b",
        "epsilon_ta_y_a",
        "epsilon_ta_z_c",
        "epsilon_sio2",
    ):
        if np.any(~np.isfinite(result[key])) or np.any(result[key].imag < 0.0):
            raise RuntimeError(f"non-passive or non-finite sampled material: {key}")
    return result


def _complex_record(value: complex) -> dict[str, float]:
    return {"real": float(value.real), "imag": float(value.imag)}


def au_fit_configuration(
    *,
    max_coefficients: int = AU_MATERIAL_MAX_COEFFICIENTS,
    tolerance: float = MATERIAL_FIT_TOLERANCE,
) -> dict[str, Any]:
    """Validate and record the independently swept Au fit parameters."""

    if not 1 <= int(max_coefficients) == max_coefficients <= 20:
        raise ValueError("Au max coefficients must be an integer in [1,20]")
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("Au fit tolerance must be finite and nonnegative")
    return {
        "max_coefficients": int(max_coefficients),
        "tolerance": float(tolerance),
        "make_fit_passive": "Lumerical default true; not disabled",
        "improve_stability": "Lumerical default true; not disabled",
        "imaginary_weight": "Lumerical default 1; not overridden",
    }


def material_contract_audit(
    *,
    au_max_coefficients: int = AU_MATERIAL_MAX_COEFFICIENTS,
    au_fit_tolerance: float = MATERIAL_FIT_TOLERANCE,
) -> dict[str, Any]:
    data = sampled_material_data()
    au_fit = au_fit_configuration(
        max_coefficients=au_max_coefficients,
        tolerance=au_fit_tolerance,
    )
    target_frequency = C0_M_S / CONTRACT.wavelength_m
    target_index = int(np.argmin(np.abs(data["frequency_hz"] - target_frequency)))
    return {
        "status": "AUDITED_4UM_DISPERSIVE_MATERIAL_INPUTS_NOT_FIT_READBACK",
        "source_wavelength_band_m": list(SOURCE_WAVELENGTH_BAND_M),
        "material_fit_wavelength_band_m": list(MATERIAL_FIT_WAVELENGTH_BAND_M),
        "material_sample_count": MATERIAL_SAMPLE_COUNT,
        "frequency_strictly_increasing": True,
        "default_non_Au_fit": {
            "max_coefficients": MATERIAL_MAX_COEFFICIENTS,
            "tolerance": MATERIAL_FIT_TOLERANCE,
            "requires_post_run_Lumerical_fit_readback": True,
        },
        "Au_fit": {
            **au_fit,
            "requires_post_run_Lumerical_fit_readback": True,
            "convergence_role": (
                "one-axis sampled-data pole-count/tolerance diagnostic; "
                "not a physical design variable"
            ),
        },
        "materials": {
            "Au": {
                "name": AU_MATERIAL,
                "model": "Sampled data; Ordal 1987 n,k converted to epsilon",
                "epsilon_nearest_4um": _complex_record(
                    data["epsilon_au"][target_index]
                ),
            },
            "TaIrTe4": {
                "name": TAIRTE4_MATERIAL,
                "model": "anisotropic Sampled 3D data",
                "axis_mapping": {"x": "b", "y": "a", "z": "c=b closure"},
                "epsilon_x_b_nearest_4um": _complex_record(
                    data["epsilon_ta_x_b"][target_index]
                ),
                "epsilon_y_a_nearest_4um": _complex_record(
                    data["epsilon_ta_y_a"][target_index]
                ),
            },
            "SiO2": {
                "name": SIO2_MATERIAL,
                "model": "Sampled data; Kitamura 2007 dielectric function",
                "epsilon_nearest_4um": _complex_record(
                    data["epsilon_sio2"][target_index]
                ),
            },
            "Si": {
                "name": SI_MATERIAL,
                "model": "installed Lumerical v261 dispersive database material",
            },
        },
        "input_sha256": {
            str(AU_TABLE.relative_to(REPOSITORY)): _sha256(AU_TABLE),
            str(TAIRTE4_TABLE.relative_to(REPOSITORY)): _sha256(TAIRTE4_TABLE),
            str(KITAMURA_IMPLEMENTATION.relative_to(REPOSITORY)): _sha256(
                KITAMURA_IMPLEMENTATION
            ),
        },
        "gates": {
            "source_band_inside_material_fit_band": bool(
                MATERIAL_FIT_WAVELENGTH_BAND_M[0]
                <= SOURCE_WAVELENGTH_BAND_M[0]
                < SOURCE_WAVELENGTH_BAND_M[1]
                <= MATERIAL_FIT_WAVELENGTH_BAND_M[1]
            ),
            "all_sampled_materials_passive": True,
            "TaIrTe4_c_equals_b": True,
            "single_frequency_constant_nk_Au_prohibited": True,
        },
    }


def add_dispersive_materials(
    fdtd: Any,
    *,
    au_max_coefficients: int = AU_MATERIAL_MAX_COEFFICIENTS,
    au_fit_tolerance: float = MATERIAL_FIT_TOLERANCE,
) -> dict[str, Any]:
    """Install the audited sampled materials into a Lumerical layout session."""

    data = sampled_material_data()
    au_fit = au_fit_configuration(
        max_coefficients=au_max_coefficients,
        tolerance=au_fit_tolerance,
    )
    for name, epsilon in (
        (AU_MATERIAL, data["epsilon_au"]),
        (SIO2_MATERIAL, data["epsilon_sio2"]),
    ):
        material = fdtd.addmaterial("Sampled data")
        fdtd.setmaterial(material, "name", name)
        fdtd.setmaterial(
            name,
            "max coefficients",
            (
                au_fit["max_coefficients"]
                if name == AU_MATERIAL
                else MATERIAL_MAX_COEFFICIENTS
            ),
        )
        fdtd.setmaterial(
            name,
            "tolerance",
            (
                au_fit["tolerance"]
                if name == AU_MATERIAL
                else MATERIAL_FIT_TOLERANCE
            ),
        )
        fdtd.setmaterial(
            name,
            "sampled data",
            np.column_stack((data["frequency_hz"], epsilon)),
        )

    material = fdtd.addmaterial("Sampled 3D data")
    fdtd.setmaterial(material, "name", TAIRTE4_MATERIAL)
    fdtd.setmaterial(TAIRTE4_MATERIAL, "anisotropy", 1)
    fdtd.setmaterial(
        TAIRTE4_MATERIAL, "max coefficients", MATERIAL_MAX_COEFFICIENTS
    )
    fdtd.setmaterial(TAIRTE4_MATERIAL, "tolerance", MATERIAL_FIT_TOLERANCE)
    fdtd.setmaterial(
        TAIRTE4_MATERIAL,
        "sampled data",
        np.column_stack(
            (
                data["frequency_hz"],
                data["epsilon_ta_x_b"],
                data["epsilon_ta_y_a"],
                data["epsilon_ta_z_c"],
            )
        ),
    )
    return material_contract_audit(
        au_max_coefficients=au_max_coefficients,
        au_fit_tolerance=au_fit_tolerance,
    )


def _add_rect(
    fdtd: Any,
    *,
    name: str,
    material: str,
    x_bounds_m: tuple[float, float],
    y_bounds_m: tuple[float, float],
    z_bounds_m: tuple[float, float],
) -> None:
    rectangle = fdtd.addrect()
    rectangle["name"] = name
    rectangle["material"] = material
    rectangle["x min"], rectangle["x max"] = x_bounds_m
    rectangle["y min"], rectangle["y max"] = y_bounds_m
    rectangle["z min"], rectangle["z max"] = z_bounds_m


def add_exact_stack_geometry(
    fdtd: Any,
    mask: np.ndarray,
    *,
    optical_x_bounds_m: tuple[float, float] = (-10.0e-6, 10.0e-6),
    optical_y_bounds_m: tuple[float, float] = (-10.0e-6, 10.0e-6),
    optical_z_min_m: float = -3.0e-6,
) -> dict[str, Any]:
    """Create the provisional Si/SiO2/TaIrTe4/exact-Au stack."""

    x_edges, y_edges = design_edges()
    z_bounds = np.asarray([0.0, CONTRACT.design_thickness_m])
    geometry_audit = exact_au_geometry_audit(
        mask,
        x_edges_m=x_edges,
        y_edges_m=y_edges,
        z_bounds_m=z_bounds,
        axis_x=CONTRACT.axis_x,
        axis_y=CONTRACT.axis_y,
    )
    substrate_x = tuple(float(item) for item in optical_x_bounds_m)
    substrate_y = tuple(float(item) for item in optical_y_bounds_m)
    flake_x = (-0.5 * CONTRACT.flake_span_x_m, 0.5 * CONTRACT.flake_span_x_m)
    flake_y = (-0.5 * CONTRACT.flake_span_y_m, 0.5 * CONTRACT.flake_span_y_m)
    _add_rect(
        fdtd,
        name="provisional_Si_substrate",
        material=SI_MATERIAL,
        x_bounds_m=substrate_x,
        y_bounds_m=substrate_y,
        z_bounds_m=(optical_z_min_m, -385.0e-9),
    )
    _add_rect(
        fdtd,
        name="provisional_285nm_SiO2",
        material=SIO2_MATERIAL,
        x_bounds_m=substrate_x,
        y_bounds_m=substrate_y,
        z_bounds_m=(-385.0e-9, -100.0e-9),
    )
    _add_rect(
        fdtd,
        name="provisional_fixed_TaIrTe4",
        material=TAIRTE4_MATERIAL,
        x_bounds_m=flake_x,
        y_bounds_m=flake_y,
        z_bounds_m=(-100.0e-9, 0.0),
    )
    rectangles = mask_rectangles(
        mask,
        x_edges_m=x_edges,
        y_edges_m=y_edges,
        z_bounds_m=z_bounds,
    )
    for index, bounds in enumerate(rectangles):
        _add_rect(
            fdtd,
            name=f"exact_Au_prism_{index:04d}",
            material=AU_MATERIAL,
            x_bounds_m=(bounds["x_min_m"], bounds["x_max_m"]),
            y_bounds_m=(bounds["y_min_m"], bounds["y_max_m"]),
            z_bounds_m=(bounds["z_min_m"], bounds["z_max_m"]),
        )
    return {
        "status": "PROVISIONAL_UNCONFIRMED_DEVICE_GEOMETRY",
        "exact_au_geometry": geometry_audit,
        "Au_rectangle_count": len(rectangles),
        "layers_z_m": {
            "Si": [optical_z_min_m, -385.0e-9],
            "SiO2": [-385.0e-9, -100.0e-9],
            "TaIrTe4": [-100.0e-9, 0.0],
            "Au": [0.0, CONTRACT.design_thickness_m],
        },
        "device_confirmation_required": True,
    }
