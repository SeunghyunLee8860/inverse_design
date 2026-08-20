#!/usr/bin/env python3
"""Finite six-PML scalar/imported Au representation control at 10 um.

This is an isolated material-representation control, not a fixed-flake device
prediction.  It reuses the certified 10-um, w0=8.5-um source wrapper and the
native-Yee Q/flux closure machinery without modifying the legacy module.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]
LEGACY = HERE.parent / "legacy_v261_optical_support" / "run_complex_material_control.py"
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from material_model import nonlinear_index_path  # noqa: E402


AU_BOUNDS = {
    "x": (-10.0e-6, 10.0e-6),
    "y": (-10.0e-6, 10.0e-6),
    "z": (0.05e-6, 0.10e-6),
}
FLUX_BOUNDS = {
    "x": (-10.5e-6, 10.5e-6),
    "y": (-10.5e-6, 10.5e-6),
    "z": (-0.45e-6, 0.60e-6),
}
AU_DZ_M = 5e-9
AU_DXY_M = 100e-9


def load_legacy():
    spec = importlib.util.spec_from_file_location("au_binary_control_base", LEGACY)
    if spec is None or spec.loader is None:
        raise ImportError(LEGACY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def au_complex_index(rho: float) -> tuple[complex, complex]:
    path = nonlinear_index_path(np.asarray([float(rho)]))
    epsilon = complex(path.epsilon[0])
    index = complex(np.sqrt(epsilon))
    if index.imag < 0.0:
        index = -index
    return epsilon, index


def imported_nodes() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nx = int(round((AU_BOUNDS["x"][1] - AU_BOUNDS["x"][0]) / AU_DXY_M)) + 1
    ny = int(round((AU_BOUNDS["y"][1] - AU_BOUNDS["y"][0]) / AU_DXY_M)) + 1
    nz = int(round((AU_BOUNDS["z"][1] - AU_BOUNDS["z"][0]) / AU_DZ_M)) + 1
    return (
        np.linspace(*AU_BOUNDS["x"], nx),
        np.linspace(*AU_BOUNDS["y"], ny),
        np.linspace(*AU_BOUNDS["z"], nz),
    )


def add_local_mesh(fdtd) -> None:
    mesh = fdtd.addmesh()
    mesh["name"] = "au_50nm_local_mesh"
    mesh["x min"], mesh["x max"] = FLUX_BOUNDS["x"]
    mesh["y min"], mesh["y max"] = FLUX_BOUNDS["y"]
    mesh["z min"], mesh["z max"] = (0.0, 0.15e-6)
    mesh["override x mesh"] = True
    mesh["override y mesh"] = True
    mesh["override z mesh"] = True
    mesh["dx"] = AU_DXY_M
    mesh["dy"] = AU_DXY_M
    mesh["dz"] = AU_DZ_M


def component_epsilon_readback(base, fdtd, q: dict[str, object]) -> dict[str, object]:
    """Read component epsilon inside a 50-nm film without a 100-nm margin."""

    spatial_shape = tuple(
        np.asarray(q["base_coordinates"][axis]).size for axis in "xyz"
    )
    result = {}
    for component in "xyz":
        index = base.frequency_slice(
            np.asarray(fdtd.getdata(base.PABS_INDEX, f"index_{component}", 1)),
            spatial_shape,
            int(q["frequency_index_zero_based"]),
            int(q["frequency_count"]),
            f"index_{component}",
        )
        epsilon = index**2
        finite = np.isfinite(epsilon)
        x = np.asarray(q["base_coordinates"]["x"], float)
        y = np.asarray(q["base_coordinates"]["y"], float)
        z = np.asarray(q["base_coordinates"]["z"], float)
        interior = (
            (x[:, None, None] >= AU_BOUNDS["x"][0] + 0.2e-6)
            & (x[:, None, None] <= AU_BOUNDS["x"][1] - 0.2e-6)
            & (y[None, :, None] >= AU_BOUNDS["y"][0] + 0.2e-6)
            & (y[None, :, None] <= AU_BOUNDS["y"][1] - 0.2e-6)
            & (z[None, None, :] >= AU_BOUNDS["z"][0] + 10e-9)
            & (z[None, None, :] <= AU_BOUNDS["z"][1] - 10e-9)
            & finite
        )
        interior_values = epsilon[interior]
        if interior_values.size == 0:
            raise RuntimeError(f"empty 50-nm Au interior epsilon readback for {component}")
        all_values = epsilon[finite]
        result[component] = {
            "shape": list(epsilon.shape),
            "epsilon_interior_median": [
                float(np.median(interior_values.real)),
                float(np.median(interior_values.imag)),
            ],
            "interior_sample_count": int(interior_values.size),
            "epsilon_real_range": [float(np.min(all_values.real)), float(np.max(all_values.real))],
            "epsilon_imag_range": [float(np.min(all_values.imag)), float(np.max(all_values.imag))],
            "all_finite": bool(np.all(finite)),
            "z_interior_margin_m": 10e-9,
        }
    return result


def configure(base) -> None:
    base.BLOCK_BOUNDS = AU_BOUNDS
    base.FLUX_BOUNDS = FLUX_BOUNDS
    base.complex_index = au_complex_index
    base.imported_nodes = imported_nodes
    base.add_local_mesh = add_local_mesh
    base.component_epsilon_readback = lambda fdtd, q: component_epsilon_readback(
        base, fdtd, q
    )


def main() -> int:
    global AU_DZ_M
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--au-dz-nm", type=float, default=5.0)
    parsed, remaining = parser.parse_known_args()
    if parsed.au_dz_nm <= 0.0:
        raise ValueError("--au-dz-nm must be positive")
    AU_DZ_M = parsed.au_dz_nm * 1e-9
    sys.argv = [sys.argv[0], *remaining]
    base = load_legacy()
    configure(base)
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
