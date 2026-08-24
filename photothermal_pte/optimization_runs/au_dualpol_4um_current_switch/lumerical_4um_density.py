"""Shared nodal density state for Lumerical Maxwell and custom GPU PDEs.

The 8-um design window has 80x80 physical 100-nm cells and therefore 81x81
nodes.  Lumerical ``importnk2`` consumes the projected nodal field on those
physical nodes.  The custom thermal/electrical models consume the exact
four-node area average on the 80x80 cells.  The transpose implemented here is
the only authorized pullback from a cell cotangent to the nodal design state.

This module builds layout objects only.  It does not run Lumerical and does
not certify the Au relaxation on a B200.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.au_density_relaxation import (
    CONTRACT as RELAXATION_CONTRACT,
    epsilon_relaxation,
    lumerical_import_index,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.contract import (
    CONTRACT,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.lumerical_4um_exact_au import (
    add_exact_stack_geometry,
    design_edges,
    exact_control_masks,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.lumerical_maxwell_contract import (
    canonical_projected_density,
)


DENSITY_IMPORT_OBJECT = "Au_topology_nk_relaxation"


def density_nodes() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the exact physical coordinates of the shared nodal state."""

    x, y = design_edges()
    z = np.asarray([0.0, CONTRACT.design_thickness_m], dtype=np.float64)
    if (x.size, y.size) != CONTRACT.design_node_shape:
        raise RuntimeError("density-node shape does not match the device contract")
    return x, y, z


def canonical_density_nodes(projected_density: np.ndarray) -> np.ndarray:
    """Validate one 81x81 projected topology state."""

    value = canonical_projected_density(projected_density)
    if value.shape != CONTRACT.design_node_shape:
        raise ValueError(
            f"projected density shape {value.shape} != {CONTRACT.design_node_shape}"
        )
    return value


def nodal_to_cell_average(projected_density: np.ndarray) -> np.ndarray:
    """Map 81x81 nodal occupancy to exact 80x80 bilinear cell averages."""

    rho = canonical_density_nodes(projected_density)
    return 0.25 * (
        rho[:-1, :-1]
        + rho[1:, :-1]
        + rho[:-1, 1:]
        + rho[1:, 1:]
    )


def nodal_to_cell_jvp(direction: np.ndarray) -> np.ndarray:
    """Apply the linear nodal-to-cell map to an unrestricted direction."""

    value = np.asarray(direction, dtype=np.float64)
    if value.shape != CONTRACT.design_node_shape or not np.all(np.isfinite(value)):
        raise ValueError("nodal direction has the wrong shape or non-finite values")
    return 0.25 * (
        value[:-1, :-1]
        + value[1:, :-1]
        + value[:-1, 1:]
        + value[1:, 1:]
    )


def nodal_to_cell_vjp(cell_cotangent: np.ndarray) -> np.ndarray:
    """Exact discrete transpose of :func:`nodal_to_cell_jvp`."""

    cotangent = np.asarray(cell_cotangent, dtype=np.float64)
    if cotangent.shape != CONTRACT.design_shape or not np.all(np.isfinite(cotangent)):
        raise ValueError("cell cotangent has the wrong shape or non-finite values")
    result = np.zeros(CONTRACT.design_node_shape, dtype=np.float64)
    result[:-1, :-1] += 0.25 * cotangent
    result[1:, :-1] += 0.25 * cotangent
    result[:-1, 1:] += 0.25 * cotangent
    result[1:, 1:] += 0.25 * cotangent
    return result


def density_state_sha256(projected_density: np.ndarray) -> str:
    """Hash nodal values, physical coordinates, axes, and constitutive law."""

    rho = canonical_density_nodes(projected_density)
    x, y, z = density_nodes()
    metadata = {
        "schema": "au-projected-nodal-density-v1",
        "density_shape": list(rho.shape),
        "density_dtype": "float64",
        "coordinate_dtype": "float64",
        "coordinate_units": "m",
        "axis_mapping": {"x": CONTRACT.axis_x, "y": CONTRACT.axis_y},
        "optical_law": RELAXATION_CONTRACT.law,
        "optical_rho_power": RELAXATION_CONTRACT.optical_rho_power,
        "cell_map": "arithmetic_mean_of_four_corner_nodes",
    }
    digest = hashlib.sha256()
    digest.update(b"au-projected-nodal-density-v1\0")
    digest.update(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("ascii")
    )
    for label, value in (("rho", rho), ("x", x), ("y", y), ("z", z)):
        digest.update(b"\0" + label.encode("ascii") + b"\0")
        digest.update(np.ascontiguousarray(value).tobytes(order="C"))
    return digest.hexdigest()


def density_state_audit(projected_density: np.ndarray) -> dict[str, Any]:
    """Return the cross-solver state contract without embedding full arrays."""

    rho = canonical_density_nodes(projected_density)
    cells = nodal_to_cell_average(rho)
    x, y, z = density_nodes()
    epsilon = epsilon_relaxation(rho)
    return {
        "schema": "au-projected-nodal-density-v1",
        "density_state_sha256": density_state_sha256(rho),
        "nodal_shape_xy": list(rho.shape),
        "pde_cell_shape_xy": list(cells.shape),
        "nodal_range": [float(np.min(rho)), float(np.max(rho))],
        "cell_range": [float(np.min(cells)), float(np.max(cells))],
        "x_bounds_m": [float(x[0]), float(x[-1])],
        "y_bounds_m": [float(y[0]), float(y[-1])],
        "z_bounds_m": [float(z[0]), float(z[-1])],
        "axis_mapping": {"x": CONTRACT.axis_x, "y": CONTRACT.axis_y},
        "optical_law": RELAXATION_CONTRACT.law,
        "optical_rho_power": RELAXATION_CONTRACT.optical_rho_power,
        "minimum_epsilon_imaginary": float(np.min(epsilon.imag)),
        "all_constitutive_maps_derive_from_this_nodal_state": True,
        "gray_state_claimed_as_fabricated_material": False,
    }


def add_density_stack_geometry(
    fdtd: Any,
    projected_density: np.ndarray,
    *,
    optical_x_bounds_m: tuple[float, float] = (-10.0e-6, 10.0e-6),
    optical_y_bounds_m: tuple[float, float] = (-10.0e-6, 10.0e-6),
    optical_z_min_m: float = -3.0e-6,
) -> dict[str, Any]:
    """Build the fixed stack and one nonlinear-nk import object in layout."""

    rho = canonical_density_nodes(projected_density)
    fixed_stack = add_exact_stack_geometry(
        fdtd,
        exact_control_masks()["empty"],
        optical_x_bounds_m=optical_x_bounds_m,
        optical_y_bounds_m=optical_y_bounds_m,
        optical_z_min_m=optical_z_min_m,
    )
    x, y, z = density_nodes()
    index = lumerical_import_index(rho, z_samples=z.size)
    fdtd.addimport({"name": DENSITY_IMPORT_OBJECT, "x": 0.0, "y": 0.0, "z": 0.0})
    result = fdtd.importnk2(index, x, y, z)
    if result is not None and int(result) != 1:
        raise RuntimeError("Lumerical importnk2 returned failure")
    return {
        "status": "AUDITED_DENSITY_LAYOUT_NOT_B200_VALIDATED",
        "density_state": density_state_audit(rho),
        "import_object": DENSITY_IMPORT_OBJECT,
        "import_index_shape_xyz": list(index.shape),
        "fixed_stack": fixed_stack,
        "Maxwell_solve_run": False,
        "remaining_gates": [
            "Lumerical 4-um endpoint response parity",
            "uniform-density field/Q resonance sweep",
            "component-Yee Jacobian FD and transpose",
            "complete latent-variable AD-FD",
        ],
    }
