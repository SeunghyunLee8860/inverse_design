#!/usr/bin/env python3
"""Fail-closed realized-grid audit for the 4 um dual-polarization model."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.contract import (
    CONTRACT,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_4um_model import (
    LAYOUT,
    MAX_IGNORED_SUBSTRATE_EPSILON_IMAG,
    build_model,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.material_fraction import (
    audit as material_fraction_audit,
)


HERE = Path(__file__).resolve().parent
OUT = HERE / "results_fdtdx_4um_runsetup"


def _bounds(grid, grid_slice: tuple[slice, slice, slice]) -> list[list[float]]:
    result = []
    for axis, part in enumerate(grid_slice):
        edges = np.asarray(grid.edges(axis), dtype=float)
        result.append([float(edges[part.start]), float(edges[part.stop])])
    return result


def _nbytes(tree, jax) -> int:
    total = 0
    for leaf in jax.tree_util.tree_leaves(tree):
        if hasattr(leaf, "size") and hasattr(leaf, "dtype"):
            total += int(leaf.size) * int(np.dtype(leaf.dtype).itemsize)
    return total


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    models = {
        polarization: build_model(polarization, include_adjoint_source=True)
        for polarization in ("Ea", "Eb")
    }
    reference = models["Ea"]
    grid = reference["grid"]
    x_edges = np.asarray(grid.edges(0), dtype=float)
    y_edges = np.asarray(grid.edges(1), dtype=float)
    z_edges = np.asarray(grid.edges(2), dtype=float)
    if not np.array_equal(x_edges, np.asarray(models["Eb"]["grid"].edges(0))):
        raise RuntimeError("polarization changed the realized grid")
    if reference["placement"] != models["Eb"]["placement"]:
        raise RuntimeError("polarization changed object placement")
    if not np.allclose(
        np.asarray(reference["placed"]["gaussian_source"].fixed_E_polarization_vector),
        np.asarray((0.0, 1.0, 0.0)),
    ):
        raise RuntimeError("Ea is not Lumerical y=a")
    if not np.allclose(
        np.asarray(models["Eb"]["placed"]["gaussian_source"].fixed_E_polarization_vector),
        np.asarray((1.0, 0.0, 0.0)),
    ):
        raise RuntimeError("Eb is not Lumerical x=b")

    placement = {
        name: {
            "slice": [list(pair) for pair in reference["placement"][name]],
            "bounds_m_xyz": _bounds(grid, value),
        }
        for name, value in reference["slices"].items()
    }
    flake_bounds = placement["fixed_tairte4"]["bounds_m_xyz"]
    au_bounds = placement["au_design"]["bounds_m_xyz"]
    source_bounds = placement["gaussian_source"]["bounds_m_xyz"]
    pml_inner = [
        [float(x_edges[LAYOUT.pml_cells]), float(x_edges[-LAYOUT.pml_cells - 1])],
        [float(y_edges[LAYOUT.pml_cells]), float(y_edges[-LAYOUT.pml_cells - 1])],
        [float(z_edges[LAYOUT.pml_cells]), float(z_edges[-LAYOUT.pml_cells - 1])],
    ]
    lateral_gap = pml_inner[0][1] - flake_bounds[0][1]
    cells = int(np.prod(grid.shape))
    base_bytes = _nbytes(reference["base"], reference["jax"])
    adjoint_profile_bytes = int(
        3
        * LAYOUT.flake_xy_cells
        * LAYOUT.flake_xy_cells
        * (LAYOUT.sio2_cells + LAYOUT.tairte4_cells + LAYOUT.au_cells)
        * np.dtype(np.complex64).itemsize
    )
    source_half = 0.5 * CONTRACT.source_aperture_span_m
    captured = math.erf(math.sqrt(2.0) * source_half / CONTRACT.gaussian_waist_m) ** 2
    realized_tairte4_thickness = flake_bounds[2][1] - flake_bounds[2][0]
    realized_au_thickness = au_bounds[2][1] - au_bounds[2][0]
    coordinate_tolerance_m = 5e-13
    audit = {
        "status": "AUDITED_FDTDX_4UM_DUALPOL_RUNSETUP_NOT_YET_SOLVED",
        "scope": "realized placement/grid/material-ADE audit only; no Maxwell solve",
        "axis_mapping": {"x": "b", "y": "a", "z": "c=b closure"},
        "polarization_vectors": {
            "Ea": [0.0, 1.0, 0.0],
            "Eb": [1.0, 0.0, 0.0],
        },
        "grid": {
            "shape_xyz": list(grid.shape),
            "cell_count": cells,
            "bounds_m_xyz": [
                [float(x_edges[0]), float(x_edges[-1])],
                [float(y_edges[0]), float(y_edges[-1])],
                [float(z_edges[0]), float(z_edges[-1])],
            ],
            "min_spacing_m_xyz": [
                float(np.min(np.diff(x_edges))),
                float(np.min(np.diff(y_edges))),
                float(np.min(np.diff(z_edges))),
            ],
            "max_spacing_m_xyz": [
                float(np.max(np.diff(x_edges))),
                float(np.max(np.diff(y_edges))),
                float(np.max(np.diff(z_edges))),
            ],
            "pml_cells_each_face": LAYOUT.pml_cells,
            "pml_inner_bounds_m_xyz": pml_inner,
            "flake_to_lateral_pml_gap_m": float(lateral_gap),
        },
        "placement": placement,
        "layer_thickness_audit": {
            "coordinate_tolerance_m": coordinate_tolerance_m,
            "requested_tairte4_m": 100e-9,
            "realized_tairte4_m": realized_tairte4_thickness,
            "tairte4_error_m": realized_tairte4_thickness - 100e-9,
            "requested_au_m": 50e-9,
            "realized_au_m": realized_au_thickness,
            "au_error_m": realized_au_thickness - 50e-9,
        },
        "source": {
            "wavelength_m": CONTRACT.wavelength_m,
            "requested_waist_m": CONTRACT.gaussian_waist_m,
            "aperture_span_m": CONTRACT.source_aperture_span_m,
            "aperture_boundary_intensity_over_peak": CONTRACT.aperture_boundary_intensity_fraction,
            "infinite_gaussian_square_captured_power_fraction": captured,
            "normal_incidence": True,
            "direction": "-z",
            "source_bounds_m_xyz": source_bounds,
        },
        "materials": {
            "epsilon_Au": [reference["epsilon"]["au"].real, reference["epsilon"]["au"].imag],
            "epsilon_TaIrTe4": {
                axis: [value.real, value.imag]
                for axis, value in reference["epsilon"]["tairte4"].items()
            },
            "epsilon_SiO2": [reference["epsilon"]["sio2"].real, reference["epsilon"]["sio2"].imag],
            "epsilon_Si": [reference["epsilon"]["silicon"].real, reference["epsilon"]["silicon"].imag],
            "ADE_fit_relative_error": {
                name: float(value["fit_relative_error"])
                for name, value in reference["fits"].items()
            },
            "au_material_fraction": material_fraction_audit(),
            "substrate_implementation": (
                "lossless uniform real epsilon; fail if epsilon.imag exceeds tolerance"
            ),
            "maximum_ignored_substrate_epsilon_imag": MAX_IGNORED_SUBSTRATE_EPSILON_IMAG,
        },
        "memory": {
            "placed_base_array_bytes": base_bytes,
            "adjoint_complex_profile_bytes": adjoint_profile_bytes,
            "sum_base_plus_adjoint_GiB": (base_bytes + adjoint_profile_bytes) / 2**30,
            "note": "lower-bound resident arrays; XLA workspace and detector outputs are additional",
        },
        "numerics": {
            "total_periods": 16,
            "phasor_window_periods": 4,
            "time_steps_total": int(reference["config"].time_steps_total),
            "time_step_s": float(reference["config"].time_step_duration),
            "backend": "gpu",
            "gradient_config": None,
            "production_gradient_method": "checkpoint-free harmonic forward + reciprocal adjoint two-solve",
        },
        "gates": {
            "same_geometry_for_Ea_Eb": True,
            "same_grid_for_Ea_Eb": True,
            "flake_inside_non_PML_with_positive_gap": lateral_gap > 0.0,
            "Au_inside_flake": all(
                au_bounds[axis][0] >= flake_bounds[axis][0]
                and au_bounds[axis][1] <= flake_bounds[axis][1]
                for axis in (0, 1)
            ),
            "exact_layer_thicknesses": (
                math.isclose(
                    realized_tairte4_thickness,
                    100e-9,
                    rel_tol=0.0,
                    abs_tol=coordinate_tolerance_m,
                )
                and math.isclose(
                    realized_au_thickness,
                    50e-9,
                    rel_tol=0.0,
                    abs_tol=coordinate_tolerance_m,
                )
            ),
            "source_boundary_intensity_lt_0p05pct": CONTRACT.aperture_boundary_intensity_fraction < 5e-4,
            "all_ADE_fit_errors_lt_1e-12": all(
                value["fit_relative_error"] < 1e-12
                for value in reference["fits"].values()
            ),
            "substrate_loss_below_implemented_tolerance": (
                0.0 <= reference["epsilon"]["sio2"].imag
                <= MAX_IGNORED_SUBSTRATE_EPSILON_IMAG
                and 0.0 <= reference["epsilon"]["silicon"].imag
                <= MAX_IGNORED_SUBSTRATE_EPSILON_IMAG
            ),
        },
    }
    if not all(audit["gates"].values()):
        audit["status"] = "FAILED_FDTDX_4UM_DUALPOL_RUNSETUP"
    (OUT / "fdtdx_4um_dualpol_runsetup.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )

    um = 1e-6
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.3), constrained_layout=True)
    ax = axes[0]
    ax.add_patch(Rectangle((-10, -10), 20, 20, color="#eaf4ff", ec="#673ab7", lw=3))
    ax.add_patch(Rectangle((-10, -10), 1, 20, color="#6a1b9a", alpha=0.55))
    ax.add_patch(Rectangle((9, -10), 1, 20, color="#6a1b9a", alpha=0.55))
    ax.add_patch(Rectangle((-9, -10), 18, 1, color="#6a1b9a", alpha=0.55))
    ax.add_patch(Rectangle((-9, 9), 18, 1, color="#6a1b9a", alpha=0.55))
    ax.add_patch(Rectangle((-8, -8), 16, 16, color="#e47b6b", alpha=0.60, label="TaIrTe4"))
    ax.add_patch(Rectangle((-4, -4), 8, 8, color="#f6bd32", alpha=0.85, label="Au design"))
    ax.add_patch(Rectangle((-8, -8), 16, 16, fill=False, ec="#1976d2", ls="--", lw=2, label="Gaussian aperture"))
    ax.set(xlim=(-10, 10), ylim=(-10, 10), aspect="equal", xlabel="x=b (um)", ylabel="y=a (um)", title="xy: finite flake, centered Au, six PML")
    ax.legend(fontsize=8)

    for ax, label in ((axes[1], "xz"), (axes[2], "yz")):
        ax.add_patch(Rectangle((-10, -3), 20, 6, color="#eaf4ff"))
        ax.add_patch(Rectangle((-10, -3), 20, 1.6, color="#5e3c99", alpha=0.45, label="z-PML"))
        ax.add_patch(Rectangle((-10, 1.4), 20, 1.6, color="#5e3c99", alpha=0.45))
        ax.add_patch(Rectangle((-10, -1.4), 20, 1.015, color="#6f8496", label="Si"))
        ax.add_patch(Rectangle((-10, -0.385), 20, 0.285, color="#8fd3d5", label="SiO2"))
        ax.add_patch(Rectangle((-8, -0.1), 16, 0.1, color="#e47b6b", label="TaIrTe4"))
        ax.add_patch(Rectangle((-4, 0), 8, 0.05, color="#f6bd32", label="Au design"))
        ax.annotate("Gaussian -z", (0, 0.1), (0, 1.0), ha="center", color="#1976d2", arrowprops=dict(arrowstyle="->", lw=2, color="#1976d2"))
        ax.set(xlim=(-10, 10), ylim=(-3, 3), xlabel=f"{label[0]} (um)", ylabel="z (um)", title=f"{label}: source/layers/PML")
    axes[1].legend(fontsize=7, loc="lower right")
    fig.suptitle("4 um dual-polarization FDTDX runsetup (no field solve yet)")
    fig.savefig(OUT / "FDTDX_4UM_DUALPOL_RUNSETUP.png", dpi=180)
    plt.close(fig)

    report = f"""# FDTDX 4 um dual-polarization runsetup audit

Status: **{audit['status']}**

This checkpoint placed the exact Ea and Eb models but did not run Maxwell,
thermal, electrical, adjoint, or optimization solves.  Both polarizations use
the same {grid.shape[0]} x {grid.shape[1]} x {grid.shape[2]} realized grid;
only the source vector changes (Ea=y, Eb=x).  The finite 16 um TaIrTe4 flake
has a {1e6*lateral_gap:.3f} um air gap to each lateral PML and the centered Au
design is 8 x 8 x 0.05 um.

The source is a normally incident scalar Gaussian with w0=4 um and a 16 um
square support.  Its requested intensity at the aperture boundary is
{100*CONTRACT.aperture_boundary_intensity_fraction:.5f}% of the peak.

The production optical-gradient implementation is the checkpoint-free harmonic
two-solve method. Its historical O3 result is not a validation of the current
shared-linear Au law. This audit does not promote the 4 um combined PTE
gradient; full-mesh convergence and a new hash-linked AD-FD certificate still
have to pass.
"""
    (OUT / "FDTDX_4UM_DUALPOL_RUNSETUP.md").write_text(report, encoding="utf-8")
    print(json.dumps(audit, indent=2), flush=True)
    return 0 if audit["status"].startswith("AUDITED_") else 2


if __name__ == "__main__":
    raise SystemExit(main())
