#!/usr/bin/env python3
"""Publish field maps for the immutable Run 005 final-binary evaluation.

This is a read-only postprocessor.  It does not run Maxwell, thermal, adjoint,
or optimization.  The full terminal-current scalar is reintegrated from the
stored 3D temperature with the exact production coefficients.  Spatial
gradient/current maps use the separately requested strict centered stencil:
all four in-plane neighbours must be inside the TaIrTe4 footprint, otherwise
the displayed value is NaN.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap, TwoSlopeNorm  # noqa: E402
import numpy as np  # noqa: E402


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]
RUN002 = HERE.parent / "run_002_gaussian10_w8p5_current_max"
for path in (REPOSITORY, RUN002):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from map_production_q_to_thermal_grid import material_masks  # noqa: E402


RAW = Path(
    "/data/seunghyun/tairte4/raw_artifacts/"
    "run005_lowbeta_topology_pilot_20260808/"
    "final_binary_g046_b2048/solver_evaluation/"
    "thresholded_binary_evaluation.npz"
)
RESULT = RAW.with_name("thresholded_binary_evaluation_result.json")
DERIVED_ROOT = Path(
    "/home/seunghyun/tairte4/raw_artifacts/"
    "run005_final_binary_derived_fields_20260808"
)
DERIVED = DERIVED_ROOT / "final_binary_derived_fields.npz"
SUMMARY = HERE / "results/final_binary_fields_summary.json"
CSV = HERE / "results/final_binary_field_metrics.csv"
REPORT = HERE / "results/FINAL_BINARY_FIELDS_REPORT.md"
STRUCTURE_PLOT = HERE / "plots/final_binary_structure_1_material_0_void.png"
ALL_FIELDS_PLOT = HERE / "plots/final_binary_Q_temperature_gradient_current.png"
CROSS_SECTION_PLOT = HERE / "plots/final_binary_Q_temperature_cross_sections.png"
CURRENT_PLOT = HERE / "plots/final_binary_pte_current_breakdown.png"
MANIFEST = HERE / "manifests/RAW_ARTIFACT_MANIFEST.json"

SIGMA_A_S_M = 4.91e5
SIGMA_B_S_M = 1.10e5
SEEBECK_A_V_K = -6.0e-6
SEEBECK_B_V_K = 27.0e-6
WEIGHTING_X_M_INV = 1.0 / 64.0e-6
WEIGHTING_Y_M_INV = 1.0 / 64.0e-6
DESIGN_HALF_SPAN_M = 9.3e-6
FLAKE_HALF_SPAN_M = 16.0e-6


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, object]:
    value = path.resolve()
    return {
        "path": str(value),
        "size_bytes": value.stat().st_size,
        "sha256": sha256(value),
    }


def centers(edges: np.ndarray) -> np.ndarray:
    return 0.5 * (edges[:-1] + edges[1:])


def signed_norm(field: np.ndarray) -> TwoSlopeNorm:
    limit = float(np.nanmax(np.abs(field)))
    if not np.isfinite(limit) or limit == 0.0:
        limit = 1.0
    return TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)


def add_design_outline(axis: plt.Axes) -> None:
    low = -DESIGN_HALF_SPAN_M * 1e6
    span = 2.0 * DESIGN_HALF_SPAN_M * 1e6
    axis.add_patch(
        plt.Rectangle(
            (low, low), span, span, fill=False, edgecolor="cyan",
            linewidth=0.8, linestyle="--",
        )
    )


def plot_structure(rho: np.ndarray) -> None:
    coordinate_um = np.linspace(-9.3, 9.3, rho.shape[0])
    z_um = np.linspace(0.0, 1.0, 101)
    cmap = ListedColormap(["#111111", "#f2c94c"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), constrained_layout=True)
    images = []
    images.append(
        axes[0].imshow(
            rho.T, origin="lower", extent=[-9.3, 9.3, -9.3, 9.3],
            interpolation="nearest", cmap=cmap, norm=norm, aspect="equal",
        )
    )
    axes[0].set_title("Top view: binary design")
    axes[0].set_xlabel("solver x (um)")
    axes[0].set_ylabel("solver y (um)")
    mid = rho.shape[0] // 2
    xz = np.repeat(rho[:, mid, None], z_um.size, axis=1)
    yz = np.repeat(rho[mid, :, None], z_um.size, axis=1)
    for axis, field, coordinate, title, label in (
        (axes[1], xz, coordinate_um, "x-z at solver y=0", "solver x (um)"),
        (axes[2], yz, coordinate_um, "y-z at solver x=0", "solver y (um)"),
    ):
        image = axis.imshow(
            field.T, origin="lower",
            extent=[coordinate[0], coordinate[-1], z_um[0], z_um[-1]],
            interpolation="nearest", cmap=cmap, norm=norm, aspect="auto",
        )
        images.append(image)
        axis.set_title(title)
        axis.set_xlabel(label)
        axis.set_ylabel("design z (um)")
    bar = fig.colorbar(images[0], ax=axes, ticks=[0, 1], shrink=0.88)
    bar.ax.set_yticklabels(["0 = air / void", "1 = SiO2 material"])
    fig.suptitle("Run 005 final exact-binary structure (material=1, void=0)")
    fig.savefig(STRUCTURE_PLOT, dpi=200)
    plt.close(fig)


def main() -> int:
    published = json.loads((HERE / "results/optimization_summary.json").read_text())
    expected = published["final_binary_NPZ"]
    if sha256(RAW) != expected["sha256"]:
        raise RuntimeError("final binary NPZ SHA mismatch")
    result = json.loads(RESULT.read_text())
    if not result.get("passed"):
        raise RuntimeError("final binary physics result is not passed")
    data = np.load(RAW)
    rho = np.asarray(data["rho_binary"], np.uint8)
    if set(np.unique(rho).tolist()) != {0, 1}:
        raise RuntimeError("density is not exactly {0,1}")

    edges = tuple(np.asarray(data[f"{axis}_edges_m"], float) for axis in "xyz")
    x, y, z = tuple(centers(edge) for edge in edges)
    dx, dy, dz = tuple(np.diff(edge) for edge in edges)
    q = np.asarray(data["Q_total_W_m3"], float)
    temperature = np.asarray(data["thermal_temperature_grid_K"], float)
    if q.shape != temperature.shape:
        raise RuntimeError("Q/temperature grids differ")
    masks = material_masks(edges, design_half_span_m=DESIGN_HALF_SPAN_M)
    flake = masks["physical_TaIrTe4"]
    design = masks["design_effective_SiO2"]
    volume = dx[:, None, None] * dy[None, :, None] * dz[None, None, :]
    mapped_power = float(np.sum(q * volume))
    expected_mapped = float(result["base_mapping"]["mapped_power_W"])
    mapping_reintegration_error = abs(mapped_power - expected_mapped) / max(
        abs(expected_mapped), np.finfo(float).tiny
    )
    if mapping_reintegration_error >= 1.0e-12:
        raise RuntimeError("stored thermal Q does not reintegrate")

    fx = np.flatnonzero(np.any(flake, axis=(1, 2)))
    fy = np.flatnonzero(np.any(flake, axis=(0, 2)))
    fz = np.flatnonzero(np.any(flake, axis=(0, 1)))
    tf = temperature[np.ix_(fx, fy, fz)]
    qf = q[np.ix_(fx, fy, fz)]
    flake_dz = dz[fz]
    thickness_m = float(np.sum(flake_dz))
    temperature_flake_average = np.sum(
        tf * flake_dz[None, None, :], axis=2
    ) / thickness_m

    # Exact production integration: three-point centered in the interior and
    # second-order one-sided at the finite flake perimeter, for each z cell.
    dtx_full = np.gradient(tf, x[fx], axis=0, edge_order=2)
    dty_full = np.gradient(tf, y[fy], axis=1, edge_order=2)
    local_x_A_m3 = -WEIGHTING_X_M_INV * SIGMA_A_S_M * SEEBECK_A_V_K * dtx_full
    local_y_A_m3 = -WEIGHTING_Y_M_INV * SIGMA_B_S_M * SEEBECK_B_V_K * dty_full
    local_total_A_m3 = local_x_A_m3 + local_y_A_m3
    flake_volume = (
        dx[fx, None, None] * dy[None, fy, None] * dz[None, None, fz]
    )
    current_x_A = float(np.sum(local_x_A_m3 * flake_volume))
    current_y_A = float(np.sum(local_y_A_m3 * flake_volume))
    current_total_A = float(np.sum(local_total_A_m3 * flake_volume))
    stored_current_A = float(result["objective_A"])
    objective_reintegration_error = abs(current_total_A - stored_current_A) / max(
        abs(stored_current_A), np.finfo(float).tiny
    )
    if objective_reintegration_error >= 1.0e-12:
        raise RuntimeError("PTE objective reintegration failed")

    # Strict display contract: both +/- x and +/- y neighbours are required.
    strict = np.zeros(temperature_flake_average.shape, bool)
    strict[1:-1, 1:-1] = True
    dtx_strict = np.full_like(temperature_flake_average, np.nan)
    dty_strict = np.full_like(temperature_flake_average, np.nan)
    dtx_strict[1:-1, 1:-1] = (
        temperature_flake_average[2:, 1:-1]
        - temperature_flake_average[:-2, 1:-1]
    ) / (x[fx][2:, None] - x[fx][:-2, None])
    dty_strict[1:-1, 1:-1] = (
        temperature_flake_average[1:-1, 2:]
        - temperature_flake_average[1:-1, :-2]
    ) / (y[fy][None, 2:] - y[fy][None, :-2])
    gradient_strict = np.hypot(dtx_strict, dty_strict)
    current_x_sheet = (
        -WEIGHTING_X_M_INV * SIGMA_A_S_M * SEEBECK_A_V_K
        * dtx_strict * thickness_m
    )
    current_y_sheet = (
        -WEIGHTING_Y_M_INV * SIGMA_B_S_M * SEEBECK_B_V_K
        * dty_strict * thickness_m
    )
    current_sheet = current_x_sheet + current_y_sheet
    strict_current_A = float(np.nansum(
        current_sheet * dx[fx, None] * dy[None, fy]
    ))
    if not (
        np.all(np.isnan(dtx_strict[~strict]))
        and np.all(np.isnan(dty_strict[~strict]))
        and np.all(np.isnan(current_sheet[~strict]))
    ):
        raise RuntimeError("strict neighbor mask was not applied")

    qxy_total = np.sum(q * dz[None, None, :], axis=2)
    qxy_flake = np.sum(q * flake * dz[None, None, :], axis=2)
    qxy_design = np.sum(q * design * dz[None, None, :], axis=2)
    qxy_flake_local = np.sum(qf * flake_dz[None, None, :], axis=2)
    tmax_xy = np.nanmax(temperature, axis=2)

    DERIVED_ROOT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        DERIVED,
        rho_material_1_void_0=rho,
        design_x_m=np.linspace(-DESIGN_HALF_SPAN_M, DESIGN_HALF_SPAN_M, rho.shape[0]),
        design_y_m=np.linspace(-DESIGN_HALF_SPAN_M, DESIGN_HALF_SPAN_M, rho.shape[1]),
        thermal_x_m=x,
        thermal_y_m=y,
        thermal_z_m=z,
        Q_depth_integrated_total_W_m2=qxy_total,
        Q_depth_integrated_TaIrTe4_W_m2=qxy_flake,
        Q_depth_integrated_design_W_m2=qxy_design,
        flake_x_m=x[fx],
        flake_y_m=y[fy],
        flake_temperature_average_rise_K=temperature_flake_average,
        strict_dTdx_K_m=dtx_strict,
        strict_dTdy_K_m=dty_strict,
        strict_gradient_magnitude_K_m=gradient_strict,
        strict_current_x_contribution_A_m2=current_x_sheet,
        strict_current_y_contribution_A_m2=current_y_sheet,
        strict_current_total_contribution_A_m2=current_sheet,
        strict_valid_mask=strict,
    )

    plot_structure(rho)

    # Consolidated top-view figure.
    fig, axes = plt.subplots(2, 4, figsize=(21, 10.5), constrained_layout=True)
    structure_image = axes[0, 0].imshow(
        rho.T, origin="lower", extent=[-9.3, 9.3, -9.3, 9.3],
        interpolation="nearest",
        cmap=ListedColormap(["#111111", "#f2c94c"]),
        norm=BoundaryNorm([-0.5, 0.5, 1.5], 2), aspect="equal",
    )
    structure_bar = fig.colorbar(structure_image, ax=axes[0, 0], ticks=[0, 1])
    structure_bar.ax.set_yticklabels(["0 void", "1 SiO2"])
    axes[0, 0].set_title("final structure")

    panels = (
        (axes[0, 1], x * 1e6, y * 1e6, qxy_total, "total depth-integrated Q", "W/m2", "inferno", None),
        (axes[0, 2], x * 1e6, y * 1e6, qxy_flake, "TaIrTe4 depth-integrated Q", "W/m2", "inferno", None),
        (axes[0, 3], x[fx] * 1e6, y[fy] * 1e6, temperature_flake_average, "TaIrTe4 thickness-avg DeltaT", "K", "magma", None),
        (axes[1, 0], x[fx] * 1e6, y[fy] * 1e6, dtx_strict, "strict centered dT/dx", "K/m", "coolwarm", signed_norm(dtx_strict)),
        (axes[1, 1], x[fx] * 1e6, y[fy] * 1e6, dty_strict, "strict centered dT/dy", "K/m", "coolwarm", signed_norm(dty_strict)),
        (axes[1, 2], x[fx] * 1e6, y[fy] * 1e6, gradient_strict, "strict |grad T|", "K/m", "viridis", None),
        (axes[1, 3], x[fx] * 1e6, y[fy] * 1e6, current_sheet, "strict local PTE contribution", "A/m2", "coolwarm", signed_norm(current_sheet)),
    )
    for axis, px, py, field, title, unit, cmap, norm in panels:
        image = axis.pcolormesh(px, py, np.ma.masked_invalid(field).T, shading="nearest", cmap=cmap, norm=norm)
        axis.set_title(title)
        axis.set_aspect("equal")
        fig.colorbar(image, ax=axis, label=unit)
        if axis in (axes[0, 1], axes[0, 2]):
            add_design_outline(axis)
    for axis in axes.flat:
        axis.set_xlabel("solver x (um)")
        axis.set_ylabel("solver y (um)")
    fig.suptitle(
        "Run 005 final binary: structure, Q, temperature, strict gradients and PTE current\n"
        f"full validated I={current_total_A:.6e} A; strict-map integral={strict_current_A:.6e} A"
    )
    fig.savefig(ALL_FIELDS_PLOT, dpi=180)
    plt.close(fig)

    # Q and temperature cross sections on literal nonuniform coordinates.
    ix0 = int(np.argmin(np.abs(x)))
    iy0 = int(np.argmin(np.abs(y)))
    z_crop = (z >= -1.0e-6) & (z <= 1.0e-6)
    fig, axes = plt.subplots(2, 3, figsize=(18, 9.5), constrained_layout=True)
    cross_panels = (
        (axes[0, 0], x * 1e6, y * 1e6, qxy_total, "Q: depth-integrated x-y", "W/m2", "inferno"),
        (axes[0, 1], x * 1e6, z[z_crop] * 1e6, q[:, iy0, :][:, z_crop], "Q: x-z at y=0", "W/m3", "inferno"),
        (axes[0, 2], y * 1e6, z[z_crop] * 1e6, q[ix0, :, :][:, z_crop], "Q: y-z at x=0", "W/m3", "inferno"),
        (axes[1, 0], x * 1e6, y * 1e6, tmax_xy, "DeltaT: maximum through z", "K", "magma"),
        (axes[1, 1], x * 1e6, z[z_crop] * 1e6, temperature[:, iy0, :][:, z_crop], "DeltaT: x-z at y=0", "K", "magma"),
        (axes[1, 2], y * 1e6, z[z_crop] * 1e6, temperature[ix0, :, :][:, z_crop], "DeltaT: y-z at x=0", "K", "magma"),
    )
    for axis, horizontal, vertical, field, title, unit, cmap in cross_panels:
        image = axis.pcolormesh(horizontal, vertical, np.ma.masked_invalid(field).T, shading="nearest", cmap=cmap)
        axis.set_title(title)
        axis.set_xlabel("solver x (um)" if "x-z" in title or "x-y" in title else "solver y (um)")
        axis.set_ylabel("solver y (um)" if "x-y" in title else "z (um)")
        fig.colorbar(image, ax=axis, label=unit)
    fig.suptitle("Run 005 final binary volumetric Q and temperature-rise cross sections")
    fig.savefig(CROSS_SECTION_PLOT, dpi=180)
    plt.close(fig)

    # Signed current decomposition and exact full-footprint totals.
    fig, axes = plt.subplots(1, 4, figsize=(21, 5.2), constrained_layout=True)
    for axis, field, title in (
        (axes[0], current_x_sheet, "strict x-term contribution"),
        (axes[1], current_y_sheet, "strict y-term contribution"),
        (axes[2], current_sheet, "strict total contribution"),
    ):
        image = axis.pcolormesh(
            x[fx] * 1e6, y[fy] * 1e6, np.ma.masked_invalid(field).T,
            shading="nearest", cmap="coolwarm", norm=signed_norm(field),
        )
        axis.set_title(title)
        axis.set_xlabel("solver x (um)")
        axis.set_ylabel("solver y (um)")
        axis.set_aspect("equal")
        fig.colorbar(image, ax=axis, label="A/m2")
    labels = ["x term", "y term", "full total", "strict map"]
    values = np.asarray([current_x_A, current_y_A, current_total_A, strict_current_A])
    axes[3].bar(labels, values / 1e-18, color=["#4c78a8", "#f58518", "#54a24b", "#b279a2"])
    axes[3].axhline(0.0, color="black", linewidth=0.8)
    axes[3].tick_params(axis="x", rotation=25)
    axes[3].set_ylabel("current (aA)")
    axes[3].set_title("integrated PTE current")
    fig.suptitle(
        "PTE equation actually used: -integral[Wx sigma_a S_a dT/dx + "
        "Wy sigma_b S_b dT/dy] dV"
    )
    fig.savefig(CURRENT_PLOT, dpi=180)
    plt.close(fig)

    summary = {
        "status": "PUBLISHED_FINAL_BINARY_Q_T_GRADIENT_CURRENT_FIELDS_WITH_AXIS_AUDIT",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_result_status": result["status"],
        "density_semantics": {
            "rho_1": "SiO2 design material present",
            "rho_0": "air/void; design material absent",
            "optical_law": "epsilon(rho)=1+rho*(epsilon_SiO2-1)",
            "thermal_law": "kappa(rho)=kappa_air+rho*(kappa_SiO2-kappa_air) for binary endpoints",
            "unique_values": np.unique(rho).tolist(),
            "solid_fraction": float(np.mean(rho)),
        },
        "coordinate_contract": {
            "plot_axes": "literal solver x/y coordinates",
            "optical_metadata": "x=b, y=a, z=c=b closure",
            "implemented_thermal_PTE_coefficients": "x uses a coefficients; y uses b coefficients",
            "interpretation_status": "UNRESOLVED_AXIS_METADATA_MISMATCH_XB_YA_VS_THERMAL_PTE_XA_YB",
            "note": "plots reproduce the immutable Run005 calculation; no axis or coefficient was silently changed",
        },
        "strict_spatial_map_contract": {
            "method": "centered two-neighbour derivative on each axis",
            "required_neighbours": ["-x", "+x", "-y", "+y"],
            "missing_neighbour_value": "NaN/masked",
            "valid_cells": int(np.count_nonzero(strict)),
            "masked_cells": int(strict.size - np.count_nonzero(strict)),
            "strict_integral_is_diagnostic": True,
        },
        "power": {
            "mapped_Q_reintegrated_W": mapped_power,
            "mapped_Q_expected_W": expected_mapped,
            "relative_reintegration_error": mapping_reintegration_error,
            "native_P_Q_W": result["base_forward"]["P_Q_W"],
            "P_six_W": result["base_forward"]["P_six_W"],
        },
        "temperature_and_gradient": {
            "maximum_temperature_rise_K": float(np.nanmax(temperature)),
            "maximum_flake_average_temperature_rise_K": float(np.nanmax(temperature_flake_average)),
            "strict_max_abs_dTdx_K_m": float(np.nanmax(np.abs(dtx_strict))),
            "strict_max_abs_dTdy_K_m": float(np.nanmax(np.abs(dty_strict))),
            "strict_max_gradient_K_m": float(np.nanmax(gradient_strict)),
        },
        "PTE_current": {
            "equation": "I=-integral[Wx*sigma_a*S_a*dTdx + Wy*sigma_b*S_b*dTdy]dV",
            "weighting_field_m_inv": [WEIGHTING_X_M_INV, WEIGHTING_Y_M_INV],
            "x_term_A": current_x_A,
            "y_term_A": current_y_A,
            "full_footprint_total_A": current_total_A,
            "stored_solver_objective_A": stored_current_A,
            "objective_reintegration_relative_error": objective_reintegration_error,
            "strict_centered_map_integral_A": strict_current_A,
            "strict_to_full_ratio": strict_current_A / current_total_A,
            "FOM_A_per_W": result["objective_A_per_incident_W"],
        },
        "inputs": {
            "evaluation_NPZ": artifact(RAW),
            "evaluation_result": artifact(RESULT),
        },
        "raw_derived_artifact": artifact(DERIVED),
        "plots": {},
        "new_solver_runs": 0,
    }
    for name, path in (
        ("structure", STRUCTURE_PLOT),
        ("all_fields", ALL_FIELDS_PLOT),
        ("cross_sections", CROSS_SECTION_PLOT),
        ("current_breakdown", CURRENT_PLOT),
    ):
        summary["plots"][name] = artifact(path)
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n")

    with CSV.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value", "unit"])
        for metric, value, unit in (
            ("solid_fraction", float(np.mean(rho)), "1"),
            ("mapped_Q", mapped_power, "W"),
            ("Tmax_rise", float(np.nanmax(temperature)), "K"),
            ("strict_max_abs_dTdx", float(np.nanmax(np.abs(dtx_strict))), "K/m"),
            ("strict_max_abs_dTdy", float(np.nanmax(np.abs(dty_strict))), "K/m"),
            ("strict_max_gradient", float(np.nanmax(gradient_strict)), "K/m"),
            ("PTE_x_term", current_x_A, "A"),
            ("PTE_y_term", current_y_A, "A"),
            ("PTE_full_total", current_total_A, "A"),
            ("PTE_strict_map_integral", strict_current_A, "A"),
            ("PTE_FOM", result["objective_A_per_incident_W"], "A/W"),
        ):
            writer.writerow([metric, f"{value:.16e}", unit])

    REPORT.write_text(
        "# Run 005 final-binary field maps\n\n"
        f"Status: `{summary['status']}`. This is read-only postprocessing of the "
        "fresh final-binary GPU Maxwell/CUDA thermal result; no solver was rerun.\n\n"
        "## Binary structure semantics\n\n"
        "- `1`: SiO2 design material is present.\n"
        "- `0`: air/void; design material is absent.\n"
        f"- Exact stored values: `{np.unique(rho).tolist()}`; material fraction: `{np.mean(rho):.9f}`.\n\n"
        "## Fields and PTE current\n\n"
        f"- mapped Q: `{mapped_power:.12e} W`; reintegration error: `{mapping_reintegration_error:.3e}`.\n"
        f"- maximum temperature rise: `{np.nanmax(temperature):.12e} K`.\n"
        f"- full-footprint current: `{current_total_A:.12e} A`; stored objective: `{stored_current_A:.12e} A`.\n"
        f"- objective reintegration error: `{objective_reintegration_error:.3e}`.\n"
        f"- x/y current terms: `{current_x_A:.12e}` / `{current_y_A:.12e} A`.\n"
        f"- strict-centered displayed-map integral: `{strict_current_A:.12e} A` "
        f"(`{strict_current_A/current_total_A:.9f}` of the boundary-aware full operator).\n\n"
        "The gradient/current maps require all four `-x,+x,-y,+y` TaIrTe4 neighbours. "
        "Every cell missing any neighbour is stored and displayed as `NaN`. The full scalar "
        "current remains the validated full-footprint operator, including its second-order "
        "one-sided perimeter stencil.\n\n"
        "## Axis audit\n\n"
        "The optical metadata says `x=b, y=a`, while the immutable Run005 thermal/PTE code "
        "uses the `a` coefficients on solver x and the `b` coefficients on solver y. The "
        "plots therefore use literal `solver x/y` labels and reproduce the existing result "
        "without silently swapping axes. Physical crystallographic interpretation remains "
        "`UNRESOLVED_AXIS_METADATA_MISMATCH_XB_YA_VS_THERMAL_PTE_XA_YB`.\n"
    )

    manifest = json.loads(MANIFEST.read_text())
    manifest["final_binary_field_visualization"] = {
        "status": summary["status"],
        "source_evaluation": artifact(RAW),
        "derived_fields": artifact(DERIVED),
        "summary": artifact(SUMMARY),
        "CSV": artifact(CSV),
        "plots": summary["plots"],
        "new_solver_runs": 0,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
