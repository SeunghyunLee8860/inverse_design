#!/usr/bin/env python3
"""Verify and publish all exact-binary beam-position spatial fields."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from photothermal_pte.optimization_runs.run_exact_binary_beam_position_fields import (
    RESULT_SCHEMA,
)
from photothermal_pte.optimization_runs.tairte4_flake_topology.beam_position_fields import (
    FIELD_SCHEMA,
)
from photothermal_pte.optimization_runs.tairte4_flake_topology.beam_response_contract import (
    CASES,
    POSITION_SWEEP_UM,
    TARGET_POWER_W,
    position_inputs,
)
from photothermal_pte.optimization_runs.tairte4_flake_topology.contract import CONTRACT
from photothermal_pte.optimization_runs.tairte4_flake_topology.thermal import (
    _piecewise_edges,
)


RUNS = tuple(sorted(CASES))
CELL_AREA_M2 = CONTRACT.design_step_m**2


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")


def expected_lookup() -> dict[str, dict[str, float | str]]:
    return {str(item["id"]): item for item in position_inputs()}


def load_results(root: Path) -> dict[int, dict[str, object]]:
    expected = expected_lookup()
    results: dict[int, dict[str, object]] = {}
    failures: list[str] = []
    for run in RUNS:
        path = root / f"run{run:03d}" / "position_fields_result.json"
        if not path.is_file():
            failures.append(f"missing {path}")
            continue
        payload = json.loads(path.read_text())
        rows = payload.get("responses", [])
        actual = {str(row.get("id")): row for row in rows}
        case = CASES[run]
        valid = (
            payload.get("schema") == RESULT_SCHEMA
            and payload.get("field_schema") == FIELD_SCHEMA
            and payload.get("status") == "COMPLETED"
            and payload.get("passed") is True
            and len(rows) == 25
            and set(actual) == set(expected)
            and all(row.get("passed") is True for row in rows)
            and payload.get("contact_axis") == case.contact_axis
            and payload.get("polarization") == case.polarization
            and payload.get("interface_scenario") == case.interface_scenario
            and payload.get("optimization_rerun") is False
            and payload.get("flake_expanded_for_scan") is False
            and payload.get("geometry", {}).get("flake_geometry_unchanged") is True
            and payload.get("geometry", {}).get("Au_entirely_inside_original_flake_xy") is True
        )
        if not valid:
            failures.append(f"invalid result contract for Run {run:03d}: {path}")
            continue
        if not all(
            np.isclose(actual[input_id]["beam_x_um"], item["x_um"])
            and np.isclose(actual[input_id]["beam_y_um"], item["y_um"])
            and np.isclose(actual[input_id]["target_waist_um"], item["waist_um"])
            for input_id, item in expected.items()
        ):
            failures.append(f"position grid mismatch for Run {run:03d}")
            continue
        results[run] = payload
    if failures:
        raise RuntimeError("incomplete spatial response set:\n" + "\n".join(failures))
    return results


def verify_field(
    run: int, contact_axis: str, row: dict[str, object]
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    raw = row["spatial_fields"]
    path = Path(str(raw["path"]))
    if not path.is_file() or sha256(path) != raw["sha256"]:
        raise RuntimeError(f"Run {run:03d} {row['id']}: field artifact hash mismatch")
    with np.load(path) as archive:
        arrays = {key: np.asarray(archive[key]) for key in archive.files}
    if str(arrays["schema"].item()) != FIELD_SCHEMA:
        raise RuntimeError(f"Run {run:03d} {row['id']}: field schema mismatch")

    contribution = np.asarray(
        arrays["terminal_current_contribution_total_A_m2"], float
    )
    total_j_weighted = np.asarray(
        arrays["total_J_weighted_contribution_total_A_m2"], float
    )
    certified = float(row["terminal_current_A"])
    contribution_current = float(np.sum(contribution) * CELL_AREA_M2)
    total_j_current = float(np.sum(total_j_weighted) * CELL_AREA_M2)
    denominator = max(
        abs(certified),
        float(np.sum(np.abs(contribution)) * CELL_AREA_M2),
        np.finfo(float).tiny,
    )
    pte_error = abs(contribution_current - certified) / denominator
    total_j_error = abs(total_j_current - certified) / denominator

    jx = np.asarray(arrays["local_J_total_x_A_m2"], float)
    jy = np.asarray(arrays["local_J_total_y_A_m2"], float)
    jx_sum = (
        np.asarray(arrays["local_J_thermoelectric_x_A_m2"], float)
        + np.asarray(arrays["local_J_conductive_x_A_m2"], float)
    )
    jy_sum = (
        np.asarray(arrays["local_J_thermoelectric_y_A_m2"], float)
        + np.asarray(arrays["local_J_conductive_y_A_m2"], float)
    )
    j_scale = max(float(np.max(np.hypot(jx, jy))), np.finfo(float).tiny)
    j_closure = max(
        float(np.max(np.abs(jx - jx_sum))),
        float(np.max(np.abs(jy - jy_sum))),
    ) / j_scale

    potential = np.asarray(arrays["short_circuit_potential_nodal_V"], float)
    terminal_values = (
        potential[(0, -1), :]
        if contact_axis == "x"
        else potential[:, (0, -1)]
    )
    terminal_potential_max = float(np.max(np.abs(terminal_values)))
    thermal_edges = _piecewise_edges()
    dx = np.diff(thermal_edges[0])
    dy = np.diff(thermal_edges[1])
    if not (
        np.allclose(
            arrays["thermal_x_cell_m"],
            0.5 * (thermal_edges[0][:-1] + thermal_edges[0][1:]),
        )
        and np.allclose(
            arrays["thermal_y_cell_m"],
            0.5 * (thermal_edges[1][:-1] + thermal_edges[1][1:]),
        )
    ):
        raise RuntimeError("thermal coordinate contract mismatch")
    area = dx[:, None] * dy[None, :]
    power_names = {
        "Au_W": "absorbed_power_density_Au_W_m2",
        "TaIrTe4_W": "absorbed_power_density_TaIrTe4_W_m2",
        "SiO2_W": "absorbed_power_density_SiO2_W_m2",
        "Si_W": "absorbed_power_density_Si_W_m2",
    }
    power_errors = {}
    for scalar_name, array_name in power_names.items():
        integrated = float(np.sum(np.asarray(arrays[array_name], float) * area))
        reference = float(row["mapped_power_at_285uW"][scalar_name])
        power_errors[scalar_name] = abs(integrated - reference) / max(
            abs(reference), np.finfo(float).tiny
        )

    allowed_nonfinite = {
        "temperature_gradient_strict_x_nodal_K_m",
        "temperature_gradient_strict_y_nodal_K_m",
        "temperature_gradient_strict_magnitude_nodal_K_m",
    }
    nonfinite = [
        key for key, value in arrays.items()
        if value.dtype.kind not in "US"
        and key not in allowed_nonfinite
        and not np.all(np.isfinite(value))
    ]
    if (
        pte_error >= 1.0e-8
        or total_j_error >= 1.0e-8
        or j_closure >= 1.0e-12
        or terminal_potential_max != 0.0
        or max(power_errors.values()) >= 1.0e-10
        or nonfinite
        or not np.isclose(float(arrays["beam_x_um"]), row["beam_x_um"])
        or not np.isclose(float(arrays["beam_y_um"]), row["beam_y_um"])
        or not np.isclose(float(arrays["terminal_current_A"]), certified)
    ):
        raise RuntimeError(
            f"Run {run:03d} {row['id']}: independent spatial-field audit failed"
        )
    audit = {
        "run": run,
        "id": row["id"],
        "beam_x_um": row["beam_x_um"],
        "beam_y_um": row["beam_y_um"],
        "terminal_current_nA": float(row["terminal_current_nA"]),
        "temperature_max_K": float(np.max(arrays["temperature_rise_nodal_K"])),
        "temperature_gradient_max_K_m": float(
            np.max(arrays["temperature_gradient_magnitude_cell_K_m"])
        ),
        "local_J_total_max_A_m2": float(
            np.max(arrays["local_J_total_magnitude_A_m2"])
        ),
        "positive_contribution_nA": float(
            np.sum(np.maximum(contribution, 0.0)) * CELL_AREA_M2 * 1.0e9
        ),
        "negative_contribution_nA": float(
            np.sum(np.minimum(contribution, 0.0)) * CELL_AREA_M2 * 1.0e9
        ),
        "pte_reintegration_relative_error": pte_error,
        "total_J_reintegration_relative_error": total_j_error,
        "local_J_constitutive_closure_relative_error": j_closure,
        "terminal_potential_max_abs_V": terminal_potential_max,
        "maximum_material_power_reintegration_relative_error": max(power_errors.values()),
        "short_circuit_continuity_residual": float(
            row["spatial_metrics"]["short_circuit_continuity_residual"]
        ),
        "reference_current_relative_error": float(row["reference_current_relative_error"]),
        "field_path": str(path),
        "field_size_bytes": int(raw["size_bytes"]),
        "field_sha256": raw["sha256"],
    }
    return audit, arrays


def rows_by_grid(rows: list[dict[str, object]]) -> dict[tuple[float, float], dict[str, object]]:
    return {
        (float(row["beam_x_um"]), float(row["beam_y_um"])): row
        for row in rows
    }


def coordinate_tag(value: float) -> str:
    sign = "p" if value >= 0.0 else "m"
    magnitude = f"{abs(value):g}".replace(".", "p")
    return f"{sign}{magnitude}"


def write_detailed_png_index(
    png_dir: Path,
    run: int,
    rows: list[dict[str, object]],
) -> None:
    positions = tuple(float(value) for value in POSITION_SWEEP_UM)
    lookup = rows_by_grid(rows)
    lines = [
        f"# Run {run:03d}: all 25 signed-current position fields",
        "",
        "Each thumbnail opens the full-resolution physical-field matrix. "
        "`I` is the signed terminal current; `I+` and `I-` are the separately "
        "integrated positive and negative local contributions.",
        "",
        "| beam y / beam x | "
        + " | ".join(f"{x:+g} um" for x in positions)
        + " |",
        "|:---:|" + ":---:|" * len(positions),
    ]
    for y in reversed(positions):
        cells = []
        for x in positions:
            row = lookup[(x, y)]
            name = (
                f"beam_x{coordinate_tag(x)}_y{coordinate_tag(y)}_fields.png"
            )
            cells.append(
                f'<a href="{name}"><img src="{name}" width="250"></a><br>'
                f'<code>I={float(row["terminal_current_nA"]):+.5g} nA</code><br>'
                f'<code>I+={float(row["positive_contribution_nA"]):+.5g}</code>, '
                f'<code>I-={float(row["negative_contribution_nA"]):+.5g} nA</code>'
            )
        lines.append(f"| **{y:+g} um** | " + " | ".join(cells) + " |")
    (png_dir / "README.md").write_text("\n".join(lines) + "\n")


def atlas(
    path: Path,
    run: int,
    rows: list[dict[str, object]],
    fields: dict[str, dict[str, np.ndarray]],
    top_key: str,
    bottom_key: str,
    top_label: str,
    bottom_label: str,
    *,
    top_cmap: str,
    bottom_cmap: str,
    bottom_centered: bool = False,
    quiver: bool = False,
    signed_current_titles: bool = False,
) -> dict[str, float]:
    positions = tuple(float(value) for value in POSITION_SWEEP_UM)
    lookup = rows_by_grid(rows)
    top_values = [np.asarray(fields[str(lookup[(x, y)]["id"])][top_key], float) for y in positions for x in positions]
    bottom_values = [np.asarray(fields[str(lookup[(x, y)]["id"])][bottom_key], float) for y in positions for x in positions]
    top_min = min(float(np.min(value)) for value in top_values)
    top_max = max(float(np.max(value)) for value in top_values)
    bottom_abs = max(float(np.max(np.abs(value))) for value in bottom_values)
    bottom_min = min(float(np.min(value)) for value in bottom_values)
    bottom_max = max(float(np.max(value)) for value in bottom_values)
    top_norm = Normalize(vmin=top_min, vmax=max(top_max, top_min + np.finfo(float).tiny))
    if bottom_centered:
        bottom_norm = TwoSlopeNorm(vmin=-bottom_abs, vcenter=0.0, vmax=bottom_abs)
    else:
        bottom_norm = Normalize(
            vmin=bottom_min,
            vmax=max(bottom_max, bottom_min + np.finfo(float).tiny),
        )

    fig, axes = plt.subplots(10, 5, figsize=(14.5, 25.0), sharex=True, sharey=True)
    top_image = None
    bottom_image = None
    for yi, y in enumerate(reversed(positions)):
        for xi, x in enumerate(positions):
            row = lookup[(x, y)]
            item = fields[str(row["id"])]
            top = np.asarray(item[top_key], float)
            bottom = np.asarray(item[bottom_key], float)
            def field_extent(value: np.ndarray) -> tuple[float, float, float, float]:
                if value.shape in ((241, 241), (240, 240)):
                    return (-12.0, 12.0, -12.0, 12.0)
                thermal_edges = _piecewise_edges()
                return (
                    float(thermal_edges[0][0] * 1e6),
                    float(thermal_edges[0][-1] * 1e6),
                    float(thermal_edges[1][0] * 1e6),
                    float(thermal_edges[1][-1] * 1e6),
                )
            top_extent = field_extent(top)
            bottom_extent = field_extent(bottom)
            top_image = axes[yi, xi].imshow(
                top.T, origin="lower", extent=top_extent, cmap=top_cmap,
                norm=top_norm, interpolation="nearest",
            )
            bottom_ax = axes[yi + 5, xi]
            bottom_image = bottom_ax.imshow(
                bottom.T, origin="lower", extent=bottom_extent, cmap=bottom_cmap,
                norm=bottom_norm, interpolation="nearest",
            )
            if quiver:
                jx = np.asarray(item["local_J_total_x_A_m2"], float)
                jy = np.asarray(item["local_J_total_y_A_m2"], float)
                stride = 8
                xq = np.asarray(item["electrical_x_cell_m"], float)[::stride] * 1e6
                yq = np.asarray(item["electrical_y_cell_m"], float)[::stride] * 1e6
                magnitude = np.hypot(jx[::stride, ::stride], jy[::stride, ::stride])
                unit_x = np.divide(
                    jx[::stride, ::stride], magnitude,
                    out=np.zeros_like(magnitude), where=magnitude > 0.0,
                )
                unit_y = np.divide(
                    jy[::stride, ::stride], magnitude,
                    out=np.zeros_like(magnitude), where=magnitude > 0.0,
                )
                axes[yi, xi].quiver(
                    xq, yq, unit_x.T, unit_y.T, color="white",
                    pivot="mid", angles="xy", scale_units="xy", scale=1.8,
                    width=0.0022, alpha=0.82,
                )
                rho = np.asarray(item["rho_exact_binary_cell"], float)
                axes[yi, xi].contour(
                    np.asarray(item["electrical_x_cell_m"], float) * 1e6,
                    np.asarray(item["electrical_y_cell_m"], float) * 1e6,
                    rho.T, levels=(0.5,), colors="cyan", linewidths=0.28,
                    alpha=0.7,
                )
            title = f"x={x:g}, y={y:g} um"
            if signed_current_titles:
                title += f"\nI={float(row['terminal_current_nA']):+.3g} nA"
            axes[yi, xi].set_title(title, fontsize=6.5, linespacing=0.95)
            if signed_current_titles:
                bottom_ax.set_title(title, fontsize=6.5, linespacing=0.95)
            for ax in (axes[yi, xi], bottom_ax):
                ax.set_xlim(-12.0, 12.0)
                ax.set_ylim(-12.0, 12.0)
                ax.tick_params(labelsize=6)
                ax.set_aspect("equal")
    for row_index in range(10):
        axes[row_index, 0].set_ylabel("y (um)", fontsize=7)
    for ax in axes[-1, :]:
        ax.set_xlabel("x (um)", fontsize=7)
    fig.subplots_adjust(left=0.07, right=0.90, bottom=0.04, top=0.95, wspace=0.06, hspace=0.28)
    fig.text(0.02, 0.73, top_label, rotation=90, va="center", fontsize=11)
    fig.text(0.02, 0.27, bottom_label, rotation=90, va="center", fontsize=11)
    fig.suptitle(
        f"Run {run:03d}: all 25 fixed-flake beam positions", fontsize=14, y=0.99,
    )
    if top_image is not None:
        top_bar = fig.colorbar(top_image, ax=axes[:5, :], fraction=0.018, pad=0.012)
        top_bar.set_label(top_label, fontsize=9)
    if bottom_image is not None:
        bottom_bar = fig.colorbar(bottom_image, ax=axes[5:, :], fraction=0.018, pad=0.012)
        bottom_bar.set_label(bottom_label, fontsize=9)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return {
        "top_min": top_min,
        "top_max": top_max,
        "bottom_min": bottom_min,
        "bottom_max": bottom_max,
    }


def field_panel(
    fig: plt.Figure,
    ax: plt.Axes,
    x_um: np.ndarray,
    y_um: np.ndarray,
    values: np.ndarray,
    title: str,
    *,
    cmap: str = "magma",
    centered: bool = False,
) -> None:
    values = np.asarray(values, float)
    finite = values[np.isfinite(values)]
    norm = None
    if centered and finite.size:
        bound = max(
            abs(float(np.min(finite))),
            abs(float(np.max(finite))),
            np.finfo(float).tiny,
        )
        norm = TwoSlopeNorm(vmin=-bound, vcenter=0.0, vmax=bound)
    image = ax.pcolormesh(
        x_um, y_um, values.T, shading="auto", cmap=cmap, norm=norm,
        rasterized=True,
    )
    ax.set_xlim(-12.0, 12.0)
    ax.set_ylim(-12.0, 12.0)
    ax.set_aspect("equal")
    ax.set_xlabel("x=b (um)", fontsize=7)
    ax.set_ylabel("y=a (um)", fontsize=7)
    ax.set_title(title, fontsize=8)
    ax.tick_params(labelsize=6)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.025)
    colorbar.ax.tick_params(labelsize=6)


def detailed_field_book(
    path: Path,
    png_dir: Path,
    run: int,
    rows: list[dict[str, object]],
    fields: dict[str, dict[str, np.ndarray]],
) -> None:
    lookup = rows_by_grid(rows)
    positions = tuple(float(value) for value in POSITION_SWEEP_UM)
    png_dir.mkdir(parents=True, exist_ok=True)

    with PdfPages(path) as pdf:
        for y in reversed(positions):
            for x in positions:
                row = lookup[(x, y)]
                item = fields[str(row["id"])]
                node_um = np.asarray(item["electrical_x_node_m"], float) * 1e6
                cell_um = np.asarray(item["electrical_x_cell_m"], float) * 1e6
                thermal_x_um = np.asarray(item["thermal_x_cell_m"], float) * 1e6
                thermal_y_um = np.asarray(item["thermal_y_cell_m"], float) * 1e6
                rho_node = np.asarray(item["rho_exact_binary_nodal"], float)
                temperature = np.asarray(item["temperature_rise_nodal_K"], float)

                fig, axes = plt.subplots(
                    3, 4, figsize=(18.0, 12.5), constrained_layout=True,
                )
                field_panel(
                    fig, axes[0, 0], node_um, node_um, rho_node,
                    "Exact binary: black=TaIrTe4", cmap="gray_r",
                )
                field_panel(
                    fig, axes[0, 1], thermal_x_um, thermal_y_um,
                    item["absorbed_power_density_total_W_m2"],
                    "All-material absorbed Q (W/m2)",
                )
                field_panel(
                    fig, axes[0, 2], thermal_x_um, thermal_y_um,
                    item["absorbed_power_density_TaIrTe4_W_m2"],
                    "TaIrTe4 absorbed Q (W/m2)",
                )
                field_panel(
                    fig, axes[0, 3], node_um, node_um,
                    np.where(rho_node > 0.5, temperature, np.nan),
                    "TaIrTe4 temperature rise (K)",
                )
                field_panel(
                    fig, axes[1, 0], node_um, node_um,
                    item["temperature_gradient_strict_x_nodal_K_m"],
                    "Strict-centered dT/db (K/m)", cmap="coolwarm", centered=True,
                )
                field_panel(
                    fig, axes[1, 1], node_um, node_um,
                    item["temperature_gradient_strict_y_nodal_K_m"],
                    "Strict-centered dT/da (K/m)", cmap="coolwarm", centered=True,
                )
                field_panel(
                    fig, axes[1, 2], node_um, node_um,
                    item["temperature_gradient_strict_magnitude_nodal_K_m"],
                    "Strict-centered |grad T| (K/m)", cmap="viridis",
                )
                field_panel(
                    fig, axes[1, 3], node_um, node_um,
                    np.where(
                        rho_node > 0.5,
                        np.asarray(item["weighting_potential_nodal"], float),
                        np.nan,
                    ),
                    "Weighting potential psi", cmap="viridis",
                )
                field_panel(
                    fig, axes[2, 0], cell_um, cell_um,
                    item["terminal_current_contribution_total_A_m2"],
                    "Terminal-current contribution: total (A/m2)",
                    cmap="coolwarm", centered=True,
                )
                field_panel(
                    fig, axes[2, 1], cell_um, cell_um,
                    item["terminal_current_contribution_x_A_m2"],
                    "Terminal-current contribution: b (A/m2)",
                    cmap="coolwarm", centered=True,
                )
                field_panel(
                    fig, axes[2, 2], cell_um, cell_um,
                    item["terminal_current_contribution_y_A_m2"],
                    "Terminal-current contribution: a (A/m2)",
                    cmap="coolwarm", centered=True,
                )
                j_ax = axes[2, 3]
                field_panel(
                    fig, j_ax, cell_um, cell_um,
                    item["local_J_total_magnitude_A_m2"],
                    "Local |J| with dense direction field (A/m2)", cmap="cividis",
                )
                jx = np.asarray(item["local_J_total_x_A_m2"], float)
                jy = np.asarray(item["local_J_total_y_A_m2"], float)
                stride = 6
                jxs = jx[::stride, ::stride]
                jys = jy[::stride, ::stride]
                magnitude = np.hypot(jxs, jys)
                ux = np.divide(
                    jxs, magnitude, out=np.zeros_like(magnitude), where=magnitude > 0.0,
                )
                uy = np.divide(
                    jys, magnitude, out=np.zeros_like(magnitude), where=magnitude > 0.0,
                )
                arrows = j_ax.quiver(
                    cell_um[::stride], cell_um[::stride], ux.T, uy.T,
                    color="white", pivot="mid", angles="xy", scale_units="xy",
                    scale=2.1, width=0.0018, alpha=0.78,
                )
                arrows.set_rasterized(True)
                positive_nA = float(row["positive_contribution_nA"])
                negative_nA = float(row["negative_contribution_nA"])
                current_nA = float(row["terminal_current_nA"])
                fig.suptitle(
                    f"Run {run:03d} | beam (x,y)=({x:+g},{y:+g}) um | "
                    f"I={current_nA:+.5g} nA | I+={positive_nA:+.5g} nA | "
                    f"I-={negative_nA:+.5g} nA",
                    fontsize=13,
                )
                pdf.savefig(fig, dpi=120)
                png_name = (
                    f"beam_x{coordinate_tag(x)}_y{coordinate_tag(y)}_fields.png"
                )
                fig.savefig(png_dir / png_name, dpi=110)
                plt.close(fig)
    write_detailed_png_index(png_dir, run, rows)


def scalar_diagnostics(
    path: Path, audits: dict[int, list[dict[str, object]]]
) -> None:
    positions = tuple(float(value) for value in POSITION_SWEEP_UM)
    quantities = (
        ("terminal_current_nA", "Terminal current (nA)", "viridis"),
        ("temperature_max_K", "Maximum temperature rise (K)", "inferno"),
        ("temperature_gradient_max_K_m", "Maximum |grad T| (K/m)", "magma"),
        ("local_J_total_max_A_m2", "Maximum |J| (A/m2)", "cividis"),
    )
    fig, axes = plt.subplots(4, 8, figsize=(21.0, 10.5), sharex=True, sharey=True)
    for column, run in enumerate(RUNS):
        lookup = {
            (float(row["beam_x_um"]), float(row["beam_y_um"])): row
            for row in audits[run]
        }
        for row_index, (key, label, cmap) in enumerate(quantities):
            values = np.asarray(
                [[lookup[(x, y)][key] for x in positions] for y in positions], float
            )
            image = axes[row_index, column].imshow(
                values, origin="lower", extent=(-12.5, 12.5, -12.5, 12.5),
                cmap=cmap, interpolation="nearest",
            )
            if row_index == 0:
                for y_index, y in enumerate(positions):
                    for x_index, x in enumerate(positions):
                        normalized = image.norm(values[y_index, x_index])
                        color = "black" if 0.58 < normalized < 0.95 else "white"
                        axes[row_index, column].text(
                            x, y, f"{values[y_index, x_index]:+.2g}",
                            ha="center", va="center", fontsize=4.2, color=color,
                        )
            if row_index == 0:
                axes[row_index, column].set_title(f"Run {run:03d}", fontsize=9)
            if column == 0:
                axes[row_index, column].set_ylabel(f"{label}\nbeam y (um)", fontsize=8)
            if row_index == 3:
                axes[row_index, column].set_xlabel("beam x (um)", fontsize=8)
            axes[row_index, column].tick_params(labelsize=6)
            colorbar = fig.colorbar(image, ax=axes[row_index, column], fraction=0.046, pad=0.02)
            colorbar.ax.tick_params(labelsize=5)
    fig.suptitle("Position-dependent scalar diagnostics from the spatial fields", fontsize=14)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    fig.savefig(path, dpi=170)
    plt.close(fig)


def write_csv(path: Path, audits: dict[int, list[dict[str, object]]]) -> None:
    fields = list(next(iter(audits.values()))[0])
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for run in RUNS:
            writer.writerows(audits[run])


def report_text(
    root: Path,
    results: dict[int, dict[str, object]],
    audits: dict[int, list[dict[str, object]]],
    figure_names: dict[int, list[str]],
) -> str:
    rows = []
    for run in RUNS:
        values = audits[run]
        rows.append(
            f"| {run:03d} | {results[run]['contact_axis']} | {results[run]['interface_scenario']} | "
            f"{results[run]['polarization']} | {min(row['terminal_current_nA'] for row in values):.3g} | "
            f"{max(row['terminal_current_nA'] for row in values):.3g} | "
            f"{max(row['temperature_max_K'] for row in values):.3g} | "
            f"{max(row['temperature_gradient_max_K_m'] for row in values):.3g} | "
            f"{max(row['local_J_total_max_A_m2'] for row in values):.3g} |"
        )
    galleries = []
    for run in RUNS:
        labels = (
            "thermal", "current", "optical/electrical", "25-page PDF",
            "25 individual PNGs",
        )
        links = " | ".join(
            f"[{label}]({name})"
            for label, name in zip(labels, figure_names[run])
        )
        galleries.append(f"| {run:03d} | {links} |")
    examples = []
    for run in RUNS:
        examples.append(
            f"### Run {run:03d}, beam x=0 um, y=0 um\n\n"
            f"[![Run {run:03d} center-position detailed fields]"
            f"(run{run:03d}_position_detailed_fields/beam_xp0_yp0_fields.png)]"
            f"(run{run:03d}_position_detailed_fields/beam_xp0_yp0_fields.png)"
        )
    return f"""# Exact-binary beam-position spatial fields with explicit Au terminals

Generated: {utc_now()}

## Scope

Runs 044, 045, 047, 048, 055, 056, 057, and 058 were evaluated at all 25 beam positions x,y = -10, -5, 0, 5, 10 um with w0 = 8.5 um and 285 uW incident power. Each calculation reuses the already optimized exact-binary structure. No optimization was rerun.

The TaIrTe4 flake remains exactly 24 x 24 um at every position. The source and transverse simulation window move; the flake, density, fixed terminal frame, and two 50 nm Au terminal rectangles do not. Every raw result records `flake_expanded_for_scan=false` and a successful Au-inside-flake geometry audit.

## Fields per position

Every one of the 200 position NPZ files contains 52 arrays/scalars: temperature rise, strict nodal and FEM-cell temperature gradients, weighting potential/gradient, short-circuit potential/electric field, thermoelectric/conductive/total local current density `J`, signed terminal-current contribution density and x/y components, total-J weighted contribution, and total/Au/TaIrTe4/SiO2/Si absorbed-power maps.

The physical current field is solved from `J = sigma E - sigma S grad(T)` with both terminals held at 0 V and insulating side boundaries. The terminal contribution is independently evaluated as `-t grad(psi) dot sigma S grad(T)`. The publisher reintegrates both field representations and rejects any point that disagrees with the certified terminal current by 1e-8 relative.

## Position extrema

| Run | contacts | interface | pol. | min I (nA) | max I (nA) | max dT (K) | max abs(grad T) (K/m) | max abs(J) (A/m2) |
|---:|:---:|:---|:---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

![Scalar diagnostics](position_scalar_diagnostics.png)

## Full 25-position atlases

Each thermal atlas shows temperature and gradient magnitude. Each current atlas shows total local `J` magnitude with dense direction arrows and the signed terminal-current contribution. Each optical/electrical atlas shows TaIrTe4+Au absorbed-power density and short-circuit potential. Every position in every atlas is labeled with the signed terminal current. The 25-page field book and 25-PNG gallery follow the detailed physical-field matrix style used by the final exact-binary report; every page shows signed `I`, positive `I+`, and negative `I-`. Color limits are shared across all 25 positions within each atlas.

| Run | Thermal | Current | Optical/electrical | Detailed PDF | Individual PNGs |
|---:|:---:|:---:|:---:|:---:|:---:|
{chr(10).join(galleries)}

## Detailed field examples

{chr(10).join(examples)}

## Audit

All 200/200 positions pass the GPU-only Maxwell, Q mapping, CUDA thermal, electrical weighting, short-circuit continuity, current identity, finite-field, and prior scalar-response agreement gates. The independent publisher also verifies the NPZ hash, constitutive identity `J_total = J_thermoelectric + J_conductive`, zero potential at both shorted terminals, material-resolved absorbed-power reintegration, and terminal-current reintegration.

Raw field root: `{root}`

Machine-readable report products: `position_fields_all.csv`, `position_fields_summary.json`, `field_dictionary.json`, and `manifest.json`.

The Au optical and thermal inputs are unchanged from the scalar beam-response report: n=12.1 and k=69.2 at 10 um; k_thermal=317 W m-1 K-1; Au/TaIrTe4 conductance=19.89 MW m-2 K-1 as an explicitly labeled Au/MoS2/sapphire surrogate. Run-specific TaIrTe4/SiO2 interface scenarios remain thermally grown for 044/045/047/048 and evaporated for 055/056/057/058.
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument(
        "--reuse-detailed",
        action="store_true",
        help="Reuse existing 25-page PDFs/PNGs while refreshing atlases and indexes.",
    )
    args = parser.parse_args()
    root = args.input_root.expanduser().resolve()
    report = args.report_dir.expanduser().resolve()
    report.mkdir(parents=True, exist_ok=True)
    results = load_results(root)

    audits: dict[int, list[dict[str, object]]] = {}
    fields: dict[int, dict[str, dict[str, np.ndarray]]] = {}
    raw_manifest = []
    for run in RUNS:
        audits[run] = []
        fields[run] = {}
        for row in results[run]["responses"]:
            audit, arrays = verify_field(run, str(results[run]["contact_axis"]), row)
            audits[run].append(audit)
            fields[run][str(row["id"])] = arrays
            raw_manifest.append({
                "run": run, "id": row["id"],
                "path": audit["field_path"], "size_bytes": audit["field_size_bytes"],
                "sha256": audit["field_sha256"],
            })
        audits[run].sort(key=lambda row: (row["beam_y_um"], row["beam_x_um"]))

    scalar_diagnostics(report / "position_scalar_diagnostics.png", audits)
    figure_names: dict[int, list[str]] = {}
    plot_limits: dict[int, dict[str, object]] = {}
    for run in RUNS:
        rows = list(results[run]["responses"])
        thermal_name = f"run{run:03d}_thermal_atlas.png"
        current_name = f"run{run:03d}_current_atlas.png"
        optical_name = f"run{run:03d}_optical_electrical_atlas.png"
        detailed_name = f"run{run:03d}_position_detailed_fields.pdf"
        detailed_png_dir = f"run{run:03d}_position_detailed_fields"
        thermal_limits = atlas(
            report / thermal_name, run, rows, fields[run],
            "temperature_rise_nodal_K", "temperature_gradient_magnitude_cell_K_m",
            "Temperature rise (K)", "Temperature-gradient magnitude (K/m)",
            top_cmap="inferno", bottom_cmap="magma",
            signed_current_titles=True,
        )
        current_limits = atlas(
            report / current_name, run, rows, fields[run],
            "local_J_total_magnitude_A_m2", "terminal_current_contribution_total_A_m2",
            "Total local J magnitude (A/m2)", "Terminal-current contribution (A/m2)",
            top_cmap="cividis", bottom_cmap="RdBu_r", bottom_centered=True,
            quiver=True, signed_current_titles=True,
        )
        for item in fields[run].values():
            item["device_absorbed_power_density_W_m2"] = (
                item["absorbed_power_density_TaIrTe4_W_m2"]
                + item["absorbed_power_density_Au_W_m2"]
            )
        optical_limits = atlas(
            report / optical_name, run, rows, fields[run],
            "device_absorbed_power_density_W_m2", "short_circuit_potential_nodal_V",
            "TaIrTe4 + Au absorbed power density (W/m2)", "Short-circuit potential (V)",
            top_cmap="plasma", bottom_cmap="RdBu_r", bottom_centered=True,
            signed_current_titles=True,
        )
        if args.reuse_detailed:
            png_dir = report / detailed_png_dir
            expected_pngs = {
                f"beam_x{coordinate_tag(x)}_y{coordinate_tag(y)}_fields.png"
                for x in POSITION_SWEEP_UM for y in POSITION_SWEEP_UM
            }
            actual_pngs = {path.name for path in png_dir.glob("*.png")}
            if not (report / detailed_name).is_file() or actual_pngs != expected_pngs:
                raise RuntimeError(
                    f"Run {run:03d}: existing detailed PDF/PNG set is incomplete"
                )
            write_detailed_png_index(png_dir, run, audits[run])
        else:
            detailed_field_book(
                report / detailed_name, report / detailed_png_dir,
                run, audits[run], fields[run],
            )
        figure_names[run] = [
            thermal_name, current_name, optical_name, detailed_name,
            detailed_png_dir,
        ]
        plot_limits[run] = {
            "thermal": thermal_limits,
            "current": current_limits,
            "optical_electrical": optical_limits,
        }
    write_csv(report / "position_fields_all.csv", audits)

    summary = {
        "schema": "exact-binary-beam-position-spatial-summary-v1",
        "generated_at_utc": utc_now(),
        "input_root": str(root),
        "runs": list(RUNS),
        "positions_per_run": 25,
        "total_positions": 200,
        "all_passed": True,
        "optimization_rerun": False,
        "flake_expanded_for_scan": False,
        "target_incident_power_W": TARGET_POWER_W,
        "plot_limits": plot_limits,
        "run_summaries": {
            str(run): {
                "contact_axis": results[run]["contact_axis"],
                "interface_scenario": results[run]["interface_scenario"],
                "polarization": results[run]["polarization"],
                "current_range_nA": [
                    min(row["terminal_current_nA"] for row in audits[run]),
                    max(row["terminal_current_nA"] for row in audits[run]),
                ],
                "maximum_temperature_K": max(row["temperature_max_K"] for row in audits[run]),
                "maximum_temperature_gradient_K_m": max(row["temperature_gradient_max_K_m"] for row in audits[run]),
                "maximum_local_J_A_m2": max(row["local_J_total_max_A_m2"] for row in audits[run]),
                "maximum_pte_reintegration_relative_error": max(row["pte_reintegration_relative_error"] for row in audits[run]),
                "maximum_total_J_reintegration_relative_error": max(row["total_J_reintegration_relative_error"] for row in audits[run]),
                "maximum_short_circuit_continuity_residual": max(row["short_circuit_continuity_residual"] for row in audits[run]),
                "maximum_reference_current_relative_error": max(row["reference_current_relative_error"] for row in audits[run]),
            }
            for run in RUNS
        },
    }
    write_json(report / "position_fields_summary.json", summary)
    field_dictionary = {
        "schema": FIELD_SCHEMA,
        "axis_contract": "Lumerical x=b, y=a, z=c",
        "NPZ_arrays": sorted(
            key for key in next(iter(fields[RUNS[0]].values()))
            if key != "device_absorbed_power_density_W_m2"
        ),
        "physical_current_definition": "J = sigma E - sigma S grad(T)",
        "terminal_contribution_definition": "-t grad(psi) dot sigma S grad(T)",
        "incident_power_W": TARGET_POWER_W,
    }
    write_json(report / "field_dictionary.json", field_dictionary)
    (report / "README.md").write_text(
        report_text(root, results, audits, figure_names)
    )

    report_files = sorted(
        path for path in report.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    )
    manifest = {
        "schema": "exact-binary-beam-position-spatial-manifest-v1",
        "generated_at_utc": utc_now(),
        "raw_artifacts": raw_manifest,
        "raw_total_size_bytes": sum(item["size_bytes"] for item in raw_manifest),
        "report_artifacts": [
            {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in report_files
        ],
    }
    write_json(report / "manifest.json", manifest)
    print(json.dumps({
        "status": "COMPLETED", "runs": len(RUNS), "positions": 200,
        "report": str(report), "raw_total_size_bytes": manifest["raw_total_size_bytes"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
