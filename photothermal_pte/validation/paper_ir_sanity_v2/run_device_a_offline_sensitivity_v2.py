#!/usr/bin/env python3
"""Offline Device-A PTE sensitivity sanity check v2.

This is a pure post-processing sanity check.  It re-uses the four completed
Device-A thermal/PTE field artifacts (isolated/perfect x E||a/E||b) and the
frozen Figure-2/3 geometry digitization.  No FDTD, no thermal solve, and no
optimization is run.

It answers three questions raised by the end-to-end disagreement
(|Ia|/|Ib| = 1.62-1.64 versus the digitized paper value 0.8366 +/- 0.0085):

1. Reproduction fidelity: can the stored terminal currents be reproduced
   bit-consistently from the stored temperature fields with the unmodified
   production integrator?  (Gates G1-G3.)
2. Declared-comparator metrics: what do the paper-comparator |dT/da| and the
   edge-normal |dT/dn| statistics actually say on the digitized off-axis
   edge band, per polarization and scenario?  (The production report used
   edge-normal statistics; the declared Figure-3G comparator is |dT/da|.)
3. Weighting-model sensitivity: how much does |Ia|/|Ib| move when the
   paper's isotropic-Laplace weighting potential (SI Eq. S7) is replaced by
   the physically motivated anisotropic-conductivity operator
   div(sigma grad psi) = 0 with sigma_b/sigma_a = 1.10e5/4.91e5 S/m?

Every variant re-uses the production ``pte_current`` integrator and the
production contact discretization unchanged; only the weighting potential
input differs between variants.  No clipping, gain, rescaling, or
polarization-dependent treatment is introduced.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from photothermal_pte.validation.paper_ir_sanity import (  # noqa: E402
    run_device_a_explicit_thermal_pte as base,
)

from scipy import sparse  # noqa: E402
import scipy.sparse.linalg as spla  # noqa: E402


PAPER_RATIO = 0.8365896980461811
PAPER_RATIO_UNCERTAINTY = 0.00852575488454707
EDGE_BANDS_UM = (0.3, 0.5, 1.0)
SIGMA_LAB_S_M = base.SIGMA_LAB_S_M  # (sigma_b, sigma_a) on lab (x, y)
SEEBECK_LAB_V_K = base.SEEBECK_LAB_V_K


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def simulation_origin_shift_um(payload: dict[str, Any]) -> np.ndarray:
    """Reproduce the frozen 60-um/50-um optical coordinate translation.

    Verbatim port of ``load_digitized_device_a_contract`` in
    ``run_lumerical_device_a_ir_q.py`` (domain_um=60, source_span_um=50).
    The end-to-end Laplace-psi reproduction gate (G3) fails closed if this
    translation or the contact segments disagree with the production run.
    """
    top = np.asarray(payload["top_metal_polygon_code_um"], float)
    bottom = np.asarray(payload["bottom_metal_polygon_code_um"], float)
    beam = np.asarray(payload["pre_registered_beam_center_code_um"], float)
    half_source = 25.0
    all_metal = np.vstack((top, bottom))
    occupied_min = np.minimum(beam - half_source, np.min(all_metal, axis=0))
    occupied_max = np.maximum(beam + half_source, np.max(all_metal, axis=0))
    return -0.5 * (occupied_min + occupied_max)


def solve_weighting_potential_conductivity(
    x_edges_m: np.ndarray,
    y_edges_m: np.ndarray,
    flake_xy: np.ndarray,
    sigma_xy_S_m: np.ndarray,
    top_segment_um: np.ndarray,
    bottom_segment_um: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Finite-volume div(sigma grad psi)=0 weighting solve.

    Identical structure, grid, contact detection, and Dirichlet penalty as
    the production ``solve_weighting_potential``; the only change is that
    every face conductance carries its axis conductivity and the contact
    penalty carries the y-axis conductivity (contacts attach through
    y-normal faces).  ``sigma_xy_S_m = (1, 1)`` must reproduce the
    production Laplace solution (identity gate G4).
    """
    x = 0.5 * (x_edges_m[:-1] + x_edges_m[1:])
    y = 0.5 * (y_edges_m[:-1] + y_edges_m[1:])
    dx, dy = np.diff(x_edges_m), np.diff(y_edges_m)
    ids = np.full(flake_xy.shape, -1, np.int64)
    ids[flake_xy] = np.arange(np.count_nonzero(flake_xy))
    n = int(np.count_nonzero(flake_xy))
    diagonal = np.zeros(n)
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    vals: list[np.ndarray] = []
    for axis in (0, 1):
        if axis == 0:
            lhs, rhs = ids[:-1], ids[1:]
            conductance = (
                sigma_xy_S_m[0]
                * 2.0
                * dy[None, :]
                / (dx[:-1, None] + dx[1:, None])
            )
        else:
            lhs, rhs = ids[:, :-1], ids[:, 1:]
            conductance = (
                sigma_xy_S_m[1]
                * 2.0
                * dx[:, None]
                / (dy[None, :-1] + dy[None, 1:])
            )
        connected = (lhs >= 0) & (rhs >= 0)
        left_id, right_id = lhs[connected], rhs[connected]
        g = np.broadcast_to(conductance, lhs.shape)[connected]
        np.add.at(diagonal, left_id, g)
        np.add.at(diagonal, right_id, g)
        rows += [left_id, right_id]
        cols += [right_id, left_id]
        vals += [-g, -g]

    neighbour_above = np.zeros_like(flake_xy)
    neighbour_above[:, :-1] = flake_xy[:, 1:]
    neighbour_below = np.zeros_like(flake_xy)
    neighbour_below[:, 1:] = flake_xy[:, :-1]
    top_boundary = flake_xy & ~neighbour_above
    bottom_boundary = flake_xy & ~neighbour_below
    top_x = np.sort(np.asarray(top_segment_um, float)[:, 0]) * 1e-6
    bottom_x = np.sort(np.asarray(bottom_segment_um, float)[:, 0]) * 1e-6
    top = top_boundary & (x[:, None] >= top_x[0]) & (x[:, None] <= top_x[-1])
    bottom = (
        bottom_boundary
        & (x[:, None] >= bottom_x[0])
        & (x[:, None] <= bottom_x[-1])
    )
    if not np.any(top) or not np.any(bottom):
        raise RuntimeError("conductivity weighting contact detection failed")
    load = np.zeros(n)
    for selected, value in ((top, 1.0), (bottom, 0.0)):
        flat = ids[selected]
        area = np.broadcast_to(dx[:, None], ids.shape)[selected]
        half_width = 0.5 * np.broadcast_to(dy[None, :], ids.shape)[selected]
        g = sigma_xy_S_m[1] * area / half_width
        np.add.at(diagonal, flat, g)
        np.add.at(load, flat, g * value)
    matrix = sparse.diags(diagonal) + sparse.coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n, n),
    ).tocsr()
    psi_active = spla.spsolve(matrix.tocsc(), load)
    residual = np.linalg.norm(matrix @ psi_active - load) / max(
        np.linalg.norm(load), np.finfo(float).tiny
    )
    psi = np.full(ids.shape, np.nan)
    psi[flake_xy] = psi_active
    grad_x, grad_y = base.cell_gradient(psi, flake_xy, x, y)
    return psi, grad_x, grad_y, {
        "top_contact_cells": int(np.count_nonzero(top)),
        "bottom_contact_cells": int(np.count_nonzero(bottom)),
        "linear_residual_relative": float(residual),
        "sigma_xy_S_m": np.asarray(sigma_xy_S_m, float),
    }


def load_case(directory: Path) -> dict[str, Any]:
    path = directory / "thermal_pte_fields.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    data = np.load(path)
    return {"path": path, "npz": data}


def edge_band_masks(
    payload: dict[str, Any],
    shift_um: np.ndarray,
    x_m: np.ndarray,
    y_m: np.ndarray,
    footprint: np.ndarray,
) -> tuple[dict[float, np.ndarray], dict[str, Any]]:
    vertices = (
        np.asarray(payload["flake_vertices_code_um"], float) + shift_um
    )
    i4, i7 = (int(v) for v in payload["off_axis_edge_vertex_indices"])
    v_a, v_b = vertices[i4], vertices[i7]
    tangent = np.asarray(payload["off_axis_edge_unit_tangent_code"], float)
    normal = np.asarray(
        payload["off_axis_edge_unit_inward_normal_code"], float
    )
    midpoint = 0.5 * (v_a + v_b)
    half_length_m = 0.5 * float(np.linalg.norm(v_b - v_a)) * 1e-6
    relative_x = x_m[:, None] - midpoint[0] * 1e-6
    relative_y = y_m[None, :] - midpoint[1] * 1e-6
    distance_normal = normal[0] * relative_x + normal[1] * relative_y
    distance_tangent = tangent[0] * relative_x + tangent[1] * relative_y
    bands = {}
    for band_um in EDGE_BANDS_UM:
        bands[band_um] = (
            footprint
            & (np.abs(distance_normal) <= band_um * 1e-6)
            & (np.abs(distance_tangent) <= half_length_m)
        )
    contract = {
        "edge_vertices_simulation_um": np.stack([v_a, v_b]),
        "unit_tangent": tangent,
        "unit_inward_normal": normal,
        "edge_half_length_um": half_length_m * 1e6,
        "band_cell_counts": {
            f"{band_um:g}um": int(np.count_nonzero(mask))
            for band_um, mask in bands.items()
        },
    }
    return bands, contract


def robust_stats(values: np.ndarray) -> dict[str, float]:
    selected = np.asarray(values, float)
    return {
        "max": float(np.max(selected)),
        "p99": float(np.percentile(selected, 99.0)),
        "rms": float(np.sqrt(np.mean(selected**2))),
    }


def comparator_metrics(
    npz: Any,
    bands: dict[float, np.ndarray],
    normal: np.ndarray,
) -> dict[str, Any]:
    grad_x = np.asarray(npz["grad_T_x_K_m"], float)
    grad_y = np.asarray(npz["grad_T_y_K_m"], float)
    grad_normal = normal[0] * grad_x + normal[1] * grad_y
    grad_magnitude = np.hypot(grad_x, grad_y)
    average_T = np.asarray(npz["temperature_flake_average_K"], float)
    result: dict[str, Any] = {}
    for band_um, mask in bands.items():
        result[f"{band_um:g}um"] = {
            "abs_grad_T_a_K_m": robust_stats(np.abs(grad_y[mask])),
            "abs_grad_T_b_K_m": robust_stats(np.abs(grad_x[mask])),
            "abs_grad_T_normal_K_m": robust_stats(np.abs(grad_normal[mask])),
            "abs_grad_T_magnitude_K_m": robust_stats(grad_magnitude[mask]),
            "band_Tmax_rise_K": float(np.max(average_T[mask])),
        }
    return result


def current_with_psi(
    npz: Any,
    geometry: SimpleNamespace,
    grad_psi_x: np.ndarray,
    grad_psi_y: np.ndarray,
) -> tuple[float, dict[str, np.ndarray]]:
    temperature = np.asarray(npz["temperature_rise_K"], float)
    return base.pte_current(temperature, geometry, grad_psi_x, grad_psi_y)


def band_decomposition(
    fields: dict[str, np.ndarray],
    geometry: SimpleNamespace,
    band_mask: np.ndarray,
) -> dict[str, float]:
    density = np.asarray(fields["shockley_ramo_integrand_A_m2"], float)
    area = (
        np.diff(geometry.x_edges_m)[:, None]
        * np.diff(geometry.y_edges_m)[None, :]
    )
    footprint = np.any(geometry.flake_mask, axis=2)
    total = float(np.sum(density[footprint] * area[footprint]))
    inside = footprint & band_mask
    edge = float(np.sum(density[inside] * area[inside]))
    return {
        "sheet_total_A": total,
        "edge_band_A": edge,
        "remainder_A": total - edge,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a-isolated", type=Path, required=True)
    parser.add_argument("--b-isolated", type=Path, required=True)
    parser.add_argument("--a-perfect", type=Path, required=True)
    parser.add_argument("--b-perfect", type=Path, required=True)
    parser.add_argument(
        "--geometry-contract-json",
        type=Path,
        default=REPOSITORY
        / "photothermal_pte/reports/paper_ir_device_a_end_to_end/device_a_geometry_digitization.json",
    )
    parser.add_argument(
        "--end-to-end-summary-json",
        type=Path,
        default=REPOSITORY
        / "photothermal_pte/reports/paper_ir_device_a_end_to_end/device_a_end_to_end_summary.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    args.report_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("MPLCONFIGDIR", str(args.output_dir / "mpl"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    payload = json.loads(args.geometry_contract_json.read_text())
    shift_um = simulation_origin_shift_um(payload)
    top_segment = (
        np.asarray(payload["top_electrical_contact_segment_code_um"], float)
        + shift_um
    )
    bottom_segment = (
        np.asarray(
            payload["bottom_electrical_contact_segment_code_um"], float
        )
        + shift_um
    )
    base.TOP_CONTACT_SEGMENT_UM = top_segment
    base.BOTTOM_CONTACT_SEGMENT_UM = bottom_segment
    base.FLAKE_VERTICES_UM = (
        np.asarray(payload["flake_vertices_code_um"], float) + shift_um
    )

    cases = {
        ("isolated", "a"): load_case(args.a_isolated),
        ("isolated", "b"): load_case(args.b_isolated),
        ("perfect", "a"): load_case(args.a_perfect),
        ("perfect", "b"): load_case(args.b_perfect),
    }

    reference = cases[("isolated", "a")]["npz"]
    x_edges = np.asarray(reference["x_edges_m"], float)
    y_edges = np.asarray(reference["y_edges_m"], float)
    z_edges = np.asarray(reference["z_edges_m"], float)
    flake_mask = np.asarray(reference["flake_mask"], bool)
    footprint = np.any(flake_mask, axis=2)
    geometry = SimpleNamespace(
        x_edges_m=x_edges,
        y_edges_m=y_edges,
        z_edges_m=z_edges,
        flake_mask=flake_mask,
    )
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])

    gates: dict[str, Any] = {}
    grid_identical = True
    for key, case in cases.items():
        npz = case["npz"]
        for name, expected in (
            ("x_edges_m", x_edges),
            ("y_edges_m", y_edges),
            ("z_edges_m", z_edges),
        ):
            if not np.array_equal(np.asarray(npz[name], float), expected):
                grid_identical = False
        if not np.array_equal(np.asarray(npz["flake_mask"], bool), flake_mask):
            grid_identical = False
    gates["G0_shared_grid_and_mask"] = grid_identical
    if not grid_identical:
        raise RuntimeError("the four cases do not share one grid/mask")

    dz = np.diff(z_edges)
    volume = (
        np.diff(x_edges)[:, None, None]
        * np.diff(y_edges)[None, :, None]
        * dz[None, None, :]
    )

    stored_currents: dict[tuple[str, str], float] = {}
    g1_errors: dict[str, float] = {}
    g2_errors: dict[str, float] = {}
    reproduction_fields: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    stored_psi = np.asarray(reference["weighting_potential"], float)
    stored_gpx = np.asarray(reference["weighting_grad_x_m_inv"], float)
    stored_gpy = np.asarray(reference["weighting_grad_y_m_inv"], float)
    for key, case in cases.items():
        npz = case["npz"]
        stored = float(npz["PTE_current_volume_integral_A"][0])
        stored_currents[key] = stored
        integrand = np.asarray(npz["shockley_ramo_integrand_A_m3_3d"], float)
        recomputed_sum = float(np.sum(integrand[flake_mask] * volume[flake_mask]))
        g1_errors["/".join(key)] = abs(recomputed_sum - stored) / abs(stored)
        current, fields = current_with_psi(
            npz, geometry, stored_gpx, stored_gpy
        )
        reproduction_fields[key] = fields
        g2_errors["/".join(key)] = abs(current - stored) / abs(stored)
    gates["G1_stored_integrand_volume_sum_relative_error"] = g1_errors
    gates["G2_pte_current_reproduction_relative_error"] = g2_errors

    laplace_psi, laplace_gpx, laplace_gpy, laplace_meta = (
        base.solve_weighting_potential(x_edges, y_edges, footprint)
    )
    gates["G3_laplace_psi_reproduction_max_abs_diff"] = float(
        np.nanmax(np.abs(laplace_psi - stored_psi))
    )

    identity_psi, _, _, _ = solve_weighting_potential_conductivity(
        x_edges,
        y_edges,
        footprint,
        np.asarray([1.0, 1.0]),
        top_segment,
        bottom_segment,
    )
    gates["G4_unit_sigma_identity_max_abs_diff"] = float(
        np.nanmax(np.abs(identity_psi - laplace_psi))
    )

    sigma_psi, sigma_gpx, sigma_gpy, sigma_meta = (
        solve_weighting_potential_conductivity(
            x_edges,
            y_edges,
            footprint,
            SIGMA_LAB_S_M,
            top_segment,
            bottom_segment,
        )
    )

    psi_variants = {
        "stored_laplace": (stored_gpx, stored_gpy),
        "laplace_resolved": (laplace_gpx, laplace_gpy),
        "sigma_weighted": (sigma_gpx, sigma_gpy),
    }
    currents: dict[str, dict[str, float]] = {}
    variant_fields: dict[tuple[str, str, str], dict[str, np.ndarray]] = {}
    for variant, (gpx, gpy) in psi_variants.items():
        table: dict[str, float] = {}
        for key, case in cases.items():
            if variant == "stored_laplace":
                current = None
                fields = reproduction_fields[key]
                current = float(
                    fields["PTE_current_volume_integral_A"][0]
                )
            else:
                current, fields = current_with_psi(
                    case["npz"], geometry, gpx, gpy
                )
            table["/".join(key)] = current
            variant_fields[(variant, *key)] = fields
        currents[variant] = table

    ratio_rows: list[dict[str, Any]] = []
    for variant in psi_variants:
        for scenario in ("isolated", "perfect"):
            current_a = currents[variant][f"{scenario}/a"]
            current_b = currents[variant][f"{scenario}/b"]
            ratio = abs(current_a) / abs(current_b)
            ratio_rows.append(
                {
                    "psi_variant": variant,
                    "scenario": scenario,
                    "I_a_A": current_a,
                    "I_b_A": current_b,
                    "abs_Ia_over_abs_Ib": ratio,
                    "paper_ratio": PAPER_RATIO,
                    "ratio_over_paper": ratio / PAPER_RATIO,
                }
            )

    bands, edge_contract = edge_band_masks(
        payload, shift_um, x_centers, y_centers, footprint
    )
    normal = np.asarray(
        payload["off_axis_edge_unit_inward_normal_code"], float
    )
    metrics = {
        "/".join(key): comparator_metrics(case["npz"], bands, normal)
        for key, case in cases.items()
    }
    comparator_ratios: dict[str, Any] = {}
    for scenario in ("isolated", "perfect"):
        entry: dict[str, Any] = {}
        for band_um in EDGE_BANDS_UM:
            band_key = f"{band_um:g}um"
            metric_a = metrics[f"{scenario}/a"][band_key]
            metric_b = metrics[f"{scenario}/b"][band_key]
            entry[band_key] = {
                field: {
                    stat: metric_a[field][stat] / metric_b[field][stat]
                    for stat in ("max", "p99", "rms")
                }
                for field in (
                    "abs_grad_T_a_K_m",
                    "abs_grad_T_b_K_m",
                    "abs_grad_T_normal_K_m",
                    "abs_grad_T_magnitude_K_m",
                )
            }
        comparator_ratios[scenario] = entry

    decomposition: dict[str, Any] = {}
    band_half_um = 0.5
    for variant in psi_variants:
        entry = {}
        for key in cases:
            fields = variant_fields[(variant, *key)]
            entry["/".join(key)] = band_decomposition(
                fields, geometry, bands[band_half_um]
            )
        decomposition[variant] = entry

    context: dict[str, Any] = {}
    if args.end_to_end_summary_json.is_file():
        summary = json.loads(args.end_to_end_summary_json.read_text())
        mapped = {}
        for case in summary.get("cases", []):
            scenario = case.get("scenario", "")
            polarization = case.get("polarization", "")
            power = case.get("P_Q_thermal_W")
            if scenario and polarization and power is not None:
                mapped[f"{scenario}/{polarization}"] = float(power)
        if mapped:
            context["mapped_power_W"] = mapped
            for scenario in ("isolated", "perfect"):
                key_a, key_b = f"{scenario}/a", f"{scenario}/b"
                if key_a in mapped and key_b in mapped:
                    context[
                        f"absorbed_power_proportional_ratio_{scenario}"
                    ] = mapped[key_a] / mapped[key_b]

    result = {
        "status": "COMPLETED_DEVICE_A_OFFLINE_SENSITIVITY_V2",
        "scope": {
            "new_fdtd_solve": False,
            "new_thermal_solve": False,
            "clipping_or_gain_or_rescaling": False,
            "polarization_dependent_treatment": False,
            "production_integrator_reused": True,
        },
        "paper_reference": {
            "digitized_abs_Ia_over_abs_Ib": PAPER_RATIO,
            "digitization_uncertainty": PAPER_RATIO_UNCERTAINTY,
        },
        "inputs": {
            "/".join(key): {
                "path": case["path"],
                "sha256": sha256(case["path"]),
            }
            for key, case in cases.items()
        },
        "geometry_contract": {
            "path": args.geometry_contract_json,
            "simulation_origin_shift_um": shift_um,
            "off_axis_edge": edge_contract,
        },
        "gates": gates,
        "weighting_meta": {
            "laplace_resolved": laplace_meta,
            "sigma_weighted": jsonable(sigma_meta),
        },
        "currents_A": currents,
        "ratio_matrix": ratio_rows,
        "comparator_metrics": metrics,
        "comparator_ratios_a_over_b": comparator_ratios,
        "edge_band_current_decomposition_0p5um": decomposition,
        "context": context,
    }
    (args.report_dir / "device_a_sanity_v2_summary.json").write_text(
        json.dumps(jsonable(result), indent=2) + "\n"
    )

    with (args.report_dir / "device_a_sanity_v2_ratio_matrix.csv").open(
        "w", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(ratio_rows[0]))
        writer.writeheader()
        writer.writerows(ratio_rows)

    np.savez_compressed(
        args.output_dir / "weighting_potential_variants.npz",
        x_edges_m=x_edges,
        y_edges_m=y_edges,
        stored_laplace_psi=stored_psi,
        laplace_resolved_psi=laplace_psi,
        sigma_weighted_psi=sigma_psi,
        sigma_gpx=sigma_gpx,
        sigma_gpy=sigma_gpy,
    )

    labels = []
    values = []
    for row in ratio_rows:
        labels.append(f"{row['psi_variant']}\n{row['scenario']}")
        values.append(row["abs_Ia_over_abs_Ib"])
    for scenario in ("isolated", "perfect"):
        key = f"absorbed_power_proportional_ratio_{scenario}"
        if key in context:
            labels.append(f"P_abs proportional\n{scenario}")
            values.append(context[key])
    figure, axis = plt.subplots(figsize=(9.6, 5.2))
    positions = np.arange(len(values))
    axis.bar(positions, values, color="#4477aa")
    axis.axhline(PAPER_RATIO, color="#cc3311", lw=1.6, label="paper 0.8366")
    axis.axhspan(
        PAPER_RATIO - PAPER_RATIO_UNCERTAINTY,
        PAPER_RATIO + PAPER_RATIO_UNCERTAINTY,
        color="#cc3311",
        alpha=0.15,
        lw=0,
    )
    axis.set_xticks(positions, labels, fontsize=8)
    axis.set_ylabel("|I_a| / |I_b|")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(args.report_dir / "V2_RATIO_MATRIX.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))
    extent = [
        x_centers[0] * 1e6,
        x_centers[-1] * 1e6,
        y_centers[0] * 1e6,
        y_centers[-1] * 1e6,
    ]
    for axis, (title, field) in zip(
        axes,
        (
            ("Laplace psi (paper Eq. S7)", laplace_psi),
            ("sigma-weighted psi", sigma_psi),
            ("difference", sigma_psi - laplace_psi),
        ),
    ):
        shown = axis.imshow(
            field.T,
            origin="lower",
            extent=extent,
            aspect="equal",
            cmap="viridis" if "difference" not in title else "coolwarm",
        )
        figure.colorbar(shown, ax=axis)
        axis.set_title(title, fontsize=9)
        axis.set_xlabel("x (b) [um]")
        axis.set_ylabel("y (a) [um]")
    figure.tight_layout()
    figure.savefig(args.report_dir / "V2_WEIGHTING_POTENTIALS.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9.0, 4.8))
    band_key = "0.5um"
    fields = (
        ("abs_grad_T_a_K_m", "|dT/da| (declared comparator)"),
        ("abs_grad_T_normal_K_m", "|dT/dn| (production metric)"),
        ("abs_grad_T_magnitude_K_m", "|grad T|"),
    )
    width = 0.25
    for offset, stat in zip((-width, 0.0, width), ("max", "p99", "rms")):
        heights = [
            comparator_ratios["isolated"][band_key][name][stat]
            for name, _ in fields
        ]
        axis.bar(
            np.arange(len(fields)) + offset,
            heights,
            width,
            label=stat,
        )
    axis.axhline(1.0, color="k", lw=1.0)
    axis.set_xticks(
        np.arange(len(fields)), [label for _, label in fields], fontsize=9
    )
    axis.set_ylabel("a/b ratio, isolated, 0.5 um edge band")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(
        args.report_dir / "V2_EDGE_COMPARATOR_METRICS.png", dpi=180
    )
    plt.close(figure)

    print(json.dumps(jsonable({
        "gates": gates,
        "ratio_matrix": ratio_rows,
        "comparator_ratios_isolated_0p5um":
            comparator_ratios["isolated"]["0.5um"],
        "context": context,
    }), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
