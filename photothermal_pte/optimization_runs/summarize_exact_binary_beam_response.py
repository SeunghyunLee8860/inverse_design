#!/usr/bin/env python3
"""Summarize fixed exact-binary beam-response sweeps with explicit Au contacts."""

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
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle
import numpy as np
from scipy.stats import spearmanr

from photothermal_pte.optimization_runs.tairte4_flake_topology.beam_response_contract import (
    AU_INTERFACE_REFERENCE,
    AU_INDEX_AT_10UM,
    AU_OPTICAL_REFERENCE,
    AU_TAIRTE4_INTERFACE_CONDUCTANCE_W_M2K,
    AU_THICKNESS_M,
    AU_THERMAL_REFERENCE,
    CASES,
    CONTACT_INNER_EDGE_M,
    FLAKE_BOUNDS_M,
    POSITION_SWEEP_UM,
    WAIST_SWEEP_UM,
    electrode_bounds_m,
    sweep_inputs,
)


RUNS = tuple(sorted(CASES))
RESULT_NAME = "beam_response_result.json"
EXPECTED_RESULT_SCHEMA = "exact-binary-fixed-flake-au-beam-response-v6"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")


def load_results(root: Path) -> dict[int, dict[str, object]]:
    results: dict[int, dict[str, object]] = {}
    failures: list[str] = []
    expected_inputs = {item["id"]: item for item in sweep_inputs()}
    for run in RUNS:
        path = root / f"run{run:03d}" / RESULT_NAME
        if not path.is_file():
            failures.append(f"missing {path}")
            continue
        payload = json.loads(path.read_text())
        if payload.get("schema") != EXPECTED_RESULT_SCHEMA:
            failures.append(
                f"run {run:03d}: schema={payload.get('schema')}, "
                f"expected {EXPECTED_RESULT_SCHEMA}"
            )
            continue
        if not payload.get("passed") or payload.get("status") != "COMPLETED":
            failures.append(f"run {run:03d}: status={payload.get('status')}")
            continue
        if len(payload.get("responses", [])) != 29:
            failures.append(
                f"run {run:03d}: expected 29 responses, got "
                f"{len(payload.get('responses', []))}"
            )
            continue
        rows = payload["responses"]
        actual_inputs = {row["id"]: row for row in rows}
        input_contract_matches = (
            len(actual_inputs) == len(expected_inputs)
            and set(actual_inputs) == set(expected_inputs)
            and all(
                actual_inputs[input_id]["kind"] == expected["kind"]
                and np.isclose(
                    actual_inputs[input_id]["target_waist_um"], expected["waist_um"]
                )
                and np.isclose(actual_inputs[input_id]["beam_x_um"], expected["x_um"])
                and np.isclose(actual_inputs[input_id]["beam_y_um"], expected["y_um"])
                for input_id, expected in expected_inputs.items()
            )
        )
        if not input_contract_matches:
            failures.append(f"run {run:03d}: response input contract mismatch")
            continue
        if not all(row.get("passed") for row in rows):
            failures.append(f"run {run:03d}: one or more response gates failed")
            continue
        case = CASES[run]
        inputs = payload.get("inputs", {})
        provenance_matches = (
            inputs.get("exact_binary_density", {}).get("sha256") == case.density_sha256
            and inputs.get("base_FSP", {}).get("sha256") == case.base_fsp_sha256
            and payload.get("contact_axis") == case.contact_axis
            and payload.get("interface_scenario") == case.interface_scenario
            and payload.get("polarization") == case.polarization
            and payload.get("optimization_rerun") is False
        )
        if not provenance_matches:
            failures.append(f"run {run:03d}: input provenance mismatch")
            continue
        au = payload.get("Au_contract", {})
        au_index = au.get("complex_index_at_10um", {})
        au_contract_matches = (
            np.isclose(au.get("thickness_m", np.nan), AU_THICKNESS_M)
            and np.isclose(au_index.get("real", np.nan), AU_INDEX_AT_10UM.real)
            and np.isclose(au_index.get("imag", np.nan), AU_INDEX_AT_10UM.imag)
            and np.isclose(
                au.get("thermal_conductivity_W_mK", np.nan),
                AU_THERMAL_REFERENCE["thermal_conductivity_W_mK"],
            )
            and np.isclose(
                au.get("Au_TaIrTe4_interface_conductance_W_m2K", np.nan),
                AU_TAIRTE4_INTERFACE_CONDUCTANCE_W_M2K,
            )
        )
        if not au_contract_matches:
            failures.append(f"run {run:03d}: Au material contract mismatch")
            continue
        geometry = payload.get("geometry", {})
        if not (
            geometry.get("flake_geometry_unchanged")
            and geometry.get("design_geometry_unchanged")
            and geometry.get("Au_entirely_inside_original_flake_xy")
            and not payload.get("flake_expanded_for_scan")
        ):
            failures.append(f"run {run:03d}: fixed-flake/Au geometry audit is not true")
            continue
        results[run] = payload
    if failures:
        raise RuntimeError("incomplete or invalid response set:\n" + "\n".join(failures))
    return results


def position_rows(payload: dict[str, object]) -> list[dict[str, object]]:
    rows = [row for row in payload["responses"] if row["kind"] == "position"]
    nominal = next(
        row
        for row in payload["responses"]
        if row["kind"] == "waist" and np.isclose(row["target_waist_um"], 8.5)
    )
    return [*rows, nominal]


def row_at(
    rows: list[dict[str, object]], x_um: float, y_um: float
) -> dict[str, object]:
    return next(
        row
        for row in rows
        if np.isclose(row["beam_x_um"], x_um)
        and np.isclose(row["beam_y_um"], y_um)
    )


def monotonic(values: np.ndarray) -> bool:
    differences = np.diff(values)
    return bool(np.all(differences >= 0.0) or np.all(differences <= 0.0))


def line_metrics(
    positions_um: np.ndarray, currents_nA: np.ndarray, center_nA: float
) -> dict[str, object]:
    rho = float(spearmanr(positions_um, currents_nA).statistic)
    slope = float((currents_nA[3] - currents_nA[1]) / 10.0)
    relative_span = float(np.ptp(currents_nA) / max(abs(center_nA), np.finfo(float).tiny))
    negative_half_monotonic = monotonic(currents_nA[:3])
    positive_half_monotonic = monotonic(currents_nA[2:])
    center_is_extremum = bool(
        center_nA >= np.max(currents_nA[[0, 1, 3, 4]])
        or center_nA <= np.min(currents_nA[[0, 1, 3, 4]])
    )
    unsigned_centering = bool(
        negative_half_monotonic
        and positive_half_monotonic
        and center_is_extremum
        and relative_span >= 0.05
    )
    return {
        "values_nA": currents_nA.tolist(),
        "monotonic": monotonic(currents_nA),
        "spearman_rho": rho,
        "central_slope_nA_per_um": slope,
        "relative_span": relative_span,
        "negative_half_monotonic": negative_half_monotonic,
        "positive_half_monotonic": positive_half_monotonic,
        "center_is_extremum": center_is_extremum,
        "unsigned_centering_assessment": (
            "promising_unsigned_displacement"
            if unsigned_centering
            else "limited_unsigned_displacement"
        ),
        "assessment": (
            "promising_1D"
            if monotonic(currents_nA) and relative_span >= 0.05
            else "limited_or_nonmonotonic_1D"
        ),
    }


def analyze_case(payload: dict[str, object]) -> dict[str, object]:
    all_responses = payload["responses"]
    waist_rows = sorted(
        (row for row in payload["responses"] if row["kind"] == "waist"),
        key=lambda row: row["target_waist_um"],
    )
    waist_um = np.asarray([row["target_waist_um"] for row in waist_rows], float)
    waist_current = np.asarray([row["terminal_current_nA"] for row in waist_rows], float)
    nominal_nA = float(waist_current[2])
    waist_rho = float(spearmanr(waist_um, waist_current).statistic)
    waist_relative_span = float(
        np.ptp(waist_current) / max(abs(nominal_nA), np.finfo(float).tiny)
    )

    rows = position_rows(payload)
    positions = np.asarray(POSITION_SWEEP_UM, float)
    current_map = np.asarray(
        [
            [row_at(rows, x_um, y_um)["terminal_current_nA"] for x_um in positions]
            for y_um in positions
        ],
        float,
    )
    x_line = current_map[2, :]
    y_line = current_map[:, 2]
    center = float(current_map[2, 2])
    gradient_x = float((current_map[2, 3] - current_map[2, 1]) / 10.0)
    gradient_y = float((current_map[3, 2] - current_map[1, 2]) / 10.0)
    map_span = float(np.ptp(current_map))
    ambiguity_tolerance = 0.01 * map_span
    flattened = current_map.ravel()
    ambiguous_pairs = int(
        sum(
            abs(float(flattened[i] - flattened[j])) <= ambiguity_tolerance
            for i in range(flattened.size)
            for j in range(i + 1, flattened.size)
        )
    )
    x_metrics = line_metrics(positions, x_line, center)
    y_metrics = line_metrics(positions, y_line, center)
    axis_metrics = x_metrics if payload["contact_axis"] == "x" else y_metrics
    cross_metrics = y_metrics if payload["contact_axis"] == "x" else x_metrics

    return {
        "run": int(payload["run"]),
        "contact_axis": payload["contact_axis"],
        "interface_scenario": payload["interface_scenario"],
        "polarization": payload["polarization"],
        "numerical": {
            "maximum_optical_closure": float(
                max(row["optical_closure"] for row in all_responses)
            ),
            "maximum_auto_shutoff": float(
                max(row["auto_shutoff"] for row in all_responses)
            ),
            "all_gates_passed": bool(all(row["passed"] for row in all_responses)),
        },
        "waist": {
            "waist_um": waist_um.tolist(),
            "current_nA": waist_current.tolist(),
            "nominal_current_nA": nominal_nA,
            "monotonic": monotonic(waist_current),
            "spearman_rho": waist_rho,
            "relative_span": waist_relative_span,
            "central_slope_nA_per_um": float((waist_current[3] - waist_current[1]) / 4.25),
            "assessment": (
                "promising"
                if monotonic(waist_current) and waist_relative_span >= 0.05
                else "limited_or_nonmonotonic"
            ),
        },
        "position": {
            "coordinates_um": positions.tolist(),
            "current_map_nA_y_by_x": current_map.tolist(),
            "center_current_nA": center,
            "map_min_nA": float(np.min(current_map)),
            "map_max_nA": float(np.max(current_map)),
            "map_span_nA": map_span,
            "relative_map_span": float(map_span / max(abs(center), np.finfo(float).tiny)),
            "central_gradient_nA_per_um": {"x": gradient_x, "y": gradient_y},
            "central_gradient_magnitude_nA_per_um": float(np.hypot(gradient_x, gradient_y)),
            "x_center_line": x_metrics,
            "y_center_line": y_metrics,
            "terminal_axis_line": axis_metrics,
            "cross_terminal_axis_line": cross_metrics,
            "ambiguous_pairs_within_1pct_map_span": ambiguous_pairs,
            "standalone_2D_assessment": "underdetermined_from_one_scalar_current",
        },
    }


def write_response_csv(
    path: Path, results: dict[int, dict[str, object]]
) -> None:
    fields = (
        "run", "contact_axis", "interface_scenario", "polarization", "id", "kind",
        "target_waist_um", "beam_x_um", "beam_y_um", "terminal_current_nA",
        "Tmax_flake_K", "Au_absorbed_W", "TaIrTe4_absorbed_W", "SiO2_absorbed_W",
        "Si_absorbed_W", "optical_closure", "Maxwell_wall_s", "passed",
    )
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for run, payload in results.items():
            for row in payload["responses"]:
                power = row["mapped_power_at_285uW"]
                writer.writerow({
                    "run": run,
                    "contact_axis": payload["contact_axis"],
                    "interface_scenario": payload["interface_scenario"],
                    "polarization": payload["polarization"],
                    "id": row["id"],
                    "kind": row["kind"],
                    "target_waist_um": row["target_waist_um"],
                    "beam_x_um": row["beam_x_um"],
                    "beam_y_um": row["beam_y_um"],
                    "terminal_current_nA": row["terminal_current_nA"],
                    "Tmax_flake_K": row["Tmax_flake_K"],
                    "Au_absorbed_W": power["Au_W"],
                    "TaIrTe4_absorbed_W": power["TaIrTe4_W"],
                    "SiO2_absorbed_W": power["SiO2_W"],
                    "Si_absorbed_W": power["Si_W"],
                    "optical_closure": row["optical_closure"],
                    "Maxwell_wall_s": row["Maxwell_wall_s"],
                    "passed": row["passed"],
                })


def plot_waist_matrix(path: Path, analyses: dict[int, dict[str, object]]) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(13.2, 6.8), sharex=True)
    for ax, run in zip(axes.ravel(), RUNS):
        data = analyses[run]["waist"]
        ax.plot(data["waist_um"], data["current_nA"], marker="o", color="#146c94", lw=1.8)
        ax.axvline(8.5, color="#d1495b", lw=1.0, ls="--")
        ax.set_title(
            f"Run {run:03d} | {analyses[run]['polarization']} | "
            f"{analyses[run]['interface_scenario'].replace('_', ' ')}",
            fontsize=9,
        )
        ax.grid(color="#d9d9d9", lw=0.6)
        ax.tick_params(labelsize=8)
    fig.supxlabel("Target beam waist w0 (um)")
    fig.supylabel("Terminal current (nA)")
    fig.suptitle("Fixed-structure beam-waist response at 285 uW", fontsize=13)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_position_matrix(
    path: Path, analyses: dict[int, dict[str, object]], normalized: bool
) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(14.8, 7.2), sharex=True, sharey=True)
    for ax, run in zip(axes.ravel(), RUNS):
        data = analyses[run]["position"]
        values = np.asarray(data["current_map_nA_y_by_x"], float)
        label = "Terminal current (nA)"
        norm = None
        if normalized:
            center = float(data["center_current_nA"])
            values = 100.0 * (values - center) / max(abs(center), np.finfo(float).tiny)
            extent = float(np.max(np.abs(values)))
            norm = TwoSlopeNorm(vmin=-extent, vcenter=0.0, vmax=extent)
            label = "Change from center (%)"
        image = ax.imshow(
            values,
            extent=(-12.5, 12.5, -12.5, 12.5),
            origin="lower",
            cmap="RdBu_r" if normalized else "viridis",
            norm=norm,
            interpolation="nearest",
        )
        ax.scatter([0], [0], marker="+", color="white", linewidths=1.2, s=35)
        ax.set_title(
            f"Run {run:03d} | contacts {analyses[run]['contact_axis']}", fontsize=9
        )
        ax.tick_params(labelsize=8)
        colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
        colorbar.ax.tick_params(labelsize=7)
        colorbar.set_label(label, fontsize=7)
    fig.supxlabel("Beam x position (um)")
    fig.supylabel("Beam y position (um)")
    title = "Normalized position response" if normalized else "Position response at w0 = 8.5 um"
    fig.suptitle(f"{title}; fixed 24 x 24 um flake", fontsize=13)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def load_density(path: Path) -> np.ndarray:
    with np.load(path) as archive:
        for key in ("rho_binary", "rho"):
            if key in archive:
                return np.asarray(archive[key], float)
    raise KeyError(f"no density array found in {path}")


def plot_fixed_geometries(path: Path) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(13.5, 7.8), sharex=True, sharey=True)
    low_um, high_um = (value * 1.0e6 for value in FLAKE_BOUNDS_M)
    inner_um = CONTACT_INNER_EDGE_M * 1.0e6
    for ax, run in zip(axes.ravel(), RUNS):
        case = CASES[run]
        density = load_density(case.density_path)
        if case.contact_axis == "y":
            extent = (low_um, high_um, -inner_um, inner_um)
        else:
            extent = (-inner_um, inner_um, low_um, high_um)
        ax.set_facecolor("#f7f7f7")
        ax.imshow(
            density.T,
            extent=extent,
            origin="lower",
            cmap="Greys",
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
        )
        for bounds in electrode_bounds_m(case.contact_axis):
            x0, x1 = (value * 1.0e6 for value in bounds["x"])
            y0, y1 = (value * 1.0e6 for value in bounds["y"])
            ax.add_patch(Rectangle(
                (x0, y0), x1 - x0, y1 - y0,
                facecolor="#d4af37", edgecolor="#7a5c00", alpha=0.78, lw=0.7,
            ))
        ax.add_patch(Rectangle(
            (low_um, low_um), high_um - low_um, high_um - low_um,
            fill=False, edgecolor="#d1495b", lw=1.2,
        ))
        ax.set_title(f"Run {run:03d} | Au on {case.contact_axis} terminals", fontsize=9)
        ax.set_aspect("equal")
        ax.set_xlim(-12.8, 12.8)
        ax.set_ylim(-12.8, 12.8)
        ax.tick_params(labelsize=8)
    fig.supxlabel("x (um)")
    fig.supylabel("y (um)")
    fig.suptitle("Exact-binary structures and 50 nm Au terminal footprints", fontsize=13)
    fig.tight_layout(h_pad=2.0)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}g}"


def write_report(
    path: Path,
    root: Path,
    analyses: dict[int, dict[str, object]],
    manifest: dict[str, object],
) -> None:
    rows = []
    for run in RUNS:
        waist = analyses[run]["waist"]
        position = analyses[run]["position"]
        rows.append(
            f"| {run:03d} | {analyses[run]['contact_axis']} | "
            f"{analyses[run]['interface_scenario']} | {analyses[run]['polarization']} | "
            f"{fmt(waist['nominal_current_nA'])} | {100.0 * waist['relative_span']:.1f}% | "
            f"{waist['monotonic']} | {fmt(waist['central_slope_nA_per_um'])} | "
            f"{100.0 * position['relative_map_span']:.1f}% | "
            f"({fmt(position['central_gradient_nA_per_um']['x'])}, "
            f"{fmt(position['central_gradient_nA_per_um']['y'])}) | "
            f"{100.0 * analyses[run]['numerical']['maximum_optical_closure']:.3f}% |"
        )
    size_promising = [
        f"{run:03d}" for run in RUNS
        if analyses[run]["waist"]["assessment"] == "promising"
    ]
    x_promising = [
        f"{run:03d}" for run in RUNS
        if analyses[run]["position"]["x_center_line"]["assessment"] == "promising_1D"
    ]
    y_promising = [
        f"{run:03d}" for run in RUNS
        if analyses[run]["position"]["y_center_line"]["assessment"] == "promising_1D"
    ]
    x_centering = [
        f"{run:03d}" for run in RUNS
        if analyses[run]["position"]["x_center_line"]["unsigned_centering_assessment"]
        == "promising_unsigned_displacement"
    ]
    y_centering = [
        f"{run:03d}" for run in RUNS
        if analyses[run]["position"]["y_center_line"]["unsigned_centering_assessment"]
        == "promising_unsigned_displacement"
    ]
    path.write_text(
        "\n".join([
            "# Exact-binary beam response with explicit Au terminals",
            "",
            f"Generated: {manifest['generated_at_utc']}",
            "",
            "## Scope and immutable geometry",
            "",
            "This report evaluates runs 044, 045, 047, 048, 055, 056, 057, and 058. "
            "Each run uses its already-optimized exact-binary density. No optimization was rerun.",
            "",
            "The TaIrTe4 flake remains exactly 24 x 24 um for every source position. "
            "Only the beam and the transverse simulation window move. The source never changes "
            "the design density, fixed TaIrTe4 terminal frames, or flake bounds.",
            "",
            "Two 50 nm Au rectangles occupy only the physical terminal strips inside the original "
            "flake footprint. The experimental paper reports a 5 nm Ti / 50 nm Au stack; this "
            "requested model includes Au only and deliberately omits the Ti adhesion layer.",
            "",
            "![Fixed geometries](fixed_geometry_and_au.png)",
            "",
            "## Material inputs",
            "",
            f"- Au at 10 um: n={AU_OPTICAL_REFERENCE['n']}, k={AU_OPTICAL_REFERENCE['k']} "
            f"from Ordal et al. ({AU_OPTICAL_REFERENCE['doi']}).",
            f"- Au thermal conductivity at {AU_THERMAL_REFERENCE['temperature_K']:.0f} K: "
            f"{AU_THERMAL_REFERENCE['thermal_conductivity_W_mK']:.0f} W m-1 K-1 "
            f"({AU_THERMAL_REFERENCE['url']}).",
            f"- Au/TaIrTe4 interface conductance: {AU_INTERFACE_REFERENCE['reported_W_m2K'] / 1e6:.2f} "
            f"MW m-2 K-1. This is explicitly a surrogate from the reported "
            f"{AU_INTERFACE_REFERENCE['measurement']} ({AU_INTERFACE_REFERENCE['doi']}); "
            "it is not presented as a direct Au/TaIrTe4 measurement.",
            "- The original run-specific TaIrTe4/SiO2 interface scenarios are preserved: "
            "thermally grown for 044/045/047/048 and evaporated for 055/056/057/058.",
            "",
            "## Sweep contract",
            "",
            f"- Equal incident power: 285 uW at 10 um.",
            f"- Target waist w0: {', '.join(fmt(value) for value in WAIST_SWEEP_UM)} um at the center.",
            f"- Position grid: x,y = {', '.join(fmt(value) for value in POSITION_SWEEP_UM)} um "
            "at w0 = 8.5 um.",
            "- Total new Maxwell inputs per run: 29 (the center position reuses the nominal-waist solve).",
            "- Absorption/flux control volume: x,y = [-14,+14] um, matching the existing "
            "250 nm illuminated-stack mesh; optical closure gate: <2%.",
            "",
            "## Results",
            "",
            "| Run | contacts | interface | pol. | I(center, 8.5 um) nA | waist span | waist monotonic | waist slope nA/um | position span | center gradient (x,y) nA/um | max closure |",
            "|---:|:---:|:---|:---:|---:|---:|:---:|---:|---:|:---|---:|",
            *rows,
            "",
            "![Waist responses](waist_response_matrix.png)",
            "",
            "![Position responses](position_response_matrix.png)",
            "",
            "![Normalized position responses](position_response_normalized_matrix.png)",
            "",
            "## Detector assessment",
            "",
            "A beam-size response is labeled promising only when all five sampled currents are "
            "monotonic and the full span is at least 5% of the nominal current. Under that "
            f"declared rule, the promising runs are: {', '.join(size_promising) or 'none'}.",
            "",
            "For a constrained one-dimensional beam path, the same monotonic and 5% rule gives "
            f"x-line candidates: {', '.join(x_promising) or 'none'}; "
            f"y-line candidates: {', '.join(y_promising) or 'none'}.",
            "",
            "For beam centering or unsigned displacement, a separate screen requires each "
            "half-line to be monotonic, the center to be an extremum, and at least 5% span. "
            f"The x-line candidates are: {', '.join(x_centering) or 'none'}; "
            f"the y-line candidates are: {', '.join(y_centering) or 'none'}. This mode cannot "
            "determine which side of center produced the current without another channel or "
            "prior position information.",
            "",
            "A single terminal-current scalar is not sufficient to infer an arbitrary 2D beam "
            "position uniquely. The maps can still be useful after constraining motion to one "
            "axis, adding a second independently patterned/current channel, or calibrating a "
            "multi-channel estimator. The summary JSON records center gradients, center-line "
            "monotonicity, Spearman rho, and the number of map-point current pairs within 1% of "
            "the map span.",
            "",
            "These labels are deterministic response-map screening results, not measured detector "
            "resolution or noise-equivalent performance. Quantifying those requires a finer "
            "calibration sweep plus readout noise, drift, fabrication variation, and experimental "
            "beam-profile uncertainty.",
            "",
            "## Numerical and provenance checks",
            "",
            "All 232 responses passed the <2% optical closure, nonnegative finite Q, auto-shutoff, "
            "Q-mapping, thermal residual/energy, electrical residual, and finite-current gates. "
            "Every result also records flake_expanded_for_scan=false and successful geometry/Au audits.",
            "",
            f"Raw result root: `{root}`",
            "",
            "Machine-readable products: `beam_response_summary.json`, `beam_response_all.csv`, "
            "and `manifest.json`.",
            "",
            "## References",
            "",
            "- M. G. Blevins et al., Advanced Functional Materials 36, e75986 (2026), "
            "https://doi.org/10.1002/adfm.75986.",
            f"- Au optical constants: {AU_OPTICAL_REFERENCE['doi']}.",
            f"- Au thermal conductivity: {AU_THERMAL_REFERENCE['url']}.",
            f"- Au-interface surrogate: {AU_INTERFACE_REFERENCE['doi']}.",
            "",
        ])
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args()

    root = args.result_root.expanduser().resolve()
    report_dir = args.report_dir.expanduser().resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    results = load_results(root)
    analyses = {run: analyze_case(payload) for run, payload in results.items()}

    csv_path = report_dir / "beam_response_all.csv"
    summary_path = report_dir / "beam_response_summary.json"
    write_response_csv(csv_path, results)
    write_json(summary_path, {
        "schema": "exact-binary-fixed-flake-au-beam-response-summary-v1",
        "generated_at_utc": utc_now(),
        "case_count": len(analyses),
        "response_count": sum(len(payload["responses"]) for payload in results.values()),
        "cases": analyses,
    })
    plot_waist_matrix(report_dir / "waist_response_matrix.png", analyses)
    plot_position_matrix(report_dir / "position_response_matrix.png", analyses, normalized=False)
    plot_position_matrix(
        report_dir / "position_response_normalized_matrix.png", analyses, normalized=True
    )
    plot_fixed_geometries(report_dir / "fixed_geometry_and_au.png")

    manifest = {
        "schema": "exact-binary-fixed-flake-au-beam-response-manifest-v1",
        "result_schema": EXPECTED_RESULT_SCHEMA,
        "generated_at_utc": utc_now(),
        "result_root": str(root),
        "flake_expanded_for_scan": False,
        "optimization_rerun": False,
        "sweep": {
            "waist_um": list(WAIST_SWEEP_UM),
            "position_x_y_um": list(POSITION_SWEEP_UM),
            "response_count_per_run": 29,
        },
        "material_references": {
            "Au_optical": AU_OPTICAL_REFERENCE,
            "Au_thermal": AU_THERMAL_REFERENCE,
            "Au_interface_surrogate": AU_INTERFACE_REFERENCE,
        },
        "runs": {},
    }
    for run, payload in results.items():
        result_path = root / f"run{run:03d}" / RESULT_NAME
        manifest["runs"][str(run)] = {
            "result": {"path": str(result_path), "sha256": sha256(result_path)},
            "density": payload["inputs"]["exact_binary_density"],
            "base_FSP": payload["inputs"]["base_FSP"],
            "response_count": len(payload["responses"]),
            "all_gates_passed": payload["all_gates_passed"],
            "geometry": payload["geometry"],
        }
    readme_path = report_dir / "README.md"
    write_report(readme_path, root, analyses, manifest)
    manifest["products"] = {
        path.name: sha256(path)
        for path in (
            readme_path,
            csv_path,
            summary_path,
            report_dir / "waist_response_matrix.png",
            report_dir / "position_response_matrix.png",
            report_dir / "position_response_normalized_matrix.png",
            report_dir / "fixed_geometry_and_au.png",
        )
    }
    write_json(report_dir / "manifest.json", manifest)
    print(json.dumps({
        "status": "COMPLETED",
        "report_dir": str(report_dir),
        "runs": list(analyses),
        "responses": sum(len(payload["responses"]) for payload in results.values()),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
