#!/usr/bin/env python3
"""Extract the weighting-free local PTE source from saved Device-A fields.

This is deliberately an offline analysis.  It reuses the temperature fields
from the nine-position/two-interface calculation and never reads or applies a
weighting potential when constructing J_loc = -sigma S grad(T).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import distance_transform_edt

from run_device_a_explicit_thermal_pte import strict_centered_cell_gradient


SIGMA_B_S_M = 1.10e5
SIGMA_A_S_M = 4.91e5
SEEBECK_B_V_K = 27.0e-6
SEEBECK_A_V_K = -6.0e-6
ILLUMINATED_EDGE_BAND_M = 1.0e-6
BEAM_RADIUS_M = 8.75e-6
SCENARIOS = ("thermally_grown", "evaporated")
POLARIZATIONS = ("a", "b")
POSITION_ORDER = (
    "outside_top",
    "outside_middle",
    "outside_bottom",
    "edge_top",
    "edge_middle",
    "edge_bottom",
    "inside_top",
    "inside_middle",
    "inside_bottom",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metrics(values: np.ndarray, selected: np.ndarray, area: np.ndarray) -> dict[str, float | int]:
    sample = np.asarray(values, float)[selected]
    weights = np.asarray(area, float)[selected]
    if sample.size == 0:
        raise RuntimeError("local-PTE metric received an empty support")
    return {
        "cell_count": int(sample.size),
        "maximum": float(np.max(sample)),
        "p99": float(np.percentile(sample, 99.0)),
        "rms_area_weighted": float(np.sqrt(np.sum(weights * sample**2) / np.sum(weights))),
        "mean_area_weighted": float(np.sum(weights * sample) / np.sum(weights)),
    }


def source_fields(npz: Any) -> dict[str, np.ndarray | float]:
    x_edges = np.asarray(npz["x_edges_m"], float)
    y_edges = np.asarray(npz["y_edges_m"], float)
    x = 0.5 * (x_edges[:-1] + x_edges[1:])
    y = 0.5 * (y_edges[:-1] + y_edges[1:])
    mask = np.any(np.asarray(npz["flake_mask"], bool), axis=2)
    temperature = np.asarray(npz["temperature_flake_average_K"], float)
    grad_b, grad_a, valid = strict_centered_cell_gradient(temperature, mask, x, y)

    saved_valid = np.asarray(npz["strict_valid_xy_mask"], bool)
    if not np.array_equal(valid, saved_valid):
        raise RuntimeError("recomputed strict four-neighbour mask differs from saved mask")
    saved_b = np.asarray(npz["grad_T_x_K_m"], float)
    saved_a = np.asarray(npz["grad_T_y_K_m"], float)
    if not np.array_equal(np.isnan(saved_b), np.isnan(grad_b)):
        raise RuntimeError("saved/recomputed b-gradient NaN support differs")
    if not np.array_equal(np.isnan(saved_a), np.isnan(grad_a)):
        raise RuntimeError("saved/recomputed a-gradient NaN support differs")
    gradient_reproduction_error = max(
        float(np.max(np.abs(saved_b[valid] - grad_b[valid]))),
        float(np.max(np.abs(saved_a[valid] - grad_a[valid]))),
    )

    # Lumerical x is crystal b and Lumerical y is crystal a.
    j_b = -SIGMA_B_S_M * SEEBECK_B_V_K * grad_b
    j_a = -SIGMA_A_S_M * SEEBECK_A_V_K * grad_a
    magnitude = np.hypot(j_b, j_a)
    area = np.diff(x_edges)[:, None] * np.diff(y_edges)[None, :]
    return {
        "x_edges_m": x_edges,
        "y_edges_m": y_edges,
        "x_m": x,
        "y_m": y,
        "flake_mask": mask,
        "valid": valid,
        "area_m2": area,
        "grad_b_K_m": grad_b,
        "grad_a_K_m": grad_a,
        "Jloc_b_A_m2": j_b,
        "Jloc_a_A_m2": j_a,
        "Jloc_magnitude_A_m2": magnitude,
        "gradient_reproduction_max_abs_K_m": gradient_reproduction_error,
    }


def illuminated_edge_mask(
    fields: dict[str, np.ndarray | float], beam_b_m: float, beam_a_m: float
) -> np.ndarray:
    mask = np.asarray(fields["flake_mask"], bool)
    valid = np.asarray(fields["valid"], bool)
    x_edges = np.asarray(fields["x_edges_m"], float)
    y_edges = np.asarray(fields["y_edges_m"], float)
    x = np.asarray(fields["x_m"], float)
    y = np.asarray(fields["y_m"], float)
    active_x = np.flatnonzero(np.any(mask, axis=1))
    active_y = np.flatnonzero(np.any(mask, axis=0))
    dx = float(np.min(np.diff(x_edges)[active_x]))
    dy = float(np.min(np.diff(y_edges)[active_y]))
    distance_inside = distance_transform_edt(mask, sampling=(dx, dy))
    xx, yy = np.meshgrid(x, y, indexing="ij")
    radial = np.hypot(xx - beam_b_m, yy - beam_a_m)
    return valid & (distance_inside <= ILLUMINATED_EDGE_BAND_M) & (radial <= BEAM_RADIUS_M)


def plot_pair(
    output: Path,
    scenario: str,
    position: str,
    beam_b_um: float,
    beam_a_um: float,
    pair: dict[str, dict[str, np.ndarray | float]],
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15.2, 9.0), constrained_layout=True)
    components = (
        ("Jloc_b_A_m2", r"$J_{\mathrm{loc},b}$"),
        ("Jloc_a_A_m2", r"$J_{\mathrm{loc},a}$"),
        ("Jloc_magnitude_A_m2", r"$|\mathbf{J}_{\mathrm{loc}}|$"),
    )
    signed_limits: dict[str, float] = {}
    magnitude_limit = 0.0
    for key, _ in components[:2]:
        signed_limits[key] = max(
            float(np.nanmax(np.abs(np.where(pair[p]["valid"], pair[p][key], np.nan))))
            for p in POLARIZATIONS
        )
    magnitude_limit = max(
        float(np.nanmax(np.where(pair[p]["valid"], pair[p]["Jloc_magnitude_A_m2"], np.nan)))
        for p in POLARIZATIONS
    )

    for row, pol in enumerate(POLARIZATIONS):
        fields = pair[pol]
        xe = np.asarray(fields["x_edges_m"]) * 1e6
        ye = np.asarray(fields["y_edges_m"]) * 1e6
        mask = np.asarray(fields["flake_mask"], bool)
        valid = np.asarray(fields["valid"], bool)
        x_centers = 0.5 * (xe[:-1] + xe[1:])
        y_centers = 0.5 * (ye[:-1] + ye[1:])
        ix, iy = np.where(mask)
        xlim = (float(x_centers[ix].min() - 1.0), float(x_centers[ix].max() + 1.0))
        ylim = (float(y_centers[iy].min() - 1.0), float(y_centers[iy].max() + 1.0))
        for col, (key, label) in enumerate(components):
            ax = axes[row, col]
            values = np.where(valid, np.asarray(fields[key], float), np.nan)
            if col < 2:
                lim = signed_limits[key]
                image = ax.pcolormesh(xe, ye, values.T, shading="auto", cmap="coolwarm", vmin=-lim, vmax=lim)
            else:
                image = ax.pcolormesh(xe, ye, values.T, shading="auto", cmap="magma", vmin=0.0, vmax=magnitude_limit)
            ax.contour(x_centers, y_centers, mask.T.astype(float), levels=[0.5], colors="cyan", linewidths=0.7)
            ax.plot(beam_b_um, beam_a_um, marker="+", color="lime", markersize=9, markeredgewidth=1.5)
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.set_aspect("equal")
            ax.set_xlabel("Lumerical x = crystal b (um)")
            ax.set_ylabel("Lumerical y = crystal a (um)")
            ax.set_title(rf"$E\parallel {pol}$: {label}")
            cbar = fig.colorbar(image, ax=ax, shrink=0.84)
            cbar.set_label(r"A m$^{-2}$")
    fig.suptitle(
        f"Weighting-free local PTE source — {scenario}, {position}\n"
        r"$\mathbf{J}_{\rm loc}=-\mathbf{\sigma}\mathbf{S}\nabla T$; no $\nabla\psi$, no terminal-current integration",
        fontsize=14,
    )
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_ratios(output: Path, paired: list[dict[str, Any]]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16.0, 10.0), constrained_layout=True)
    specs = (
        ("full_maximum_Ea_over_Eb", "full-flake strict maximum"),
        ("full_p99_Ea_over_Eb", "full-flake strict P99"),
        ("edge_maximum_Ea_over_Eb", "illuminated-edge maximum"),
        ("edge_p99_Ea_over_Eb", "illuminated-edge P99"),
    )
    x = np.arange(len(POSITION_ORDER))
    width = 0.36
    for ax, (key, title) in zip(axes.flat, specs):
        for offset, scenario, color in (
            (-width / 2, "thermally_grown", "tab:blue"),
            (width / 2, "evaporated", "tab:orange"),
        ):
            lookup = {r["position"]: r[key] for r in paired if r["scenario"] == scenario}
            ax.bar(x + offset, [lookup[p] for p in POSITION_ORDER], width, label=scenario, color=color)
        ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0)
        ax.set_xticks(x, POSITION_ORDER, rotation=35, ha="right")
        ax.set_ylabel(r"$|\mathbf{J}_{loc}(E\parallel a)|/|\mathbf{J}_{loc}(E\parallel b)|$")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        ax.legend()
    fig.suptitle("Device-A weighting-free local-PTE-source polarization ratios", fontsize=15)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("photothermal_pte/reports/paper_ir_device_a_nine_position_two_interface"),
    )
    parser.add_argument("--skip-sha-verification", action="store_true")
    args = parser.parse_args()
    report_dir = args.report_dir.resolve()
    manifest_path = report_dir / "RAW_ARTIFACT_MANIFEST.json"
    summary_path = report_dir / "device_a_nine_position_two_interface_summary.json"
    manifest = json.loads(manifest_path.read_text())
    old_summary = json.loads(summary_path.read_text())
    metadata = {
        (r["scenario"], r["position"], r["polarization"]): r
        for r in old_summary["rows"]
    }

    # Analytic sanity check of the strict-centered gradient and constitutive law.
    coords = np.array([0.0, 1.0, 2.0])
    xx, yy = np.meshgrid(coords, coords, indexing="ij")
    gb, ga, valid = strict_centered_cell_gradient(2.0 * xx + 3.0 * yy, np.ones((3, 3), bool), coords, coords)
    if not (valid[1, 1] and gb[1, 1] == 2.0 and ga[1, 1] == 3.0):
        raise RuntimeError("strict-centered analytic gradient self-test failed")

    cases: list[dict[str, Any]] = []
    field_pairs: dict[tuple[str, str], dict[str, dict[str, np.ndarray | float]]] = {}
    references: list[dict[str, Any]] = []
    for record in manifest["raw_thermal_field_artifacts"]:
        key = (record["scenario"], record["position"], record["polarization"])
        if key not in metadata:
            raise RuntimeError(f"missing metadata for {key}")
        path = Path(record["path"])
        size_ok = path.stat().st_size == int(record["size_bytes"])
        digest = None if args.skip_sha_verification else sha256(path)
        sha_ok = None if digest is None else digest == record["sha256"]
        if not size_ok or sha_ok is False:
            raise RuntimeError(f"raw thermal artifact provenance failed: {path}")
        references.append({**record, "size_verified": size_ok, "sha256_verified": sha_ok})
        with np.load(path) as npz:
            fields = source_fields(npz)
        meta = metadata[key]
        edge = illuminated_edge_mask(
            fields,
            float(meta["beam_x_b_um"]) * 1e-6,
            float(meta["beam_y_a_um"]) * 1e-6,
        )
        valid_mask = np.asarray(fields["valid"], bool)
        area = np.asarray(fields["area_m2"], float)
        mag = np.asarray(fields["Jloc_magnitude_A_m2"], float)
        abs_b = np.abs(np.asarray(fields["Jloc_b_A_m2"], float))
        abs_a = np.abs(np.asarray(fields["Jloc_a_A_m2"], float))
        full_metrics = metrics(mag, valid_mask, area)
        edge_metrics = metrics(mag, edge, area)
        row: dict[str, Any] = {
            "scenario": record["scenario"],
            "position": record["position"],
            "polarization": record["polarization"],
            "beam_x_b_um": meta["beam_x_b_um"],
            "beam_y_a_um": meta["beam_y_a_um"],
            "Jloc_full_max_A_m2": full_metrics["maximum"],
            "Jloc_full_p99_A_m2": full_metrics["p99"],
            "Jloc_full_rms_A_m2": full_metrics["rms_area_weighted"],
            "Jloc_full_mean_A_m2": full_metrics["mean_area_weighted"],
            "Jloc_edge_max_A_m2": edge_metrics["maximum"],
            "Jloc_edge_p99_A_m2": edge_metrics["p99"],
            "Jloc_edge_rms_A_m2": edge_metrics["rms_area_weighted"],
            "Jloc_edge_mean_A_m2": edge_metrics["mean_area_weighted"],
            "abs_Jloc_b_full_max_A_m2": metrics(abs_b, valid_mask, area)["maximum"],
            "abs_Jloc_a_full_max_A_m2": metrics(abs_a, valid_mask, area)["maximum"],
            "strict_valid_cell_count": full_metrics["cell_count"],
            "illuminated_edge_cell_count": edge_metrics["cell_count"],
            "saved_gradient_reproduction_max_abs_K_m": fields["gradient_reproduction_max_abs_K_m"],
            "weighting_potential_used": False,
            "terminal_current_integration_performed": False,
        }
        cases.append(row)
        field_pairs.setdefault((record["scenario"], record["position"]), {})[record["polarization"]] = fields

    paired: list[dict[str, Any]] = []
    case_lookup = {(r["scenario"], r["position"], r["polarization"]): r for r in cases}
    for scenario in SCENARIOS:
        for position in POSITION_ORDER:
            a = case_lookup[(scenario, position, "a")]
            b = case_lookup[(scenario, position, "b")]
            paired.append({
                "scenario": scenario,
                "position": position,
                "beam_x_b_um": a["beam_x_b_um"],
                "beam_y_a_um": a["beam_y_a_um"],
                "full_maximum_Ea_A_m2": a["Jloc_full_max_A_m2"],
                "full_maximum_Eb_A_m2": b["Jloc_full_max_A_m2"],
                "full_p99_Ea_A_m2": a["Jloc_full_p99_A_m2"],
                "full_p99_Eb_A_m2": b["Jloc_full_p99_A_m2"],
                "edge_maximum_Ea_A_m2": a["Jloc_edge_max_A_m2"],
                "edge_maximum_Eb_A_m2": b["Jloc_edge_max_A_m2"],
                "edge_p99_Ea_A_m2": a["Jloc_edge_p99_A_m2"],
                "edge_p99_Eb_A_m2": b["Jloc_edge_p99_A_m2"],
                "full_maximum_Ea_over_Eb": a["Jloc_full_max_A_m2"] / b["Jloc_full_max_A_m2"],
                "full_p99_Ea_over_Eb": a["Jloc_full_p99_A_m2"] / b["Jloc_full_p99_A_m2"],
                "full_rms_Ea_over_Eb": a["Jloc_full_rms_A_m2"] / b["Jloc_full_rms_A_m2"],
                "edge_maximum_Ea_over_Eb": a["Jloc_edge_max_A_m2"] / b["Jloc_edge_max_A_m2"],
                "edge_p99_Ea_over_Eb": a["Jloc_edge_p99_A_m2"] / b["Jloc_edge_p99_A_m2"],
                "edge_rms_Ea_over_Eb": a["Jloc_edge_rms_A_m2"] / b["Jloc_edge_rms_A_m2"],
                "ratio_is_polarization_indexed_not_vector_component_ratio": True,
            })

    panels = report_dir / "local_pte_source_case_panels"
    panels.mkdir(exist_ok=True)
    for scenario, position in field_pairs:
        pair = field_pairs[(scenario, position)]
        if set(pair) != set(POLARIZATIONS):
            raise RuntimeError(f"incomplete polarization pair: {scenario}, {position}")
        meta = metadata[(scenario, position, "a")]
        plot_pair(
            panels / f"{scenario}_{position}_LOCAL_PTE_SOURCE.png",
            scenario,
            position,
            float(meta["beam_x_b_um"]),
            float(meta["beam_y_a_um"]),
            pair,
        )
    plot_ratios(report_dir / "LOCAL_PTE_SOURCE_EA_OVER_EB.png", paired)

    write_csv(report_dir / "device_a_local_pte_source_cases.csv", cases)
    write_csv(report_dir / "device_a_local_pte_source_polarization_ratios.csv", paired)
    result = {
        "status": "COMPLETED_WEIGHTING_FREE_LOCAL_PTE_SOURCE_EXTRACTION",
        "source_report": str(summary_path),
        "coordinate_contract": "Lumerical x=crystal b; Lumerical y=crystal a",
        "definition": {
            "Jloc_b": "-sigma_b*S_b*dT/db",
            "Jloc_a": "-sigma_a*S_a*dT/da",
            "Jloc_magnitude": "sqrt(Jloc_b^2 + Jloc_a^2)",
            "sigma_b_S_m": SIGMA_B_S_M,
            "sigma_a_S_m": SIGMA_A_S_M,
            "S_b_V_K": SEEBECK_B_V_K,
            "S_a_V_K": SEEBECK_A_V_K,
            "temperature_field": "dz-weighted TaIrTe4 thickness average",
            "gradient_stencil": "strict centered +/-b,+/-a; any missing neighbor is NaN",
            "weighting_potential_used": False,
            "electrochemical_potential_solved": False,
            "terminal_current_integration_performed": False,
        },
        "illuminated_edge_roi": {
            "inside_flake_boundary_band_m": ILLUMINATED_EDGE_BAND_M,
            "within_beam_center_radius_m": BEAM_RADIUS_M,
            "interpretation": "robust local edge diagnostic; not a terminal-current integral",
        },
        "cases": cases,
        "paired_polarization_ratios": paired,
        "provenance": {
            "input_manifest": str(manifest_path),
            "input_manifest_sha256": sha256(manifest_path),
            "raw_thermal_artifacts": references,
            "new_FDTD_runs": 0,
            "new_thermal_solves": 0,
        },
    }
    (report_dir / "device_a_local_pte_source_summary.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )

    edge_rows = [r for r in paired if r["position"].startswith("edge_")]
    table = "\n".join(
        "| {scenario} | {position} | {full_maximum_Ea_A_m2:.6e} | "
        "{full_maximum_Eb_A_m2:.6e} | {full_maximum_Ea_over_Eb:.6f} | "
        "{edge_p99_Ea_A_m2:.6e} | {edge_p99_Eb_A_m2:.6e} | "
        "{edge_p99_Ea_over_Eb:.6f} |".format(**r)
        for r in edge_rows
    )
    report = f"""# Device-A weighting-free local PTE source

Status: `COMPLETED_WEIGHTING_FREE_LOCAL_PTE_SOURCE_EXTRACTION`

This offline extraction reuses all 36 saved temperature artifacts from the
nine-position/two-interface calculation.  It performs **zero new FDTD runs**
and **zero new thermal solves**.

## Quantity calculated

The coordinate contract is Lumerical `x = crystal b`, `y = crystal a`:

\\[
J_{{\\mathrm{{loc}},b}}=-\\sigma_bS_b\\,\\partial_bT,\\qquad
J_{{\\mathrm{{loc}},a}}=-\\sigma_aS_a\\,\\partial_aT.
\\]

The calculation uses `sigma_b={SIGMA_B_S_M:.6g} S/m`,
`S_b={SEEBECK_B_V_K:.6g} V/K`, `sigma_a={SIGMA_A_S_M:.6g} S/m`, and
`S_a={SEEBECK_A_V_K:.6g} V/K`.  `T` is the dz-weighted TaIrTe4
thickness-average temperature.  Both derivatives require all four
`+/-b,+/-a` TaIrTe4 neighbours; every incomplete stencil is `NaN`.

No weighting potential, no `Jloc dot grad(psi)`, and no area/volume terminal
current integration is used.  These maps have units of A/m2 and are local PTE
source-density diagnostics, not amperes measured at a remote electrode.

## Off-axis edge results

Ratios below are polarization-indexed:
`|Jloc(E parallel a)| / |Jloc(E parallel b)|`.  They are not the component
ratio `|Jloc,a|/|Jloc,b|` within one illumination case.

| interface | beam position | full max Ea (A/m2) | full max Eb (A/m2) | max Ea/Eb | edge P99 Ea (A/m2) | edge P99 Eb (A/m2) | P99 Ea/Eb |
|---|---|---:|---:|---:|---:|---:|---:|
{table}

The illuminated-edge diagnostic contains strict-valid cells within 1 um of
the flake boundary and within 8.75 um of the saved beam center.  Maximum and
P99 are both retained because a single-cell maximum is not a robust spatial
metric.

![Weighting-free local source polarization ratios](LOCAL_PTE_SOURCE_EA_OVER_EB.png)

## Spatial maps

Every scenario/position panel uses the same color limit for `E parallel a`
and `E parallel b`.  The green plus is the saved beam center and cyan is the
flake boundary.  The white/blank one-cell rim is intentional strict-stencil
masking, not zero current.

Spatial panels are in [`local_pte_source_case_panels/`](local_pte_source_case_panels/).

## Machine-readable outputs

- [`device_a_local_pte_source_summary.json`](device_a_local_pte_source_summary.json)
- [`device_a_local_pte_source_cases.csv`](device_a_local_pte_source_cases.csv)
- [`device_a_local_pte_source_polarization_ratios.csv`](device_a_local_pte_source_polarization_ratios.csv)
- [`LOCAL_PTE_SOURCE_RAW_REFERENCE_MANIFEST.json`](LOCAL_PTE_SOURCE_RAW_REFERENCE_MANIFEST.json)

## Scope warning for Figure 3J

This extraction supplies the weighting-free `Jloc=-sigma S grad(T)` requested
for diagnosis.  It does not relabel the earlier `total_current_A` values: those
remain Shockley-Ramo terminal currents.  The paper's Figure 3J compares
measured off-axis SPCM current ratios against a calculated temperature-gradient
trend, so this local-source result and the terminal-current result must remain
separate columns.
"""
    (report_dir / "DEVICE_A_LOCAL_PTE_SOURCE_REPORT.md").write_text(report)
    reference_manifest = {
        "status": result["status"],
        "analysis_command": (
            "/home/eidl/miniconda3/envs/EIDL-Lumapi/bin/python "
            "photothermal_pte/validation/paper_ir_sanity/analyze_device_a_local_pte_source.py"
        ),
        "raw_inputs_are_referenced_not_modified": True,
        "source_manifest_path": str(manifest_path),
        "source_manifest_sha256": result["provenance"]["input_manifest_sha256"],
        "raw_thermal_artifacts": references,
    }
    (report_dir / "LOCAL_PTE_SOURCE_RAW_REFERENCE_MANIFEST.json").write_text(
        json.dumps(reference_manifest, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
