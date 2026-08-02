#!/usr/bin/env python3
"""Dual-weighting re-integration of the Fig.-3I edge line scan.

For every scan thermal artifact (2 polarizations x 6 beam positions),
recompute the Shockley-Ramo terminal current under BOTH weighting
models using the stored temperature fields (no GPU, no thermal solve):

* isotropic Laplace psi (paper SI Eq. S7) - the primary comparator;
  cross-checked bit-consistently against the artifact's stored current;
* anisotropic div(sigma grad psi) = 0 with the published
  sigma_b/sigma_a = 1.10e5/4.91e5 S/m.

The edge-lobe extremum (position and value) is located independently
per polarization AND per weighting, because the weighting model can
move the extremum position, not just its value.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
V2_SPEC = importlib.util.spec_from_file_location(
    "offline_v2", HERE / "run_device_a_offline_sensitivity_v2.py"
)
offline_v2 = importlib.util.module_from_spec(V2_SPEC)
V2_SPEC.loader.exec_module(offline_v2)
base = offline_v2.base

PAPER_RATIO = 0.8365896980461811
PAPER_RATIO_UNCERTAINTY = 0.00852575488454707
POSITIONS = {
    "sm1p5": -1.5,
    "s0": 0.0,
    "s1": 1.0,
    "s2": 2.0,
    "s3": 3.0,
    "s5": 5.0,
}


def scan_frame_shift_um(payload: dict[str, Any], half_source_um: float) -> np.ndarray:
    top = np.asarray(payload["top_metal_polygon_code_um"], float)
    bottom = np.asarray(payload["bottom_metal_polygon_code_um"], float)
    beam = np.asarray(payload["pre_registered_beam_center_code_um"], float)
    all_metal = np.vstack((top, bottom))
    occupied_min = np.minimum(beam - half_source_um, np.min(all_metal, axis=0))
    occupied_max = np.maximum(beam + half_source_um, np.max(all_metal, axis=0))
    return -0.5 * (occupied_min + occupied_max)


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path(
            "/data/seunghyun/tairte4/artifacts/paper_ir_device_a_end_to_end"
        ),
    )
    parser.add_argument("--stamp", default="20260801")
    parser.add_argument(
        "--geometry-contract-json",
        type=Path,
        default=Path(
            "/home/seunghyun/tairte4/pte_inverse_design_adfd/photothermal_pte/"
            "reports/paper_ir_device_a_end_to_end/"
            "device_a_geometry_digitization.json"
        ),
    )
    parser.add_argument(
        "--source-span-um",
        type=float,
        default=40.0,
        help="span of the scan optical contract (fixes the frame shift)",
    )
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    payload = json.loads(args.geometry_contract_json.read_text())
    shift = scan_frame_shift_um(payload, 0.5 * args.source_span_um)
    top_segment = (
        np.asarray(payload["top_electrical_contact_segment_code_um"], float)
        + shift
    )
    bottom_segment = (
        np.asarray(
            payload["bottom_electrical_contact_segment_code_um"], float
        )
        + shift
    )
    base.TOP_CONTACT_SEGMENT_UM = top_segment
    base.BOTTOM_CONTACT_SEGMENT_UM = bottom_segment
    base.FLAKE_VERTICES_UM = (
        np.asarray(payload["flake_vertices_code_um"], float) + shift
    )

    rows: list[dict[str, Any]] = []
    reference_grid: dict[str, np.ndarray] | None = None
    sigma_solution = None
    for pol in ("a", "b"):
        for label, s_um in POSITIONS.items():
            directory = (
                args.artifact_root / f"scan40_thermal_{pol}_{label}_{args.stamp}"
            )
            npz_path = directory / "thermal_pte_fields.npz"
            summary_path = directory / "summary.json"
            if not npz_path.is_file() or not summary_path.is_file():
                rows.append(
                    {"polarization": pol, "label": label, "s_um": s_um,
                     "status": "MISSING"}
                )
                continue
            data = np.load(npz_path)
            summary = json.loads(summary_path.read_text())
            x_edges = np.asarray(data["x_edges_m"], float)
            y_edges = np.asarray(data["y_edges_m"], float)
            z_edges = np.asarray(data["z_edges_m"], float)
            flake_mask = np.asarray(data["flake_mask"], bool)
            footprint = np.any(flake_mask, axis=2)
            geometry = SimpleNamespace(
                x_edges_m=x_edges,
                y_edges_m=y_edges,
                z_edges_m=z_edges,
                flake_mask=flake_mask,
            )
            if reference_grid is None:
                reference_grid = {
                    "x": x_edges, "y": y_edges, "z": z_edges,
                    "footprint": footprint,
                }
                laplace_psi, laplace_gpx, laplace_gpy, _ = (
                    base.solve_weighting_potential(x_edges, y_edges, footprint)
                )
                sigma_psi, sigma_gpx, sigma_gpy, sigma_meta = (
                    offline_v2.solve_weighting_potential_conductivity(
                        x_edges,
                        y_edges,
                        footprint,
                        offline_v2.SIGMA_LAB_S_M,
                        top_segment,
                        bottom_segment,
                    )
                )
                stored_psi = np.asarray(data["weighting_potential"], float)
                laplace_reproduction = float(
                    np.nanmax(np.abs(laplace_psi - stored_psi))
                )
            else:
                for name, expected in (
                    ("x_edges_m", reference_grid["x"]),
                    ("y_edges_m", reference_grid["y"]),
                    ("z_edges_m", reference_grid["z"]),
                ):
                    if not np.array_equal(
                        np.asarray(data[name], float), expected
                    ):
                        raise RuntimeError(
                            f"{directory.name}: thermal grid differs from "
                            "the reference case - dual weighting reuse is "
                            "invalid"
                        )
            temperature = np.asarray(data["temperature_rise_K"], float)
            current_laplace, _ = base.pte_current(
                temperature, geometry, laplace_gpx, laplace_gpy
            )
            current_sigma, _ = base.pte_current(
                temperature, geometry, sigma_gpx, sigma_gpy
            )
            stored_current = float(summary["PTE_current_A_at_285uW_incident"])
            rows.append(
                {
                    "polarization": pol,
                    "label": label,
                    "s_um": s_um,
                    "status": "OK",
                    "I_laplace_A": float(current_laplace),
                    "I_sigma_A": float(current_sigma),
                    "I_stored_A": stored_current,
                    "laplace_vs_stored_rel": float(
                        abs(current_laplace - stored_current)
                        / max(abs(stored_current), np.finfo(float).tiny)
                    ),
                }
            )

    usable = [r for r in rows if r.get("status") == "OK"]
    profiles: dict[str, dict[str, dict[float, float]]] = {
        "laplace": {"a": {}, "b": {}},
        "sigma": {"a": {}, "b": {}},
    }
    for row in usable:
        profiles["laplace"][row["polarization"]][row["s_um"]] = row[
            "I_laplace_A"
        ]
        profiles["sigma"][row["polarization"]][row["s_um"]] = row["I_sigma_A"]

    verdict: dict[str, Any] = {}
    for weighting in ("laplace", "sigma"):
        entry: dict[str, Any] = {}
        table_a = profiles[weighting]["a"]
        table_b = profiles[weighting]["b"]
        common = sorted(set(table_a) & set(table_b))
        # PRIMARY comparator: the same-position ratio profile
        # r(s) = |I_a(s)| / |I_b(s)| at every scanned position, quoted
        # at the common edge-lobe peak (argmax of the summed |I_a|+|I_b|).
        # Comparing per-polarization extrema at DIFFERENT positions would
        # conflate position dependence with polarization dependence.
        if common:
            entry["pointwise_ratio_by_s_um"] = {
                f"{s:g}": abs(table_a[s]) / abs(table_b[s])
                for s in common
            }
            summed = np.asarray(
                [abs(table_a[s]) + abs(table_b[s]) for s in common]
            )
            s_peak = common[int(np.argmax(summed))]
            entry["common_peak_s_um"] = float(s_peak)
            entry["same_position_ratio_at_common_peak"] = abs(
                table_a[s_peak]
            ) / abs(table_b[s_peak])
        # SECONDARY reference only (positions reported): each
        # polarization's own |I| extremum.
        for pol, table in (("a", table_a), ("b", table_b)):
            if table:
                s_values = np.asarray(sorted(table))
                currents = np.asarray([table[s] for s in s_values])
                index = int(np.argmax(np.abs(currents)))
                entry.setdefault("per_polarization_extrema", {})[pol] = {
                    "s_um": float(s_values[index]),
                    "I_A": float(currents[index]),
                }
        extrema = entry.get("per_polarization_extrema", {})
        if "a" in extrema and "b" in extrema:
            entry["secondary_extremum_abs_Ia_over_abs_Ib"] = abs(
                extrema["a"]["I_A"]
            ) / abs(extrema["b"]["I_A"])
        verdict[weighting] = entry

    result = {
        "status": "DUAL_WEIGHTING_EDGE_SCAN",
        "paper_ratio": PAPER_RATIO,
        "paper_ratio_uncertainty": PAPER_RATIO_UNCERTAINTY,
        "frame_shift_um": shift,
        "laplace_psi_reproduction_max_abs_diff_first_case": (
            laplace_reproduction if reference_grid is not None else None
        ),
        "sigma_weighting_meta": jsonable(sigma_meta)
        if reference_grid is not None
        else None,
        "rows": rows,
        "extrema": verdict,
    }
    (args.report_dir / "dual_weighting_edge_scan_summary.json").write_text(
        json.dumps(jsonable(result), indent=2) + "\n"
    )
    with (args.report_dir / "dual_weighting_edge_scan.csv").open(
        "w", newline=""
    ) as stream:
        fieldnames = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharex=True)
    for axis, weighting, title in (
        (axes[0], "laplace", "Laplace psi (paper Eq. S7, primary)"),
        (axes[1], "sigma", "sigma-weighted psi (div sigma grad psi = 0)"),
    ):
        for pol, color in (("a", "#cc3311"), ("b", "#4477aa")):
            table = profiles[weighting][pol]
            if table:
                s_values = sorted(table)
                axis.plot(
                    s_values,
                    [table[s] * 1e9 for s in s_values],
                    "o-",
                    color=color,
                    label=f"E||{pol}",
                )
        ratio = verdict[weighting].get("same_position_ratio_at_common_peak")
        axis.set_title(
            f"{title}\nsame-position r(s_peak) = "
            f"{ratio:.3f}" if ratio else title
        )
        axis.axvline(0.0, color="k", lw=1.0, ls=":")
        axis.set_xlabel("s (um, + into flake)")
        axis.grid(alpha=0.25)
        axis.legend()
    axes[0].set_ylabel("terminal current at 285 uW (nA)")
    figure.tight_layout()
    figure.savefig(args.report_dir / "DUAL_WEIGHTING_EDGE_SCAN.png", dpi=180)
    plt.close(figure)

    print(json.dumps(jsonable({"extrema": verdict}), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
