#!/usr/bin/env python3
"""Summarize fail-closed Device-A finite-Q fast-mesh comparisons.

The raw Lumerical arrays remain external.  Every spatial comparison first
restricts both cases to their common physical control volume and maps the
reference cell energies conservatively onto the candidate dual-cell grid.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from photothermal_pte.validation.paper_ir_sanity.summarize_w12_edge_a_xy_refinement import (
    bounded_dual_cells,
    overlap_fraction,
    remap_energy,
    volume,
)


GATE = 5.0e-3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-5um", type=Path, required=True)
    parser.add_argument("--case-7um", type=Path, required=True)
    parser.add_argument("--case-9um", type=Path, required=True)
    parser.add_argument("--case-12um", type=Path, required=True)
    parser.add_argument("--case-9um-intermediate", type=Path, required=True)
    parser.add_argument("--interrupted-case", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_case(path: Path) -> dict[str, Any]:
    result_path = path / "case_result.json"
    artifact_path = path / "finite_q_on_artifact.npz"
    if not result_path.is_file() or not artifact_path.is_file():
        raise FileNotFoundError(f"incomplete finite case: {path}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("status") != "COMPLETED":
        raise RuntimeError(f"finite case did not complete: {path}")
    return {
        "directory": path.resolve(),
        "result_path": result_path.resolve(),
        "artifact_path": artifact_path.resolve(),
        "result": result,
    }


def finite_grid_and_runtime(case: dict[str, Any]) -> dict[str, Any]:
    run = case["result"]["run_result"]
    log_path = case["directory"] / "finite_2um_optical_q_p0.log"
    if not log_path.is_file():
        log_path = Path(run["auto_shutoff"]["log_path"])
    text = log_path.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"Simulation size in gridpoints:\s*(\d+) x (\d+) x (\d+)", text)
    wall = re.search(r"Overall wall time measurements in seconds:\s*([0-9.eE+-]+)", text)
    gpu = re.search(r"Total:\s*([0-9.]+) GiB\nnumber of processors", text)
    if match is None or wall is None:
        raise RuntimeError(f"missing runtime audit in {log_path}")
    shape = [int(value) for value in match.groups()]
    return {
        "shape": shape,
        "gridpoints": int(np.prod(shape, dtype=np.int64)),
        "wall_time_s": float(wall.group(1)),
        "estimated_GPU_memory_GiB": None if gpu is None else float(gpu.group(1)),
        "log_path": str(log_path.resolve()),
    }


def compare(candidate: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    candidate_run = candidate["result"]["run_result"]
    reference_run = reference["result"]["run_result"]
    bounds = {}
    for axis in "xyz":
        candidate_bounds = candidate_run["native_Yee_mesh_audit"][
            "Q_quadrature_control_volume_bounds_m"
        ][axis]
        reference_bounds = reference_run["native_Yee_mesh_audit"][
            "Q_quadrature_control_volume_bounds_m"
        ][axis]
        bounds[axis] = [
            max(candidate_bounds[0], reference_bounds[0]),
            min(candidate_bounds[1], reference_bounds[1]),
        ]
    with np.load(candidate["artifact_path"], allow_pickle=False) as left, np.load(
        reference["artifact_path"], allow_pickle=False
    ) as right:
        arrays = {"candidate": left, "reference": right}
        cells = {
            name: tuple(
                bounded_dual_cells(
                    np.asarray(archive[f"{axis}_m"], float),
                    *bounds[axis],
                )
                for axis in "xyz"
            )
            for name, archive in arrays.items()
        }
        edges = {
            name: tuple(value[1] for value in values)
            for name, values in cells.items()
        }
        volumes = {name: volume(value) for name, value in edges.items()}
        operators = tuple(
            overlap_fraction(target, source)
            for target, source in zip(edges["candidate"], edges["reference"])
        )
        metrics = {}
        for component, key in (
            ("x", "Qx_W_m3"),
            ("y", "Qy_W_m3"),
            ("z", "Qz_W_m3"),
            ("total", "Q_on_W_m3"),
        ):
            candidate_q = np.asarray(
                left[key][np.ix_(*[value[0] for value in cells["candidate"]])],
                float,
            )
            reference_q = np.asarray(
                right[key][np.ix_(*[value[0] for value in cells["reference"]])],
                float,
            )
            candidate_energy = candidate_q * volumes["candidate"]
            reference_energy = reference_q * volumes["reference"]
            mapped_reference = remap_energy(reference_energy, operators)
            candidate_power = float(np.sum(candidate_energy))
            reference_power = float(np.sum(reference_energy))
            mapped_power = float(np.sum(mapped_reference))
            candidate_normalized = candidate_energy / candidate_power
            reference_normalized = mapped_reference / mapped_power
            candidate_lateral = np.sum(candidate_energy, axis=2) / candidate_power
            reference_lateral = np.sum(mapped_reference, axis=2) / mapped_power
            candidate_depth = np.sum(candidate_energy, axis=(0, 1)) / candidate_power
            reference_depth = np.sum(mapped_reference, axis=(0, 1)) / mapped_power
            metrics[component] = {
                "candidate_power_W": candidate_power,
                "reference_power_W": reference_power,
                "relative_power_change": abs(candidate_power - reference_power)
                / abs(reference_power),
                "conservative_remap_power_error": abs(mapped_power - reference_power)
                / abs(reference_power),
                "equal_power_full_3D_NRMSE": float(
                    np.linalg.norm(candidate_normalized - reference_normalized)
                    / np.linalg.norm(reference_normalized)
                ),
                "equal_power_lateral_NRMSE": float(
                    np.linalg.norm(candidate_lateral - reference_lateral)
                    / np.linalg.norm(reference_lateral)
                ),
                "equal_power_depth_NRMSE": float(
                    np.linalg.norm(candidate_depth - reference_depth)
                    / np.linalg.norm(reference_depth)
                ),
                "correlation": float(
                    np.corrcoef(
                        candidate_normalized.reshape(-1),
                        reference_normalized.reshape(-1),
                    )[0, 1]
                ),
            }
    total = metrics["total"]
    gates = {
        "total_power_change_lt_0p5_percent": total["relative_power_change"] < GATE,
        "lateral_Q_NRMSE_lt_0p5_percent": total["equal_power_lateral_NRMSE"] < GATE,
        "full_3D_Q_NRMSE_lt_0p5_percent": total["equal_power_full_3D_NRMSE"] < GATE,
        "depth_Q_NRMSE_lt_0p5_percent": total["equal_power_depth_NRMSE"] < GATE,
        "conservative_remap_error_lt_1e_minus_12": total[
            "conservative_remap_power_error"
        ] < 1.0e-12,
        "both_closure_lt_0p5_percent": max(
            candidate_run["six_face_relative_closure"],
            reference_run["six_face_relative_closure"],
        ) < GATE,
        "both_auto_shutoff_lt_1e_minus_5": max(
            candidate_run["auto_shutoff"]["final_value"],
            reference_run["auto_shutoff"]["final_value"],
        ) <= 1.0e-5,
    }
    return {
        "candidate": str(candidate["directory"]),
        "reference": str(reference["directory"]),
        "common_control_volume_bounds_m": bounds,
        "component_metrics": metrics,
        "candidate_closure": candidate_run["six_face_relative_closure"],
        "reference_closure": reference_run["six_face_relative_closure"],
        "gates": gates,
        "all_gates_pass": all(gates.values()),
    }


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = {
        "50nm_half_span_5um": load_case(args.case_5um),
        "50nm_half_span_7um": load_case(args.case_7um),
        "50nm_half_span_9um": load_case(args.case_9um),
        "50nm_half_span_12um": load_case(args.case_12um),
        "50nm_half_span_9um_100nm_to_12um": load_case(
            args.case_9um_intermediate
        ),
    }
    pairs = {
        "5um_vs_7um": compare(cases["50nm_half_span_5um"], cases["50nm_half_span_7um"]),
        "7um_vs_9um": compare(cases["50nm_half_span_7um"], cases["50nm_half_span_9um"]),
        "9um_vs_12um": compare(cases["50nm_half_span_9um"], cases["50nm_half_span_12um"]),
        "9um_plus_100nm_transition_vs_12um": compare(
            cases["50nm_half_span_9um_100nm_to_12um"],
            cases["50nm_half_span_12um"],
        ),
    }
    case_summary = {}
    for name, case in cases.items():
        result = case["result"]
        run = result["run_result"]
        mesh = result["pre_run_contract"]["mesh"]
        case_summary[name] = {
            "directory": str(case["directory"]),
            "P_Q_W": run["P_Q_W"],
            "P_six_W": run["P_six_face_W"],
            "six_face_relative_closure": run["six_face_relative_closure"],
            "auto_shutoff": run["auto_shutoff"]["final_value"],
            "component_power_W": run["component_power_W"],
            "flake_dz_nm": result["flake_dz_nm"],
            "outer_xy_nm": mesh["outer_local_xy_mesh_m"] * 1e9,
            "fine_xy_nm": mesh["local_xy_mesh_m"] * 1e9,
            "fine_half_span_um": mesh["refinement_half_span_m"] * 1e6,
            "intermediate_xy_nm": None
            if mesh["intermediate_local_xy_mesh_m"] is None
            else mesh["intermediate_local_xy_mesh_m"] * 1e9,
            "intermediate_half_span_um": None
            if mesh["intermediate_half_span_m"] is None
            else mesh["intermediate_half_span_m"] * 1e6,
            "finite_solver": finite_grid_and_runtime(case),
        }
    status = "FAILED_FAST_DEVICE_A_SPATIAL_Q_CONVERGENCE"
    summary = {
        "status": status,
        "classification": (
            "one-polarization finite Device-A optical mesh diagnostic; not a "
            "promoted production Q and not a thermal/PTE result"
        ),
        "fixed_contract": {
            "wavelength_um": 11.0,
            "waist_um": 8.75,
            "domain_um": 60.0,
            "source_span_um": 50.0,
            "polarization": "E parallel a",
            "outer_xy_nm": 200.0,
            "fine_xy_nm": 50.0,
            "flake_dz_nm": 10.0,
            "mesh_accuracy": 3,
            "six_boundaries": "PML",
            "substrate": "Lumerical v261 Palik SiO2/Si at 11 um",
        },
        "cases": case_summary,
        "comparisons": pairs,
        "interpretation": {
            "total_power_can_hide_spatial_nonconvergence": True,
            "no_fast_candidate_promoted": True,
            "thermal_PTE_adjoint_optimization_run": False,
            "dz10_vs_dz5_not_yet_isolated": True,
        },
    }
    summary_path = args.output_dir / "device_a_fast_finite_mesh_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    csv_path = args.output_dir / "device_a_fast_finite_mesh_cases.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "comparison",
                "power_change_percent",
                "lateral_NRMSE_percent",
                "full_3D_NRMSE_percent",
                "depth_NRMSE_percent",
                "candidate_runtime_s",
                "reference_runtime_s",
                "all_gates_pass",
            ],
        )
        writer.writeheader()
        for name, comparison in pairs.items():
            metric = comparison["component_metrics"]["total"]
            candidate_name = next(
                key for key, value in case_summary.items()
                if value["directory"] == comparison["candidate"]
            )
            reference_name = next(
                key for key, value in case_summary.items()
                if value["directory"] == comparison["reference"]
            )
            writer.writerow(
                {
                    "comparison": name,
                    "power_change_percent": 100 * metric["relative_power_change"],
                    "lateral_NRMSE_percent": 100 * metric["equal_power_lateral_NRMSE"],
                    "full_3D_NRMSE_percent": 100 * metric["equal_power_full_3D_NRMSE"],
                    "depth_NRMSE_percent": 100 * metric["equal_power_depth_NRMSE"],
                    "candidate_runtime_s": case_summary[candidate_name]["finite_solver"]["wall_time_s"],
                    "reference_runtime_s": case_summary[reference_name]["finite_solver"]["wall_time_s"],
                    "all_gates_pass": comparison["all_gates_pass"],
                }
            )

    labels = list(pairs)
    x = np.arange(len(labels))
    lateral = [100 * pairs[name]["component_metrics"]["total"]["equal_power_lateral_NRMSE"] for name in labels]
    full = [100 * pairs[name]["component_metrics"]["total"]["equal_power_full_3D_NRMSE"] for name in labels]
    power = [100 * pairs[name]["component_metrics"]["total"]["relative_power_change"] for name in labels]
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    width = 0.25
    axes[0].bar(x - width, power, width, label="total power")
    axes[0].bar(x, lateral, width, label="lateral Q")
    axes[0].bar(x + width, full, width, label="full 3D Q")
    axes[0].axhline(0.5, color="black", linestyle="--", label="0.5% gate")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("relative metric (%)")
    axes[0].set_xticks(x, labels, rotation=20, ha="right")
    axes[0].legend()
    names = list(case_summary)
    display_names = {
        "50nm_half_span_5um": "50 nm within +/-5 um",
        "50nm_half_span_7um": "50 nm within +/-7 um",
        "50nm_half_span_9um": "50 nm within +/-9 um",
        "50nm_half_span_12um": "50 nm within +/-12 um",
        "50nm_half_span_9um_100nm_to_12um": (
            "50 nm +/-9 um; 100 nm to +/-12 um"
        ),
    }
    runtimes = [case_summary[name]["finite_solver"]["wall_time_s"] / 60 for name in names]
    points = [case_summary[name]["finite_solver"]["gridpoints"] / 1e6 for name in names]
    axes[1].scatter(points, runtimes, s=70)
    for name, px, py in zip(names, points, runtimes):
        axes[1].annotate(
            display_names[name],
            (px, py),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )
    axes[1].set_xlabel("finite FDTD grid points (million)")
    axes[1].set_ylabel("solver wall time (min)")
    axes[1].grid(alpha=0.25)
    figure.suptitle("Device-A fast finite-Q mesh: speed versus spatial convergence")
    figure.tight_layout()
    figure.savefig(args.output_dir / "device_a_fast_finite_mesh_convergence.png", dpi=180)
    plt.close(figure)

    manifest_entries = []
    for name, case in cases.items():
        for filename in (
            "case_result.json",
            "finite_q_on_artifact.npz",
            "finite_2um_optical_q.fsp",
            "finite_2um_optical_q_p0.log",
        ):
            path = case["directory"] / filename
            if path.is_file():
                manifest_entries.append(
                    {
                        "case": name,
                        "server_path": str(path),
                        "size_bytes": path.stat().st_size,
                        "sha256": sha256(path),
                        "generation_command": case["result"][
                            "generation_command"
                        ],
                        "committed_to_git": False,
                    }
                )
    if args.interrupted_case is not None:
        manifest_entries.append(
            {
                "case": "interrupted_12um_empty_precheck",
                "server_path": str(args.interrupted_case.resolve()),
                "exists": args.interrupted_case.exists(),
                "removed_after_interrupt": not args.interrupted_case.exists(),
                "committed_to_git": False,
                "promoted": False,
                "note": (
                    "interrupted after runtime proved incompatible with the "
                    "fast-test intent; removed to recover disk space"
                ),
            }
        )
    manifest = {
        "status": status,
        "raw_artifacts_committed_to_git": False,
        "artifacts": manifest_entries,
    }
    (args.output_dir / "RAW_ARTIFACT_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    final_pair = pairs["9um_plus_100nm_transition_vs_12um"]
    final_metric = final_pair["component_metrics"]["total"]
    report = f"""# Device-A fast finite optical mesh validation

Status: `{status}`

This is a one-polarization (`E || a`) finite Device-A optical mesh diagnostic.
It is not a promoted production heat source and no thermal, PTE, adjoint, or
optimization calculation was run.

## Fixed contract

- Scalar Gaussian at 11 um, explicitly assumed `w0=8.75 um`.
- 60 x 60 um lateral domain, 50 x 50 um source aperture, six PML boundaries.
- Palik SiO2/Si and paper-derived anisotropic TaIrTe4 with `epsilon_z=epsilon_b`.
- Conformal variant 1, mesh accuracy 3, TaIrTe4 `dz=10 nm`.
- Fine optical x/y mesh is 50 nm; the full Device-A/Q outer region is 200 nm.
- No clipping, smoothing, gain, global rescaling, tiling, or source deletion.

`half-span` means the half-width of the square 50-nm refinement window around
the registered beam centre.  It is not a convection coefficient.

## Result

All five GPU calculations completed with auto-shutoff <= 1e-5 and six-face
closure below 0.5%.  Total absorbed power appears converged much earlier than
the spatial source.  The most useful fast candidate used 50 nm within +/-9 um,
100 nm from 9--12 um, and 200 nm outside.  Relative to the +/-12 um all-50-nm
reference it achieved:

- total-power change: `{final_metric['relative_power_change']:.4%}`;
- depth-profile NRMSE: `{final_metric['equal_power_depth_NRMSE']:.4%}`;
- lateral-Q NRMSE: `{final_metric['equal_power_lateral_NRMSE']:.4%}`;
- full-3D-Q NRMSE: `{final_metric['equal_power_full_3D_NRMSE']:.4%}`;
- conservative-remap power error: `{final_metric['conservative_remap_power_error']:.3e}`;
- runtime: `{case_summary['50nm_half_span_9um_100nm_to_12um']['finite_solver']['wall_time_s']:.1f} s`
  versus `{case_summary['50nm_half_span_12um']['finite_solver']['wall_time_s']:.1f} s`.

The candidate saves about 20% solver time, but its lateral and full-3D Q
metrics exceed the 0.5% gate.  It is therefore not promoted.  The 5, 7, and
9 um direct 50-to-200-nm transitions also fail spatial convergence even when
their total powers are close.  Total power alone would have produced a false
pass.

## Interpretation and next minimal test

The current data do not isolate `dz=10 nm` versus `dz=5 nm`; every case in this
checkpoint uses 10 nm.  The observed failure is the x/y refinement-window
sensitivity.  A next test should keep the 50-nm illuminated region but compare
a 100-nm outer Device-A mesh against the current 200-nm outer mesh on one
polarization.  It should not proceed to thermal/PTE unless the spatial optical
gate is resolved or an explicitly approved downstream-observable gate replaces
the strict raw-Q gate.
"""
    (args.output_dir / "DEVICE_A_FAST_FINITE_MESH_REPORT.md").write_text(
        report, encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
