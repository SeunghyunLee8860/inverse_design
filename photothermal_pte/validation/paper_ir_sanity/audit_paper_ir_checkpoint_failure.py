#!/usr/bin/env python3
"""Audit the existing paper-IR checkpoints without launching a new solver.

This command is intentionally read-only with respect to raw artifacts.  It
joins the 200/100/50 nm paper-reduced thermal results, the exact-coordinate
edge comparator, and the failed GPU run evidence into compact report files.
It does not open Lumerical or run thermal, optical, PTE, adjoint, or
optimization calculations.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

import h5py
import numpy as np


STATUS = (
    "AUDITED_EXISTING_PAPER_IR_CHECKPOINTS_"
    "UNRESOLVED_ENGINE_TERMINATION_AND_EDGE_METRIC"
)
EXPECTED_HEAD = "651797f"
MESHES_NM = (200, 100, 50)
POLARIZATIONS = ("a", "b")
SOURCES = ("analytic_paper_source", "legacy_Lumerical_edge_Q")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument(
        "--audit-start-clean",
        action="store_true",
        help=(
            "record the separately observed clean worktree at audit start; "
            "use only when the audit script itself is the sole later change"
        ),
    )
    parser.add_argument(
        "--observed-fdtd-process-memory-mib",
        type=float,
        default=None,
        help="optional one-time nvidia-smi observation; not a peak",
    )
    parser.add_argument(
        "--observed-gpu-total-used-mib",
        type=float,
        default=None,
        help="optional one-time nvidia-smi observation; includes other jobs",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path: Path, role: str) -> dict[str, Any]:
    return {
        "role": role,
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def run_directory(root: Path, polarization: str, mesh_nm: int) -> Path:
    suffix = "_v2" if mesh_nm == 200 else ""
    return root / (
        f"audit_paper_reduced_{polarization}_core{mesh_nm}_"
        f"20260730{suffix}"
    )


def rel_change(value: float, reference: float) -> float:
    return abs(value - reference) / max(
        abs(value),
        abs(reference),
        np.finfo(float).tiny,
    )


def coordinate_summary(values: np.ndarray) -> dict[str, Any]:
    coordinate = np.asarray(values, float).reshape(-1)
    if coordinate.size < 2:
        return {
            "count": int(coordinate.size),
            "minimum_m": float(coordinate[0]) if coordinate.size else None,
            "maximum_m": float(coordinate[-1]) if coordinate.size else None,
            "minimum_step_m": None,
            "median_step_m": None,
            "maximum_step_m": None,
        }
    steps = np.diff(coordinate)
    return {
        "count": int(coordinate.size),
        "minimum_m": float(coordinate[0]),
        "maximum_m": float(coordinate[-1]),
        "minimum_step_m": float(np.min(steps)),
        "median_step_m": float(np.median(steps)),
        "maximum_step_m": float(np.max(steps)),
    }


def primary_robust_rows(path: Path) -> dict[tuple[str, str, int, str], dict[str, float]]:
    result: dict[tuple[str, str, int, str], dict[str, float]] = {}
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row["fit_band"] != "primary":
                continue
            key = (
                row["source"],
                row["polarization"],
                int(row["mesh_nm"]),
                row["component"],
            )
            result[key] = {
                name: float(row[name])
                for name in (
                    "maximum_abs_K_m",
                    "p99_abs_K_m",
                    "edge_strip_mean_abs_K_m",
                    "edge_integrated_abs_K",
                    "fit_residual_p99",
                )
            }
    return result


def thermal_row(
    summary: dict[str, Any],
    robust: dict[tuple[str, str, int, str], dict[str, float]],
    profile_path: Path,
    *,
    source: str,
    polarization: str,
    mesh_nm: int,
) -> dict[str, Any]:
    if source == "analytic_paper_source":
        solve = summary["analytic_offset_cases"][0]
        thermal = solve["thermal"]
        metrics = solve["straight_edge_metrics"]
    else:
        thermal = summary["remapped_Lumerical_thermal_solve"]
        metrics = thermal["straight_edge_metrics"]
    fitted_x = robust[(source, polarization, mesh_nm, "x")]
    fitted_n = robust[(source, polarization, mesh_nm, "n")]
    fixed_roi = metrics.get("fixed_24um_ROI_area_average_rise_K")
    if fixed_roi is None:
        field_name = (
            "analytic_temperature_flake_average_K"
            if source == "analytic_paper_source"
            else "remapped_Lumerical_temperature_flake_average_K"
        )
        with np.load(profile_path, allow_pickle=False) as raw:
            x_edges = np.asarray(raw["x_edges_m"], float)
            y_edges = np.asarray(raw["y_edges_m"], float)
            temperature = np.asarray(raw[field_name], float)
        x = 0.5 * (x_edges[:-1] + x_edges[1:])
        y = 0.5 * (y_edges[:-1] + y_edges[1:])
        mask = (
            (y[None, :] <= x[:, None] + 1e-15)
            & (np.abs(x[:, None]) <= 12.0e-6)
            & (np.abs(y[None, :]) <= 12.0e-6)
            & np.isfinite(temperature)
        )
        area = (
            np.diff(x_edges)[:, None]
            * np.diff(y_edges)[None, :]
        )
        fixed_roi = float(
            np.sum(temperature[mask] * area[mask])
            / np.sum(area[mask])
        )
    return {
        "source": source,
        "polarization": polarization,
        "mesh_nm": mesh_nm,
        "Tmax_rise_K": float(metrics["Tmax_rise_K"]),
        "fixed_24um_ROI_average_rise_K": float(
            fixed_roi
        ),
        "raw_max_abs_dTdx_K_m": float(metrics["max_abs_grad_T_x_K_m"]),
        "raw_max_abs_dTdn_K_m": float(
            metrics["max_abs_edge_normal_gradient_K_m"]
        ),
        "raw_p99_abs_dTdn_K_m": float(
            metrics["p99_abs_edge_normal_gradient_K_m"]
        ),
        "fitted_x_max_abs_K_m": fitted_x["maximum_abs_K_m"],
        "fitted_x_p99_abs_K_m": fitted_x["p99_abs_K_m"],
        "fitted_x_strip_mean_abs_K_m": fitted_x[
            "edge_strip_mean_abs_K_m"
        ],
        "fitted_x_integrated_abs_K": fitted_x[
            "edge_integrated_abs_K"
        ],
        "fitted_n_max_abs_K_m": fitted_n["maximum_abs_K_m"],
        "fitted_n_p99_abs_K_m": fitted_n["p99_abs_K_m"],
        "fitted_n_strip_mean_abs_K_m": fitted_n[
            "edge_strip_mean_abs_K_m"
        ],
        "fitted_n_integrated_abs_K": fitted_n[
            "edge_integrated_abs_K"
        ],
        "fit_residual_p99": max(
            fitted_x["fit_residual_p99"],
            fitted_n["fit_residual_p99"],
        ),
        "energy_balance_relative_error": float(
            thermal["energy_balance_relative_error"]
        ),
        "linear_residual_relative": float(
            thermal["linear_residual_relative"]
        ),
    }


def normalized_contract(summary: dict[str, Any]) -> dict[str, Any]:
    geometry = dict(summary["geometry"])
    geometry.pop("grid_shape", None)
    geometry.pop("core_step_nm", None)
    source = dict(summary["source_contract"])
    optical = source.get("optical_constants", {})
    return {
        "geometry_except_intended_mesh": geometry,
        "thermal_model_contract": summary["thermal_model_contract"],
        "source_fixed_values": {
            "incident_power_W": source["incident_power_W"],
            "wavelength_m": source["wavelength_m"],
            "waist_radius_m": source["waist_radius_m"],
            "TMM_full_plane_absorption": source[
                "TMM_full_plane_absorption"
            ],
            "normalization": source["normalization"],
            "optical_constants": optical,
        },
        "mapping_operations": summary["saved_Lumerical_mapping"][
            "mapping_operations"
        ],
        "optical_artifact_sha256": summary["saved_Lumerical_mapping"][
            "optical_artifact_sha256"
        ],
    }


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=args.repository,
        text=True,
    ).strip()
    audit_basis = subprocess.check_output(
        ["git", "rev-parse", EXPECTED_HEAD],
        cwd=args.repository,
        text=True,
    ).strip()
    current_descends_from_basis = (
        subprocess.run(
            [
                "git",
                "merge-base",
                "--is-ancestor",
                audit_basis,
                head,
            ],
            cwd=args.repository,
            check=False,
        ).returncode
        == 0
    )
    porcelain = subprocess.check_output(
        ["git", "status", "--porcelain=v1"],
        cwd=args.repository,
        text=True,
    )
    current_porcelain = [line for line in porcelain.splitlines() if line]
    audit_start_porcelain = [] if args.audit_start_clean else current_porcelain
    robust_dir = args.artifact_root / "robust_edge_gradient_20260730"
    robust = primary_robust_rows(
        robust_dir / "robust_edge_gradient_cases.csv"
    )
    rows: list[dict[str, Any]] = []
    summaries: dict[tuple[str, int], dict[str, Any]] = {}
    input_records: list[dict[str, Any]] = []
    contracts_by_pol: dict[str, list[dict[str, Any]]] = {
        polarization: [] for polarization in POLARIZATIONS
    }
    for polarization in POLARIZATIONS:
        for mesh_nm in MESHES_NM:
            directory = run_directory(
                args.artifact_root,
                polarization,
                mesh_nm,
            )
            summary_path = directory / "summary.json"
            profile_path = directory / "straight_edge_profiles.npz"
            summary = load_json(summary_path)
            summaries[(polarization, mesh_nm)] = summary
            contracts_by_pol[polarization].append(
                normalized_contract(summary)
            )
            input_records.extend(
                (
                    record(summary_path, "paper_reduced_summary"),
                    record(profile_path, "paper_reduced_fields"),
                )
            )
            for source in SOURCES:
                rows.append(
                    thermal_row(
                        summary,
                        robust,
                        profile_path,
                        source=source,
                        polarization=polarization,
                        mesh_nm=mesh_nm,
                    )
                )

    table_path = args.output_dir / "paper_reduced_thermal_audit_cases.csv"
    with table_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    row_index = {
        (row["source"], row["polarization"], row["mesh_nm"]): row
        for row in rows
    }
    convergence_metrics = (
        "Tmax_rise_K",
        "fixed_24um_ROI_average_rise_K",
        "raw_max_abs_dTdx_K_m",
        "raw_max_abs_dTdn_K_m",
        "raw_p99_abs_dTdn_K_m",
        "fitted_x_max_abs_K_m",
        "fitted_x_p99_abs_K_m",
        "fitted_x_strip_mean_abs_K_m",
        "fitted_x_integrated_abs_K",
        "fitted_n_max_abs_K_m",
        "fitted_n_p99_abs_K_m",
        "fitted_n_strip_mean_abs_K_m",
        "fitted_n_integrated_abs_K",
    )
    convergence: dict[str, Any] = {}
    for source in SOURCES:
        for polarization in POLARIZATIONS:
            coarse = row_index[(source, polarization, 100)]
            fine = row_index[(source, polarization, 50)]
            key = f"{source}_{polarization}_100_to_50"
            values = {
                metric: rel_change(coarse[metric], fine[metric])
                for metric in convergence_metrics
            }
            convergence[key] = {
                "relative_changes": values,
                "global_T_and_fixed_ROI_lt_1pct": (
                    values["Tmax_rise_K"] < 0.01
                    and values["fixed_24um_ROI_average_rise_K"] < 0.01
                ),
                "fitted_x_all_lt_1pct": all(
                    values[metric] < 0.01
                    for metric in (
                        "fitted_x_max_abs_K_m",
                        "fitted_x_p99_abs_K_m",
                        "fitted_x_strip_mean_abs_K_m",
                        "fitted_x_integrated_abs_K",
                    )
                ),
                "fitted_n_all_lt_1pct": all(
                    values[metric] < 0.01
                    for metric in (
                        "fitted_n_max_abs_K_m",
                        "fitted_n_p99_abs_K_m",
                        "fitted_n_strip_mean_abs_K_m",
                        "fitted_n_integrated_abs_K",
                    )
                ),
                "raw_maxima_role": "diagnostic_only",
            }

    retry_dir = (
        args.artifact_root
        / "straight45_a_w6p5_dz10_L48_cclosure_gpu2_retry4_20260730"
    )
    engine_log = retry_dir / "finite_2um_optical_q_p0.log"
    case_result_path = retry_dir / "case_result.json"
    partial_h5 = (
        retry_dir
        / "finite_2um_optical_q"
        / "finite_2um_optical_q_output.h5"
    )
    log_text = engine_log.read_text(errors="replace")
    case_result = load_json(case_result_path)
    progress = [
        float(value)
        for value in re.findall(
            r"([0-9]+(?:\.[0-9]+)?)% complete",
            log_text,
        )
    ]
    grid_match = re.search(
        r"Simulation size in gridpoints: (\d+) x (\d+) x (\d+)",
        log_text,
    )
    if grid_match is None:
        raise RuntimeError("failed GPU log has no grid-size readback")
    grid_shape = tuple(int(value) for value in grid_match.groups())
    estimates: dict[str, float] = {}
    for key, pattern in (
        (
            "initial_total_GiB",
            r"Maximum memory estimate for GPU:.*?Total:\s+([0-9.]+) GiB",
        ),
        (
            "precise_total_GiB",
            r"Estimated memory use on GPU 2 \(precise\):.*?"
            r"Total:\s+([0-9.]+) GiB",
        ),
        (
            "peak_CPU_after_meshing_GiB",
            r"Peak CPU memory used by end of meshing:\s+([0-9.]+) GiB",
        ),
        (
            "host_available_GiB",
            r"host available memory \(GiB\)\s+([0-9.]+)",
        ),
    ):
        match = re.search(pattern, log_text, re.S)
        estimates[key] = float(match.group(1)) if match else float("nan")

    coordinate_rows: list[dict[str, Any]] = []
    with h5py.File(partial_h5, "r") as handle:
        for monitor in sorted(handle):
            for axis in "xyz":
                if axis not in handle[monitor]:
                    continue
                # External-engine HDF5 monitor coordinates are stored in um,
                # unlike lumapi getdata/getresult arrays, which return SI.
                values = np.asarray(handle[monitor][axis], float) * 1.0e-6
                coordinate_rows.append(
                    {
                        "partial_monitor_id": monitor,
                        "axis": axis,
                        **coordinate_summary(values),
                        "semantic_monitor_name": (
                            "unavailable_from_incomplete_external-engine_HDF5"
                        ),
                        "coordinate_role": (
                            "monitor_sampling_grid_not_native_solver_mesh"
                        ),
                    }
                )
    coordinate_path = (
        args.output_dir / "partial_h5_monitor_coordinate_audit.csv"
    )
    with coordinate_path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(coordinate_rows[0]),
        )
        writer.writeheader()
        writer.writerows(coordinate_rows)

    contracts_match = {
        polarization: all(
            contract == contracts_by_pol[polarization][0]
            for contract in contracts_by_pol[polarization][1:]
        )
        for polarization in POLARIZATIONS
    }
    kernel_evidence = {
        "wall_time_UTC": "2026-07-30 07:53:00",
        "process": "lum::RemoteMess[1829144]",
        "binary": "fdtd-solutions-app",
        "event": "segfault at address 8",
        "contemporaneous_other_API_host_segfaults": [
            "lum::RemoteMess[1790129] at 07:52:56",
            "lum::RemoteMess[1780725] at 07:52:56",
        ],
        "engine_pid_exit_code": None,
        "interpretation": (
            "an API-host remote-messenger segfault is confirmed, but the "
            "external fdtd-engine exit code was not captured; engine "
            "termination cause remains unresolved"
        ),
    }
    failure = {
        "classification": "UNRESOLVED_ENGINE_TERMINATION",
        "confirmed_secondary_event": (
            "FDTD_SOLUTIONS_REMOTE_MESSENGER_SEGFAULT"
        ),
        "license": {
            "acquired_for_retry4": (
                "Ansys Lumerical 2026 R1.2 FDTD Solver" in log_text
                and "Beginning initialization of 3D Simulation" in log_text
            ),
            "earlier_attempts_failed_license": 3,
            "release_directly_traceable_to_engine_pid": False,
        },
        "phase": {
            "meshing_completed": "Meshing complete" in log_text,
            "time_stepping_started": (
                "Starting 39362 total iterations" in log_text
            ),
            "maximum_logged_progress_percent": max(progress),
            "normal_solver_completion": False,
            "post_run_Q_extraction_started": False,
        },
        "grid_shape": list(grid_shape),
        "total_gridpoints": int(np.prod(grid_shape)),
        "nominal_lateral_average_step_m": (
            48.0e-6 / max(grid_shape[0] - 1, 1)
        ),
        "memory": {
            **estimates,
            "GPU_capacity_MiB": 49140,
            "GPU_free_at_precise_initialization_MiB": 48640,
            "observed_fdtd_process_memory_MiB": (
                args.observed_fdtd_process_memory_mib
            ),
            "observed_GPU_total_used_MiB": (
                args.observed_gpu_total_used_mib
            ),
            "observation_is_peak": False,
            "GPU_total_used_includes_other_processes": True,
        },
        "OOM_evidence_at_failure_time": False,
        "GPU_reset_or_Xid_evidence_at_failure_time": False,
        "timeout_evidence": False,
        "solver_exit_code": None,
        "native_solver_coordinate_readback": (
            "unavailable because the session/API host failed before "
            "post-run getresult(FDTD,x/y/z)"
        ),
        "partial_HDF5_coordinate_role": (
            "monitor sampling coordinates only; never promoted as native mesh"
        ),
        "kernel_evidence": kernel_evidence,
        "exception_type": case_result.get("exception_type"),
        "exception": case_result.get("exception"),
    }
    selected_records = [
        *input_records,
        record(
            robust_dir / "robust_edge_gradient_cases.csv",
            "robust_comparator_table",
        ),
        record(engine_log, "failed_GPU_engine_log"),
        record(case_result_path, "failed_GPU_case_result"),
        record(partial_h5, "incomplete_GPU_external_engine_HDF5"),
        record(
            args.artifact_root
            / "contract_straight45_a_cclosure_retry_20260730"
            / "case_result.json",
            "production_material_contract_readback",
        ),
        record(
            args.artifact_root
            / "straight45_a_w6p5_dz10_L48_gpu4_20260730"
            / "finite_q_on_artifact.npz",
            "legacy_epsilon_c_16_Q_diagnostic",
        ),
    ]
    audit = {
        "status": STATUS,
        "repository": {
            "branch": subprocess.check_output(
                ["git", "branch", "--show-current"],
                cwd=args.repository,
                text=True,
            ).strip(),
            "audit_basis_commit": audit_basis,
            "expected_audit_basis_prefix": EXPECTED_HEAD,
            "audit_basis_matches": audit_basis.startswith(EXPECTED_HEAD),
            "report_generation_head": head,
            "report_generation_head_descends_from_audit_basis": (
                current_descends_from_basis
            ),
            "dirty_or_untracked_at_audit_start": audit_start_porcelain,
            "current_dirty_or_untracked_after_adding_audit_code": (
                current_porcelain
            ),
            "checkpoint_commits": [
                "0ba30f7 Define paper IR optical closure and edge controls",
                "184ecdc Summarize paper IR material and edge controls",
                "4d79124 Publish paper IR material and edge control diagnostics",
                "651797f Record GPU smoke runtime blocker",
            ],
        },
        "same_contract_200_100_50": {
            "per_polarization": contracts_match,
            "allowed_differences": [
                "core_step_nm",
                "grid_shape implied by core_step_nm",
            ],
            "geometry": "48 um square straight y=x half-plane",
            "boundary": (
                "paper-reduced flake-only Robin: bottom 7.37e6, top 1 "
                "W/(m2 K); no lateral Dirichlet"
            ),
            "remap": (
                "same conservative physical-3D-nearest support projection; "
                "no clipping, smoothing, gain, global rescaling, or tiling"
            ),
        },
        "convergence": convergence,
        "failure": failure,
        "artifacts": selected_records,
        "forbidden_operations": {
            "new_FDTD_run": False,
            "new_thermal_run": False,
            "CPU_FDTD_fallback": False,
            "Q_rescaling": False,
            "PTE": False,
            "adjoint": False,
            "optimization": False,
        },
    }
    summary_path = args.output_dir / "paper_ir_checkpoint_failure_audit.json"
    summary_path.write_text(json.dumps(audit, indent=2) + "\n")

    analytic_a = convergence[
        "analytic_paper_source_a_100_to_50"
    ]
    legacy_b = convergence[
        "legacy_Lumerical_edge_Q_b_100_to_50"
    ]
    report = f"""# Existing paper-IR checkpoint and GPU-failure audit

**Status: `{STATUS}`**

## Repository and provenance

- Branch: `{audit['repository']['branch']}`
- Immutable audit basis: `{audit_basis}`
- Report-generation HEAD: `{head}` (descends from the audit basis:
  `{current_descends_from_basis}`)
- Dirty/untracked at audit start: `{audit['repository']['dirty_or_untracked_at_audit_start']}`
- All six 200/100/50 nm paper-reduced cases use the same geometry, Robin
  boundary, source, and remap contract within each polarization.  Only the
  intended lateral core step and resulting grid shape differ.
- Exact paths, byte sizes, and SHA-256 values are in the audit JSON.

## Thermal comparator decision

The full per-case table is `paper_reduced_thermal_audit_cases.csv`.
For the analytic a-polarization source, the 100-to-50 nm fitted-x strip mean
changes by `{100*analytic_a['relative_changes']['fitted_x_strip_mean_abs_K_m']:.6f}%`,
but its fitted normal strip mean changes by
`{100*analytic_a['relative_changes']['fitted_n_strip_mean_abs_K_m']:.6f}%`.
For legacy Maxwell-Q b polarization, the fitted-x strip mean changes by
`{100*legacy_b['relative_changes']['fitted_x_strip_mean_abs_K_m']:.6f}%`.
Raw maxima remain diagnostic only.  Therefore the local edge-gradient gate is
not promoted even where a single fitted-x aggregate is below 1%.

## GPU termination evidence

Retry 4 acquired the solve license, completed meshing, initialized GPU 2, and
started 39,362 time steps.  The log stops at
`{failure['phase']['maximum_logged_progress_percent']}%`.
The kernel recorded an `fdtd-solutions-app` remote-messenger segfault at the
same wall time, but the external `fdtd-engine` exit code was not captured.
The formal classification is therefore `UNRESOLVED_ENGINE_TERMINATION`, with
a confirmed secondary `FDTD_SOLUTIONS_REMOTE_MESSENGER_SEGFAULT`.

There is no contemporaneous OOM, GPU Xid/reset, timeout, or license failure in
retry 4.  The engine estimated `{estimates['precise_total_GiB']:.3f} GiB` GPU
memory against 49,140 MiB capacity, and the host reported
`{estimates['host_available_GiB']:.3f} GiB` available.  This makes memory
exhaustion unlikely but does not prove the fdtd-engine's internal cause.

## Why the run became large

The logged grid is `{grid_shape[0]} x {grid_shape[1]} x {grid_shape[2]}` =
`{int(np.prod(grid_shape))}` gridpoints.  There is no x/y mesh override; only
the flake-region z mesh is fixed to 10 nm.  The production straight-edge
TaIrTe4 half-plane spans the 48-um lateral domain, so accuracy-5 automatic
meshing resolves its high complex index laterally across most of the domain.
The nominal average lateral interval is
`{1e9*failure['nominal_lateral_average_step_m']:.3f} nm`.

The incomplete HDF5 contains monitor sampling coordinates, not the native
FDTD mesh.  They are reported separately in
`partial_h5_monitor_coordinate_audit.csv`; native x/y/z min/median/max steps
remain unavailable because the API failed before the post-run solver-mesh
readback.

No new solver calculation was performed by this audit.
"""
    (args.output_dir / "PAPER_IR_CHECKPOINT_FAILURE_AUDIT.md").write_text(
        report
    )
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
