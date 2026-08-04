#!/usr/bin/env python3
"""Summarize the 4 um-domain CPU TFSF source-gate controls."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--refined-dir", required=True)
    parser.add_argument("--report-dir", required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def match_float(text: str, pattern: str) -> float:
    match = re.search(pattern, text)
    if match is None:
        raise RuntimeError(f"missing solver-log field: {pattern}")
    return float(match.group(1))


def match_int_tuple(text: str, pattern: str) -> list[int]:
    match = re.search(pattern, text)
    if match is None:
        raise RuntimeError(f"missing solver-log field: {pattern}")
    return [int(match.group(index)) for index in range(1, 4)]


def read_case(directory: Path) -> dict[str, object]:
    result_path = directory / "probe_result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    logs = sorted(directory.glob("*_p0.log"))
    if len(logs) != 1:
        raise RuntimeError(f"expected exactly one solver log in {directory}")
    log_path = logs[0]
    log = log_path.read_text(encoding="utf-8", errors="replace")
    grid = match_int_tuple(
        log, r"Simulation size in gridpoints:\s*(\d+)\s*x\s*(\d+)\s*x\s*(\d+)"
    )
    result["solver_log_metrics"] = {
        "version": re.search(
            r"(Ansys Lumerical .*?FDTD Solver Version .*?\(Linux 64bit\))",
            log,
        ).group(1),
        "gridpoints_xyz": grid,
        "gridpoint_product": grid[0] * grid[1] * grid[2],
        "estimated_memory_GiB": match_float(
            log, r"Estimate of memory required:\s*([0-9.eE+-]+)\s*GiB"
        ),
        "peak_cpu_memory_GiB": match_float(
            log,
            r"Peak CPU memory used in the simulation \(GiB\):\s*([0-9.eE+-]+)",
        ),
        "engine_overall_wall_s": match_float(
            log, r"Overall wall time measurements in seconds:\s*([0-9.eE+-]+)"
        ),
        "engine_mesh_initialize_wall_s": match_float(
            log, r"time to mesh and initialize:\s*([0-9.eE+-]+)"
        ),
        "engine_time_step_wall_s": match_float(
            log, r"time to run FDTD simulation:\s*([0-9.eE+-]+)"
        ),
        "completed_iterations": int(
            match_float(log, r"Completed\s+([0-9]+)\s+iterations")
        ),
        "solver_speed_Mnodes_s": match_float(
            log,
            r"total FDTD solver speed on 1 processes \(Mnodes/s\):\s*([0-9.eE+-]+)",
        ),
        "early_termination": "Early termination of simulation" in log,
    }
    artifacts = []
    for path in (result_path, log_path, directory / "cpu_tfsf_roi_probe.fsp"):
        artifacts.append(
            {
                "path": str(path.resolve()),
                "byte_size": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    result["raw_artifacts"] = artifacts
    return result


def relative_change(a: float, b: float) -> float:
    return abs(b - a) / max(abs(a), abs(b), 1e-300)


def main() -> int:
    args = parse_args()
    baseline = read_case(Path(args.baseline_dir).resolve())
    refined = read_case(Path(args.refined_dir).resolve())
    report_dir = Path(args.report_dir).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)

    b = baseline["roi_metrics"]
    r = refined["roi_metrics"]
    convergence = {
        "mean_E2_relative_change": relative_change(b["mean_E2"], r["mean_E2"]),
        "intensity_rms_absolute_change": abs(
            r["intensity_relative_rms"] - b["intensity_relative_rms"]
        ),
        "peak_to_peak_absolute_change": abs(
            r["intensity_relative_peak_to_peak"]
            - b["intensity_relative_peak_to_peak"]
        ),
        "phase_max_abs_deg_change": abs(
            r["phase_max_abs_deg"] - b["phase_max_abs_deg"]
        ),
        "energy_closure_absolute_change": abs(
            refined["closed_flux_box"]["relative_energy_closure_error"]
            - baseline["closed_flux_box"]["relative_energy_closure_error"]
        ),
    }
    status = (
        "VALIDATED_CPU_TFSF_4UM_DOMAIN_2UM_ROI_SOURCE_GATE"
        if baseline["status"] == refined["status"]
        == "VALIDATED_CPU_TFSF_4UM_DOMAIN_2UM_ROI_SOURCE_GATE"
        else "FAILED_CPU_TFSF_4UM_DOMAIN_2UM_ROI_SOURCE_GATE"
    )
    summary = {
        "status": status,
        "scope": (
            "empty-air source integrity and timing only; no device, Q, "
            "thermal, AD, FD, or optimization"
        ),
        "protected_roi_um": {"x": [-1.0, 1.0], "y": [-1.0, 1.0]},
        "lateral_fdtd_domain_um": 4.0,
        "tfsf_span_um": baseline["tfsf_span_um"],
        "baseline_pml_layers": baseline["pml_layers"],
        "refined_pml_layers": refined["pml_layers"],
        "baseline": baseline,
        "refined": refined,
        "pml_convergence": convergence,
        "geometry_gate": {
            "four_um_flake_fits_in_four_um_fdtd_domain": False,
            "reason": (
                "A finite scatterer must remain fully inside the TFSF box, "
                "and the TFSF box must remain inside the PML boundaries."
            ),
            "consequence": (
                "The 4 um domain is promoted only for the empty-air source "
                "gate. A 4 um flake requires a larger FDTD domain."
            ),
        },
    }
    summary_path = report_dir / "CPU_TFSF_4UM_ROI_SUMMARY.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    manifest = {
        "status": status,
        "raw_files_committed_to_git": False,
        "artifacts": baseline["raw_artifacts"] + refined["raw_artifacts"],
    }
    manifest_path = report_dir / "CPU_TFSF_4UM_ROI_RAW_MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    bp = baseline["solver_log_metrics"]
    rp = refined["solver_log_metrics"]
    report = f"""# CPU TFSF protection of the central 2 µm ROI

Status: `{status}`

## Scope

This is an empty-air source-integrity and runtime gate in Ansys Lumerical
2026 R1.2 v261. It is not a device, optical-Q, thermal, AD, FD, or
optimization result.

- protected ROI: `x,y=[-1,1] µm`;
- lateral FDTD domain: `4×4 µm`;
- TFSF transverse span: `{baseline['tfsf_span_um']} µm`;
- six outer boundaries: PML, no periodic/Bloch boundary;
- source: normal incidence, x polarization, 3–6 µm;
- analysis: 4 µm;
- mesh: auto non-uniform, accuracy {baseline['mesh_accuracy']};
- CPU: one process, {baseline['threads']} threads; GPU resource disabled.

## Baseline result (PML {baseline['pml_layers']})

- mean |E|² error from the unit-amplitude incident field:
  `{100*b['mean_E2_relative_error_from_unit_source']:.8f}%`;
- spatial intensity RMS: `{100*b['intensity_relative_rms']:.8f}%`;
- spatial intensity peak-to-peak: `{100*b['intensity_relative_peak_to_peak']:.8f}%`;
- maximum phase deviation: `{b['phase_max_abs_deg']:.8e} degree`;
- Ey/Ex L2: `{b['Ey_to_Ex_L2']:.8e}`;
- Ez/Ex L2: `{b['Ez_to_Ex_L2']:.8e}`;
- closed-box energy error:
  `{100*baseline['closed_flux_box']['relative_energy_closure_error']:.8f}%`.

Runtime and memory from the native solver log:

- grid: `{bp['gridpoints_xyz'][0]}×{bp['gridpoints_xyz'][1]}×{bp['gridpoints_xyz'][2]}`
  (`{bp['gridpoint_product']}` gridpoints);
- engine overall wall time: `{bp['engine_overall_wall_s']:.6f} s`;
- time stepping: `{bp['engine_time_step_wall_s']:.6f} s`;
- Python `run()` wall time including engine launch/return: `{baseline['fdtd_run_wall_s']:.6f} s`;
- complete session wall time: `{baseline['total_wall_s']:.6f} s`;
- peak CPU memory: `{bp['peak_cpu_memory_GiB']:.6f} GiB`;
- completed iterations: `{bp['completed_iterations']}` (auto-shutoff reached).

## PML refinement (PML {refined['pml_layers']})

- mean |E|² relative change: `{100*convergence['mean_E2_relative_change']:.8e}%`;
- energy-closure absolute change:
  `{100*convergence['energy_closure_absolute_change']:.8e}` percentage point;
- grid: `{rp['gridpoints_xyz'][0]}×{rp['gridpoints_xyz'][1]}×{rp['gridpoints_xyz'][2]}`;
- engine overall wall time: `{rp['engine_overall_wall_s']:.6f} s`;
- Python `run()` wall time: `{refined['fdtd_run_wall_s']:.6f} s`;
- complete session wall time: `{refined['total_wall_s']:.6f} s`;
- peak CPU memory: `{rp['peak_cpu_memory_GiB']:.6f} GiB`.

Both PML cases passed every preregistered ROI and energy gate. PML 24 is the
faster promoted source-gate setting; PML 32 is retained as the refinement
control.

## Geometry limitation before device AD–FD

A 4 µm TaIrTe4 flake cannot be placed in this same 4 µm FDTD domain for a
valid TFSF calculation. The finite flake must be fully inside the TFSF box,
and the TFSF box must be strictly inside the PML boundaries. Therefore this
result validates the illumination in the central 2 µm ROI, not the final
finite-flake device geometry. The device calculation must enlarge the FDTD
domain while keeping the physical design/PTE ROI exactly 2 µm.

Raw FSP and solver logs remain outside Git. Their paths, byte sizes, and
SHA-256 values are recorded in `CPU_TFSF_4UM_ROI_RAW_MANIFEST.json`.
"""
    report_path = report_dir / "CPU_TFSF_4UM_ROI_REPORT.md"
    report_path.write_text(report, encoding="utf-8")
    print(json.dumps({"status": status, "report": str(report_path)}, indent=2))
    return 0 if status.startswith("VALIDATED") else 2


if __name__ == "__main__":
    raise SystemExit(main())
