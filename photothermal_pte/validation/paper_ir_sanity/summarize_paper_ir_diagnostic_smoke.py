#!/usr/bin/env python3
"""Publish the failed reduced paper-IR GPU smoke without altering raw data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any


OFFICIAL_STATUS = (
    "PARTIAL_PAPER_IR_CONTROL_VALIDATION_"
    "BLOCKED_OPTICAL_RUNTIME_AND_UNRESOLVED_EDGE_METRIC"
)
DIAGNOSTIC_STATUS = (
    "FAILED_DIAGNOSTIC_ONE_POL_GPU_SMOKE_SIX_FACE_CLOSURE"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def last_float(pattern: str, text: str) -> float:
    matches = re.findall(pattern, text, flags=re.MULTILINE)
    if not matches:
        raise RuntimeError(f"missing log pattern: {pattern}")
    return float(matches[-1])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--report-dir", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    report_dir = Path(args.report_dir).expanduser().resolve()
    report_dir.mkdir(parents=True, exist_ok=True)

    case_path = run_dir / "case_result.json"
    post_path = run_dir / "read_only_closure_postprocess.json"
    raw_manifest_path = run_dir / "RAW_ARTIFACT_MANIFEST.json"
    log_path = run_dir / "finite_2um_optical_q_p0.log"
    case = read_json(case_path)
    post = read_json(post_path)
    raw_manifest = read_json(raw_manifest_path)
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    run = case["run_result"]

    if case["status"] != "FAILED_ACCEPTANCE":
        raise RuntimeError("expected failed-acceptance diagnostic")
    if run["six_face_relative_closure"] < 0.005:
        raise RuntimeError("refusing to label a passing closure as failed")
    if not post["FDTD_solve_called"] is False:
        raise RuntimeError("postprocess provenance is not read-only")

    grid_match = re.search(
        r"Simulation size in gridpoints: (\d+) x (\d+) x (\d+)",
        log_text,
    )
    if grid_match is None:
        raise RuntimeError("missing logged grid size")
    logged_grid = [int(value) for value in grid_match.groups()]
    gpu_memory_gib = last_float(
        r"Estimated memory use on GPU 4 \(precise\):[\s\S]*?"
        r"Total: ([0-9.]+) GiB",
        log_text,
    )
    solver_wall_s = last_float(
        r"Overall wall time measurements in seconds: ([0-9.]+)",
        log_text,
    )
    gpu_run_s = last_float(
        r"time to run GPU simulation: ([0-9.]+)",
        log_text,
    )
    final_auto_shutoff = last_float(
        r"Auto Shutoff: ([0-9.eE+-]+)",
        log_text,
    )
    iterations = int(
        last_float(r"Completed (\d+) iterations", log_text)
    )
    solver_completed = "Simulation completed successfully" in log_text

    p_common = float(post["P_Q_common_grid_W"])
    p_native = float(post["P_Q_native_component_grid_W"])
    p_six = float(post["P_six_face_W"])
    mismatch = abs(p_native - p_six)
    absolute_face_sum = float(
        post["six_face_native"]["sum_absolute_face_power_W"]
    )
    cancellation_conditioned_mismatch = mismatch / absolute_face_sum
    net_to_absolute_face_ratio = abs(p_six) / absolute_face_sum

    q_bounds = run["native_Yee_mesh_audit"][
        "common_Q_output_coordinates"
    ]
    flux_bounds = case["pre_run_contract"]["geometry"][
        "six_face_absorption_box_bounds_m"
    ]
    lateral_control_volume_gap = {
        axis: {
            "Q_common_bounds_m": q_bounds[axis]["bounds_m"],
            "flux_box_bounds_m": flux_bounds[axis],
            "low_side_gap_m": (
                q_bounds[axis]["bounds_m"][0] - flux_bounds[axis][0]
            ),
            "high_side_gap_m": (
                flux_bounds[axis][1] - q_bounds[axis]["bounds_m"][1]
            ),
        }
        for axis in ("x", "y")
    }

    artifact_names = [
        "case_result.json",
        "diagnostic_q_native_artifact.npz",
        "finite_2um_optical_q.fsp",
        "finite_2um_optical_q_output.h5",
        "finite_2um_optical_q_p0.log",
        "native_yee_mesh_coordinates.npz",
        "read_only_closure_postprocess.json",
    ]
    artifacts = {}
    for name in artifact_names:
        path = run_dir / name
        if path.is_file():
            artifacts[name] = {
                "server_path": str(path),
                "exists_after_session_close": True,
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        else:
            artifacts[name] = {
                "server_path": str(path),
                "exists_after_session_close": False,
                "size_bytes": None,
                "sha256": None,
                "note": (
                    "transient Lumerical engine H5 was not retained after "
                    "the session closed; no hash is claimed"
                ),
            }
    published_manifest = {
        "policy": (
            "Raw NPZ/FSP/H5/log files remain outside Git. This manifest "
            "records immutable paths, sizes, hashes, and generation commands."
        ),
        "official_status": OFFICIAL_STATUS,
        "diagnostic_status": DIAGNOSTIC_STATUS,
        "generation_commit": case["generation_commit"],
        "generation_command": case["generation_command"],
        "read_only_postprocess_command": (
            "see read_only_closure_postprocess.json provenance; no FDTD solve "
            "or runanalysis was called"
        ),
        "source_raw_manifest_path": str(raw_manifest_path),
        "source_raw_manifest_sha256": sha256(raw_manifest_path),
        "artifacts": artifacts,
    }
    manifest_output = (
        report_dir / "PAPER_IR_DIAGNOSTIC_RAW_ARTIFACT_MANIFEST.json"
    )
    write_json(manifest_output, published_manifest)

    face_csv = report_dir / "paper_ir_diagnostic_six_face_fluxes.csv"
    with face_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "face",
                "monitor",
                "normalized_signed_axis_flux",
                "signed_axis_power_native_W",
                "outward_power_native_W",
                "absolute_fraction_of_face_sum",
            ],
        )
        writer.writeheader()
        for face, values in sorted(
            post["six_face_native"]["faces"].items()
        ):
            outward = float(values["outward_power_W_at_1_W_m2"])
            writer.writerow(
                {
                    "face": face,
                    "monitor": values["monitor"],
                    "normalized_signed_axis_flux": values[
                        "normalized_signed_axis_flux"
                    ],
                    "signed_axis_power_native_W": values[
                        "signed_axis_power_W_at_1_W_m2"
                    ],
                    "outward_power_native_W": outward,
                    "absolute_fraction_of_face_sum": (
                        abs(outward) / absolute_face_sum
                    ),
                }
            )

    summary = {
        "official_status": OFFICIAL_STATUS,
        "diagnostic_status": DIAGNOSTIC_STATUS,
        "validated": False,
        "production_paper_like_result": False,
        "thermal_run": False,
        "PTE_run": False,
        "adjoint_run": False,
        "gradient_run": False,
        "optimization_run": False,
        "execution": {
            "generation_commit": case["generation_commit"],
            "GPU": "GPU 4, NVIDIA RTX 6000 Ada Generation",
            "CPU_FDTD_fallback": False,
            "solver_completed_normally": solver_completed,
            "logged_gridpoints": logged_grid,
            "logged_gridpoint_product": (
                logged_grid[0] * logged_grid[1] * logged_grid[2]
            ),
            "precise_GPU_memory_estimate_GiB": gpu_memory_gib,
            "solver_wall_time_s": solver_wall_s,
            "GPU_simulation_time_s": gpu_run_s,
            "iterations": iterations,
            "requested_simulation_time_s": 1.2e-12,
            "requested_auto_shutoff_min": 1.0e-5,
            "final_logged_auto_shutoff": final_auto_shutoff,
            "auto_shutoff_gate_reached": final_auto_shutoff <= 1.0e-5,
        },
        "contract": {
            "classification": run["classification"],
            "domain_um": case["domain_um"],
            "source_span_um": case["source_span_um"],
            "waist_um": case["waist_um"],
            "polarization": run["polarization"],
            "pml_layers": case["pml_layers"],
            "flake_dz_nm": case["flake_dz_nm"],
            "material": case["pre_run_contract"]["material"],
            "boundaries": case["pre_run_contract"]["boundaries"],
            "diagnostic_difference_from_production": case[
                "pre_run_contract"
            ]["geometry"]["diagnostic_difference_from_production"],
        },
        "absorption": {
            "source_power_native_W": post["source_power_native_W"],
            "component_power_common_grid_native_W": run[
                "component_power_native_W"
            ],
            "component_power_native_Yee_grid_W": post[
                "native_component_power_W"
            ],
            "P_Q_common_grid_native_W": p_common,
            "P_Q_native_Yee_grid_W": p_native,
            "P_six_face_native_W": p_six,
            "common_grid_six_face_relative_closure": post[
                "common_grid_vs_six_face_relative_closure"
            ],
            "native_grid_six_face_relative_closure": post[
                "native_component_vs_six_face_relative_closure"
            ],
            "native_common_relative_difference": post[
                "native_to_common_relative_difference"
            ],
            "six_face_sum_absolute_power_W": absolute_face_sum,
            "net_to_absolute_face_ratio": net_to_absolute_face_ratio,
            "mismatch_relative_to_absolute_face_sum": (
                cancellation_conditioned_mismatch
            ),
            "Q_hotspot": run["Q_hotspot"],
            "support_analysis": run["support_analysis"],
            "lateral_control_volume_gap": lateral_control_volume_gap,
            "Q_operations": run["artifact_metadata"]["Q_operations"],
        },
        "acceptance": run["acceptance"],
        "failure_analysis": {
            "confirmed": [
                (
                    "The six-face closure is 9.18%, above the 0.5% gate."
                ),
                (
                    "The Q monitor and flux surface do not enclose the same "
                    "lateral control volume for the extended half-plane flake."
                ),
                (
                    "The run reached 1.2 ps with final auto shutoff "
                    "1.81e-5, above the requested 1e-5 threshold."
                ),
                (
                    "The net absorbed flux is a 4.04% remainder of the sum "
                    "of absolute face powers, amplifying small DFT errors."
                ),
                (
                    "Native-to-common Q integration differs by only 0.0299%, "
                    "so common-grid interpolation cannot explain 9.18%."
                ),
            ],
            "not_proven_without_another_solve": [
                (
                    "How much of the closure error is caused separately by "
                    "the lateral shell versus finite-time DFT convergence."
                ),
                (
                    "Whether a matched control volume and longer decay time "
                    "would pass 0.5%."
                ),
            ],
            "next_contract_fix_before_any_rerun": [
                (
                    "Use exactly matched Q-volume and six-face lateral bounds "
                    "for the extended half-plane."
                ),
                (
                    "Increase simulation time or tighten convergence so the "
                    "requested auto-shutoff threshold is actually reached."
                ),
                (
                    "Keep one polarization and the reduced diagnostic label; "
                    "do not promote it to the 48 um production result."
                ),
            ],
        },
        "raw_artifact_manifest": manifest_output.name,
        "six_face_csv": face_csv.name,
    }
    summary_path = (
        report_dir / "paper_ir_diagnostic_gpu_smoke_summary.json"
    )
    write_json(summary_path, summary)

    report = f"""# Paper-IR reduced GPU diagnostic smoke

## Status

- Official project status: `{OFFICIAL_STATUS}`
- Diagnostic substatus: `{DIAGNOSTIC_STATUS}`
- This is **not** a paper-like production optical result.
- No thermal, PTE, adjoint, gradient, or optimization calculation ran.
- No CPU FDTD fallback, Q clipping, smoothing, gain, rescaling, tiling, or
  source deletion was used.

## What ran

The single approved smoke used a 12 x 12 um straight-45-degree half-plane,
one `a` polarization, 6 um Gaussian aperture, 2 um waist, six PML boundaries,
24 PML layers, and 10 nm flake-region z mesh.  It retained the production
material closure `epsilon_x=epsilon_b`, `epsilon_y=epsilon_a`,
`epsilon_z=epsilon_b`, but deliberately reduced the lateral optical geometry
and monitor set.  It ran on GPU 4, an NVIDIA RTX 6000 Ada Generation; GPU 2
was not used because only 4.8 GB was free before launch.

The engine completed {iterations:,d} iterations normally.  Logged FDTD size
was {logged_grid[0]} x {logged_grid[1]} x {logged_grid[2]}
({logged_grid[0] * logged_grid[1] * logged_grid[2]:,d} gridpoints), precise GPU
memory estimate was {gpu_memory_gib:.3f} GiB, solver wall time was
{solver_wall_s:.3f} s, and GPU stepping took {gpu_run_s:.3f} s.

## Result

| Metric | Value |
|---|---:|
| native source power | {post["source_power_native_W"]:.12e} W |
| common-grid P_Q | {p_common:.12e} W |
| native-component-grid P_Q | {p_native:.12e} W |
| six-face P | {p_six:.12e} W |
| common-grid closure | {100.0 * post["common_grid_vs_six_face_relative_closure"]:.6f}% |
| native-grid closure | {100.0 * post["native_component_vs_six_face_relative_closure"]:.6f}% |
| native/common Q difference | {100.0 * post["native_to_common_relative_difference"]:.6f}% |
| mismatch / sum(abs(face power)) | {100.0 * cancellation_conditioned_mismatch:.6f}% |
| final auto shutoff | {final_auto_shutoff:.6e} |

Component powers on the common Q grid are:

- Qx: {run["component_power_native_W"]["x"]:.12e} W
- Qy: {run["component_power_native_W"]["y"]:.12e} W
- Qz: {run["component_power_native_W"]["z"]:.12e} W

Qz is finite and nonzero, as required by the lossy
`epsilon_z=epsilon_b` closure.  The hotspot is
({run["Q_hotspot"]["x_m"] * 1e6:.6f},
{run["Q_hotspot"]["y_m"] * 1e6:.6f},
{run["Q_hotspot"]["z_m"] * 1e9:.6f}) in (um, um, nm).

## Why the gate failed

The 9.18% closure failure is real and is not corrected empirically.

1. The control volumes are not identical.  The six-face box is x/y = +/-5
   um, while the actual common Q output ends near +/-4.542 um.  Because the
   TaIrTe4 half-plane continues laterally, the surface encloses a lossy shell
   that the volume-Q monitor does not integrate.
2. The run exhausted its 1.2 ps simulation time with final auto shutoff
   {final_auto_shutoff:.6e}, above the requested 1e-5.  The DFT convergence
   criterion therefore was not reached.
3. The six-face net power is only
   {100.0 * net_to_absolute_face_ratio:.3f}% of the sum of absolute face
   powers.  This cancellation makes the absorption closure sensitive to small
   residual face errors.
4. Native-component and common-grid P_Q differ by only
   {100.0 * post["native_to_common_relative_difference"]:.4f}%.  Component
   interpolation is not large enough to explain the closure failure.

The existing data cannot separate the shell contribution from finite-time DFT
error.  Doing so requires another solve with a matched control volume and
sufficient decay time.  Per the one-smoke fail-closed contract, that solve was
not started.

## Provenance

- Generation commit: `{case["generation_commit"]}`
- Raw Q NPZ: `{artifacts["diagnostic_q_native_artifact.npz"]["server_path"]}`
- Raw Q NPZ size: {artifacts["diagnostic_q_native_artifact.npz"]["size_bytes"]:,d} bytes
- Raw Q NPZ SHA-256: `{artifacts["diagnostic_q_native_artifact.npz"]["sha256"]}`
- FSP, log, coordinate NPZ, case JSON, and read-only postprocess hashes are
  recorded in `{manifest_output.name}`.  The transient engine H5 was not
  retained after session close, so the manifest explicitly records it as
  absent and does not invent a hash.
- Individual face powers are in `{face_csv.name}`.

The raw per-case JSON and raw solver artifacts were not modified.
"""
    (report_dir / "PAPER_IR_DIAGNOSTIC_GPU_SMOKE_REPORT.md").write_text(
        report,
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
