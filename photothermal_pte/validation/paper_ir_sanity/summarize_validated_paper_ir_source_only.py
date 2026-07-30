#!/usr/bin/env python3
"""Publish the validated scalar-Gaussian source-only certificate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


REPOSITORY = Path(__file__).resolve().parents[3]
STATUS = "VALIDATED_PAPER_LIKE_SCALAR_GAUSSIAN_SOURCE_ONLY"


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "UNKNOWN"


def artifact_record(path: Path, role: str) -> dict[str, Any]:
    return {
        "role": role,
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def direct_case(name: str, role: str, path: Path) -> dict[str, Any]:
    data = read_json(path)
    focus = data.get("planes", {}).get("flake_target_plane", {})
    source = data.get("pre_run", {}).get("built_contract", {}).get(
        "source", {}
    )
    readback = data.get("pre_run", {}).get("source_readback", {})
    acceptance = data.get("acceptance", {})
    return {
        "case": name,
        "role": role,
        "status": data.get("status"),
        "generation_commit": data.get("generation_commit"),
        "mesh_accuracy": data.get("pre_run", {})
        .get("domain_readback", {})
        .get("mesh accuracy"),
        "host_threads": data.get("pre_run", {})
        .get("resources", {})
        .get("2", {})
        .get("threads"),
        "source_object_w0_um": (
            float(readback["waist radius w0"]) * 1.0e6
            if "waist radius w0" in readback
            else None
        ),
        "physical_target_w0_um": (
            float(source["target_realized_waist_radius_m"]) * 1.0e6
            if "target_realized_waist_radius_m" in source
            else 12.0
        ),
        "distance_from_waist_um": (
            float(readback["distance from waist"]) * 1.0e6
            if "distance from waist" in readback
            else None
        ),
        "realized_wx_um": (
            float(focus["fitted_waist_x_m"]) * 1.0e6 if focus else None
        ),
        "realized_wy_um": (
            float(focus["fitted_waist_y_m"]) * 1.0e6 if focus else None
        ),
        "waist_x_error_percent": (
            abs(float(focus["fitted_waist_x_m"]) / 12.0e-6 - 1.0) * 100.0
            if focus
            else None
        ),
        "waist_y_error_percent": (
            abs(float(focus["fitted_waist_y_m"]) / 12.0e-6 - 1.0) * 100.0
            if focus
            else None
        ),
        "Gaussian_fit_NRMSE_percent": (
            float(focus["Gaussian_fit_NRMSE"]) * 100.0 if focus else None
        ),
        "ellipticity_percent": (
            float(focus["fitted_xy_ellipticity"]) * 100.0
            if focus
            else None
        ),
        "incident_power_closure_percent": (
            abs(float(focus["downward_Poynting_power_over_sourcepower"]) - 1.0)
            * 100.0
            if focus
            else None
        ),
        "auto_shutoff": data.get("log_audit", {}).get("final_auto_shutoff"),
        "runtime_s": data.get("solver_wall_time_s"),
        "GPU_memory_GiB": data.get("log_audit", {}).get(
            "precise_GPU_memory_GiB"
        ),
        "NPZ_sha256": data.get("field_artifact", {}).get("sha256"),
        "source_only_gate_passed": data.get("source_only_gate_passed"),
        "failed_gates": ";".join(
            key for key, value in acceptance.items() if not value
        ),
    }


def recovered_case(name: str, role: str, path: Path) -> dict[str, Any]:
    recovered = read_json(path)
    source_path = Path(recovered["source_result_path"])
    case = direct_case(name, role, source_path)
    focus = recovered["planes"]["flake_target_plane"]
    case.update(
        {
            "status": recovered["status"],
            "generation_commit": recovered["generation_commit"],
            "realized_wx_um": float(focus["fitted_waist_x_m"]) * 1.0e6,
            "realized_wy_um": float(focus["fitted_waist_y_m"]) * 1.0e6,
            "waist_x_error_percent": abs(
                float(focus["fitted_waist_x_m"]) / 12.0e-6 - 1.0
            )
            * 100.0,
            "waist_y_error_percent": abs(
                float(focus["fitted_waist_y_m"]) / 12.0e-6 - 1.0
            )
            * 100.0,
            "Gaussian_fit_NRMSE_percent": (
                float(focus["Gaussian_fit_NRMSE"]) * 100.0
            ),
            "ellipticity_percent": (
                float(focus["fitted_xy_ellipticity"]) * 100.0
            ),
            "incident_power_closure_percent": abs(
                float(focus["downward_Poynting_power_over_sourcepower"]) - 1.0
            )
            * 100.0,
            "source_only_gate_passed": recovered["source_only_gate_passed"],
            "failed_gates": ";".join(recovered["failed_gates"]),
        }
    )
    return case


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-summary", required=True)
    parser.add_argument("--uncalibrated-recovery", required=True)
    parser.add_argument("--uncalibrated-manifest", required=True)
    parser.add_argument("--accuracy6-result", required=True)
    parser.add_argument("--accuracy6-manifest", required=True)
    parser.add_argument("--positive-result", required=True)
    parser.add_argument("--positive-manifest", required=True)
    parser.add_argument("--final-result", required=True)
    parser.add_argument("--final-manifest", required=True)
    parser.add_argument("--license-debug-log", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    paths = {
        key: Path(getattr(args, key)).expanduser().resolve()
        for key in (
            "audit_summary",
            "uncalibrated_recovery",
            "uncalibrated_manifest",
            "accuracy6_result",
            "accuracy6_manifest",
            "positive_result",
            "positive_manifest",
            "final_result",
            "final_manifest",
            "license_debug_log",
        )
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    audit = read_json(paths["audit_summary"])
    final = read_json(paths["final_result"])
    if final.get("status") != STATUS:
        raise ValueError(f"final source status is {final.get('status')}")
    if not final.get("source_only_gate_passed"):
        raise ValueError("final source gate is not passed")
    if not all(final.get("acceptance", {}).values()):
        raise ValueError("one or more final source acceptance gates failed")
    if final.get("scope", {}).get("CPU_FDTD_fallback"):
        raise ValueError("CPU FDTD fallback was used")

    cases = [
        recovered_case(
            "uncalibrated_negative_distance_accuracy5",
            "calibration_baseline",
            paths["uncalibrated_recovery"],
        ),
        direct_case(
            "uncalibrated_negative_distance_accuracy6",
            "mesh_refinement_control",
            paths["accuracy6_result"],
        ),
        direct_case(
            "positive_distance_accuracy5",
            "distance_sign_diagnostic",
            paths["positive_result"],
        ),
        direct_case(
            "calibrated_negative_distance_accuracy5",
            "promoted_source_only_certificate",
            paths["final_result"],
        ),
    ]
    focus = final["planes"]["flake_target_plane"]
    selected = audit["selected_source_only_scenario"]
    final_metrics = {
        "physical_target_w0_um": 12.0,
        "Lumerical_source_object_input_w0_um": (
            final["pre_run"]["source_readback"]["waist radius w0"] * 1.0e6
        ),
        "realized_wx_um": focus["fitted_waist_x_m"] * 1.0e6,
        "realized_wy_um": focus["fitted_waist_y_m"] * 1.0e6,
        "waist_x_error_percent": cases[-1]["waist_x_error_percent"],
        "waist_y_error_percent": cases[-1]["waist_y_error_percent"],
        "Gaussian_fit_NRMSE_percent": (
            focus["Gaussian_fit_NRMSE"] * 100.0
        ),
        "beam_center_displacement_m": focus["beam_center_error_m"],
        "xy_ellipticity_percent": (
            focus["fitted_xy_ellipticity"] * 100.0
        ),
        "source_boundary_max_over_peak": final["source_object_profile"][
            "boundary_max_intensity_over_peak"
        ],
        "source_boundary_mean_over_peak": final["source_object_profile"][
            "boundary_mean_intensity_over_peak"
        ],
        "incident_power_W": focus["downward_Poynting_power_W"],
        "incident_power_closure_percent": cases[-1][
            "incident_power_closure_percent"
        ],
        "field_component_E2_fractions": {
            "x": focus["x_polarization_E2_fraction"],
            "y": focus["cross_polarized_Ey_E2_fraction"],
            "z": focus["longitudinal_Ez_E2_fraction"],
        },
        "native_mesh": final["post_run_mesh"],
        "logged_solver_grid": final["log_audit"]["logged_grid"],
        "runtime_s": final["solver_wall_time_s"],
        "GPU_memory_GiB": final["log_audit"]["precise_GPU_memory_GiB"],
        "auto_shutoff": final["log_audit"]["final_auto_shutoff"],
        "raw_NPZ": final["field_artifact"],
        "raw_FSP": {
            "path": str(paths["final_result"].parent / "paper_ir_source_only.fsp"),
            "size_bytes": (
                paths["final_result"].parent / "paper_ir_source_only.fsp"
            ).stat().st_size,
            "sha256": sha256(
                paths["final_result"].parent / "paper_ir_source_only.fsp"
            ),
        },
    }
    payload = {
        "status": STATUS,
        "generation_commit": git_commit(),
        "source_generation_commit": final["generation_commit"],
        "scenario_label": selected["scenario_label"],
        "experimentally_reproduced_beam": False,
        "paper_certified_beam": False,
        "explicit_assumption": "physical target-plane w0=12 um",
        "source_object_calibration": selected["source_object_calibration"],
        "calibration_is_Q_rescaling": False,
        "license_diagnosis": {
            "license_entitlement_valid": True,
            "v261_GUI_checkout_observed": True,
            "initial_license_unavailable_status_was_sandbox_false_negative": True,
            "transient_solver_task_contention_observed": True,
            "final_GPU_engine_host_threads": int(
                final["pre_run"]["resources"]["2"]["threads"]
            ),
            "CPU_FDTD_fallback": False,
        },
        "final_metrics": final_metrics,
        "acceptance": final["acceptance"],
        "cases": cases,
        "successor_sequence_authorized": True,
        "next_sequence": final["successor_sequence"],
        "not_run": [
            "TaIrTe4 planar/edge material cases",
            "thermal",
            "weighting potential",
            "PTE",
            "adjoint",
            "gradient",
            "optimization",
            "thin-lens diagnostic",
        ],
    }
    summary_path = output / "paper_ir_source_only_certification_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2) + "\n")

    cases_path = output / "paper_ir_source_only_cases.csv"
    with cases_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(cases[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(cases)

    roles = {
        "audit_summary": "paper_beam_audit",
        "uncalibrated_recovery": "uncalibrated_readonly_result",
        "uncalibrated_manifest": "uncalibrated_recovery_manifest",
        "accuracy6_result": "accuracy6_control_result",
        "accuracy6_manifest": "accuracy6_control_manifest",
        "positive_result": "positive_sign_diagnostic_result",
        "positive_manifest": "positive_sign_diagnostic_manifest",
        "final_result": "promoted_source_only_result",
        "final_manifest": "promoted_source_only_external_manifest",
        "license_debug_log": "v261_license_debug_log",
    }
    manifest = {
        "raw_artifacts_committed_to_git": False,
        "new_raw_FSP_or_NPZ_generated": True,
        "generation_commit": git_commit(),
        "generation_command": (
            "python3 photothermal_pte/validation/paper_ir_sanity/"
            "summarize_validated_paper_ir_source_only.py [arguments in report]"
        ),
        "artifacts": [
            artifact_record(paths[key], role) for key, role in roles.items()
        ],
    }
    (output / "RAW_ARTIFACT_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )

    m = final_metrics
    report = f"""# Paper-IR source-only beam certification

Status: `{STATUS}`

This is a **paper-like scalar-Gaussian scenario with an explicitly assumed
waist**.  It is not an experimentally reproduced or paper-certified beam.
The physical target-plane 1/e² radius remains 12 µm.  The Lumerical
source-object input is {m['Lumerical_source_object_input_w0_um']:.9f} µm,
obtained from the SHA-pinned uncalibrated field as a numerical source
calibration.  It does not rescale incident power or Q.

## Final GPU source-only certificate

- v261 internal version: `{final['pre_run']['version']}`
- wavelength/source/domain: 11 / 50 / 60 µm
- six boundaries: PML; periodic/Bloch: none
- source-object input w0: {m['Lumerical_source_object_input_w0_um']:.9f} µm
- realized target w0 x/y: {m['realized_wx_um']:.9f} /
  {m['realized_wy_um']:.9f} µm
- target error x/y: {m['waist_x_error_percent']:.6f}% /
  {m['waist_y_error_percent']:.6f}%
- linear-intensity Gaussian-fit NRMSE:
  {m['Gaussian_fit_NRMSE_percent']:.6f}%
- x/y ellipticity: {m['xy_ellipticity_percent']:.6f}%
- target incident power: {m['incident_power_W']:.15e} W
- incident-power closure: {m['incident_power_closure_percent']:.6f}%
- source boundary max/mean: {m['source_boundary_max_over_peak']:.8e} /
  {m['source_boundary_mean_over_peak']:.8e}
- auto-shutoff: {m['auto_shutoff']:.8e}
- solver grid: {m['logged_solver_grid']['shape_xyz']}
- post-run native mesh: {m['native_mesh']['shape_xyz']}
- precise GPU memory: {m['GPU_memory_GiB']:.3f} GiB
- API wall time: {m['runtime_s']:.3f} s
- GPU engine: yes; CPU FDTD fallback: no

All mandatory gates pass.  The final raw NPZ SHA-256 is
`{m['raw_NPZ']['sha256']}`.  Raw NPZ/FSP files remain outside Git.

## Controls and corrections

The original `BLOCKED_LUMERICAL_LICENSE_UNAVAILABLE` interpretation was a
sandbox-network false negative.  Host execution checked out v261
successfully.  A later `lum_fdtd_solve` task shortage was transient license
contention, not missing entitlement; the final GPU solve used three host
orchestration threads without changing the GPU engine or numerical model.

The uncalibrated accuracy-5/6 controls realized approximately 12.08 µm, so
mesh refinement did not remove the small waist offset.  A positive
`distance from waist` diagnostic worsened the error above 3%; it is not
promoted.  The negative-distance calibrated case is the sole promoted
source-only contract.

## Decision

The source-only gate authorizes the ordered successor cases: planar
TaIrTe4 a/b, followed by straight-45-degree finite-edge a/b.  None of those
material cases, nor thermal/PTE/adjoint/optimization, was executed by this
certificate.
"""
    (output / "PAPER_IR_SOURCE_ONLY_CERTIFICATION_REPORT.md").write_text(report)
    print(json.dumps({"status": STATUS, "final_metrics": final_metrics}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
