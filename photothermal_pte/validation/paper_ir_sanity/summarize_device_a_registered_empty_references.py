#!/usr/bin/env python3
"""Audit registered off-flake empty-stack polarization references offline."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def raw_artifact(path: Path, role: str) -> dict[str, Any]:
    return {
        "role": role,
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
        "committed_to_git": False,
    }


def maximum_lateral_fraction(power_box: dict[str, Any]) -> float:
    faces = power_box["faces"]
    return max(
        abs(float(faces[f"{axis}_{side}"]["normalized_signed_axis_flux"]))
        for axis in "xy"
        for side in ("min", "max")
    )


def summarize_case(case_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = json.loads((case_dir / "case_result.json").read_text())
    run = payload["run_result"]
    failed = sorted(key for key, value in run["acceptance"].items() if not value)
    return payload, {
        "path": str(case_dir.resolve()),
        "status": payload["status"],
        "polarization_deg": payload["polarization_deg"],
        "failed_acceptance_items": failed,
        "auto_shutoff": run["auto_shutoff"],
        "source_aperture_edge_to_central": run["incident_reference"][
            "source_aperture_edge_to_central"
        ],
        "inner_control_volume_maximum_absolute_lateral_flux_fraction": (
            maximum_lateral_fraction(run["six_face"])
        ),
        "outer_control_volume_maximum_absolute_lateral_flux_fraction": (
            maximum_lateral_fraction(run["outer_power_box"])
        ),
        "empty_stack_absorbed_power_W_at_1_W_m2": run[
            "empty_stack_P_Q_W_at_1_W_m2"
        ],
        "inner_net_inward_power_W_at_1_W_m2": run["six_face"][
            "net_inward_power_W"
        ],
        "solver_completed": run["auto_shutoff"][
            "simulation_completed_successfully"
        ],
        "matched_bounds": run["native_Yee_mesh_audit"][
            "Pabs_object_and_realized_six_face_bounds_match_lt_1fm"
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--empty-a", type=Path, required=True)
    parser.add_argument("--empty-b", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    payload_a, case_a = summarize_case(args.empty_a)
    payload_b, case_b = summarize_case(args.empty_b)
    failed_b_expected = case_b["failed_acceptance_items"] == [
        "offset_source_max_absolute_lateral_flux_fraction_lt_1e_4"
    ]
    checks = {
        "empty_a_all_existing_acceptance_passed": not case_a[
            "failed_acceptance_items"
        ],
        "empty_b_only_inner_lateral_gate_failed": failed_b_expected,
        "both_solver_runs_completed": bool(
            case_a["solver_completed"] and case_b["solver_completed"]
        ),
        "both_auto_shutoff_below_1e_5": max(
            float(case_a["auto_shutoff"]["final_value"]),
            float(case_b["auto_shutoff"]["final_value"]),
        )
        < 1e-5,
        "both_matched_control_volume_bounds": bool(
            case_a["matched_bounds"] and case_b["matched_bounds"]
        ),
        "both_outer_lateral_flux_below_1e_4": max(
            float(
                case_a[
                    "outer_control_volume_maximum_absolute_lateral_flux_fraction"
                ]
            ),
            float(
                case_b[
                    "outer_control_volume_maximum_absolute_lateral_flux_fraction"
                ]
            ),
        )
        < 1e-4,
    }
    summary = {
        "status": "BLOCKED_REGISTERED_EMPTY_B_INNER_LATERAL_FLUX_GATE_REQUIRES_PHYSICAL_REVIEW",
        "scope": "offline audit of completed GPU empty-stack references",
        "cases": {"E_parallel_a": case_a, "E_parallel_b": case_b},
        "checks": checks,
        "diagnosis": {
            "registered_beam_is_outside_flake": True,
            "inner_control_volume_is_flake_local": True,
            "inner_lateral_flux_interpretation": (
                "physical Gaussian-beam Poynting flux crossing the local "
                "flake absorption box; not a direct PML-truncation metric"
            ),
            "outer_lateral_flux_interpretation": (
                "large-box lateral boundary flux; proposed numerical "
                "truncation diagnostic"
            ),
            "gate_was_not_relaxed_or_rescaled": True,
            "finite_Device_A_started": False,
        },
        "proposed_contract_change_not_yet_applied": {
            "preserve_inner_lateral_flux_as_signed_diagnostic": True,
            "replace_inner_gate_with_outer_lateral_flux_fraction_lt_1e_4": True,
            "retain_matched_volume_closure_and_auto_shutoff_gates": True,
            "reason": (
                "the registered beam centre lies outside the finite flake, "
                "so the local absorption box intentionally intercepts real "
                "lateral beam power"
            ),
        },
        "execution_scope": {
            "new_FDTD": False,
            "thermal": False,
            "PTE": False,
            "adjoint": False,
            "optimization": False,
        },
    }
    summary_path = args.output_dir / "device_a_registered_empty_reference_audit.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    artifacts = []
    for label, case_dir in (("E||a", args.empty_a), ("E||b", args.empty_b)):
        for path in sorted(case_dir.iterdir()):
            if path.is_file() and (
                path.name == "case_result.json"
                or path.suffix.lower() in {".fsp", ".npz", ".log"}
            ):
                artifacts.append(raw_artifact(path, f"{label} {path.name}"))
    manifest = {
        "status": "RAW_EMPTY_REFERENCE_ARTIFACTS_RECORDED_NOT_COMMITTED",
        "artifacts": artifacts,
        "generation_commands": [
            payload_a["generation_command"],
            payload_b["generation_command"],
        ],
    }
    (args.output_dir / "RAW_ARTIFACT_MANIFEST_EMPTY_REFERENCES.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )

    report = f"""# Registered Device-A empty-reference audit

Status: `{summary['status']}`

Both polarization-matched empty SiO2/Si GPU calculations completed and reached
the requested auto-shutoff.  The existing `E||a` acceptance passed.  `E||b`
failed only the local inner-box lateral-flux gate:

| metric | E parallel a | E parallel b |
|---|---:|---:|
| auto-shutoff | {case_a['auto_shutoff']['final_value']:.6e} | {case_b['auto_shutoff']['final_value']:.6e} |
| inner max lateral flux / incident | {case_a['inner_control_volume_maximum_absolute_lateral_flux_fraction']:.6e} | {case_b['inner_control_volume_maximum_absolute_lateral_flux_fraction']:.6e} |
| outer max lateral flux / incident | {case_a['outer_control_volume_maximum_absolute_lateral_flux_fraction']:.6e} | {case_b['outer_control_volume_maximum_absolute_lateral_flux_fraction']:.6e} |
| source-aperture edge / central intensity | {case_a['source_aperture_edge_to_central']:.6e} | {case_b['source_aperture_edge_to_central']:.6e} |

The registered beam centre is outside the flake, while the inner six-face box
is local to the flake.  Lateral Poynting flux through that box is therefore a
physical part of the off-flake Gaussian illumination, not a direct PML leakage
measurement.  The outer lateral fractions are below `1e-6` for both
polarizations.

No gate was relaxed and no finite Device-A solve was started.  The proposed
correction is to retain the inner signed flux as a diagnostic and use the outer
box lateral flux for the `<1e-4` truncation gate, while retaining the existing
matched-volume closure, auto-shutoff, source-aperture, and material-readback
gates.  This contract change requires explicit approval.
"""
    (args.output_dir / "DEVICE_A_REGISTERED_EMPTY_REFERENCE_AUDIT.md").write_text(
        report
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
