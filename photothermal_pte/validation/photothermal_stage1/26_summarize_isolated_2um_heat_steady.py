#!/usr/bin/env python3
"""Write compact reports for the isolated 2 um steady-state HEAT controls."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import config_stage1 as config
from lumerical_api import write_json


BASELINE_COMMIT = "be2cbc2c9c77bbcc0265ce2c293affdbb08105de"
REPORT_NAME = "isolated_2um_heat_steady"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-results", required=True)
    parser.add_argument(
        "--report-dir",
        default=str(config.REPOSITORY_ROOT / "reports" / REPORT_NAME),
    )
    parser.add_argument(
        "--latest-status",
        default=str(config.REPOSITORY_ROOT / "reports" / "LATEST_STATUS.md"),
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return value


def percent(value: Any) -> str:
    return "unavailable" if value is None else f"{100.0 * float(value):.9f}%"


def write_cases_csv(path: Path, control: dict[str, Any]) -> None:
    fields = (
        "case_id",
        "execution_status",
        "lateral_span_um",
        "Si_depth_um",
        "mesh_label",
        "G_top_W_m2K",
        "G_bottom_W_m2K",
        "G_oxide_Si_W_m2K",
        "top_boundary",
        "generated_power_W",
        "escaped_power_W",
        "energy_residual",
        "T_max_K",
        "TaIrTe4_average_temperature_K",
        "hotspot_x_m",
        "hotspot_y_m",
        "hotspot_z_m",
        "top_interface_temperature_jump_K",
        "bottom_interface_temperature_jump_K",
        "bottom_outflow_W",
        "lateral_outflow_W",
        "top_convection_outflow_W",
        "raw_artifact_manifest",
        "notes",
    )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "case_id": "controls_gate",
                "execution_status": control.get("status", "unknown"),
                "G_oxide_Si_W_m2K": control.get("sweep_contract", {}).get(
                    "G_oxide_Si_W_m2K"
                ),
                "notes": "; ".join(control.get("blockers", [])),
            }
        )


def api_report(control: dict[str, Any]) -> str:
    prior = control.get("prior_v261_api_evidence", {})
    live = control.get("live_v261_api_probe", {})
    interface = live.get("interface_G", {})
    live_exception = live.get("exception")
    if live_exception:
        live_exception = (
            "DEVICE startup failed: Ansys license-sharing client did not "
            "publish its server port"
        )
    else:
        live_exception = "none"
    return "\n".join(
        [
            "# Anisotropic kappa and interface-G API report",
            "",
            f"Optical baseline: `{BASELINE_COMMIT}`.",
            "",
            "## TaIrTe4 conductivity tensor",
            "",
            f"- Requested diagonal: `{prior.get('diagonal_request_W_mK')}` W/(m K)",
            f"- v261 round-trip value: `{prior.get('diagonal_return')}`",
            f"- Scalar control passed: `{prior.get('scalar_round_trip')}`",
            f"- Diagonal round trip passed: `{prior.get('diagonal_round_trip')}`",
            f"- Gate: `{prior.get('status')}`",
            "- No isotropic average or isotropic override was used.",
            "- `kz = 1.0 W/(m K)` is an estimated value.",
            "",
            "The installed Solid material property exposes the constant thermal",
            "conductivity field used by the prior probe. The official scripting",
            "example documents a scalar `thermal conductivity.constant`:",
            "https://optics.ansys.com/hc/en-us/articles/360034919233-Creating-and-modifying-thermal-materials-from-a-script",
            "",
            "## Fresh v261 probe",
            "",
            f"- Attempted: `{live.get('attempted')}`",
            f"- Status: `{live.get('status')}`",
            f"- Exception summary: `{live_exception}`",
            "",
            "A fresh DEVICE session could not be established on the current host",
            "because the Ansys license-sharing client did not publish its server",
            "port. The prior v261 result remains direct solver-API evidence, but a",
            "new round trip could not be recorded.",
            "",
            "## Interface conductance",
            "",
            f"- Internal domain-to-domain G verified: `{interface.get('verified', False)}`",
            f"- Gate: `{interface.get('status', 'BLOCKED_INTERFACE_G_UNVERIFIED')}`",
            "",
            "The requested finite conductances cannot be represented by silently",
            "assuming perfect contact. The official HEAT boundary documentation",
            "defines thermal impedance as a boundary thermal insulance in m2 K/W:",
            "https://optics.ansys.com/hc/en-us/articles/360034398314-Boundary-Conditions-in-HEAT-Simulation-Object",
            "",
            "A live two-domain analytic solve must still demonstrate that the",
            "chosen internal-interface API realizes `DeltaT = q''/G` and that the",
            "configured value survives save/load. No interface-G full-device case",
            "was started.",
            "",
        ]
    )


def final_report(control: dict[str, Any]) -> str:
    q = control.get("q_import_control", {})
    analytics = control.get("analytic_controls", {})
    lines = [
        "# Isolated 2 um TaIrTe4 steady-state HEAT validation",
        "",
        f"**Status: {control.get('status', 'unknown')}.**",
        "",
        f"- Optical baseline: `{BASELINE_COMMIT}`",
        "- Optical production code and validated Q values: unchanged",
        "- Normalization: UNIT_RESPONSE_MODE, 1 W/m2",
        "- Reported thermal quantity when unblocked: DeltaT / incident intensity",
        "- Transient, PTE current, adjoint, gradient, and optimization: not run",
        "",
        "## Mandatory gate results",
        "",
        "| Gate | Result | Evidence |",
        "|---|---|---|",
        (
            "| Validated-Q full-grid reintegration | PASS | "
            f"relative error `{q.get('full_grid_reintegration_relative_error')}` |"
        ),
        (
            "| Q compatibility with 2 um footprint | FAIL | "
            f"inside `{percent(q.get('inside_power_fraction'))}`; "
            f"outside `{percent(q.get('outside_power_fraction'))}` |"
        ),
    ]
    for name, item in analytics.items():
        lines.append(
            f"| {name} offline reference | "
            f"{'PASS' if item.get('offline_reference_passed') else 'FAIL'} | "
            f"solver verified `{item.get('solver_verified')}` |"
        )
    prior = control.get("prior_v261_api_evidence", {})
    lines.extend(
        [
            (
                "| TaIrTe4 diagonal kappa API round trip | FAIL | "
                f"requested `{prior.get('diagonal_request_W_mK')}`, "
                f"returned `{prior.get('diagonal_return')}` |"
            ),
            "| Interface-G analytic solver control | NOT RUN | live DEVICE unavailable |",
            "",
            "## Fail-closed blockers",
            "",
        ]
    )
    for blocker in control.get("blockers", []):
        lines.append(f"- `{blocker}`")
    lines.extend(
        [
            "",
            "The immutable production Q grid spans 6 um by 6 um. Its total",
            f"power is `{q.get('P_Q_full_validated_grid_W')} W`, while the power",
            "inside the requested 2 um by 2 um TaIrTe4 footprint is",
            f"`{q.get('P_Q_inside_requested_2um_TaIrTe4_W')} W`. Restricting the",
            "source to the finite flake would discard most of the validated",
            "power and violate the 0.5% conservation limit. Cropping, tiling,",
            "gain, smoothing, and rescaling were not used.",
            "",
            "## Full-device result",
            "",
            "No full-device HEAT case was executed. Consequently there are no",
            "claims for T(x,y,z), heat flux, energy balance, lateral/depth",
            "convergence, or interface-G sweeps. This is required behavior because",
            "the task states that any failed control stops the workflow before the",
            "full 3-D model.",
            "",
            "To unblock the physical run, both of the following are required:",
            "",
            "1. A validated non-periodic Q artifact generated for the exact 2 um",
            "   by 2 um TaIrTe4 volume, preserving the <0.5% FDTD-to-HEAT power",
            "   identity without post-processing.",
            "2. A HEAT solver/version or verified material route that stores and",
            "   executes diag(14.4, 3.8, 1.0) W/(m K), plus a verified internal",
            "   interface-G route that passes the analytic temperature-jump test.",
            "",
        ]
    )
    return "\n".join(lines)


def latest_status(control: dict[str, Any]) -> str:
    q = control.get("q_import_control", {})
    return "\n".join(
        [
            "# Latest photothermal validation status",
            "",
            "## Isolated 2 um TaIrTe4 steady-state HEAT",
            "",
            "- Branch: `agent/validate-isolated-2um-heat-steady`",
            f"- Optical baseline: `{BASELINE_COMMIT}`",
            "- Optical code and validated production Q: unchanged",
            f"- Status: `{control.get('status')}`",
            "- Full-device HEAT cases executed: `false`",
            "- Transient/PTE/adjoint/gradient/optimization executed: `false`",
            "",
            "### Active blockers",
            "",
            *[f"- `{item}`" for item in control.get("blockers", [])],
            "",
            "### Key measurements",
            "",
            f"- Validated full-grid Q power: `{q.get('P_Q_full_validated_grid_W')} W`",
            (
                "- Q power inside requested 2 um footprint: "
                f"`{q.get('P_Q_inside_requested_2um_TaIrTe4_W')} W`"
            ),
            f"- Predicted import mismatch: `{percent(q.get('outside_power_fraction'))}`",
            "- Allowed import mismatch: `0.5%`",
            "- v261 diagonal-kappa request `[14.4, 3.8, 1.0]` returned `[0.0]`",
            "",
            "No isotropic fallback, Q clipping, gain, smoothing, rescaling, or",
            "periodic tiling is permitted.",
            "",
        ]
    )


def main() -> int:
    args = parse_args()
    control_path = Path(args.control_results).expanduser().resolve()
    report_dir = Path(args.report_dir).expanduser().resolve()
    latest_path = Path(args.latest_status).expanduser().resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    control = read_json(control_path)
    raw_manifest_path = control_path.with_name("artifact_manifest.json")

    compact_summary = {
        "status": control.get("status"),
        "validated": False,
        "baseline_optical_commit": BASELINE_COMMIT,
        "blockers": control.get("blockers", []),
        "unit_response_mode": control.get("q_import_control", {})
        .get("normalization", {})
        .get("unit_response_mode"),
        "incident_intensity_W_m2": control.get("q_import_control", {})
        .get("normalization", {})
        .get("incident_intensity_W_m2"),
        "Q_power": {
            key: control.get("q_import_control", {}).get(key)
            for key in (
                "P_Q_full_validated_grid_W",
                "P_Q_inside_requested_2um_TaIrTe4_W",
                "inside_power_fraction",
                "outside_power_fraction",
                "predicted_FDTD_to_HEAT_import_relative_error",
                "acceptance_limit",
            )
        },
        "anisotropic_kappa": {
            key: control.get("prior_v261_api_evidence", {}).get(key)
            for key in (
                "diagonal_request_W_mK",
                "diagonal_return",
                "diagonal_round_trip",
                "status",
            )
        },
        "analytic_controls": control.get("analytic_controls", {}),
        "full_device_executed": False,
        "cases_completed": 0,
        "domain_converged": False,
        "energy_balance_passed": False,
        "sweep_contract": control.get("sweep_contract", {}),
        "raw_control_results_manifest": {
            "filename": control_path.name,
            "not_committed_reason": "raw run output remains outside Git",
        },
    }
    write_json(report_dir / f"{REPORT_NAME}_summary.json", compact_summary)
    if raw_manifest_path.is_file():
        write_json(
            report_dir / "RAW_ARTIFACT_MANIFEST.json",
            read_json(raw_manifest_path),
        )
    write_cases_csv(report_dir / f"{REPORT_NAME}_cases.csv", control)
    (report_dir / "ANISOTROPIC_K_AND_INTERFACE_G_API_REPORT.md").write_text(
        api_report(control)
    )
    (report_dir / "ISOLATED_2UM_HEAT_STEADY_REPORT.md").write_text(
        final_report(control)
    )
    latest_path.write_text(latest_status(control))
    print(json.dumps(compact_summary, indent=2))
    return 0 if control.get("status") == "READY_FOR_FULL_DEVICE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
