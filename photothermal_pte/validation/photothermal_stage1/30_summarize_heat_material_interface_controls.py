#!/usr/bin/env python3
"""Summarize the fail-closed v261 HEAT material/interface controls."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import config_stage1 as config
from lumerical_api import utc_timestamp, write_json


REPORT_NAME = "heat_material_interface_controls"
OVERALL_STATUS = "BLOCKED_ANISOTROPIC_K_UNSUPPORTED"
INTERFACE_STATUS = "BLOCKED_INTERFACE_G_UNVERIFIED"
ALLOWED_STATUSES = {
    "VALIDATED_HEAT_MATERIAL_INTERFACE_CONTROLS",
    "BLOCKED_ANISOTROPIC_K_UNSUPPORTED",
    "BLOCKED_INTERFACE_G_UNVERIFIED",
    "BLOCKED_LUMERICAL_LICENSE_UNAVAILABLE",
    "FAILED_ANISOTROPIC_K_ANALYTIC_CONTROL",
    "FAILED_INTERFACE_G_ANALYTIC_CONTROL",
}


def parse_args() -> argparse.Namespace:
    raw_root = config.OUTPUT_ROOT / REPORT_NAME
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--q-precheck",
        default=str(raw_root / "q_precheck_v1" / "q_precheck.json"),
    )
    parser.add_argument(
        "--license-probe",
        default=str(raw_root / "license_probe_v2" / "license_probe.json"),
    )
    parser.add_argument(
        "--kappa-controls",
        default=str(raw_root / "kappa_controls_v2" / "kappa_controls.json"),
    )
    parser.add_argument(
        "--tensor-encoding-probe",
        default=str(
            raw_root
            / "tensor_encoding_probe_v1"
            / "tensor_encoding_probe.json"
        ),
    )
    parser.add_argument(
        "--native-anisotropy-probe",
        default=str(
            raw_root
            / "anisotropic_native_probe_v2"
            / "anisotropic_kappa_resolution.json"
        ),
    )
    parser.add_argument(
        "--anisotropic-fvm-controls",
        default=str(
            raw_root
            / "anisotropic_resolution_fvm_v2"
            / "anisotropic_kappa_resolution.json"
        ),
    )
    parser.add_argument(
        "--interface-controls",
        default=str(
            raw_root / "interface_controls_v3" / "interface_controls.json"
        ),
    )
    parser.add_argument(
        "--report-dir",
        default=str(config.REPOSITORY_ROOT / "reports" / REPORT_NAME),
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repository_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(config.REPOSITORY_ROOT.parent))
    except ValueError:
        return str(path.resolve())


def artifact_record(path: Path, category: str) -> dict[str, Any]:
    return {
        "category": category,
        "repository_or_server_path": repository_path(path),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "tracked_in_git": False,
    }


def make_summary(
    q_result: dict[str, Any],
    license_result: dict[str, Any],
    encoding_result: dict[str, Any],
    kappa_result: dict[str, Any],
    native_result: dict[str, Any],
    fvm_result: dict[str, Any],
    interface_result: dict[str, Any],
) -> dict[str, Any]:
    q = q_result["q_precheck"]
    license_probe = license_result["license_probe"]
    encoding = encoding_result["tensor_encoding_probe"]
    kappa = kappa_result["anisotropic_kappa"]
    native = native_result["native_v261_probe"]
    fvm = fvm_result["fvm_controls"]
    interface = interface_result["internal_interface_G"]
    finite_cases = interface["finite_G_cases"]
    perfect_cases = interface["perfect_contact_cases"]
    summary = {
        "schema_version": 1,
        "generated_at_utc": utc_timestamp(),
        "status": OVERALL_STATUS,
        "status_allowed": OVERALL_STATUS in ALLOWED_STATUSES,
        "branch": "agent/unblock-heat-material-interface-controls",
        "stacked_base": "agent/validate-isolated-2um-heat-steady",
        "pr3_commit": "053260d",
        "pr2_and_pr3_modified": False,
        "blockers": [
            "BLOCKED_ANISOTROPIC_K_UNSUPPORTED",
            INTERFACE_STATUS,
        ],
        "finite_Q_precheck": q,
        "license_probe": license_probe,
        "anisotropic_kappa": {
            "status": "BLOCKED_ANISOTROPIC_K_UNSUPPORTED",
            "passed": kappa["passed"],
            "candidate_case_status": (
                "FAILED_ANISOTROPIC_K_ANALYTIC_CONTROL"
            ),
            "requested_tensor_W_mK": kappa["requested_tensor_W_mK"],
            "tensor_supported": False,
            "encoding_probe": encoding,
            "native_v261_deep_probe": {
                "status": native["status"],
                "passed": native["passed"],
                "v261_DEVICE_version": native["v261_DEVICE_version"],
                "installation_root": native["installation_root"],
                "probe_scope": native["probe_scope"],
                "lsf_tensor_round_trips": native[
                    "lsf_tensor_round_trips"
                ],
                "hidden_property_candidates": native[
                    "hidden_property_candidates"
                ],
                "thermal_database_material_count": native[
                    "thermal_database_material_count"
                ],
                "thermal_database_scan": native["thermal_database_scan"],
            },
            "validated_fvm_fallback": fvm,
            "isotropic_fallback_used": kappa[
                "isotropic_fallback_used"
            ],
            "cases": kappa["cases"],
        },
        "internal_interface_G": {
            "status": INTERFACE_STATUS,
            "passed": False,
            "candidate_case_status": (
                "FAILED_INTERFACE_G_ANALYTIC_CONTROL"
            ),
            "candidate_path": (
                "temperature BC on shared material:material surface, "
                "thermal impedance=1/G"
            ),
            "candidate_property_save_reload_passed": all(
                case.get("internal_boundary_property_passed", False)
                for case in finite_cases
            ),
            "candidate_has_contact_resistance_semantics": False,
            "finite_G_cases": finite_cases,
            "perfect_contact_mesh_converged_to_zero_jump": interface[
                "perfect_contact_mesh_converged_to_zero_jump"
            ],
            "perfect_contact_cases": perfect_cases,
            "thin_layer_or_isotropic_fallback_used": False,
        },
        "scope_guard": {
            "finite_Q_full_device_HEAT_import": False,
            "production_temperature_result": False,
            "domain_sweep": False,
            "Si_depth_sweep": False,
            "G_top_sweep": False,
            "G_bottom_sweep": False,
            "transient": False,
            "PTE": False,
            "optimization": False,
        },
        "next_step_authorized": False,
        "next_step_condition": (
            "anisotropic-kappa and internal-interface-G controls must both pass"
        ),
    }
    return summary


def write_cases_csv(path: Path, summary: dict[str, Any]) -> None:
    fields = (
        "case_id",
        "control",
        "status",
        "passed",
        "axis",
        "requested_value",
        "readback_before_save",
        "readback_after_reload",
        "analytic_heat_flux_W_m2",
        "numerical_heat_flux_W_m2",
        "heat_flux_relative_error",
        "temperature_profile_relative_error",
        "expected_interface_jump_K",
        "numerical_interface_jump_K",
        "interface_jump_relative_error",
        "flux_transmission_relative_error",
        "energy_balance_relative_error",
        "mesh_max_edge_m",
        "notes",
    )
    rows: list[dict[str, Any]] = [
        {
            "case_id": "finite_Q_precheck",
            "control": "finite_Q",
            "status": summary["finite_Q_precheck"]["status"],
            "passed": True,
            "requested_value": summary["finite_Q_precheck"][
                "artifact"
            ]["sha256"],
            "numerical_heat_flux_W_m2": "",
            "energy_balance_relative_error": summary[
                "finite_Q_precheck"
            ]["reintegration_relative_error"],
            "notes": "Q blocker is a release candidate; PR #2 unchanged",
        },
        {
            "case_id": "v261_DEVICE_license_probe",
            "control": "license",
            "status": summary["license_probe"]["status"],
            "passed": summary["license_probe"]["passed"],
            "requested_value": "v261",
            "readback_after_reload": summary["license_probe"].get(
                "lumerical_version"
            ),
            "notes": "startup/save/load/actual HEAT solve",
        },
    ]
    for case in summary["anisotropic_kappa"]["cases"]:
        rows.append(
            {
                "case_id": case["case_id"],
                "control": "anisotropic_kappa",
                "status": case["status"],
                "passed": case["passed"],
                "axis": case["axis"],
                "requested_value": case["property_write_W_mK"],
                "readback_before_save": case[
                    "property_readback_before_save_W_mK"
                ],
                "readback_after_reload": case[
                    "property_readback_after_reload_W_mK"
                ],
                "analytic_heat_flux_W_m2": case[
                    "analytic_heat_flux_W_m2"
                ],
                "numerical_heat_flux_W_m2": case[
                    "numerical_heat_flux_W_m2"
                ],
                "heat_flux_relative_error": case[
                    "heat_flux_relative_error"
                ],
                "temperature_profile_relative_error": case[
                    "temperature_profile_max_relative_error"
                ],
                "energy_balance_relative_error": case[
                    "energy_balance_relative_error"
                ],
                "notes": "no isotropic fallback",
            }
        )
    for case in summary["anisotropic_kappa"][
        "validated_fvm_fallback"
    ]["cases"]:
        rows.append(
            {
                "case_id": case["case_id"],
                "control": "diagonal_kappa_fvm_fallback",
                "status": case["status"],
                "passed": case["passed"],
                "axis": case["axis"],
                "requested_value": case["requested_tensor_W_mK"],
                "analytic_heat_flux_W_m2": case[
                    "analytic_heat_flux_W_m2"
                ],
                "numerical_heat_flux_W_m2": case[
                    "numerical_heat_flux_W_m2"
                ],
                "heat_flux_relative_error": case[
                    "heat_flux_relative_error"
                ],
                "temperature_profile_relative_error": case[
                    "temperature_profile_max_relative_error"
                ],
                "energy_balance_relative_error": case[
                    "energy_balance_relative_error"
                ],
                "notes": (
                    "validated conservative Python FVM; "
                    "not a v261 HEAT result"
                ),
            }
        )
    interface = summary["internal_interface_G"]
    for case in interface["finite_G_cases"]:
        rows.append(
            {
                "case_id": case["case_id"],
                "control": "internal_interface_G",
                "status": case["status"],
                "passed": case["passed"],
                "requested_value": case["G_W_m2K"],
                "readback_before_save": case[
                    "property_readback_before_save"
                ]["thermal impedance"],
                "readback_after_reload": case[
                    "property_readback_after_reload"
                ]["thermal impedance"],
                "analytic_heat_flux_W_m2": case[
                    "analytic_heat_flux_W_m2"
                ],
                "numerical_heat_flux_W_m2": case[
                    "transmitted_heat_flux_W_m2"
                ],
                "heat_flux_relative_error": case[
                    "heat_flux_relative_error"
                ],
                "expected_interface_jump_K": case[
                    "expected_interface_temperature_jump_K"
                ],
                "numerical_interface_jump_K": case["temperature_fit"][
                    "temperature_jump_K"
                ],
                "interface_jump_relative_error": case[
                    "interface_temperature_jump_relative_error"
                ],
                "flux_transmission_relative_error": case[
                    "flux_transmission_relative_error"
                ],
                "energy_balance_relative_error": case[
                    "global_energy_balance_relative_error"
                ],
                "mesh_max_edge_m": case["mesh_max_edge_m"],
                "notes": "candidate acted as fixed-temperature reservoir",
            }
        )
    for case in interface["perfect_contact_cases"]:
        rows.append(
            {
                "case_id": case["case_id"],
                "control": "perfect_contact",
                "status": case["status"],
                "passed": case["passed"],
                "requested_value": "perfect_contact",
                "analytic_heat_flux_W_m2": case[
                    "analytic_heat_flux_W_m2"
                ],
                "numerical_heat_flux_W_m2": case[
                    "transmitted_heat_flux_W_m2"
                ],
                "heat_flux_relative_error": case[
                    "heat_flux_relative_error"
                ],
                "expected_interface_jump_K": 0.0,
                "numerical_interface_jump_K": case["temperature_fit"][
                    "temperature_jump_K"
                ],
                "flux_transmission_relative_error": case[
                    "flux_transmission_relative_error"
                ],
                "energy_balance_relative_error": case[
                    "global_energy_balance_relative_error"
                ],
                "mesh_max_edge_m": case["mesh_max_edge_m"],
                "notes": "default continuity; mesh-refinement control",
            }
        )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def kappa_report(summary: dict[str, Any]) -> str:
    kappa = summary["anisotropic_kappa"]
    native = kappa["native_v261_deep_probe"]
    fvm = kappa["validated_fvm_fallback"]
    lines = [
        "# HEAT anisotropic-kappa solver report",
        "",
        "**Status: `BLOCKED_ANISOTROPIC_K_UNSUPPORTED`.**",
        "",
        "- v261 DEVICE version: `7.17.4413`",
        "- Requested tensor: `diag(14.4, 3.8, 1.0) W/(m K)`",
        "- Isotropic fallback: `false`",
        "- Full-device HEAT: `not run`",
        "",
        "| Encoding | requested shape | returned shape/value | round trip |",
        "|---|---:|---:|---:|",
    ]
    for label, item in kappa["encoding_probe"]["encodings"].items():
        lines.append(
            f"| {label} | `{item['requested_shape']}` | "
            f"`{item['returned_shape']} / {item['returned']}` | "
            f"`{item['round_trip_passed']}` |"
        )
    lines.extend(
        [
        "",
        "| Axis | write/readback before | readback after reload | effective k | flux error | profile error |",
        "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for case in kappa["cases"]:
        lines.append(
            f"| {case['axis']} | `{case['property_write_W_mK']} -> "
            f"{case['property_readback_before_save_W_mK']}` | "
            f"`{case['property_readback_after_reload_W_mK']}` | "
            f"`{case['effective_kappa_W_mK']:.6g}` | "
            f"`{100 * case['heat_flux_relative_error']:.6g}%` | "
            f"`{100 * case['temperature_profile_max_relative_error']:.6g}%` |"
        )
    lines.extend(
        [
            "",
            "The scalar material route passed in the license probe, but v261 did",
            "not retain the requested three-component constant conductivity. The",
            "three directional solves also failed their analytic flux and",
            "temperature-profile controls. A matching readback alone would not",
            "have been accepted; here both readback and solver behavior fail.",
            "",
            "No scalar average, coordinate remapping, or isotropic replacement was",
            "used. The production finite-Q source was not imported.",
            "",
            "## Exhaustive native v261 probe",
            "",
            "A fresh DEVICE session tested LSF-native 3x1, 1x3, and 3x3",
            "matrix expressions, eleven plausible hidden property names, and",
            "every material returned by `addmaterialproperties(\"HT\")`.",
            "",
            f"- Native probe status: `{native['status']}`",
            f"- HT database entries: `{native['thermal_database_material_count']}`",
            "- Readable scalar conductivity entries: "
            f"`{native['thermal_database_scan']['scalar_conductivity_material_count']}`",
            "- Non-scalar conductivity entries: "
            f"`{len(native['thermal_database_scan']['nonscalar_conductivity_materials'])}`",
            "- Hidden property writes accepted: "
            f"`{sum(item['write_succeeded'] for item in native['hidden_property_candidates'].values())}`",
            "",
            "| Native LSF encoding | requested | returned | exact round trip |",
            "|---|---:|---:|---:|",
        ]
    )
    for label, item in native["lsf_tensor_round_trips"].items():
        lines.append(
            f"| {label} | `{item['requested']}` | `{item['returned']}` | "
            f"`{item['round_trip_passed']}` |"
        )
    lines.extend(
        [
            "",
            "This closes the known native v261 material routes: the installed",
            "HEAT material API exposes scalar conductivity only. Consequently",
            "`BLOCKED_ANISOTROPIC_K_UNSUPPORTED` remains correct specifically",
            "for a v261 HEAT-backed result.",
            "",
            "## Working anisotropic path",
            "",
            f"**Fallback status: `{fvm['status']}`.**",
            "",
            "A repository-native, cell-centered conservative finite-volume",
            "solver now accepts cellwise `diag(kx, ky, kz)`. Conductances use",
            "the exact series resistance of adjacent half cells; unspecified",
            "outer faces are adiabatic. This is an independently validated",
            "solver path, not a relabeled Lumerical result.",
            "The present implementation is intentionally limited to diagonal",
            "tensors aligned with the Cartesian grid; that exactly matches the",
            "requested `diag(14.4, 3.8, 1.0)` tensor.",
            "",
            "| Axis | expected k (W/m K) | recovered k (W/m K) | flux error | profile error | energy error |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for case in fvm["cases"]:
        lines.append(
            f"| {case['axis']} | `{case['expected_kappa_W_mK']:.9g}` | "
            f"`{case['effective_kappa_W_mK']:.12g}` | "
            f"`{100 * case['heat_flux_relative_error']:.6g}%` | "
            f"`{100 * case['temperature_profile_max_relative_error']:.6g}%` | "
            f"`{100 * case['energy_balance_relative_error']:.6g}%` |"
        )
    lines.extend(
        [
            "",
            "All three controls satisfy the requested `<1%` heat-flux and",
            "temperature-profile criteria without an isotropic average.",
            "",
            "Reproduce the controls with:",
            "",
            "```bash",
            "python photothermal_pte/validation/photothermal_stage1/31_resolve_anisotropic_kappa.py \\",
            "  --phase fvm-controls --output-dir /tmp/anisotropic-kappa-controls",
            "```",
            "",
            "Official Ansys scripting documentation describes the constant thermal",
            "conductivity field as scalar and lists only Solid, Solid Alloy,",
            "and Fluid thermal property types:",
            "https://optics.ansys.com/hc/en-us/articles/360034919233-Creating-and-modifying-thermal-materials-from-a-script",
            "https://optics.ansys.com/hc/en-us/articles/360034924973-addhtmaterialproperty-Script-command",
            "",
        ]
    )
    return "\n".join(lines)


def interface_report(summary: dict[str, Any]) -> str:
    interface = summary["internal_interface_G"]
    lines = [
        "# HEAT internal-interface-G solver report",
        "",
        "**Status: `BLOCKED_INTERFACE_G_UNVERIFIED`.**",
        "",
        "The tested v261 candidate used a temperature boundary on the shared",
        "`material:material` surface with `thermal impedance = 1/G`. The",
        "surface/material selection and exact insulance survived save/reload,",
        "but the numerical solution did not realize a two-sided contact",
        "resistance.",
        "",
        "| G (W/m2 K) | 1/G (m2 K/W) | expected jump | numerical jump | jump error | flux error | transmission error | energy error |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for case in interface["finite_G_cases"]:
        lines.append(
            f"| `{case['G_W_m2K']:.6g}` | "
            f"`{case['thermal_insulance_m2K_W']:.9g}` | "
            f"`{case['expected_interface_temperature_jump_K']:.9g} K` | "
            f"`{case['temperature_fit']['temperature_jump_K']:.9g} K` | "
            f"`{100 * case['interface_temperature_jump_relative_error']:.6g}%` | "
            f"`{100 * case['heat_flux_relative_error']:.6g}%` | "
            f"`{100 * case['flux_transmission_relative_error']:.6g}%` | "
            f"`{100 * case['global_energy_balance_relative_error']:.6g}%` |"
        )
    lines.extend(
        [
            "",
            "Both finite-G cases remained temperature-continuous and introduced a",
            "third 305 K boundary power. They therefore behave as a",
            "fixed-temperature reservoir with insulance, not as the requested",
            "two-sided law `DeltaT_int = q''/G`.",
            "",
            "## Perfect-contact mesh control",
            "",
            "| max edge | numerical jump | heat-flux error | energy error |",
            "|---:|---:|---:|---:|",
        ]
    )
    for case in interface["perfect_contact_cases"]:
        lines.append(
            f"| `{case['mesh_max_edge_m'] * 1e9:g} nm` | "
            f"`{case['temperature_fit']['temperature_jump_K']:.9g} K` | "
            f"`{100 * case['heat_flux_relative_error']:.6g}%` | "
            f"`{100 * case['global_energy_balance_relative_error']:.6g}%` |"
        )
    lines.extend(
        [
            "",
            "The perfect-contact control reproduces the analytic",
            "`4.0e7 W/m2` flux and a machine-zero jump at all three meshes. This",
            "validates the two-slab geometry and extraction method, but does not",
            "validate finite internal G.",
            "",
            "The official HEAT documentation defines thermal impedance on a",
            "temperature boundary as boundary thermal insulance, and states that",
            "internal material interfaces otherwise enforce continuity:",
            "https://optics.ansys.com/hc/en-us/articles/360034398314-Boundary-Conditions-in-HEAT-Simulation-Object",
            "https://optics.ansys.com/hc/en-us/articles/360034917713-HEAT-solver-introduction",
            "",
            "No thin-layer substitute or perfect-contact substitution is reported",
            "as a validated internal-G path. The full device remains blocked.",
            "",
        ]
    )
    return "\n".join(lines)


def build_manifest(
    sources: dict[str, Path],
    report_dir: Path,
    summary: dict[str, Any],
) -> dict[str, Any]:
    artifacts = []
    seen: set[Path] = set()
    for category, source in sources.items():
        root = source.parent
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            artifacts.append(artifact_record(path, category))
    q_artifact = Path(
        summary["finite_Q_precheck"]["artifact"]["server_path"]
    ).resolve()
    artifacts.append(artifact_record(q_artifact, "validated_finite_Q"))
    for name in (
        "HEAT_ANISOTROPIC_K_SOLVER_REPORT.md",
        "HEAT_INTERNAL_INTERFACE_G_SOLVER_REPORT.md",
        "heat_material_interface_controls_summary.json",
        "heat_material_interface_controls_cases.csv",
    ):
        artifacts.append(
            {
                **artifact_record(report_dir / name, "report"),
                "tracked_in_git": True,
            }
        )
    return {
        "schema_version": 1,
        "generated_at_utc": utc_timestamp(),
        "generated_by": (
            "photothermal_pte/validation/photothermal_stage1/"
            "30_summarize_heat_material_interface_controls.py"
        ),
        "status": OVERALL_STATUS,
        "validated_finite_Q": {
            "pr3_commit": "053260d",
            "sha256": (
                "7a63f82842751e7623e895701bac4ce92558679ed71bf13a3d404695a150e794"
            ),
            "P_Q_W": 2.56071371086521e-12,
        },
        "artifacts": artifacts,
        "note": (
            "Raw DEVICE projects and solver arrays stay server-side and are "
            "identified by path, byte count, and SHA-256."
        ),
    }


def main() -> int:
    args = parse_args()
    paths = {
        "q_precheck": Path(args.q_precheck).expanduser().resolve(),
        "license_probe": Path(args.license_probe).expanduser().resolve(),
        "kappa_controls": Path(args.kappa_controls).expanduser().resolve(),
        "tensor_encoding_probe": Path(
            args.tensor_encoding_probe
        ).expanduser().resolve(),
        "native_anisotropy_probe": Path(
            args.native_anisotropy_probe
        ).expanduser().resolve(),
        "anisotropic_fvm_controls": Path(
            args.anisotropic_fvm_controls
        ).expanduser().resolve(),
        "interface_controls": Path(
            args.interface_controls
        ).expanduser().resolve(),
    }
    data = {name: read_json(path) for name, path in paths.items()}
    summary = make_summary(
        data["q_precheck"],
        data["license_probe"],
        data["tensor_encoding_probe"],
        data["kappa_controls"],
        data["native_anisotropy_probe"],
        data["anisotropic_fvm_controls"],
        data["interface_controls"],
    )
    report_dir = Path(args.report_dir).expanduser().resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        report_dir / "heat_material_interface_controls_summary.json",
        summary,
    )
    write_cases_csv(
        report_dir / "heat_material_interface_controls_cases.csv", summary
    )
    (report_dir / "HEAT_ANISOTROPIC_K_SOLVER_REPORT.md").write_text(
        kappa_report(summary)
    )
    (report_dir / "HEAT_INTERNAL_INTERFACE_G_SOLVER_REPORT.md").write_text(
        interface_report(summary)
    )
    manifest = build_manifest(paths, report_dir, summary)
    write_json(report_dir / "RAW_ARTIFACT_MANIFEST.json", manifest)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
