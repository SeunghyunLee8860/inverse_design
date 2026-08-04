#!/usr/bin/env python3
"""Summarize the registered sparse Device-A Maxwell/thermal/PTE scan.

The input directories are immutable raw artifacts.  This script reads their
JSON/NPZ provenance, recomputes the position-matched reference audit, and
writes only compact publishable summaries and plots.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CASES = (
    (-1.0, "a", "scan_dm1_finite_a_20260802", "scan_dm1_thermal_a_intersection_20260802"),
    (1.0, "a", "scan_d1_finite_a_retry_20260802", "scan_d1_thermal_a_intersection_20260802"),
    (1.0, "b", "scan_d1_finite_b_20260802", "scan_d1_thermal_b_intersection_20260802"),
    (3.0, "a", "phase1_finite_a_64um_20260802", "thermal_a_intersection_expanded_60um_100nm_20260802"),
    (3.0, "b", "phase1_finite_b_64um_20260802", "thermal_b_intersection_expanded_60um_100nm_20260802"),
    (5.0, "a", "scan_d5_finite_a_20260802", "scan_d5_thermal_a_intersection_20260802"),
    (5.0, "b", "scan_d5_finite_b_20260802", "scan_d5_thermal_b_intersection_20260802"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path: Path, role: str) -> dict[str, Any]:
    return {
        "role": role,
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
        "committed_to_git": False,
    }


def source_contract(case: dict[str, Any]) -> dict[str, Any]:
    return case["pre_run_contract"]["geometry"]["source"]


def recompute_reference_audit(case: dict[str, Any]) -> dict[str, Any]:
    reference_path = Path(
        case["run_result"]["normalization"]["incident_reference"]["case_result_path"]
    )
    reference = json.loads(reference_path.read_text())
    finite_source = source_contract(case)
    reference_source = source_contract(reference)
    finite_geometry = case["run_result"]["normalization"]["incident_reference"][
        "active_geometry"
    ]
    checks = {
        "polarization": reference_source["polarization_axis"]
        == finite_source["polarization_axis"],
        "beam_center_lt_1fm": bool(
            np.max(
                np.abs(
                    np.asarray(reference_source["beam_center_m"])
                    - np.asarray(finite_source["beam_center_m"])
                )
            )
            < 1e-15
        ),
        "physical_target_waist": reference_source["physical_target_waist_radius_m"]
        == finite_source["physical_target_waist_radius_m"],
        "source_object_waist": reference_source["Lumerical_source_object_waist_radius_m"]
        == finite_source["Lumerical_source_object_waist_radius_m"],
        "source_span": reference_source["source_span_m"] == finite_source["source_span_m"],
        "pulse_band": reference_source["numerical_pulse_band_m"]
        == finite_source["numerical_pulse_band_m"],
        "domain_um": reference["domain_um"] == finite_geometry["domain_um"],
        "pml_layers": reference["pml_layers"] == finite_geometry["pml_layers"],
        "flake_dz_nm": reference["flake_dz_nm"] == finite_geometry["flake_dz_nm"],
        "substrate_model": reference["pre_run_contract"]["geometry"]["substrate_optical_contract"]["model"]
        == case["pre_run_contract"]["geometry"]["substrate_optical_contract"]["model"],
    }
    return {
        "reference_case_result": str(reference_path.resolve()),
        "finite_beam_center_m": finite_source["beam_center_m"],
        "reference_beam_center_m": reference_source["beam_center_m"],
        "checks": checks,
        "passed": all(checks.values()),
    }


def load_record(
    artifact_root: Path,
    distance_um: float,
    polarization: str,
    optical_name: str,
    thermal_name: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    optical_dir = artifact_root / optical_name
    thermal_dir = artifact_root / thermal_name
    optical_path = optical_dir / "case_result.json"
    thermal_path = thermal_dir / "summary.json"
    optical = json.loads(optical_path.read_text())
    thermal = json.loads(thermal_path.read_text())
    run = optical["run_result"]
    audit = recompute_reference_audit(optical)
    reference_path = Path(audit["reference_case_result"])
    record = {
        "scan_distance_um": distance_um,
        "polarization": polarization,
        "beam_center_x_um": audit["finite_beam_center_m"][0] * 1e6,
        "beam_center_y_um": audit["finite_beam_center_m"][1] * 1e6,
        "optical_status": optical["status"],
        "position_matched_reference_passed": audit["passed"],
        "P_Q_W_at_unit_central_intensity": run["P_Q_W"],
        "P_six_W_at_unit_central_intensity": run["P_six_face_W"],
        "six_face_closure": run["six_face_relative_closure"],
        "auto_shutoff": run["auto_shutoff"]["final_value"],
        "P_Qx_W": run["component_power_W"]["x"],
        "P_Qy_W": run["component_power_W"]["y"],
        "P_Qz_W": run["component_power_W"]["z"],
        "TaIrTe4_exact_support_power_W_at_unit_central_intensity": run[
            "material_resolved_absorption"
        ]["P_Q_TaIrTe4_exact_support_W"],
        "mapped_TaIrTe4_power_W_at_284p40uW": thermal["mapping"]["P_Q_target_W"],
        "mapping_relative_power_error": thermal["mapping"]["mapping_relative_power_error"],
        "mapped_power_outside_TaIrTe4_W": thermal["mapping"]["mapped_power_outside_flake_W"],
        "Tmax_rise_K": thermal["thermal"]["Tmax_rise_K"],
        "TaIrTe4_volume_average_rise_K": thermal["thermal"][
            "TaIrTe4_volume_average_rise_K"
        ],
        "thermal_linear_residual_relative": thermal["thermal"][
            "linear_residual_relative"
        ],
        "thermal_energy_balance_relative_error": thermal["thermal"][
            "energy_balance_relative_error"
        ],
        "PTE_current_A": thermal["PTE_current_A_at_requested_incident_power"],
        "PTE_current_nA": thermal["PTE_current_A_at_requested_incident_power"] * 1e9,
        "predicted_resistance_ohm": thermal["two_terminal_resistance_audit"][
            "predicted_resistance_ohm"
        ],
        "measured_resistance_ohm": thermal["two_terminal_resistance_audit"][
            "published_measured_device_A_resistance_ohm"
        ],
        "optical_case_result_path": str(optical_path.resolve()),
        "thermal_summary_path": str(thermal_path.resolve()),
        "reference_audit": audit,
    }
    artifacts = [
        artifact(optical_path, f"d={distance_um:g} um E||{polarization} optical result"),
        artifact(optical_dir / "finite_q_on_artifact.npz", f"d={distance_um:g} um E||{polarization} raw optical NPZ"),
        artifact(optical_dir / "finite_2um_optical_q.fsp", f"d={distance_um:g} um E||{polarization} raw FSP"),
        artifact(reference_path, f"d={distance_um:g} um E||{polarization} matched empty-stack result"),
        artifact(thermal_path, f"d={distance_um:g} um E||{polarization} thermal/PTE result"),
        artifact(thermal_dir / "thermal_pte_fields.npz", f"d={distance_um:g} um E||{polarization} thermal/PTE fields"),
    ]
    return record, artifacts


def plot_scan(path: Path, records: list[dict[str, Any]]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    metrics = (
        ("PTE_current_nA", "Integrated PTE current", "nA"),
        ("mapped_TaIrTe4_power_W_at_284p40uW", "Mapped TaIrTe₄ power", "µW"),
        ("Tmax_rise_K", "Maximum temperature rise", "K"),
    )
    styles = {"a": ("o-", "tab:blue"), "b": ("s-", "tab:orange")}
    for axis, (key, title, unit) in zip(axes.flat[:3], metrics):
        for polarization in ("a", "b"):
            subset = sorted(
                (row for row in records if row["polarization"] == polarization),
                key=lambda row: row["scan_distance_um"],
            )
            x = np.asarray([row["scan_distance_um"] for row in subset])
            y = np.asarray([row[key] for row in subset])
            if key == "mapped_TaIrTe4_power_W_at_284p40uW":
                y = y * 1e6
            fmt, color = styles[polarization]
            axis.plot(x, y, fmt, color=color, linewidth=1.8, label=f"E || {polarization}")
            peak = int(np.argmax(np.abs(y)))
            axis.plot(x[peak], y[peak], "*", color=color, markersize=13)
        axis.set_title(title)
        axis.set_xlabel("registered Figure-3I scan coordinate d (µm)")
        axis.set_ylabel(unit)
        axis.grid(alpha=0.3)
        axis.legend()

    paired = {}
    for row in records:
        paired.setdefault(row["scan_distance_um"], {})[row["polarization"]] = row
    distances = sorted(distance for distance, value in paired.items() if set(value) == {"a", "b"})
    ratios = [
        abs(paired[distance]["b"]["PTE_current_A"])
        / abs(paired[distance]["a"]["PTE_current_A"])
        for distance in distances
    ]
    axes[1, 1].plot(distances, ratios, "d-", color="tab:purple", linewidth=1.8)
    axes[1, 1].axhline(1.0, color="black", linestyle="--", label="equal magnitude")
    axes[1, 1].set_title("Paired-point |Ib| / |Ia|")
    axes[1, 1].set_xlabel("registered Figure-3I scan coordinate d (µm)")
    axes[1, 1].set_ylabel("ratio")
    axes[1, 1].grid(alpha=0.3)
    axes[1, 1].legend()
    fig.suptitle(
        "Device-A registered sparse scan\n"
        "GPU Maxwell → literal material intersection → explicit 3D FVM → integrated PTE",
        fontsize=14,
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    raw_artifacts: list[dict[str, Any]] = []
    for case in CASES:
        record, artifacts = load_record(args.artifact_root, *case)
        records.append(record)
        raw_artifacts.extend(artifacts)

    # Preserve the shared-resource race as a diagnostic rather than hiding it.
    failed_race = args.artifact_root / "scan_d1_finite_a_20260802" / "case_result.json"
    if failed_race.exists():
        raw_artifacts.append(
            artifact(failed_race, "fail-closed parallel Lumerical shared-resource race diagnostic")
        )

    optical_gate = all(
        row["optical_status"] == "COMPLETED"
        and row["position_matched_reference_passed"]
        and row["six_face_closure"] < 0.005
        and row["auto_shutoff"] < 1e-5
        for row in records
    )
    thermal_gate = all(
        row["mapping_relative_power_error"] < 1e-12
        and row["mapped_power_outside_TaIrTe4_W"] == 0.0
        and row["thermal_linear_residual_relative"] < 1e-8
        and row["thermal_energy_balance_relative_error"] < 0.01
        for row in records
    )
    maxima = {}
    for polarization in ("a", "b"):
        subset = [row for row in records if row["polarization"] == polarization]
        maxima[polarization] = max(subset, key=lambda row: abs(row["PTE_current_A"]))
    sampled_peak_ratio = abs(maxima["b"]["PTE_current_A"]) / abs(
        maxima["a"]["PTE_current_A"]
    )
    paired_ratios = {
        str(distance): abs(next(row for row in records if row["scan_distance_um"] == distance and row["polarization"] == "b")["PTE_current_A"])
        / abs(next(row for row in records if row["scan_distance_um"] == distance and row["polarization"] == "a")["PTE_current_A"])
        for distance in (1.0, 3.0, 5.0)
    }

    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, cwd=Path.cwd()
        ).strip()
    except (subprocess.SubprocessError, OSError):
        commit = "UNKNOWN"
    status = "PARTIAL_REGISTERED_DEVICE_A_SPARSE_SCAN_CURRENT_TREND_OPPOSITE_PAPER"
    summary = {
        "status": status,
        "generation_commit": commit,
        "scenario": {
            "beam": "paper-like scalar-Gaussian scenario with explicitly assumed w0=8.75 um",
            "beam_is_paper_certified": False,
            "device_geometry": "Figure-digitized Device-A approximation, not unpublished CAD",
            "scan_coordinate": "Figure-3I registered nominal d; source y offset is d-3 um",
            "optical_solver": "Lumerical v261 GPU FDTD only",
            "thermal_solver": "independent conservative explicit 3D Cartesian FVM",
            "thermal_source": "TaIrTe4-only literal optical-cell/material intersection density",
            "SiO2_and_Si_role": "thermal materials; optical absorption excluded from this current comparison",
            "current": "full flake-volume Shockley-Ramo integral, never a one-point gradient",
            "incident_power_W": 284.40e-6,
        },
        "records": records,
        "gates": {
            "all_optical_cases_pass_closure_auto_shutoff_and_matched_reference": optical_gate,
            "all_thermal_cases_pass_mapping_residual_and_energy_balance": thermal_gate,
            "absolute_current_certified": False,
        },
        "sampled_maxima_not_continuous_peak_fit": {
            "a": {key: maxima["a"][key] for key in ("scan_distance_um", "PTE_current_A", "PTE_current_nA")},
            "b": {key: maxima["b"][key] for key in ("scan_distance_um", "PTE_current_A", "PTE_current_nA")},
            "abs_Ib_over_abs_Ia": sampled_peak_ratio,
            "paper_Figure3I_visual_reference_pA": {"a": 122.0, "b": 143.0, "abs_Ib_over_abs_Ia": 143.0 / 122.0},
        },
        "paired_point_abs_Ib_over_abs_Ia": paired_ratios,
        "interpretation": {
            "single_position_artifact_excluded": "a and b sampled maxima occur at different registered positions",
            "result": "sampled current trend remains |Ia|>|Ib|, opposite the paper Figure-3I trend",
            "not_claimed": "the 2-um sparse spacing is not a continuous peak-convergence certificate",
            "absolute_current_blocker": "predicted two-terminal resistance is about 14.11 ohm versus measured 213 ohm; no fitting or rescaling",
            "likely_remaining_physical_uncertainty": "unpublished exact CAD/contact resistance, scan registration, metal thermalization/interface, and resulting weighting-field overlap",
        },
        "runtime_diagnostic": {
            "parallel_lumerical_runs_supported": False,
            "observation": "two concurrent sessions raced through Lumerical's user-level shared GPU-resource configuration; the mismatched session failed closed before time stepping",
            "production_rule": "run Lumerical sessions sequentially; independent thermal solves may still run separately",
        },
        "forbidden_operations": {
            "Q_clipping": False,
            "smoothing": False,
            "gain": False,
            "global_rescaling": False,
            "nearest_cell_relocation": False,
            "empirical_current_or_resistance_fit": False,
        },
    }

    plot_path = args.output_dir / "DEVICE_A_REGISTERED_SPARSE_SCAN.png"
    plot_scan(plot_path, records)
    (args.output_dir / "device_a_registered_sparse_scan_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    csv_path = args.output_dir / "device_a_registered_sparse_scan_cases.csv"
    csv_fields = [key for key in records[0] if key not in {"reference_audit"}]
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=csv_fields, lineterminator="\n")
        writer.writeheader()
        for record in records:
            writer.writerow({key: record[key] for key in csv_fields})

    manifest = {
        "status": status,
        "raw_artifacts_committed_to_git": False,
        "artifact_count": len(raw_artifacts),
        "artifacts": raw_artifacts,
        "generation_command": (
            "python photothermal_pte/validation/paper_ir_sanity/"
            "summarize_device_a_registered_sparse_scan.py "
            f"--artifact-root {args.artifact_root.resolve()} "
            f"--output-dir {args.output_dir.resolve()}"
        ),
    }
    (args.output_dir / "RAW_ARTIFACT_MANIFEST_SPARSE_SCAN.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )

    report = f"""# Registered Device-A sparse Maxwell–thermal–PTE scan

Status: `{status}`

This is a **sparse registered diagnostic**, not a paper-certified beam, exact
Device-A reproduction, or continuous scan-peak fit.  The optical source is the
explicitly assumed scalar Gaussian `w0=8.75 um` scenario.  The flake/electrode
geometry is digitized from the published figures because exact CAD is not
available.

## Result

| sampled maximum | coordinate d | integrated current | paper visual reference |
|---|---:|---:|---:|
| `|Ia|` | {maxima['a']['scan_distance_um']:.1f} um | {abs(maxima['a']['PTE_current_nA']):.6f} nA | about 122 pA |
| `|Ib|` | {maxima['b']['scan_distance_um']:.1f} um | {abs(maxima['b']['PTE_current_nA']):.6f} nA | about 143 pA |

The sampled-maximum ratio is

`max_sampled(|Ib|) / max_sampled(|Ia|) = {sampled_peak_ratio:.6f}`.

Figure 3I visually gives the opposite trend, about `143/122 = {143/122:.6f}`.
Because the `a` and `b` sampled maxima occur at different coordinates, this
result rules out the earlier concern that the reversal came only from comparing
one common position.  It does **not** prove sub-micrometre continuous peak
convergence: the present spacing is 2 um.

## Numerical gates

- All seven finite optical cases pass matched-volume closure `<0.5%`, final
  auto-shutoff `<1e-5`, and the independently recomputed position-matched
  empty-stack reference audit: `{optical_gate}`.
- Every source uses the literal optical-cell/TaIrTe4 intersection-density
  mapping.  Non-overlapping air/SiO2 power is not forced into TaIrTe4.
- Every thermal case has mapping error `<1e-12`, zero mapped power outside the
  TaIrTe4 support, linear residual `<1e-8`, and energy error `<1%`: `{thermal_gate}`.
- Current is the full flake-volume Shockley–Ramo integral.  It is not sampled
  from one temperature-gradient point.
- No Q clipping, smoothing, gain, global rescaling, nearest-cell relocation,
  current fit, or resistance fit was used.

## Physical interpretation and limits

The mapped absorbed power remains generally larger for `E || b`, but the
spatial heat-source/temperature/weighting-field overlap gives a larger
integrated current for `E || a` throughout the paired sparse points.  The
current discrepancy therefore remains a physical-model/geometry problem, not
a failed optical closure or conservative-mapping problem.

Absolute current is blocked: the digitized conductivity geometry predicts
`{records[0]['predicted_resistance_ohm']:.6f} ohm`, whereas Device A measured
`{records[0]['measured_resistance_ohm']:.0f} ohm`.  Exact CAD, contact
resistance/geometry, metal thermalization/interface data, and absolute scan
metrology are not published.  The nA values above are consequently not called
experimental predictions and were not rescaled to the paper's pA values.

One attempted parallel optical launch is preserved as a fail-closed diagnostic:
concurrent Lumerical sessions raced through the shared user-level GPU resource
configuration.  Production optical sessions were therefore run sequentially.

The next useful step, if a tighter comparison is required, is a denser local
scan around each sampled maximum together with sensitivity to the digitized
contact/weighting geometry.  AD–FD or optimization would not resolve this
paper-reproduction discrepancy.
"""
    (args.output_dir / "DEVICE_A_REGISTERED_SPARSE_SCAN_REPORT.md").write_text(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
