#!/usr/bin/env python3
"""Publish the no-new-FDTD paper-IR Q/thermal/remap checkpoints."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any


REPOSITORY = Path(__file__).resolve().parents[3]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPOSITORY, text=True
    ).strip()


def artifact_record(path: Path, role: str) -> dict[str, Any]:
    return {
        "role": role,
        "path": str(path.resolve()),
        "byte_size": path.stat().st_size,
        "sha256": sha256(path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--q-convergence-dir", type=Path, required=True)
    parser.add_argument("--thermal-convergence-dir", type=Path, required=True)
    parser.add_argument("--remap-control-dir", type=Path, required=True)
    parser.add_argument("--three-source-audit-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    figures = args.output_dir / "figures"
    figures.mkdir()

    q_path = args.q_convergence_dir / "q_observable_convergence.json"
    thermal_path = (
        args.thermal_convergence_dir / "paper_cutcell_thermal_summary.json"
    )
    remap_path = (
        args.remap_control_dir / "analytic_q_remap_control_summary.json"
    )
    audit_path = (
        args.three_source_audit_dir / "saved_three_source_q_audit.json"
    )
    q = load_json(q_path)
    thermal = load_json(thermal_path)
    remap = load_json(remap_path)
    audit = load_json(audit_path)

    figure_sources = {
        "Q_OBSERVABLE_CONVERGENCE.png": (
            args.q_convergence_dir / "q_observable_convergence.png"
        ),
        "PAPER_CUTCELL_THERMAL_MAPS_50NM.png": (
            args.thermal_convergence_dir / "paper_cutcell_thermal_maps_50nm.png"
        ),
        "PAPER_CUTCELL_GRADIENT_CONVERGENCE.png": (
            args.thermal_convergence_dir
            / "paper_cutcell_gradient_convergence.png"
        ),
        "ANALYTIC_Q_REMAP_CONTROL.png": (
            args.remap_control_dir / "analytic_q_remap_control.png"
        ),
        "SAVED_THREE_SOURCE_Q_AUDIT.png": (
            args.three_source_audit_dir / "saved_three_source_q_audit.png"
        ),
    }
    for name, source in figure_sources.items():
        shutil.copy2(source, figures / name)

    q_power = q["power"]
    q_spatial = q["normalized_spatial_Q"]
    auto = q["acceptance"]["auto_shutoff_gate"]
    worst_remap = remap["worst"]
    worst_remap_published = {
        **worst_remap,
        "Q_thermal_grid_NRMSE": max(
            case["Q_thermal_grid_NRMSE"] for case in remap["cases"]
        ),
    }
    ratios = thermal["ratios_b_over_a"]
    convergence = thermal["convergence"]
    status = (
        "PARTIAL_OFFLINE_PAPER_IR_VALIDATION_"
        "BLOCKED_PLANAR_Q_AND_FIG3HI"
    )
    summary = {
        "status": status,
        "validated_subgates": {
            "diagnostic_Q_observable_convergence": q["status"],
            "paper_reduced_cutcell_thermal_trend": thermal["status"],
            "analytic_Q_Yee_like_remap": remap["status"],
        },
        "unresolved_or_blocked": {
            "auto_shutoff": {
                "status": "FAILED_AUTO_SHUTOFF_GATE",
                **auto,
            },
            "raw_local_peak_mesh_convergence": {
                "status": "UNRESOLVED_DIAGNOSTIC_RAW_PEAK_CONVERGENCE",
                "thermal_100_to_50_raw_max_grad_x_change": {
                    pol: convergence[pol]["100_to_50"][
                        "raw_max_abs_grad_T_x"
                    ]
                    for pol in ("a", "b")
                },
                "remap_raw_peak_change_max": worst_remap[
                    "diagnostic_raw_peak_relative_change"
                ],
            },
            "three_source_decomposition": audit["status"],
            "Figure_3H_I": {
                "status": "NOT_RUN_SEQUENCED_AFTER_BLOCKED_THREE_SOURCE_AUDIT",
                "reason": audit["sequencing"]["reason"],
            },
        },
        "q_1p2_vs_4ps": {
            "P_Q_1p2ps_W": q_power["P_Q_1p2ps_W"],
            "P_Q_4ps_W": q_power["P_Q_4ps_W"],
            "P_Q_relative_change": q_power["relative_change"],
            "component_relative_change": {
                name: item["relative_change"]
                for name, item in q_power["components"].items()
            },
            "normalized_spatial_Q_NRMSE": q_spatial[
                "volume_weighted_NRMSE"
            ],
            "spatial_correlation": q_spatial[
                "volume_weighted_Pearson_correlation"
            ],
            "centroid_shift_m": q["centroid_shift_m"],
            "hotspot_shift_m": q["hotspot_shift_m"],
            "sigma_relative_change": q["sigma_relative_change"],
            "edge_normal_profile": q["edge_normal_profile"],
            "diagnostic_heat_source_only": True,
            "promoted_to_production_Q": False,
        },
        "paper_Fig3F_G_thermal": {
            "model": (
                "paper Supplement Eq. S4 reduced TaIrTe4 sheet with "
                "G_bottom=7.37e6 and G_top=1 W/(m2 K)"
            ),
            "meshes_nm": [200, 100, 50],
            "b_over_a": ratios,
            "robust_x_100_to_50_relative_change": {
                pol: convergence[pol]["100_to_50"][
                    "robust_x_strip_mean"
                ]
                for pol in ("a", "b")
            },
            "Tmax_100_to_50_relative_change": {
                pol: convergence[pol]["100_to_50"]["Tmax"]
                for pol in ("a", "b")
            },
            "interpretation": (
                "the requested b>a gradient trend and robust exact-edge "
                "x-gradient comparator converge below 1%; raw cell maxima "
                "and Tmax remain separately unresolved diagnostics"
            ),
        },
        "remap_control": {
            "source_grid_role": remap["cases"][0]["source_grid"]["role"],
            "worst": worst_remap_published,
            "acceptance": remap["acceptance"],
            "no_equal_power_rescaling": remap["comparison"][
                "no_equal_power_rescaling"
            ],
        },
        "three_source_audit": {
            "status": audit["status"],
            "planar_stack_artifact_available": audit[
                "planar_stack_artifact_available"
            ],
            "candidate_audit": audit["candidate_audit"],
            "provenance_only_plot": True,
        },
        "execution_scope": {
            "new_FDTD_run": False,
            "thermal_FVM_run": True,
            "PTE_run": False,
            "adjoint_run": False,
            "gradient_run": False,
            "optimization_run": False,
            "generation_commit": git_commit(),
        },
    }
    summary_path = args.output_dir / "paper_ir_offline_controls_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    rows: list[dict[str, Any]] = []
    for mesh in ("200nm", "100nm", "50nm"):
        ratio = ratios[mesh]
        rows.append(
            {
                "category": "paper_cutcell_thermal",
                "case": mesh,
                **ratio,
            }
        )
    rows.extend(
        {
            "category": "saved_source_audit",
            "case": item["name"],
            "P_Q_W": item.get("P_Q_W"),
            "P_Qx_W": item.get("P_Qx_W"),
            "P_Qy_W": item.get("P_Qy_W"),
            "P_Qz_W": item.get("P_Qz_W"),
            "usable_as_planar_TaIrTe4_stack": item[
                "usable_as_planar_TaIrTe4_stack"
            ],
            "rejection_reason": item["rejection_reason"],
        }
        for item in audit["candidate_audit"]
    )
    csv_path = args.output_dir / "paper_ir_offline_controls_cases.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=sorted({key for row in rows for key in row}),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    raw_files = [
        (q_path, "saved 1.2-vs-4 ps observable-Q comparison"),
        (
            args.q_convergence_dir / "q_observable_convergence_profiles.npz",
            "Q profiles and comparison fields",
        ),
        (thermal_path, "paper cut-cell thermal convergence summary"),
        (
            args.thermal_convergence_dir
            / "paper_cutcell_thermal_cases.csv",
            "paper cut-cell thermal cases",
        ),
        (remap_path, "analytic-Q remap summary"),
        (
            args.remap_control_dir / "analytic_q_remap_control_fields.npz",
            "analytic-Q direct/remapped fields",
        ),
        (audit_path, "saved three-source availability audit"),
        (
            args.three_source_audit_dir / "saved_three_source_q_audit.csv",
            "saved source candidates",
        ),
    ]
    manifest = {
        "status": status,
        "raw_artifacts_committed_to_git": False,
        "generation_command": (
            "python summarize_offline_q_thermal_controls.py "
            "--q-convergence-dir <saved-comparison> "
            "--thermal-convergence-dir <cutcell-thermal> "
            "--remap-control-dir <remap-control> "
            "--three-source-audit-dir <saved-source-audit> "
            "--output-dir <report-dir>"
        ),
        "inputs": [artifact_record(path, role) for path, role in raw_files],
        "published": [
            artifact_record(summary_path, "published summary JSON"),
            artifact_record(csv_path, "published case table"),
        ],
    }
    manifest_path = args.output_dir / "RAW_ARTIFACT_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    report = f"""# Paper-IR offline Q, thermal, and remap controls

Status: `{status}`

No new FDTD solve was run. The saved 1.2/4 ps artifacts were read only.
Thermal FVM calculations used the analytic source. No PTE, adjoint,
gradient, or optimization calculation ran.

## Saved 1.2 ps versus 4 ps Q

The observable-Q subgate passes for diagnostic heat-source use:

- P_Q: `{q_power['P_Q_1p2ps_W']:.12e}` to
  `{q_power['P_Q_4ps_W']:.12e} W`
- relative P_Q change: `{q_power['relative_change']:.6%}`
- normalized spatial-Q NRMSE:
  `{q_spatial['volume_weighted_NRMSE']:.6%}`
- spatial Pearson correlation:
  `{q_spatial['volume_weighted_Pearson_correlation']:.12f}`
- centroid shift: `{q['centroid_shift_m']:.6e} m`
- hotspot shift: `{q['hotspot_shift_m']:.6e} m`

The FDTD gate remains failed: final auto-shutoff is
`{auto['1p2ps_final']:.6e}` at 1.2 ps and `{auto['4ps_final']:.6e}` at
4 ps, both above `{auto['threshold']:.1e}`. Observable convergence does not
promote this artifact to production Q.

## Paper-like Figure 3F/G thermal control

The source is the analytic 11-um Gaussian--Beer--Lambert law on a 130-nm
TaIrTe4 y<=x half-plane. The thermal model is the paper-reduced Robin model:
bottom `G=7.37e6 W/(m2 K)`, top air `G=1 W/(m2 K)`, insulating lateral
material edge, and paper anisotropic kappa.

At 200/100/50 nm, the robust exact-edge x-gradient b/a ratios are
`{ratios['200nm']['robust_x_strip_mean_b_over_a']:.6f}`,
`{ratios['100nm']['robust_x_strip_mean_b_over_a']:.6f}`, and
`{ratios['50nm']['robust_x_strip_mean_b_over_a']:.6f}`. Thus the requested
`|grad T|_b > |grad T|_a` trend is reproduced. The 100-to-50 nm robust-x
changes are `{convergence['a']['100_to_50']['robust_x_strip_mean']:.3%}`
and `{convergence['b']['100_to_50']['robust_x_strip_mean']:.3%}`.

This is not a blanket local-maximum convergence claim. Raw max-dT/dx changes
by `{convergence['a']['100_to_50']['raw_max_abs_grad_T_x']:.3%}` and
`{convergence['b']['100_to_50']['raw_max_abs_grad_T_x']:.3%}`, while Tmax
changes by `{convergence['a']['100_to_50']['Tmax']:.3%}` and
`{convergence['b']['100_to_50']['Tmax']:.3%}`. Those diagnostics remain
unresolved.

## Yee-like remap control

The exact cut-cell analytic source was compared with the same law sampled on
a 33.898-nm Yee-like Cartesian layout and passed through the current
conservative remap. This was not a Maxwell solve and is not claimed to be the
exact v261 Yee mesh.

- worst Q_T NRMSE:
  `{worst_remap_published['Q_thermal_grid_NRMSE']:.3%}`
- worst T-field NRMSE: `{worst_remap['temperature_field_NRMSE']:.3%}`
- worst gradient-field NRMSE:
  `{worst_remap['gradient_field_NRMSE']:.3%}`
- worst primary-metric change:
  `{worst_remap['primary_metric_relative_change']:.3%}`
- worst raw-cell peak change:
  `{worst_remap['diagnostic_raw_peak_relative_change']:.3%}`
  (diagnostic only)

No clipping, smoothing, gain, global rescaling, tiling, or source deletion
was used. The earlier centre-sampled diagonal control was wrong because a
cell cut by y=x was filled completely; the published control uses the exact
half-cell measure.

## Three-source decomposition and Figure 3H/I

The required edge-free planar TaIrTe4-stack Q artifact is absent.
The saved empty-stack case contains no TaIrTe4; the finite-centre case is a
digitized Device-A polygon with edges. Neither is relabeled as planar. The
saved straight-edge Q is also a legacy `epsilon_c=16` diagnostic with
exactly zero Qz, not the production `epsilon_c=epsilon_b` material closure.

Therefore the analytic/planar/edge decomposition is
`{audit['status']}`. The available distributions are plotted only for
provenance; they do not establish the requested causal decomposition.
Following the approved order, Figure 3H/I was not started after this
blocker. Doing so requires either an explicitly approved new planar-stack
FDTD artifact or a revised order/contract.
"""
    (args.output_dir / "PAPER_IR_OFFLINE_Q_THERMAL_CONTROLS_REPORT.md").write_text(
        report
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
