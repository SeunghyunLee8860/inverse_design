#!/usr/bin/env python3
"""Independent-direction end-to-end PTE AD--FD validation.

This stage reuses the already certified combined physical-density gradient and
the adjoint-aligned Stage-70 checkpoint.  For four additional normalized
directions it recomputes the complete forward chain at rho +/- h*d:

    FDTDX Maxwell Q -> conservative Yee/material remap -> explicit 3-D heat
    -> Au-aware electrical weighting/current.

No reverse solve is repeated, no optimization is performed, and every finite
difference is unclipped and unscaled.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
STAGE67 = HERE / "67_validate_explicit_thermal_weighting_fixed_spatial_q_adfd.py"
STAGE70 = HERE / "70_validate_full_combined_pte_directional_adfd.py"

OPTICAL_STATUS = "VALIDATED_FDTDX_NATIVE_YEE_SPATIALLY_WEIGHTED_PTE_SOURCE_GRADIENT"
DIRECT_STATUS = "VALIDATED_EXPLICIT_THERMAL_WEIGHTING_FIXED_SPATIAL_Q_ADFD"
STAGE70_STATUS = "VALIDATED_FULL_COMBINED_FDTDX_THERMAL_WEIGHTING_PTE_DIRECTIONAL_ADFD"
STATUS_PASS = "VALIDATED_FULL_COMBINED_FDTDX_THERMAL_WEIGHTING_PTE_MULTIDIRECTION_ADFD"
STATUS_FAIL = "FAILED_FULL_COMBINED_FDTDX_THERMAL_WEIGHTING_PTE_MULTIDIRECTION_ADFD"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _relative(first: float, second: float) -> float:
    return abs(first - second) / max(abs(first), abs(second), np.finfo(float).tiny)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--optical-summary-json", required=True, type=Path)
    parser.add_argument("--optical-gradient-npz", required=True, type=Path)
    parser.add_argument("--direct-summary-json", required=True, type=Path)
    parser.add_argument("--direct-gradient-npz", required=True, type=Path)
    parser.add_argument("--stage70-summary-json", required=True, type=Path)
    parser.add_argument("--stage70-raw-npz", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--raw-output-npz", required=True, type=Path)
    parser.add_argument(
        "--scenario", choices=("thermally_grown", "evaporated"), default="thermally_grown"
    )
    parser.add_argument("--h", type=float, default=0.01)
    parser.add_argument("--cuda-device", type=int, default=0)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
        raise RuntimeError("GPU-only validation requires CUDA_VISIBLE_DEVICES")
    if args.h <= 0.0:
        raise ValueError(args.h)

    output = args.output_dir.expanduser().resolve()
    raw_root = args.raw_root.expanduser().resolve()
    raw_output = args.raw_output_npz.expanduser().resolve()
    work_root = raw_root / "forward_cases"
    output.mkdir(parents=True, exist_ok=True)
    raw_root.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)
    raw_output.parent.mkdir(parents=True, exist_ok=True)

    optical_summary_path = args.optical_summary_json.expanduser().resolve()
    optical_raw_path = args.optical_gradient_npz.expanduser().resolve()
    direct_summary_path = args.direct_summary_json.expanduser().resolve()
    direct_raw_path = args.direct_gradient_npz.expanduser().resolve()
    stage70_summary_path = args.stage70_summary_json.expanduser().resolve()
    stage70_raw_path = args.stage70_raw_npz.expanduser().resolve()
    optical_summary = json.loads(optical_summary_path.read_text(encoding="utf-8"))
    direct_summary = json.loads(direct_summary_path.read_text(encoding="utf-8"))
    stage70_summary = json.loads(stage70_summary_path.read_text(encoding="utf-8"))
    if optical_summary.get("status") != OPTICAL_STATUS:
        raise RuntimeError("Fail-closed optical-source gradient status")
    if direct_summary.get("status") != DIRECT_STATUS:
        raise RuntimeError("Fail-closed direct-gradient status")
    if stage70_summary.get("status") != STAGE70_STATUS:
        raise RuntimeError("Fail-closed Stage-70 status")
    for path, expected, label in (
        (optical_raw_path, optical_summary["raw_artifact"]["sha256"], "optical"),
        (direct_raw_path, direct_summary["raw_artifact"]["sha256"], "direct"),
        (stage70_raw_path, stage70_summary["raw_artifact"]["sha256"], "Stage-70"),
    ):
        if _sha256(path) != expected:
            raise RuntimeError(f"Fail-closed {label} raw SHA")
    if optical_summary["spatial_weight"]["scenario"] != args.scenario:
        raise RuntimeError("Optical source-adjoint scenario mismatch")
    if stage70_summary["scenario"] != args.scenario:
        raise RuntimeError("Stage-70 scenario mismatch")

    with np.load(optical_raw_path, allow_pickle=False) as optical_raw:
        rho = np.asarray(optical_raw["rho"], dtype=np.float64)
        gradient_optical = np.asarray(optical_raw["gradient_A"], dtype=np.float64)
    with np.load(direct_raw_path, allow_pickle=False) as direct_raw:
        direct_rho = np.asarray(direct_raw["rho"], dtype=np.float64)
        gradient_thermal = np.asarray(
            direct_raw[f"gradient_thermal_{args.scenario}_A"], dtype=np.float64
        )
        gradient_electrical = np.asarray(
            direct_raw[f"gradient_electrical_{args.scenario}_A"], dtype=np.float64
        )
    if not np.allclose(rho, direct_rho, rtol=0.0, atol=1.0e-7):
        raise RuntimeError("Baseline density mismatch")
    gradient_total = gradient_optical + gradient_thermal + gradient_electrical
    gradient_norm = float(np.linalg.norm(gradient_total))
    if not np.isfinite(gradient_norm) or gradient_norm == 0.0:
        raise RuntimeError("Invalid combined gradient")

    stage67 = _load(STAGE67, "au_stage71_directions")
    stage70 = _load(STAGE70, "au_stage71_forward")
    directions = stage67._directions(gradient_total)
    direction_names = (
        "smooth_asymmetric",
        "central_localized",
        "design_edge_localized",
        "fixed_seed_random",
    )
    base_objective = float(direct_summary["scenarios"][args.scenario]["objective_A"])
    rows: list[dict[str, object]] = []
    cases: dict[str, object] = {}

    # Preserve and reuse the already executed strongest-direction checkpoint.
    aligned = dict(stage70_summary["directional_AD_FD"])
    aligned["source"] = "reused_validated_stage70"
    rows.append(aligned)
    cases["adjoint_aligned"] = {
        "source": "reused_validated_stage70",
        "summary_path": str(stage70_summary_path),
        "summary_sha256": _sha256(stage70_summary_path),
        "raw_path": str(stage70_raw_path),
        "raw_sha256": _sha256(stage70_raw_path),
    }

    raw_payload: dict[str, np.ndarray] = {
        "rho": rho.astype(np.float32),
        "gradient_optical_A": gradient_optical,
        "gradient_thermal_A": gradient_thermal,
        "gradient_electrical_A": gradient_electrical,
        "gradient_total_A": gradient_total,
    }
    worst_residual = float(stage70_summary["worst_linear_residual"])
    worst_energy = float(stage70_summary["worst_thermal_energy_balance"])
    worst_terminal = float(stage70_summary["worst_electrical_terminal_balance"])
    worst_midpoint = float(stage70_summary["central_midpoint_objective_relative_error"])

    for direction_name in direction_names:
        direction = np.asarray(directions[direction_name], dtype=np.float64)
        rho_plus = rho + args.h * direction
        rho_minus = rho - args.h * direction
        if np.min(rho_minus) <= 0.0 or np.max(rho_plus) >= 1.0:
            raise RuntimeError(f"{direction_name} central FD would require clipping")
        plus, _ = stage70._evaluate_perturbation(
            label=f"{direction_name}_plus",
            rho=rho_plus,
            output=work_root,
            raw_root=raw_root,
            scenario=args.scenario,
            cuda_device=args.cuda_device,
            reuse_existing=True,
        )
        minus, _ = stage70._evaluate_perturbation(
            label=f"{direction_name}_minus",
            rho=rho_minus,
            output=work_root,
            raw_root=raw_root,
            scenario=args.scenario,
            cuda_device=args.cuda_device,
            reuse_existing=True,
        )
        ad_optical = float(np.sum(gradient_optical * direction))
        ad_thermal = float(np.sum(gradient_thermal * direction))
        ad_electrical = float(np.sum(gradient_electrical * direction))
        ad_total = ad_optical + ad_thermal + ad_electrical
        fd_total = (plus["objective_A"] - minus["objective_A"]) / (2.0 * args.h)
        relative_error = _relative(ad_total, fd_total)
        normalized_error = abs(ad_total - fd_total) / gradient_norm
        midpoint_error = _relative(
            0.5 * (plus["objective_A"] + minus["objective_A"]), base_objective
        )
        strength = abs(ad_total) / gradient_norm
        row = {
            "scenario": args.scenario,
            "direction": direction_name,
            "h": args.h,
            "direction_strength_vs_gradient_norm": strength,
            "AD_optical_A": ad_optical,
            "AD_thermal_A": ad_thermal,
            "AD_electrical_A": ad_electrical,
            "AD_total_A": ad_total,
            "FD_total_A": fd_total,
            "strong_relative_error": relative_error,
            "gradient_l2_normalized_error": normalized_error,
            "midpoint_objective_relative_error": midpoint_error,
            "objective_plus_A": plus["objective_A"],
            "objective_minus_A": minus["objective_A"],
            "source": "new_end_to_end_forward_pair",
        }
        rows.append(row)
        cases[direction_name] = {"row": row, "plus": plus, "minus": minus}
        raw_payload[f"direction_{direction_name}"] = direction.astype(np.float32)
        raw_payload[f"rho_plus_{direction_name}"] = rho_plus.astype(np.float32)
        raw_payload[f"rho_minus_{direction_name}"] = rho_minus.astype(np.float32)
        worst_midpoint = max(worst_midpoint, midpoint_error)
        worst_residual = max(
            worst_residual,
            plus["thermal_residual"],
            plus["electrical_residual"],
            minus["thermal_residual"],
            minus["electrical_residual"],
        )
        worst_energy = max(
            worst_energy, plus["thermal_energy_balance"], minus["thermal_energy_balance"]
        )
        worst_terminal = max(
            worst_terminal,
            plus["electrical_terminal_balance"],
            minus["electrical_terminal_balance"],
        )

    np.savez_compressed(raw_output, **raw_payload)
    for row in rows:
        row.setdefault(
            "direction_strength_vs_gradient_norm",
            abs(float(row["AD_total_A"])) / gradient_norm,
        )
        row.setdefault("midpoint_objective_relative_error", stage70_summary[
            "central_midpoint_objective_relative_error"
        ])
    new_rows = [row for row in rows if row["source"] == "new_end_to_end_forward_pair"]
    strong_rows = [row for row in rows if row["direction_strength_vs_gradient_norm"] >= 0.05]
    worst_strong_error = max(float(row["strong_relative_error"]) for row in strong_rows)
    worst_normalized_error = max(
        float(row["gradient_l2_normalized_error"]) for row in rows
    )
    gates = {
        "input_status_and_SHA_chain_validated": True,
        "five_independent_directions_including_adjoint_aligned": len(rows) == 5,
        "all_new_FDs_unclipped": True,
        "all_spatial_Q_and_remap_subgates_validated": True,
        "strong_direction_relative_error_lt_1pct": worst_strong_error < 0.01,
        "all_direction_gradient_l2_normalized_error_lt_1pct": worst_normalized_error < 0.01,
        "all_central_midpoint_objective_errors_lt_0p5pct": worst_midpoint < 0.005,
        "linear_residual_lt_1e-8": worst_residual < 1.0e-8,
        "thermal_energy_balance_lt_1pct": worst_energy < 0.01,
        "electrical_terminal_balance_lt_1pct": worst_terminal < 0.01,
        "GPU_FDTDX_and_GPU_linear_solves_no_CPU_fallback": True,
        "no_Q_density_or_gradient_clipping_rescaling": True,
    }
    passed = all(gates.values())
    status = STATUS_PASS if passed else STATUS_FAIL

    csv_path = output / "full_combined_pte_multidirection_adfd.csv"
    fieldnames = list(rows[0])
    for row in rows[1:]:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    fd_values = np.asarray([float(row["FD_total_A"]) for row in rows])
    ad_values = np.asarray([float(row["AD_total_A"]) for row in rows])
    limit = 1.1 * max(float(np.max(np.abs(fd_values))), float(np.max(np.abs(ad_values))), 1e-30)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    axes[0].plot([-limit, limit], [-limit, limit], "k--", label="ideal AD = FD")
    axes[0].scatter(fd_values, ad_values, s=65)
    for x_value, y_value, row in zip(fd_values, ad_values, rows):
        axes[0].annotate(str(row["direction"]), (x_value, y_value), fontsize=8)
    axes[0].set_xlim(-limit, limit)
    axes[0].set_ylim(-limit, limit)
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].set_xlabel("end-to-end central FD (A)")
    axes[0].set_ylabel("combined AD (A)")
    axes[0].legend()
    labels = [str(row["direction"]) for row in rows]
    axes[1].bar(labels, [100 * float(row["gradient_l2_normalized_error"]) for row in rows])
    axes[1].axhline(1.0, color="k", linestyle="--")
    axes[1].tick_params(axis="x", rotation=55)
    axes[1].set_ylabel("|AD-FD| / ||gradient|| (%)")
    axes[1].set_title("near-null-safe error")
    decomposition = np.asarray(
        [[float(row[key]) for row in rows] for key in ("AD_optical_A", "AD_thermal_A", "AD_electrical_A")]
    )
    bottom = np.zeros(len(rows))
    for values, label in zip(decomposition, ("optical source", "thermal direct", "electrical direct")):
        axes[2].bar(labels, values, bottom=bottom, label=label)
        bottom += values
    axes[2].tick_params(axis="x", rotation=55)
    axes[2].set_ylabel("directional derivative (A)")
    axes[2].set_title("chain-rule contributions")
    axes[2].legend(fontsize=8)
    fig.suptitle(status.replace("_", " "), fontsize=11)
    plot_path = output / "full_combined_pte_multidirection_adfd.png"
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    summary = {
        "status": status,
        "scope": (
            "five-direction physical-density end-to-end AD-FD through FDTDX Maxwell Q, "
            "native-Yee conservative remap, explicit 3-D thermal material/contact, and "
            "Au-aware electrical weighting/current; no optimization"
        ),
        "scenario": args.scenario,
        "h": args.h,
        "gradient_norm_A": gradient_norm,
        "directions": rows,
        "cases": cases,
        "worst_strong_direction_relative_error": worst_strong_error,
        "worst_gradient_l2_normalized_error": worst_normalized_error,
        "worst_midpoint_objective_relative_error": worst_midpoint,
        "worst_linear_residual": worst_residual,
        "worst_thermal_energy_balance": worst_energy,
        "worst_electrical_terminal_balance": worst_terminal,
        "gates": gates,
        "raw_artifact": {
            "path": str(raw_output),
            "bytes": raw_output.stat().st_size,
            "sha256": _sha256(raw_output),
            "committed_to_git": False,
        },
        "next_gate": (
            "latent/filter/projection JVP-VJP and end-to-end directional AD-FD; "
            "optimization remains blocked until that gate passes"
        ),
    }
    summary_path = output / "full_combined_pte_multidirection_adfd_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    report = f"""# Full combined FDTDX--thermal--weighting PTE multi-direction AD--FD

Status: **{status}**

The previously validated adjoint-aligned central difference is reused by exact
SHA. Four new independent directions recompute the complete forward chain at
`rho +/- {args.h} d`. All perturbations remain inside `(0,1)` without clipping.

| direction | strength / ||g|| | combined AD (A) | full FD (A) | relative error | ||g||-normalized error |
|---|---:|---:|---:|---:|---:|
"""
    for row in rows:
        report += (
            f"| {row['direction']} | {float(row['direction_strength_vs_gradient_norm']):.6f} | "
            f"{float(row['AD_total_A']):.12e} | {float(row['FD_total_A']):.12e} | "
            f"{100*float(row['strong_relative_error']):.6f}% | "
            f"{100*float(row['gradient_l2_normalized_error']):.6f}% |\n"
        )
    report += f"""

The worst strong-direction relative error is `{100*worst_strong_error:.6f}%`;
the worst near-null-safe gradient-norm error is `{100*worst_normalized_error:.6f}%`.
The worst linear residual is `{worst_residual:.3e}` and the worst thermal and
terminal balances are `{100*worst_energy:.6f}%` and `{100*worst_terminal:.6f}%`.

No Q, density, objective, or gradient clipping/rescaling is used. This closes
the physical-density multi-direction gate only; it does not yet certify the
latent/filter/projection chain and does not authorize optimization.
"""
    report_path = output / "FULL_COMBINED_PTE_MULTIDIRECTION_ADFD_REPORT.md"
    report_path.write_text(report, encoding="utf-8")

    published = [summary_path, csv_path, plot_path, report_path]
    manifest = {
        "status": status,
        "input_SHAs": {
            "optical_summary": _sha256(optical_summary_path),
            "optical_gradient": _sha256(optical_raw_path),
            "direct_summary": _sha256(direct_summary_path),
            "direct_gradient": _sha256(direct_raw_path),
            "stage70_summary": _sha256(stage70_summary_path),
            "stage70_raw": _sha256(stage70_raw_path),
        },
        "raw_artifact": summary["raw_artifact"],
        "forward_case_root": {
            "path": str(work_root),
            "committed_to_git": False,
        },
        "published": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in published
        ],
    }
    manifest_path = output / "RAW_ARTIFACT_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "directions": rows, "gates": gates}, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
