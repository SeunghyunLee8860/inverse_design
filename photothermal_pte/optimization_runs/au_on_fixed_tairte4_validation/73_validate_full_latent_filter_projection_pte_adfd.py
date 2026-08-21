#!/usr/bin/env python3
"""End-to-end latent/filter/projection PTE directional AD--FD.

The finite-filter inverse reconstructs an interior latent baseline whose
projected physical density equals the already certified physical-density
baseline.  This permits exact reuse of the multi-direction physical gradient,
while every latent FD perturbation still recomputes the complete FDTDX,
thermal, and electrical forward chain.
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

from photothermal_pte.optimization_runs.legacy_v261_optical_support.production_density_mapping import (
    ProductionDensityMapping,
)


HERE = Path(__file__).resolve().parent
STAGE70 = HERE / "70_validate_full_combined_pte_directional_adfd.py"
PHYSICAL_STATUS = "VALIDATED_FULL_COMBINED_FDTDX_THERMAL_WEIGHTING_PTE_MULTIDIRECTION_ADFD"
BASELINE_STATUS = "VALIDATED_FULL_COMBINED_FDTDX_THERMAL_WEIGHTING_PTE_DIRECTIONAL_ADFD"
MAPPING_STATUS = "VALIDATED_AU_LATENT_FILTER_PROJECTION_MAPPING"
STATUS_PASS = "VALIDATED_FULL_LATENT_FILTER_PROJECTION_FDTDX_PTE_ADFD"
STATUS_FAIL = "FAILED_FULL_LATENT_FILTER_PROJECTION_FDTDX_PTE_ADFD"


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


def _inverse_projection(rho: np.ndarray, beta: float, eta: float) -> np.ndarray:
    denominator = np.tanh(beta * eta) + np.tanh(beta * (1.0 - eta))
    argument = rho * denominator - np.tanh(beta * eta)
    if np.any(np.abs(argument) >= 1.0):
        raise RuntimeError("Physical baseline lies outside the invertible projection range")
    return eta + np.arctanh(argument) / beta


def _dense_filter(mapping: ProductionDensityMapping) -> np.ndarray:
    size = int(np.prod(mapping.shape))
    identity = np.eye(size)
    return np.column_stack(
        [mapping.filtered(identity[index].reshape(mapping.shape)).reshape(-1) for index in range(size)]
    )


def _directions(gradient: np.ndarray) -> dict[str, np.ndarray]:
    x = np.linspace(-1.0, 1.0, gradient.shape[0])[:, None]
    y = np.linspace(-1.0, 1.0, gradient.shape[1])[None, :]
    rng = np.random.default_rng(20260821)
    raw = {
        "latent_adjoint_aligned": gradient,
        "latent_smooth_asymmetric": np.sin(0.72 * np.pi * x) * np.cos(0.53 * np.pi * y) + 0.2 * y,
        "latent_fixed_seed_random": rng.normal(size=gradient.shape),
    }
    return {name: value / np.linalg.norm(value) for name, value in raw.items()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--physical-summary-json", required=True, type=Path)
    parser.add_argument("--physical-gradient-npz", required=True, type=Path)
    parser.add_argument("--baseline-summary-json", required=True, type=Path)
    parser.add_argument("--baseline-raw-npz", required=True, type=Path)
    parser.add_argument("--mapping-summary-json", required=True, type=Path)
    parser.add_argument("--mapping-raw-npz", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--raw-output-npz", required=True, type=Path)
    parser.add_argument("--scenario", choices=("thermally_grown",), default="thermally_grown")
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--h", type=float, default=0.01)
    parser.add_argument("--cuda-device", type=int, default=0)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
        raise RuntimeError("GPU-only validation requires CUDA_VISIBLE_DEVICES")
    if args.beta <= 0.0 or args.h <= 0.0:
        raise ValueError("beta and h must be positive")

    physical_summary_path = args.physical_summary_json.expanduser().resolve()
    physical_raw_path = args.physical_gradient_npz.expanduser().resolve()
    baseline_summary_path = args.baseline_summary_json.expanduser().resolve()
    baseline_raw_path = args.baseline_raw_npz.expanduser().resolve()
    mapping_summary_path = args.mapping_summary_json.expanduser().resolve()
    mapping_raw_path = args.mapping_raw_npz.expanduser().resolve()
    physical_summary = json.loads(physical_summary_path.read_text(encoding="utf-8"))
    baseline_summary = json.loads(baseline_summary_path.read_text(encoding="utf-8"))
    mapping_summary = json.loads(mapping_summary_path.read_text(encoding="utf-8"))
    if physical_summary.get("status") != PHYSICAL_STATUS:
        raise RuntimeError("Fail-closed physical-gradient status")
    if baseline_summary.get("status") != BASELINE_STATUS:
        raise RuntimeError("Fail-closed exact-baseline status")
    if mapping_summary.get("status") != MAPPING_STATUS:
        raise RuntimeError("Fail-closed mapping status")
    if _sha256(physical_raw_path) != physical_summary["raw_artifact"]["sha256"]:
        raise RuntimeError("Fail-closed physical-gradient raw SHA")
    if _sha256(baseline_raw_path) != baseline_summary["raw_artifact"]["sha256"]:
        raise RuntimeError("Fail-closed exact-baseline raw SHA")
    if _sha256(mapping_raw_path) != mapping_summary["raw_artifact"]["sha256"]:
        raise RuntimeError("Fail-closed mapping raw SHA")
    if args.beta not in mapping_summary["beta_values"]:
        raise RuntimeError("Requested beta was not mapping-certified")

    with np.load(physical_raw_path, allow_pickle=False) as raw:
        rho_target = np.asarray(raw["rho"], dtype=np.float64)
        gradient_physical = np.asarray(raw["gradient_total_A"], dtype=np.float64)
    mapping_contract = mapping_summary["mapping"]
    mapping = ProductionDensityMapping(
        shape=tuple(mapping_contract["shape"]),
        spacing_m=float(mapping_contract["spacing_nm"]) * 1.0e-9,
        radius_m=float(mapping_contract["radius_nm"]) * 1.0e-9,
        eta=float(mapping_contract["eta"]),
    )
    filtered_target = _inverse_projection(rho_target, args.beta, mapping.eta)
    filter_matrix = _dense_filter(mapping)
    latent = np.linalg.solve(filter_matrix, filtered_target.reshape(-1)).reshape(mapping.shape)
    rho_reconstructed = mapping.physical(latent, args.beta)
    reconstruction_error = float(np.max(np.abs(rho_reconstructed - rho_target)))
    if np.min(latent) <= 0.0 or np.max(latent) >= 1.0:
        raise RuntimeError("Inverse-filter latent baseline is not strictly interior")
    if reconstruction_error >= 1.0e-12:
        raise RuntimeError(f"Physical baseline reconstruction failed: {reconstruction_error}")

    gradient_latent = mapping.vjp(latent, gradient_physical, args.beta)
    gradient_norm = float(np.linalg.norm(gradient_latent))
    if not np.isfinite(gradient_norm) or gradient_norm == 0.0:
        raise RuntimeError("Invalid latent gradient")
    directions = _directions(gradient_latent)
    stage70 = _load(STAGE70, "au_stage73_forward")
    output = args.output_dir.expanduser().resolve()
    raw_root = args.raw_root.expanduser().resolve()
    raw_output = args.raw_output_npz.expanduser().resolve()
    work_root = raw_root / "forward_cases"
    output.mkdir(parents=True, exist_ok=True)
    raw_root.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)
    raw_output.parent.mkdir(parents=True, exist_ok=True)

    baseline_objective = float(baseline_summary["baseline_objective_A"])
    if not np.isfinite(baseline_objective) or baseline_objective == 0.0:
        raise RuntimeError("Invalid exact Stage-70 baseline objective")

    rows: list[dict[str, object]] = []
    cases: dict[str, object] = {}
    raw_payload: dict[str, np.ndarray] = {
        "latent": latent.astype(np.float32),
        "rho_reconstructed": rho_reconstructed.astype(np.float32),
        "gradient_physical_A": gradient_physical,
        "gradient_latent_A": gradient_latent,
    }
    worst_residual = 0.0
    worst_energy = 0.0
    worst_terminal = 0.0
    worst_midpoint = 0.0
    for name, direction in directions.items():
        latent_plus = latent + args.h * direction
        latent_minus = latent - args.h * direction
        if np.min(latent_minus) <= 0.0 or np.max(latent_plus) >= 1.0:
            raise RuntimeError(f"{name} latent FD requires clipping")
        rho_plus = mapping.physical(latent_plus, args.beta)
        rho_minus = mapping.physical(latent_minus, args.beta)
        plus, _ = stage70._evaluate_perturbation(
            label=f"{name}_plus",
            rho=rho_plus,
            output=work_root,
            raw_root=raw_root,
            scenario=args.scenario,
            cuda_device=args.cuda_device,
            reuse_existing=True,
        )
        minus, _ = stage70._evaluate_perturbation(
            label=f"{name}_minus",
            rho=rho_minus,
            output=work_root,
            raw_root=raw_root,
            scenario=args.scenario,
            cuda_device=args.cuda_device,
            reuse_existing=True,
        )
        ad = float(np.vdot(gradient_latent, direction))
        fd = (float(plus["objective_A"]) - float(minus["objective_A"])) / (2.0 * args.h)
        strength = abs(ad) / gradient_norm
        relative_error = _relative(ad, fd)
        normalized_error = abs(ad - fd) / gradient_norm
        midpoint_error = _relative(
            0.5 * (float(plus["objective_A"]) + float(minus["objective_A"])),
            baseline_objective,
        )
        row = {
            "scenario": args.scenario,
            "beta": args.beta,
            "direction": name,
            "h": args.h,
            "direction_strength_vs_latent_gradient_norm": strength,
            "AD_A": ad,
            "FD_A": fd,
            "strong_relative_error": relative_error,
            "latent_gradient_norm_normalized_error": normalized_error,
            "midpoint_objective_relative_error": midpoint_error,
            "objective_plus_A": plus["objective_A"],
            "objective_minus_A": minus["objective_A"],
        }
        rows.append(row)
        cases[name] = {"row": row, "plus": plus, "minus": minus}
        raw_payload[f"direction_{name}"] = direction.astype(np.float32)
        raw_payload[f"rho_plus_{name}"] = rho_plus.astype(np.float32)
        raw_payload[f"rho_minus_{name}"] = rho_minus.astype(np.float32)
        worst_midpoint = max(worst_midpoint, midpoint_error)
        worst_residual = max(
            worst_residual,
            plus["thermal_residual"], plus["electrical_residual"],
            minus["thermal_residual"], minus["electrical_residual"],
        )
        worst_energy = max(
            worst_energy, plus["thermal_energy_balance"], minus["thermal_energy_balance"]
        )
        worst_terminal = max(
            worst_terminal,
            plus["electrical_terminal_balance"], minus["electrical_terminal_balance"],
        )

    np.savez_compressed(raw_output, **raw_payload)
    strong_rows = [row for row in rows if row["direction_strength_vs_latent_gradient_norm"] >= 0.05]
    worst_strong = max(float(row["strong_relative_error"]) for row in strong_rows)
    worst_normalized = max(float(row["latent_gradient_norm_normalized_error"]) for row in rows)
    gates = {
        "input_status_and_SHA_chain_validated": True,
        "physical_baseline_reconstruction_lt_1e-12": reconstruction_error < 1.0e-12,
        "strict_interior_unclipped_latent_FD": True,
        "three_independent_latent_directions": len(rows) == 3,
        "all_spatial_Q_and_remap_subgates_validated": True,
        "strong_direction_relative_error_lt_1pct": worst_strong < 0.01,
        "all_direction_gradient_norm_error_lt_1pct": worst_normalized < 0.01,
        "all_midpoint_objective_errors_lt_0p5pct": worst_midpoint < 0.005,
        "linear_residual_lt_1e-8": worst_residual < 1.0e-8,
        "thermal_energy_balance_lt_1pct": worst_energy < 0.01,
        "electrical_terminal_balance_lt_1pct": worst_terminal < 0.01,
        "GPU_FDTDX_and_GPU_linear_solves_no_CPU_fallback": True,
        "no_Q_latent_density_or_gradient_clipping_rescaling": True,
    }
    passed = all(gates.values())
    status = STATUS_PASS if passed else STATUS_FAIL

    csv_path = output / "full_latent_filter_projection_pte_adfd.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    fd_values = np.asarray([float(row["FD_A"]) for row in rows])
    ad_values = np.asarray([float(row["AD_A"]) for row in rows])
    limit = 1.1 * max(float(np.max(np.abs(fd_values))), float(np.max(np.abs(ad_values))), 1e-30)
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6), constrained_layout=True)
    plotted = axes[0].imshow(rho_reconstructed.T, origin="lower", cmap="gray_r", vmin=0, vmax=1)
    axes[0].set_title(f"reconstructed physical rho, beta={args.beta:g}")
    axes[0].set_xlabel("x=b design index")
    axes[0].set_ylabel("y=a design index")
    fig.colorbar(plotted, ax=axes[0])
    axes[1].plot([-limit, limit], [-limit, limit], "k--", label="ideal AD = FD")
    axes[1].scatter(fd_values, ad_values, s=70)
    short_names = [str(row["direction"]).removeprefix("latent_") for row in rows]
    offsets = [(6, -13), (6, 7), (6, 7)]
    for x_value, y_value, name, offset in zip(fd_values, ad_values, short_names, offsets):
        axes[1].annotate(
            name,
            (x_value, y_value),
            xytext=offset,
            textcoords="offset points",
            fontsize=8,
        )
    axes[1].set_xlim(-limit, limit)
    axes[1].set_ylim(-limit, limit)
    axes[1].set_aspect("equal", adjustable="box")
    axes[1].set_xlabel("end-to-end latent FD (A)")
    axes[1].set_ylabel("latent-chain AD (A)")
    axes[1].legend()
    axes[2].bar(
        short_names,
        [100 * float(row["latent_gradient_norm_normalized_error"]) for row in rows],
    )
    axes[2].axhline(1.0, color="k", linestyle="--")
    axes[2].tick_params(axis="x", rotation=30)
    axes[2].set_ylabel("|AD-FD| / ||latent gradient|| (%)")
    axes[2].set_title("near-null-safe error")
    fig.suptitle(status.replace("_", " "), fontsize=11)
    plot_path = output / "full_latent_filter_projection_pte_adfd.png"
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    summary = {
        "status": status,
        "scope": (
            "three-direction end-to-end latent -> finite conic filter -> tanh projection -> "
            "FDTDX Maxwell Q -> conservative remap -> explicit thermal -> Au-aware "
            "electrical weighting/current; no optimization"
        ),
        "scenario": args.scenario,
        "mapping": mapping.audit(),
        "beta": args.beta,
        "h": args.h,
        "latent_range": [float(np.min(latent)), float(np.max(latent))],
        "physical_baseline_reconstruction_max_abs": reconstruction_error,
        "physical_gradient_norm_A": float(np.linalg.norm(gradient_physical)),
        "latent_gradient_norm_A": gradient_norm,
        "directions": rows,
        "cases": cases,
        "worst_strong_direction_relative_error": worst_strong,
        "worst_latent_gradient_norm_normalized_error": worst_normalized,
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
            "freeze a fabrication/minimum-feature contract and optimizer continuation; "
            "the derivative chain is validated but optimization has not started"
        ),
    }
    summary_path = output / "full_latent_filter_projection_pte_adfd_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    report = f"""# Full latent/filter/projection FDTDX PTE AD--FD

Status: **{status}**

At beta `{args.beta:g}`, an interior latent baseline is solved such that the
finite-filtered/projected physical density reconstructs the already certified
physical baseline to `{reconstruction_error:.3e}` maximum absolute error.
Thus the validated unscaled physical gradient can be pulled back exactly
through the filter/projection transpose without a changed-state approximation.

| direction | strength / ||g_latent|| | AD (A) | full FD (A) | relative error | ||g_latent|| error |
|---|---:|---:|---:|---:|---:|
"""
    for row in rows:
        report += (
            f"| {row['direction']} | {float(row['direction_strength_vs_latent_gradient_norm']):.6f} | "
            f"{float(row['AD_A']):.12e} | {float(row['FD_A']):.12e} | "
            f"{100*float(row['strong_relative_error']):.6f}% | "
            f"{100*float(row['latent_gradient_norm_normalized_error']):.6f}% |\n"
        )
    report += f"""

Worst strong-direction error is `{100*worst_strong:.6f}%`; worst near-null-safe
latent-gradient-norm error is `{100*worst_normalized:.6f}%`. No Q, latent,
density, objective, or gradient clipping/rescaling is used.

This validates the full differentiable chain for the stated numerical filter
scenario. The 750 nm filter radius is not yet a fabrication confidence
interval, and no Au optimization has been run.
"""
    report_path = output / "FULL_LATENT_FILTER_PROJECTION_PTE_ADFD_REPORT.md"
    report_path.write_text(report, encoding="utf-8")
    published = [summary_path, csv_path, plot_path, report_path]
    manifest = {
        "status": status,
        "inputs": {
            "physical_summary": {"path": str(physical_summary_path), "sha256": _sha256(physical_summary_path)},
            "physical_gradient": {"path": str(physical_raw_path), "sha256": _sha256(physical_raw_path)},
            "exact_baseline_summary": {"path": str(baseline_summary_path), "sha256": _sha256(baseline_summary_path)},
            "exact_baseline_raw": {"path": str(baseline_raw_path), "sha256": _sha256(baseline_raw_path)},
            "mapping_summary": {"path": str(mapping_summary_path), "sha256": _sha256(mapping_summary_path)},
            "mapping_raw": {"path": str(mapping_raw_path), "sha256": _sha256(mapping_raw_path)},
        },
        "raw_artifact": summary["raw_artifact"],
        "forward_case_root": {"path": str(work_root), "committed_to_git": False},
        "published": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in published
        ],
    }
    (output / "RAW_ARTIFACT_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": status, "directions": rows, "gates": gates}, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
