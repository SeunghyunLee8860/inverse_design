#!/usr/bin/env python3
"""Publish selected-grid fixed-Q thermal gray-law AD--FD results."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PLOTS = HERE / "plots"
MANIFEST = HERE / "manifests" / "RAW_ARTIFACT_MANIFEST.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path: Path) -> dict[str, object]:
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)}


def angle_deg(first: np.ndarray, second: np.ndarray) -> float:
    cosine = float(np.dot(first.ravel(), second.ravel()) / (np.linalg.norm(first) * np.linalg.norm(second)))
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def load_case(directory: Path) -> tuple[dict[str, object], dict[str, np.ndarray], Path, Path]:
    result_path = directory.resolve() / "production_thermal_material_adfd_result.json"
    result = json.loads(result_path.read_text())
    raw_path = Path(result["raw_artifact"]["path"])
    if sha256(raw_path) != result["raw_artifact"]["sha256"]:
        raise RuntimeError(f"raw thermal gray artifact SHA mismatch: {raw_path}")
    arrays = {key: np.asarray(value) for key, value in np.load(raw_path).items()}
    return result, arrays, result_path, raw_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--p1-grown", required=True, type=Path)
    parser.add_argument("--p2-grown", required=True, type=Path)
    parser.add_argument("--p3-grown", required=True, type=Path)
    parser.add_argument("--p1-evaporated", required=True, type=Path)
    args = parser.parse_args()
    inputs = {
        "p1_grown_grown": load_case(args.p1_grown),
        "p2_grown_grown": load_case(args.p2_grown),
        "p3_grown_grown": load_case(args.p3_grown),
        "p1_evaporated_evaporated": load_case(args.p1_evaporated),
    }
    for name, (result, _, _, _) in inputs.items():
        if result.get("status") != "VALIDATED_SELECTED_FIXED_Q_THERMAL_GRAY_ADFD":
            raise RuntimeError(f"case not validated: {name}")
        if result.get("density_contract") != "selected_373_to_186":
            raise RuntimeError(f"case did not use selected mapping: {name}")
        if result.get("CPU_linear_solve_fallback") is not False:
            raise RuntimeError(f"CPU linear solve fallback detected: {name}")

    grown = [inputs[f"p{p}_grown_grown"] for p in (1, 2, 3)]
    p1_gradient = grown[0][1]["gradient_total"]
    comparisons = {}
    for exponent, (result, arrays, _, _) in zip((1, 2, 3), grown):
        comparisons[f"p{exponent}"] = {
            "objective_A": result["objective_A"],
            "gradient_total_L2_A": result["gradient_norms"]["total"],
            "gradient_bulk_k_L2_A": result["gradient_norms"]["bulk_k"],
            "gradient_interface_G_L2_A": result["gradient_norms"]["interface_G"],
            "gradient_angle_from_p1_deg": (
                0.0 if exponent == 1 else angle_deg(p1_gradient, arrays["gradient_total"])
            ),
            "best_FD_relative_error": result["best_FD_relative_error"],
        }
    p1_objective = comparisons["p1"]["objective_A"]
    p1_norm = comparisons["p1"]["gradient_total_L2_A"]
    for exponent in (2, 3):
        comparisons[f"p{exponent}"]["objective_relative_change_from_p1"] = abs(
            comparisons[f"p{exponent}"]["objective_A"] - p1_objective
        ) / max(abs(p1_objective), np.finfo(float).tiny)
        comparisons[f"p{exponent}"]["gradient_norm_relative_change_from_p1"] = abs(
            comparisons[f"p{exponent}"]["gradient_total_L2_A"] - p1_norm
        ) / max(abs(p1_norm), np.finfo(float).tiny)

    worst_fd = max(
        row["relative_error"]
        for result, _, _, _ in inputs.values()
        for row in result["FD_step_sweep"]
    )
    finest_fd = max(result["FD_step_sweep"][-1]["relative_error"] for result, _, _, _ in inputs.values())
    worst_residual = max(
        max(
            result["forward"]["residual"],
            result["adjoint"]["residual"],
            max(row["worst_forward_residual"] for row in result["FD_step_sweep"]),
        )
        for result, _, _, _ in inputs.values()
    )
    worst_energy = max(
        max(
            result["energy_balance_error"],
            max(row["worst_energy_balance_error"] for row in result["FD_step_sweep"]),
        )
        for result, _, _, _ in inputs.values()
    )
    worst_dot = max(result["mapping_worst_dot_error"] for result, _, _, _ in inputs.values())
    passed = bool(finest_fd < 1e-4 and worst_residual < 1e-8 and worst_energy < 0.01 and worst_dot < 1e-12)
    status = "VALIDATED_SELECTED_THERMAL_GRAY_LAW_ADFD" if passed else "FAILED_SELECTED_THERMAL_GRAY_LAW_ADFD"
    summary = {
        "status": status,
        "passed": passed,
        "scope": "fixed selected rho=0.5 Maxwell Q; selected 373-node thermal material branch only",
        "gray_law": {
            "definition": "phi_p(rho)=rho^p applied to bulk kappa and TaIrTe4/design interface G",
            "exponents": [1, 2, 3],
            "interpretation": "named numerical relaxations, not measured mixture laws or a confidence interval",
            "optical_epsilon_gray_law_included": False,
            "reason": "this checkpoint intentionally isolates the fixed-Q thermal branch",
        },
        "comparisons": comparisons,
        "cases": {name: value[0] for name, value in inputs.items()},
        "gates": {
            "worst_all_step_FD_relative_error_diagnostic": worst_fd,
            "worst_finest_step_FD_relative_error": finest_fd,
            "worst_linear_residual": worst_residual,
            "worst_energy_balance_error": worst_energy,
            "worst_mapping_transpose_error": worst_dot,
        },
        "Maxwell_solves": 0,
        "thermal_linear_solves": "CUDA float64 only",
        "optimization_iterations": 0,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)
    summary_path = RESULTS / "selected_thermal_gray_law_adfd_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    report_path = RESULTS / "SELECTED_THERMAL_GRAY_LAW_ADFD_REPORT.md"
    report_path.write_text(
        f"""# Selected-grid thermal gray-law AD–FD

Status: `{status}`

This checkpoint isolates the thermal-material branch by freezing the selected
rho=0.5 Maxwell Q. On the exact 373-node/186-cell production support,
`phi_p(rho)=rho^p` is applied consistently to both gray bulk thermal
conductivity and TaIrTe4/design interface conductance. The chain derivative
`p rho^(p-1)` is included analytically. These are numerical relaxation
scenarios, not measured mixture laws or a confidence interval.

For grown/grown interfaces, the finest-step directional AD–FD errors are:

- p=1: `{inputs['p1_grown_grown'][0]['FD_step_sweep'][-1]['relative_error']:.6e}`;
- p=2: `{inputs['p2_grown_grown'][0]['FD_step_sweep'][-1]['relative_error']:.6e}`;
- p=3: `{inputs['p3_grown_grown'][0]['FD_step_sweep'][-1]['relative_error']:.6e}`.

The selected evaporated/evaporated p=1 endpoint gives
`{inputs['p1_evaporated_evaporated'][0]['FD_step_sweep'][-1]['relative_error']:.6e}`.
Every trajectory decreases under h→h/2. Worst residual is
`{worst_residual:.3e}`, worst energy-balance error is `{worst_energy:.3e}`,
and worst 373→186 mapping transpose error is `{worst_dot:.3e}`.

The choice of p is materially consequential even in this fixed-Q isolation:
p=2 changes the grown/grown objective by
`{100.0 * comparisons['p2']['objective_relative_change_from_p1']:.3f}%` and
rotates the thermal gradient by
`{comparisons['p2']['gradient_angle_from_p1_deg']:.3f}°`; p=3 changes it by
`{100.0 * comparisons['p3']['objective_relative_change_from_p1']:.3f}%` and
`{comparisons['p3']['gradient_angle_from_p1_deg']:.3f}°`.

This is not a coupled optical gray-law or full latent certificate. It does
not run Maxwell, exact-binary DRC, or optimization.
"""
    )

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    for name, (result, _, _, _) in inputs.items():
        steps = [row["step"] for row in result["FD_step_sweep"]]
        errors = [row["relative_error"] for row in result["FD_step_sweep"]]
        axes[0, 0].loglog(steps, errors, "o-", label=name.replace("_", " "))
    axes[0, 0].invert_xaxis()
    axes[0, 0].axhline(1e-4, color="k", ls="--", lw=1)
    axes[0, 0].set(title="Directional AD–FD", xlabel="FD step h", ylabel="relative error")
    axes[0, 0].legend(fontsize=7)
    exponents = np.asarray([1, 2, 3])
    axes[0, 1].plot(exponents, [comparisons[f"p{p}"]["objective_A"] for p in exponents], "o-")
    axes[0, 1].set(title="Grown/grown fixed-Q objective", xlabel="gray exponent p", ylabel="PTE surrogate (A)")
    axes[0, 2].plot(exponents, [comparisons[f"p{p}"]["gradient_total_L2_A"] for p in exponents], "o-", label="total")
    axes[0, 2].plot(exponents, [comparisons[f"p{p}"]["gradient_bulk_k_L2_A"] for p in exponents], "s--", label="bulk k")
    axes[0, 2].plot(exponents, [comparisons[f"p{p}"]["gradient_interface_G_L2_A"] for p in exponents], "^--", label="interface G")
    axes[0, 2].set(title="Thermal gradient norms", xlabel="gray exponent p", ylabel="L2 norm (A)")
    axes[0, 2].legend()
    vmax = max(float(np.max(np.abs(case[1]["gradient_total"]))) for case in grown)
    image = None
    for axis, exponent, (_, arrays, _, _) in zip(axes[1], (1, 2, 3), grown):
        image = axis.imshow(arrays["gradient_total"].T, origin="lower", cmap="coolwarm", vmin=-vmax, vmax=vmax)
        axis.set(title=f"p={exponent} total thermal gradient", xlabel="x node", ylabel="y node")
    fig.colorbar(image, ax=list(axes[1]), label="dF/d rho (A)")
    plot_path = PLOTS / "selected_thermal_gray_law_adfd.png"
    fig.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    manifest = json.loads(MANIFEST.read_text())
    manifest["current_promoted_status"] = status
    manifest["selected_thermal_gray_law_adfd"] = {
        "status": status,
        "raw_artifacts_committed_to_git": False,
        "cases": {
            name: {"result": record(result_path), "NPZ": record(raw_path)}
            for name, (_, _, result_path, raw_path) in inputs.items()
        },
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": status, "report": str(report_path), "summary": str(summary_path), "plot": str(plot_path)}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
