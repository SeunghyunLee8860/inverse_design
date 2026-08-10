#!/usr/bin/env python3
"""Compare the final binary design under the two orthogonal polarizations.

No solver is called here.  The script reads the independently completed GPU
Maxwell/CUDA thermal evaluations, reconstructs the exact full-footprint PTE
current, and publishes common-scale field maps.  Spatial derivatives shown in
the maps require all four in-plane neighbours; missing-neighbour cells are NaN.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402
import numpy as np  # noqa: E402


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]
RUN002 = HERE.parent / "run_002_gaussian10_w8p5_current_max"
for path in (HERE, REPOSITORY, RUN002):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from map_production_q_to_thermal_grid import material_masks  # noqa: E402
from photothermal_pte.finite_inverse_design.native_yee_q import integrate_xyz  # noqa: E402
from plot_final_binary_fields import (  # noqa: E402
    DESIGN_HALF_SPAN_M,
    SEEBECK_A_V_K,
    SEEBECK_B_V_K,
    SIGMA_A_S_M,
    SIGMA_B_S_M,
    WEIGHTING_X_M_INV,
    WEIGHTING_Y_M_INV,
    artifact,
    centers,
)


X_RAW = Path(
    "/data/seunghyun/tairte4/raw_artifacts/run005_lowbeta_topology_pilot_20260808/"
    "final_binary_g046_b2048/solver_evaluation/thresholded_binary_evaluation.npz"
)
X_RESULT = X_RAW.with_name("thresholded_binary_evaluation_result.json")
Y_RAW = Path(
    "/home/seunghyun/tairte4/raw_artifacts/"
    "run005_final_binary_y_polarization_resume_20260808/"
    "thresholded_binary_evaluation.npz"
)
Y_RESULT = Y_RAW.with_name("thresholded_binary_evaluation_result.json")
DERIVED_ROOT = Path(
    "/home/seunghyun/tairte4/raw_artifacts/"
    "run005_final_binary_both_polarizations_20260808"
)
DERIVED = DERIVED_ROOT / "final_binary_both_polarizations_derived_fields.npz"

PLOT_FIELDS = HERE / "plots/final_binary_both_polarizations_Q_T_gradient_current.png"
PLOT_METRICS = HERE / "plots/final_binary_both_polarizations_metrics.png"
SUMMARY = HERE / "results/final_binary_polarization_comparison_summary.json"
CSV_PATH = HERE / "results/final_binary_polarization_comparison_metrics.csv"
REPORT = HERE / "results/FINAL_BINARY_POLARIZATION_COMPARISON_REPORT.md"
MANIFEST = HERE / "manifests/RAW_ARTIFACT_MANIFEST.json"

AXIS_BLOCKER = "UNRESOLVED_AXIS_METADATA_MISMATCH_XB_YA_VS_THERMAL_PTE_XA_YB"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_norm(fields: list[np.ndarray]) -> TwoSlopeNorm:
    limit = max(float(np.nanmax(np.abs(field))) for field in fields)
    return TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)


def native_component_power(result: dict) -> dict[str, float]:
    path = Path(result["base_forward"]["native_Q"]["path"])
    expected = result["base_forward"]["native_Q"]["sha256"]
    if sha256(path) != expected:
        raise RuntimeError(f"native Q SHA mismatch: {path}")
    data = np.load(path)
    power = {}
    for component in "xyz":
        power[component] = integrate_xyz(
            np.asarray(data[f"Q{component}_W_m3"], float),
            *(
                np.asarray(data[f"Q{component}_{axis}_m"], float)
                for axis in "xyz"
            ),
        )
    return power


def process_case(raw_path: Path, result_path: Path, optical_label: str) -> dict:
    result = json.loads(result_path.read_text())
    if not result.get("passed"):
        raise RuntimeError(f"source evaluation failed: {result_path}")
    data = np.load(raw_path)
    rho = np.asarray(data["rho_binary"], np.uint8)
    if set(np.unique(rho).tolist()) != {0, 1}:
        raise RuntimeError("density is not exact binary")
    edges = tuple(np.asarray(data[f"{axis}_edges_m"], float) for axis in "xyz")
    x, y, z = tuple(centers(edge) for edge in edges)
    dx, dy, dz = tuple(np.diff(edge) for edge in edges)
    q = np.asarray(data["Q_total_W_m3"], float)
    temperature = np.asarray(data["thermal_temperature_grid_K"], float)
    masks = material_masks(edges, design_half_span_m=DESIGN_HALF_SPAN_M)
    flake = masks["physical_TaIrTe4"]
    fx = np.flatnonzero(np.any(flake, axis=(1, 2)))
    fy = np.flatnonzero(np.any(flake, axis=(0, 2)))
    fz = np.flatnonzero(np.any(flake, axis=(0, 1)))
    tf = temperature[np.ix_(fx, fy, fz)]
    flake_dz = dz[fz]
    thickness = float(np.sum(flake_dz))
    tavg = np.sum(tf * flake_dz[None, None, :], axis=2) / thickness

    # Exact existing production scalar: boundary-aware gradient at every z.
    dtx_full = np.gradient(tf, x[fx], axis=0, edge_order=2)
    dty_full = np.gradient(tf, y[fy], axis=1, edge_order=2)
    current_x_density = -WEIGHTING_X_M_INV * SIGMA_A_S_M * SEEBECK_A_V_K * dtx_full
    current_y_density = -WEIGHTING_Y_M_INV * SIGMA_B_S_M * SEEBECK_B_V_K * dty_full
    volume = dx[fx, None, None] * dy[None, fy, None] * dz[None, None, fz]
    current_x = float(np.sum(current_x_density * volume))
    current_y = float(np.sum(current_y_density * volume))
    current = current_x + current_y
    stored = float(result["objective_A"])
    reintegration = abs(current - stored) / max(abs(stored), np.finfo(float).tiny)
    if reintegration >= 1e-12:
        raise RuntimeError(f"current reintegration failed: {reintegration}")

    strict = np.zeros(tavg.shape, bool)
    strict[1:-1, 1:-1] = True
    dtx = np.full_like(tavg, np.nan)
    dty = np.full_like(tavg, np.nan)
    dtx[1:-1, 1:-1] = (
        tavg[2:, 1:-1] - tavg[:-2, 1:-1]
    ) / (x[fx][2:, None] - x[fx][:-2, None])
    dty[1:-1, 1:-1] = (
        tavg[1:-1, 2:] - tavg[1:-1, :-2]
    ) / (y[fy][None, 2:] - y[fy][None, :-2])
    gradient = np.hypot(dtx, dty)
    current_map = thickness * (
        -WEIGHTING_X_M_INV * SIGMA_A_S_M * SEEBECK_A_V_K * dtx
        -WEIGHTING_Y_M_INV * SIGMA_B_S_M * SEEBECK_B_V_K * dty
    )
    if not (
        np.all(np.isnan(dtx[~strict]))
        and np.all(np.isnan(dty[~strict]))
        and np.all(np.isnan(current_map[~strict]))
    ):
        raise RuntimeError("strict four-neighbour mask failed")

    qxy = np.sum(q * flake * dz[None, None, :], axis=2)[np.ix_(fx, fy)]
    mapped_power = float(np.sum(q * dx[:, None, None] * dy[None, :, None] * dz[None, None, :]))
    expected_power = float(result["base_mapping"]["mapped_power_W"])
    if abs(mapped_power - expected_power) / expected_power >= 1e-12:
        raise RuntimeError("mapped Q reintegration failed")
    component_power = native_component_power(result)
    metrics = {
        "optical_label": optical_label,
        "native_P_Q_W": float(result["base_forward"]["P_Q_W"]),
        "P_six_W": float(result["base_forward"]["P_six_W"]),
        "closure": float(result["base_forward"]["closure"]),
        "mapped_Q_W": mapped_power,
        "native_component_power_W": component_power,
        "maximum_temperature_rise_K": float(np.nanmax(temperature)),
        "maximum_flake_average_temperature_rise_K": float(np.nanmax(tavg)),
        "strict_max_abs_dTdx_K_m": float(np.nanmax(np.abs(dtx))),
        "strict_max_abs_dTdy_K_m": float(np.nanmax(np.abs(dty))),
        "strict_max_gradient_K_m": float(np.nanmax(gradient)),
        "current_x_term_A": current_x,
        "current_y_term_A": current_y,
        "full_current_A": current,
        "FOM_A_per_W": float(result["objective_A_per_incident_W"]),
        "objective_reintegration_error": reintegration,
        "auto_shutoff": float(result["gates"]["base_auto_shutoff"]),
        "thermal_residual": float(result["gates"]["thermal_residual"]),
        "thermal_energy_balance": float(result["gates"]["thermal_energy_balance"]),
    }
    return {
        "rho": rho,
        "x": x[fx],
        "y": y[fy],
        "Q": qxy,
        "T": tavg,
        "dTdx": dtx,
        "dTdy": dty,
        "grad": gradient,
        "current": current_map,
        "strict": strict,
        "metrics": metrics,
        "result": result,
    }


def main() -> int:
    # Optical source x maps to crystal b; source y maps to crystal a under the
    # immutable optical metadata.  The separate thermal/PTE axis mismatch is
    # reported, not silently corrected here.
    cases = [
        process_case(X_RAW, X_RESULT, "E || b (source x, 0 deg)"),
        process_case(Y_RAW, Y_RESULT, "E || a (source y, 90 deg)"),
    ]
    if not np.array_equal(cases[0]["rho"], cases[1]["rho"]):
        raise RuntimeError("polarizations used different binary structures")
    for key in ("x", "y"):
        if not np.array_equal(cases[0][key], cases[1][key]):
            raise RuntimeError(f"polarizations used different thermal {key} grid")

    DERIVED_ROOT.mkdir(parents=True, exist_ok=True)
    arrays = {"rho_material_1_void_0": cases[0]["rho"]}
    for prefix, case in zip(("E_parallel_b", "E_parallel_a"), cases):
        arrays[f"{prefix}_x_m"] = case["x"]
        arrays[f"{prefix}_y_m"] = case["y"]
        for key in ("Q", "T", "dTdx", "dTdy", "grad", "current", "strict"):
            arrays[f"{prefix}_{key}"] = case[key]
    np.savez_compressed(DERIVED, **arrays)

    qmax = max(float(np.nanmax(case["Q"])) for case in cases)
    tmax = max(float(np.nanmax(case["T"])) for case in cases)
    dxnorm = strict_norm([case["dTdx"] for case in cases])
    dynorm = strict_norm([case["dTdy"] for case in cases])
    gmax = max(float(np.nanmax(case["grad"])) for case in cases)
    jnorm = strict_norm([case["current"] for case in cases])
    fig, axes = plt.subplots(2, 6, figsize=(27, 9.3), constrained_layout=True)
    specs = (
        ("Q", "TaIrTe4 depth-integrated Q", "W/m2", "inferno", 0.0, qmax, None),
        ("T", "thickness-avg DeltaT", "K", "magma", 0.0, tmax, None),
        ("dTdx", "strict dT/dx", "K/m", "coolwarm", None, None, dxnorm),
        ("dTdy", "strict dT/dy", "K/m", "coolwarm", None, None, dynorm),
        ("grad", "strict |grad T|", "K/m", "viridis", 0.0, gmax, None),
        ("current", "strict local PTE contribution", "A/m2", "coolwarm", None, None, jnorm),
    )
    for row, case in enumerate(cases):
        for col, (key, title, unit, cmap, vmin, vmax, norm) in enumerate(specs):
            image = axes[row, col].pcolormesh(
                case["x"] * 1e6,
                case["y"] * 1e6,
                np.ma.masked_invalid(case[key]).T,
                shading="nearest",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                norm=norm,
            )
            axes[row, col].set_title(f"{case['metrics']['optical_label']}\n{title}")
            axes[row, col].set_xlabel("solver x (um)")
            axes[row, col].set_ylabel("solver y (um)")
            axes[row, col].set_aspect("equal")
            fig.colorbar(image, ax=axes[row, col], label=unit, shrink=0.82)
    fig.suptitle(
        "Run 005 exact-binary final structure: both optical polarizations\n"
        "common scale by column; strict maps mask every missing -x,+x,-y,+y neighbour"
    )
    fig.savefig(PLOT_FIELDS, dpi=170)
    plt.close(fig)

    labels = ["E||b\nsource x", "E||a\nsource y"]
    metrics = [case["metrics"] for case in cases]
    fig, axes = plt.subplots(2, 3, figsize=(17, 9.5), constrained_layout=True)
    axes[0, 0].bar(labels, [m["native_P_Q_W"] * 1e15 for m in metrics], label="P_Q")
    axes[0, 0].scatter(labels, [m["P_six_W"] * 1e15 for m in metrics], color="black", marker="x", label="P_six")
    axes[0, 0].set_ylabel("power (fW)")
    axes[0, 0].set_title("native absorption and six-face power")
    axes[0, 0].legend()
    width = 0.24
    index = np.arange(2)
    for offset, component in zip((-width, 0.0, width), "xyz"):
        axes[0, 1].bar(index + offset, [m["native_component_power_W"][component] * 1e15 for m in metrics], width, label=f"Q{component}")
    axes[0, 1].set_xticks(index, labels)
    axes[0, 1].set_ylabel("component power (fW)")
    axes[0, 1].set_title("native Yee Q components")
    axes[0, 1].legend()
    axes[0, 2].bar(labels, [m["maximum_flake_average_temperature_rise_K"] * 1e9 for m in metrics])
    axes[0, 2].set_ylabel("maximum flake-average DeltaT (nK)")
    axes[0, 2].set_title("temperature response")
    axes[1, 0].bar(labels, [m["strict_max_gradient_K_m"] for m in metrics])
    axes[1, 0].set_ylabel("strict max |grad T| (K/m)")
    axes[1, 0].set_title("four-neighbour gradient diagnostic")
    for offset, key, name in ((-0.18, "current_x_term_A", "solver-x term"), (0.0, "current_y_term_A", "solver-y term"), (0.18, "full_current_A", "full total")):
        axes[1, 1].bar(index + offset, [m[key] * 1e18 for m in metrics], 0.18, label=name)
    axes[1, 1].set_xticks(index, labels)
    axes[1, 1].set_ylabel("current (aA)")
    axes[1, 1].set_title("existing full-footprint PTE operator")
    axes[1, 1].legend()
    ratio_p = metrics[1]["native_P_Q_W"] / metrics[0]["native_P_Q_W"]
    ratio_i = metrics[1]["full_current_A"] / metrics[0]["full_current_A"]
    ratio_f = metrics[1]["FOM_A_per_W"] / metrics[0]["FOM_A_per_W"]
    axes[1, 2].axis("off")
    axes[1, 2].text(
        0.02,
        0.95,
        "Ratios (E||a / E||b)\n\n"
        f"P_Q ratio = {ratio_p:.6f}\n"
        f"current ratio = {ratio_i:.6f}\n"
        f"FOM ratio = {ratio_f:.6f}\n\n"
        "Numerical gates: PASS for both\n"
        f"Physical axis interpretation:\n{AXIS_BLOCKER}",
        va="top",
        family="monospace",
        fontsize=10,
    )
    fig.suptitle("Final-binary two-polarization scalar and component comparison")
    fig.savefig(PLOT_METRICS, dpi=190)
    plt.close(fig)

    ratios = {
        "E_parallel_a_over_E_parallel_b": {
            "native_P_Q": metrics[1]["native_P_Q_W"] / metrics[0]["native_P_Q_W"],
            "mapped_Q": metrics[1]["mapped_Q_W"] / metrics[0]["mapped_Q_W"],
            "maximum_flake_average_temperature": metrics[1]["maximum_flake_average_temperature_rise_K"] / metrics[0]["maximum_flake_average_temperature_rise_K"],
            "strict_max_gradient": metrics[1]["strict_max_gradient_K_m"] / metrics[0]["strict_max_gradient_K_m"],
            "full_current": metrics[1]["full_current_A"] / metrics[0]["full_current_A"],
            "FOM": metrics[1]["FOM_A_per_W"] / metrics[0]["FOM_A_per_W"],
        }
    }
    summary = {
        "status": "VALIDATED_FINAL_BINARY_BOTH_OPTICAL_POLARIZATIONS_NUMERICALLY_WITH_AXIS_INTERPRETATION_BLOCKED",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "same exact-binary structure, source power, geometry, mesh, conservative remap, and CUDA thermal/PTE operator",
        "polarization_contract": {
            "source_x_0_deg": "E || b under optical metadata x=b",
            "source_y_90_deg": "E || a under optical metadata y=a",
            "no_power_matching_or_rescaling": True,
        },
        "coordinate_audit": {
            "optical_metadata": "x=b, y=a, z=c=b closure",
            "implemented_thermal_PTE_coefficients": "solver x uses a coefficients; solver y uses b coefficients",
            "interpretation_status": AXIS_BLOCKER,
            "action": "no silent axis swap; all spatial plots use literal solver x/y",
        },
        "cases": {"E_parallel_b": metrics[0], "E_parallel_a": metrics[1]},
        "ratios": ratios,
        "inputs": {
            "E_parallel_b_result": artifact(X_RESULT),
            "E_parallel_b_evaluation_NPZ": artifact(X_RAW),
            "E_parallel_a_result": artifact(Y_RESULT),
            "E_parallel_a_evaluation_NPZ": artifact(Y_RAW),
        },
        "derived_fields": artifact(DERIVED),
        "plots": {"field_comparison": artifact(PLOT_FIELDS), "metric_comparison": artifact(PLOT_METRICS)},
        "new_solver_counts": {"E_parallel_b_forward": 0, "E_parallel_a_forward": 1, "thermal_forward": 1, "thermal_adjoint": 1},
    }
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n")

    with CSV_PATH.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["polarization", "metric", "value", "unit"])
        units = {
            "native_P_Q_W": "W", "P_six_W": "W", "closure": "1", "mapped_Q_W": "W",
            "maximum_temperature_rise_K": "K", "maximum_flake_average_temperature_rise_K": "K",
            "strict_max_abs_dTdx_K_m": "K/m", "strict_max_abs_dTdy_K_m": "K/m",
            "strict_max_gradient_K_m": "K/m", "current_x_term_A": "A", "current_y_term_A": "A",
            "full_current_A": "A", "FOM_A_per_W": "A/W", "auto_shutoff": "1",
            "thermal_residual": "1", "thermal_energy_balance": "1",
        }
        for label, values in (("E_parallel_b", metrics[0]), ("E_parallel_a", metrics[1])):
            for key, unit in units.items():
                writer.writerow([label, key, f"{values[key]:.16e}", unit])

    ratio = ratios["E_parallel_a_over_E_parallel_b"]
    REPORT.write_text(
        "# Run 005 final-binary two-polarization comparison\n\n"
        f"Status: `{summary['status']}`.\n\n"
        "The earlier final-binary solve used source x polarization. Under the frozen optical "
        "metadata `x=b, y=a`, it is **E || b**, not E || a. A fresh source-y, 90-degree "
        "GPU forward and CUDA thermal/PTE evaluation now supplies **E || a**. Both use the "
        "same exact-binary structure and incident power; no polarization matching, Q "
        "rescaling, clipping, smoothing, or gain was used.\n\n"
        "## Numerical results\n\n"
        "| metric | E || b (source x) | E || a (source y) | a/b |\n"
        "|---|---:|---:|---:|\n"
        f"| P_Q (W) | {metrics[0]['native_P_Q_W']:.12e} | {metrics[1]['native_P_Q_W']:.12e} | {ratio['native_P_Q']:.6f} |\n"
        f"| mapped Q (W) | {metrics[0]['mapped_Q_W']:.12e} | {metrics[1]['mapped_Q_W']:.12e} | {ratio['mapped_Q']:.6f} |\n"
        f"| max flake-average DeltaT (K) | {metrics[0]['maximum_flake_average_temperature_rise_K']:.12e} | {metrics[1]['maximum_flake_average_temperature_rise_K']:.12e} | {ratio['maximum_flake_average_temperature']:.6f} |\n"
        f"| strict max gradient (K/m) | {metrics[0]['strict_max_gradient_K_m']:.12e} | {metrics[1]['strict_max_gradient_K_m']:.12e} | {ratio['strict_max_gradient']:.6f} |\n"
        f"| full current (A) | {metrics[0]['full_current_A']:.12e} | {metrics[1]['full_current_A']:.12e} | {ratio['full_current']:.6f} |\n"
        f"| FOM (A/W) | {metrics[0]['FOM_A_per_W']:.12e} | {metrics[1]['FOM_A_per_W']:.12e} | {ratio['FOM']:.6f} |\n\n"
        "Both cases pass optical closure, auto-shutoff, conservative-remap power, CUDA "
        "thermal residual, and energy-balance gates. Spatial derivative/current maps use "
        "NaN wherever any one of `-x,+x,-y,+y` neighbours is missing.\n\n"
        "## Axis interpretation blocker\n\n"
        "The optical metadata says `x=b, y=a`, but the existing immutable thermal/PTE "
        "operator applies the `a` coefficients to solver x and the `b` coefficients to "
        f"solver y. Therefore the numerical two-polarization comparison is valid, while "
        f"its crystallographic current interpretation remains `{AXIS_BLOCKER}`. No "
        "coefficient or axis was silently swapped in this postprocessing.\n"
    )

    manifest = json.loads(MANIFEST.read_text())
    manifest["final_binary_two_polarization_comparison"] = {
        "status": summary["status"],
        "E_parallel_b": summary["inputs"]["E_parallel_b_evaluation_NPZ"],
        "E_parallel_a": summary["inputs"]["E_parallel_a_evaluation_NPZ"],
        "E_parallel_a_FSP": artifact(Path(cases[1]["result"]["base_forward"]["project"]["path"])),
        "E_parallel_a_native_Q": artifact(Path(cases[1]["result"]["base_forward"]["native_Q"]["path"])),
        "derived_fields": artifact(DERIVED),
        "summary": artifact(SUMMARY),
        "CSV": artifact(CSV_PATH),
        "report": artifact(REPORT),
        "plots": summary["plots"],
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
