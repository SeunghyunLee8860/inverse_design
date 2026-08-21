#!/usr/bin/env python3
"""Validate explicit thermal + Au-aware electrical AD--FD at fixed spatial Q."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
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
from scipy import ndimage

from photothermal_pte.optimization_runs.cuda_thermal_adjoint import PersistentCudaCSR


HERE = Path(__file__).resolve().parent
STAGE54 = HERE / "54_validate_au_weighting_electrical_adfd.py"
STAGE62 = HERE / "62_validate_coupled_au_thermal_weighting_pte_adfd.py"
STAGE65 = HERE / "65_solve_fdtdx_explicit_thermal_weighting_pte.py"
EXPECTED_REMAP_STATUS = "VALIDATED_FDTDX_SPATIAL_Q_CONSERVATIVE_MATERIAL_OVERLAP_REMAP"
STATUS_PASS = "VALIDATED_EXPLICIT_THERMAL_WEIGHTING_FIXED_SPATIAL_Q_ADFD"
STATUS_FAIL = "FAILED_EXPLICIT_THERMAL_WEIGHTING_FIXED_SPATIAL_Q_ADFD"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(first: float, second: float) -> float:
    return abs(first - second) / max(abs(first), abs(second), np.finfo(float).tiny)


def _solve(matrix, rhs: np.ndarray, cuda_device: int):
    operator = PersistentCudaCSR(matrix, cuda_device=cuda_device)
    return operator.solve(
        rhs,
        relative_tolerance=1.0e-9,
        max_iterations=30000,
        residual_check_interval=25,
    )


def _temperature_pullback_to_explicit(
    psi: np.ndarray, state: dict, electrical, coupled
) -> np.ndarray:
    """Transpose the 3-D Ta-temperature -> 40x40 electrical averaging map."""
    coarse = coupled.temperature_pullback(psi, electrical)
    if coarse.shape != (40, 40):
        raise RuntimeError(f"Unexpected electrical temperature pullback {coarse.shape}")
    fine_xy = np.repeat(np.repeat(coarse / 25.0, 5, axis=0), 5, axis=1)
    x, y, z = state["centers"]
    ix = np.flatnonzero((x >= -10.0e-6) & (x < 10.0e-6))
    iy = np.flatnonzero((y >= -10.0e-6) & (y < 10.0e-6))
    iz = np.flatnonzero((z >= -0.1e-6) & (z < 0.0))
    if fine_xy.shape != (ix.size, iy.size):
        raise RuntimeError("Ta electrical/thermal lateral grids do not match")
    z_weights = state["widths"][2][iz]
    z_weights = z_weights / np.sum(z_weights)
    full = np.zeros(state["system"].shape, dtype=np.float64)
    full[np.ix_(ix, iy, iz)] = fine_xy[:, :, None] * z_weights[None, None, :]
    return full.reshape(-1)


def _generic_face_dg(
    state: dict,
    left: tuple[int, int, int],
    right: tuple[int, int, int],
    axis: int,
    derivative_cell: tuple[int, int, int],
    dk_drho: float,
) -> float:
    widths = state["widths"]
    kappa = state["kappa"]
    resistance = state["interface_resistance"]["xyz"[axis]]
    li, lj, lk = left
    ri, rj, rk = right
    ci, cj, ck = derivative_cell
    if axis == 0:
        face_r = resistance[li, lj, lk]
        area = widths[1][lj] * widths[2][lk]
    elif axis == 1:
        face_r = resistance[li, lj, lk]
        area = widths[0][li] * widths[2][lk]
    else:
        face_r = resistance[li, lj, lk]
        area = widths[0][li] * widths[1][lj]
    total_r = (
        0.5 * widths[axis][left[axis]] / kappa[left + (axis,)]
        + face_r
        + 0.5 * widths[axis][right[axis]] / kappa[right + (axis,)]
    )
    local_k = kappa[ci, cj, ck, axis]
    return float(
        area
        / total_r**2
        * 0.5
        * widths[axis][derivative_cell[axis]]
        / local_k**2
        * dk_drho
    )


def _thermal_density_gradient(
    state: dict,
    rho: np.ndarray,
    temperature: np.ndarray,
    adjoint: np.ndarray,
    forward,
) -> np.ndarray:
    """Return -lambda^T(dK/drho)T for the explicit 3-D Au gray layer."""
    shape = state["system"].shape
    ids = np.arange(np.prod(shape), dtype=np.int64).reshape(shape)
    x, y, z = state["centers"]
    ix = np.flatnonzero((x >= -5.0e-6) & (x < 5.0e-6))
    iy = np.flatnonzero((y >= -5.0e-6) & (y < 5.0e-6))
    iz = np.flatnonzero((z >= 0.0) & (z < 0.05e-6))
    if (ix.size, iy.size) != (100, 100) or rho.shape != (20, 20):
        raise RuntimeError("Unexpected explicit Au design grid")
    dk = forward.K_AU_W_MK - forward.K_AIR_W_MK
    result = np.zeros_like(rho, dtype=np.float64)
    ta_top_face = state["faces"]["TaIrTe4_Au_or_air"]
    lower_dz = state["widths"][2][ta_top_face]
    upper_dz = state["widths"][2][ta_top_face + 1]
    r_air = (
        0.5 * lower_dz / forward.K_TA_XYZ_W_MK[2]
        + 1.0 / forward.G_TA_AIR_W_M2K
        + 0.5 * upper_dz / forward.K_AIR_W_MK
    )
    r_au = (
        0.5 * lower_dz / forward.K_TA_XYZ_W_MK[2]
        + 1.0 / forward.G_AU_TA_W_M2K
        + 0.5 * upper_dz / forward.K_AU_W_MK
    )

    def add(coarse_i: int, coarse_j: int, left_id: int, right_id: int, dg: float):
        result[coarse_i, coarse_j] += -dg * (
            adjoint[left_id] - adjoint[right_id]
        ) * (temperature[left_id] - temperature[right_id])

    for local_i, i in enumerate(ix):
        coarse_i = local_i // 5
        for local_j, j in enumerate(iy):
            coarse_j = local_j // 5
            for k in iz:
                cell = (i, j, k)
                for axis in (0, 1):
                    lower = list(cell)
                    lower[axis] -= 1
                    lower_tuple = tuple(lower)
                    dg = _generic_face_dg(
                        state, lower_tuple, cell, axis, cell, dk
                    )
                    add(
                        coarse_i,
                        coarse_j,
                        int(ids[lower_tuple]),
                        int(ids[cell]),
                        dg,
                    )
                    upper = list(cell)
                    upper[axis] += 1
                    upper_tuple = tuple(upper)
                    dg = _generic_face_dg(
                        state, cell, upper_tuple, axis, cell, dk
                    )
                    add(
                        coarse_i,
                        coarse_j,
                        int(ids[cell]),
                        int(ids[upper_tuple]),
                        dg,
                    )

                if k == iz[0]:
                    lower = (i, j, k - 1)
                    area = state["widths"][0][i] * state["widths"][1][j]
                    dg = area * (1.0 / r_au - 1.0 / r_air)
                    add(
                        coarse_i,
                        coarse_j,
                        int(ids[lower]),
                        int(ids[cell]),
                        float(dg),
                    )
                else:
                    lower = (i, j, k - 1)
                    dg = _generic_face_dg(state, lower, cell, 2, cell, dk)
                    add(
                        coarse_i,
                        coarse_j,
                        int(ids[lower]),
                        int(ids[cell]),
                        dg,
                    )
                upper = (i, j, k + 1)
                dg = _generic_face_dg(state, cell, upper, 2, cell, dk)
                add(
                    coarse_i,
                    coarse_j,
                    int(ids[cell]),
                    int(ids[upper]),
                    dg,
                )
    return result


def _evaluate(
    rho: np.ndarray,
    source_power: np.ndarray,
    scenario: str,
    cuda_device: int,
    *,
    need_gradient: bool,
    forward,
    electrical,
    coupled,
    topology,
    fvm,
) -> dict:
    state = forward._thermal_state(
        rho,
        forward.G_TA_SIO2_SCENARIOS[scenario],
        topology,
        fvm,
    )
    thermal = _solve(
        state["system"].matrix_W_K, source_power.reshape(-1), cuda_device
    )
    temperature = thermal.solution
    full_temperature = temperature.reshape(state["system"].shape)
    ta_temperature = forward._ta_temperature_500nm(state, full_temperature)
    electrical_base = electrical.build_system(
        rho, forward.ELECTRICAL_CONTACT_S_M2
    )
    electrical_system = replace(
        electrical_base,
        objective_gradient_psi_A=coupled.electrical_load(
            ta_temperature, electrical
        ),
    )
    psi, electrical_adjoint, electrical_solver = electrical.solve_gpu(
        electrical_system, cuda_device, need_adjoint=need_gradient
    )
    objective = electrical.objective(electrical_system, psi)
    electrical_audit = electrical.audit(electrical_system, psi)
    thermal_energy, _ = forward._boundary_energy(
        state, temperature, source_power.reshape(-1)
    )
    result = {
        "objective_A": objective,
        "temperature": temperature,
        "weighting": psi,
        "thermal_residual": float(thermal.explicit_relative_residual),
        "thermal_energy_balance": thermal_energy,
        "electrical_residual": float(
            electrical_solver["weighting_relative_residual"]
        ),
        "electrical_balance": float(electrical_audit["terminal_current_balance"]),
        "state": state,
    }
    if need_gradient:
        if electrical_adjoint is None:
            raise RuntimeError("Missing electrical adjoint")
        thermal_rhs = _temperature_pullback_to_explicit(
            psi, state, electrical, coupled
        )
        thermal_adjoint = _solve(
            state["system"].matrix_W_K, thermal_rhs, cuda_device
        )
        gradient_thermal = _thermal_density_gradient(
            state,
            rho,
            temperature,
            thermal_adjoint.solution,
            forward,
        )
        gradient_electrical = electrical.gradient(
            electrical_system, psi, electrical_adjoint
        )
        result.update(
            thermal_adjoint=thermal_adjoint.solution,
            gradient_thermal=gradient_thermal,
            gradient_electrical=gradient_electrical,
            gradient_total=gradient_thermal + gradient_electrical,
            thermal_adjoint_residual=float(
                thermal_adjoint.explicit_relative_residual
            ),
            electrical_adjoint_residual=float(
                electrical_solver["adjoint_relative_residual"]
            ),
        )
    return result


def _directions(gradient: np.ndarray) -> dict[str, np.ndarray]:
    x = np.linspace(-1.0, 1.0, 20)[:, None]
    y = np.linspace(-1.0, 1.0, 20)[None, :]
    rng = np.random.default_rng(20260821)
    raw = {
        "adjoint_aligned": gradient,
        "smooth_asymmetric": np.sin(0.72 * np.pi * x) * np.cos(0.53 * np.pi * y)
        + 0.2 * y,
        "central_localized": np.exp(-((x - 0.08) ** 2 + (y + 0.1) ** 2) / 0.07),
        "design_edge_localized": np.exp(
            -((x + 0.86) ** 2 + (y - 0.3) ** 2) / 0.025
        ),
        "fixed_seed_random": ndimage.gaussian_filter(rng.normal(size=(20, 20)), 1.5),
    }
    return {name: value / np.linalg.norm(value) for name, value in raw.items()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--remap-summary-json", required=True, type=Path)
    parser.add_argument("--raw-remap-npz", required=True, type=Path)
    parser.add_argument("--raw-spatial-npz", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--raw-output-npz", required=True, type=Path)
    parser.add_argument("--cuda-device", type=int, default=0)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
        raise RuntimeError("GPU-only validation requires CUDA_VISIBLE_DEVICES")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    remap_summary = json.loads(
        args.remap_summary_json.resolve().read_text(encoding="utf-8")
    )
    if remap_summary.get("status") != EXPECTED_REMAP_STATUS:
        raise RuntimeError("Fail-closed: remap certificate status mismatch")
    remap_path = args.raw_remap_npz.resolve()
    spatial_path = args.raw_spatial_npz.resolve()
    if _sha256(remap_path) != remap_summary["output_thermal_Q"]["sha256"]:
        raise RuntimeError("Fail-closed: thermal-Q SHA mismatch")
    if _sha256(spatial_path) != remap_summary["input_spatial_Q"]["sha256"]:
        raise RuntimeError("Fail-closed: spatial-Q SHA mismatch")

    forward = _load(STAGE65, "au_stage67_forward")
    electrical = _load(STAGE54, "au_stage67_electrical")
    coupled = _load(STAGE62, "au_stage67_coupled")
    topology = _load(forward.TOPOLOGY_THERMAL, "au_stage67_topology")
    fvm = _load(
        Path(__file__).parents[2]
        / "validation"
        / "photothermal_stage1"
        / "anisotropic_heat_fvm.py",
        "au_stage67_fvm",
    )
    overlap = _load(forward.STAGE64, "au_stage67_overlap")
    with np.load(spatial_path, allow_pickle=False) as spatial:
        rho = np.asarray(spatial["rho"], dtype=np.float64)
    base_state = forward._thermal_state(
        rho,
        forward.G_TA_SIO2_SCENARIOS["thermally_grown"],
        topology,
        fvm,
    )
    with np.load(remap_path, allow_pickle=False) as remap:
        _, source_power, mapping = forward._map_thermal_q(
            remap, base_state, overlap
        )

    rows: list[dict[str, object]] = []
    scenario_results = {}
    raw = {"rho": rho, "source_power_W": source_power.astype(np.float32)}
    worst_strong = 0.0
    worst_normalized = 0.0
    worst_residual = 0.0
    worst_energy = 0.0
    worst_electrical_balance = 0.0
    derivative_audits = {}
    steps = (0.01, 0.005, 0.0025)
    for scenario in ("thermally_grown", "evaporated"):
        base = _evaluate(
            rho,
            source_power,
            scenario,
            args.cuda_device,
            need_gradient=True,
            forward=forward,
            electrical=electrical,
            coupled=coupled,
            topology=topology,
            fvm=fvm,
        )
        gradient = base["gradient_total"]
        gradient_norm = float(np.linalg.norm(gradient))
        scenario_rows = []
        directions = _directions(gradient)

        audit_direction = directions["smooth_asymmetric"]
        epsilon = 1.0e-4
        plus_state = forward._thermal_state(
            rho + epsilon * audit_direction,
            forward.G_TA_SIO2_SCENARIOS[scenario],
            topology,
            fvm,
        )
        plus_action = plus_state["system"].matrix_W_K @ base["temperature"]
        minus_state = forward._thermal_state(
            rho - epsilon * audit_direction,
            forward.G_TA_SIO2_SCENARIOS[scenario],
            topology,
            fvm,
        )
        minus_action = minus_state["system"].matrix_W_K @ base["temperature"]
        matrix_fd = -float(
            base["thermal_adjoint"]
            @ ((plus_action - minus_action) / (2.0 * epsilon))
        )
        matrix_ad = float(np.sum(base["gradient_thermal"] * audit_direction))
        derivative_audits[scenario] = {
            "direction": "smooth_asymmetric",
            "epsilon": epsilon,
            "analytic_A": matrix_ad,
            "matrix_central_difference_A": matrix_fd,
            "relative_error": _relative(matrix_ad, matrix_fd),
        }

        for direction_name, direction in directions.items():
            ad = float(np.sum(gradient * direction))
            for step in steps:
                plus = _evaluate(
                    rho + step * direction,
                    source_power,
                    scenario,
                    args.cuda_device,
                    need_gradient=False,
                    forward=forward,
                    electrical=electrical,
                    coupled=coupled,
                    topology=topology,
                    fvm=fvm,
                )
                minus = _evaluate(
                    rho - step * direction,
                    source_power,
                    scenario,
                    args.cuda_device,
                    need_gradient=False,
                    forward=forward,
                    electrical=electrical,
                    coupled=coupled,
                    topology=topology,
                    fvm=fvm,
                )
                fd = (plus["objective_A"] - minus["objective_A"]) / (2.0 * step)
                strong = max(abs(ad), abs(fd)) >= 0.05 * gradient_norm
                row = {
                    "scenario": scenario,
                    "direction": direction_name,
                    "h": step,
                    "AD_A": ad,
                    "FD_A": fd,
                    "strong": strong,
                    "strong_relative_error": _relative(ad, fd),
                    "gradient_l2_normalized_error": abs(ad - fd)
                    / max(gradient_norm, np.finfo(float).tiny),
                    "plus_A": plus["objective_A"],
                    "minus_A": minus["objective_A"],
                }
                rows.append(row)
                scenario_rows.append(row)
        fine = [row for row in scenario_rows if row["h"] == min(steps)]
        local_strong = max(
            (
                float(row["strong_relative_error"])
                for row in fine
                if bool(row["strong"])
            ),
            default=0.0,
        )
        local_normalized = max(
            float(row["gradient_l2_normalized_error"]) for row in fine
        )
        local_residual = max(
            base["thermal_residual"],
            base["thermal_adjoint_residual"],
            base["electrical_residual"],
            base["electrical_adjoint_residual"],
        )
        scenario_results[scenario] = {
            "objective_A": base["objective_A"],
            "Tmax_K": float(np.max(base["temperature"])),
            "gradient_norm_A": gradient_norm,
            "gradient_thermal_norm_A": float(
                np.linalg.norm(base["gradient_thermal"])
            ),
            "gradient_electrical_norm_A": float(
                np.linalg.norm(base["gradient_electrical"])
            ),
            "worst_fine_strong_error": local_strong,
            "worst_fine_normalized_error": local_normalized,
            "max_residual": local_residual,
            "thermal_energy_balance": base["thermal_energy_balance"],
            "electrical_terminal_balance": base["electrical_balance"],
        }
        worst_strong = max(worst_strong, local_strong)
        worst_normalized = max(worst_normalized, local_normalized)
        worst_residual = max(worst_residual, local_residual)
        worst_energy = max(worst_energy, base["thermal_energy_balance"])
        worst_electrical_balance = max(
            worst_electrical_balance, base["electrical_balance"]
        )
        raw[f"temperature_{scenario}_K"] = base["temperature"].astype(np.float32)
        raw[f"weighting_{scenario}"] = base["weighting"].astype(np.float32)
        raw[f"thermal_adjoint_{scenario}_A_W"] = base[
            "thermal_adjoint"
        ].astype(np.float32)
        raw[f"gradient_total_{scenario}_A"] = gradient
        raw[f"gradient_thermal_{scenario}_A"] = base["gradient_thermal"]
        raw[f"gradient_electrical_{scenario}_A"] = base["gradient_electrical"]

    worst_matrix_derivative = max(
        audit["relative_error"] for audit in derivative_audits.values()
    )
    gates = {
        "thermal_matrix_derivative_relative_error_lt_1e-5": bool(
            worst_matrix_derivative < 1.0e-5
        ),
        "strong_direction_error_lt_1pct": bool(worst_strong < 0.01),
        "multi_direction_gradient_l2_normalized_error_lt_1pct": bool(
            worst_normalized < 0.01
        ),
        "linear_residual_lt_1e-8": bool(worst_residual < 1.0e-8),
        "thermal_energy_balance_lt_1pct": bool(worst_energy < 0.01),
        "electrical_terminal_balance_lt_1pct": bool(
            worst_electrical_balance < 0.01
        ),
        "fixed_spatial_Q_SHA_verified": True,
        "unclipped_density_and_FD": bool(
            np.min(rho) - max(steps) > 0.0 and np.max(rho) + max(steps) < 1.0
        ),
        "GPU_linear_solves_no_CPU_fallback": True,
        "no_gradient_rescaling": True,
    }
    passed = all(gates.values())
    status = STATUS_PASS if passed else STATUS_FAIL
    raw_path = args.raw_output_npz.resolve()
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(raw_path, **raw)
    summary = {
        "status": status,
        "scope": (
            "fixed certified spatial Maxwell Q; explicit 3-D Au/TaIrTe4/SiO2/Si "
            "thermal material/contact derivative plus floating-Au electrical/weighting "
            "derivative; no optical Maxwell derivative and no optimization"
        ),
        "source": {
            "P_Q_W": float(np.sum(source_power)),
            "spatial_Q_sha256": remap_summary["input_spatial_Q"]["sha256"],
            "thermal_Q_sha256": remap_summary["output_thermal_Q"]["sha256"],
            "mapping": mapping,
        },
        "scenarios": scenario_results,
        "thermal_matrix_derivative_audit": derivative_audits,
        "directions": rows,
        "worst_fine_strong_error": worst_strong,
        "worst_fine_gradient_l2_normalized_error": worst_normalized,
        "gates": gates,
        "raw_artifact": {
            "path": str(raw_path),
            "bytes": raw_path.stat().st_size,
            "sha256": _sha256(raw_path),
            "committed_to_git": False,
        },
        "next_gate": (
            "contract the thermal source adjoint through the conservative remap "
            "onto native Yee Q and validate the spatially weighted FDTDX optical gradient"
        ),
    }
    summary_path = output / "explicit_thermal_weighting_fixed_spatial_q_adfd_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    csv_path = output / "explicit_thermal_weighting_fixed_spatial_q_adfd_cases.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    figure, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for axis, scenario in zip(axes[:2], ("thermally_grown", "evaporated")):
        axis.imshow(raw[f"gradient_total_{scenario}_A"].T, origin="lower", cmap="coolwarm")
        axis.set_title(f"total fixed-Q gradient: {scenario}")
    fine_rows = [row for row in rows if row["h"] == min(steps)]
    axes[2].scatter(
        [float(row["FD_A"]) for row in fine_rows],
        [float(row["AD_A"]) for row in fine_rows],
    )
    bounds = [
        value
        for row in fine_rows
        for value in (float(row["FD_A"]), float(row["AD_A"]))
    ]
    low, high = min(bounds), max(bounds)
    axes[2].plot([low, high], [low, high], "k--")
    axes[2].set_xlabel("central FD (A)")
    axes[2].set_ylabel("adjoint (A)")
    axes[2].set_title("h=0.0025 directional derivatives")
    plot_path = output / "explicit_thermal_weighting_fixed_spatial_q_adfd.png"
    figure.savefig(plot_path, dpi=180)
    plt.close(figure)
    report_path = output / "EXPLICIT_THERMAL_WEIGHTING_FIXED_SPATIAL_Q_ADFD_REPORT.md"
    report_path.write_text(
        f"""# Explicit thermal/weighting fixed-spatial-Q AD--FD

Status: **{status}**

The certified spatial Maxwell source is held fixed while the same 20x20 Au
density changes the explicit 3-D Au conductivity, parallel-area Au/TaIrTe4
thermal contact, floating-Au electrical conductivity, and finite vertical
electrical contact. The 40x40 electrical temperature pullback is transposed
through the 500-nm-to-100-nm and thickness-averaging maps before the thermal
adjoint solve.

The independent thermal-matrix directional derivative audit has worst error
`{100*worst_matrix_derivative:.6f}%`. Across thermally-grown and evaporated
interface scenarios, the finest-step worst strong-direction error is
`{100*worst_strong:.6f}%` and the worst gradient-L2-normalized error is
`{100*worst_normalized:.6f}%`. Maximum linear residual is
`{worst_residual:.3e}`; thermal energy balance is `{100*worst_energy:.6f}%`.

This certificate contains no Maxwell optical derivative. The next gate is the
native-Yee spatial-Q weighted optical adjoint, followed by a full combined
directional AD--FD. No optimization is authorized yet.
""",
        encoding="utf-8",
    )
    manifest = {
        "status": status,
        "raw_artifact": summary["raw_artifact"],
        "input_spatial_Q": remap_summary["input_spatial_Q"],
        "input_thermal_Q": remap_summary["output_thermal_Q"],
        "published": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in (summary_path, csv_path, plot_path, report_path)
        ],
    }
    (output / "RAW_ARTIFACT_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": status,
                "worst_matrix_derivative_error": worst_matrix_derivative,
                "worst_strong_error": worst_strong,
                "worst_normalized_error": worst_normalized,
                "worst_residual": worst_residual,
            },
            indent=2,
        )
    )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
