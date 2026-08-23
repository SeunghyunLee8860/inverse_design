#!/usr/bin/env python3
"""Certify the full physical-density PTE gradient for both polarizations."""

from __future__ import annotations

import csv
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.combined_4um import (
    CompiledOpticalRunner,
    combined_gradient,
    evaluate_forward_multiphysics,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.contract import (
    CONTRACT,
)


HERE = Path(__file__).resolve().parent
OUT = HERE / "results_4um_combined_adfd"
RAW = Path("/home/seunghyun/tairte4/raw/au_dualpol_4um_current_switch/combined_adfd")
CALIBRATION = HERE / "results_fdtdx_4um_source_calibration/fdtdx_4um_source_calibration.json"
STEPS = (0.01, 0.005, 0.0025)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directions(gradient: np.ndarray) -> dict[str, np.ndarray]:
    x = (np.arange(CONTRACT.design_shape[0]) + 0.5) * CONTRACT.design_pitch_m - 4e-6
    y = (np.arange(CONTRACT.design_shape[1]) + 0.5) * CONTRACT.design_pitch_m - 4e-6
    xx, yy = np.meshgrid(x, y, indexing="ij")
    rng = np.random.default_rng(20260823)
    random = rng.standard_normal(CONTRACT.design_shape)
    random = (
        random
        + np.roll(random, 1, 0)
        + np.roll(random, -1, 0)
        + np.roll(random, 1, 1)
        + np.roll(random, -1, 1)
    ) / 5.0
    candidates = {
        "adjoint_aligned": gradient,
        "central_localized": np.exp(-(xx**2 + yy**2) / (2.0 * (0.75e-6) ** 2)),
        "asymmetric_smooth": np.sin(0.61e6 * xx + 0.37) * np.cos(0.43e6 * yy - 0.21),
        "fixed_seed_random": random,
    }
    result = {}
    for name, value in candidates.items():
        maximum = float(np.max(np.abs(value)))
        if not maximum > 0.0 or not np.isfinite(maximum):
            raise RuntimeError(f"invalid direction {name}")
        result[name] = np.asarray(value / maximum, dtype=np.float64)
    return result


def directional_angle(rows: list[dict[str, object]], step: float = 0.005) -> float:
    selected = [row for row in rows if float(row["step"]) == step]
    ad = np.asarray([float(row["AD_A"]) for row in selected])
    fd = np.asarray([float(row["FD_A"]) for row in selected])
    cosine = float(np.dot(ad, fd) / max(np.linalg.norm(ad) * np.linalg.norm(fd), 1e-30))
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def run_case(pol: str, scale: float) -> dict[str, object]:
    rho = np.full(CONTRACT.design_shape, 0.5, dtype=np.float64)
    print(f"[{pol}] compiling checkpoint-free forward and adjoint", flush=True)
    runner = CompiledOpticalRunner.create(pol, rho)
    start = time.perf_counter()
    base = combined_gradient(runner, rho, scale, 0)
    gradient = np.asarray(base["gradient_total_A"], dtype=np.float64)
    rows: list[dict[str, object]] = []
    for direction_name, direction in directions(gradient).items():
        ad = float(np.vdot(gradient, direction))
        print(f"[{pol}] direction={direction_name} AD={ad:.6e} A", flush=True)
        for step in STEPS:
            if np.min(rho - step * direction) <= 0.0 or np.max(rho + step * direction) >= 1.0:
                raise RuntimeError("density perturbation would clip")
            plus = evaluate_forward_multiphysics(
                runner, rho + step * direction, scale, 0, need_gradient=False
            )
            minus = evaluate_forward_multiphysics(
                runner, rho - step * direction, scale, 0, need_gradient=False
            )
            fd = (float(plus["objective_A"]) - float(minus["objective_A"])) / (
                2.0 * step
            )
            row = {
                "polarization": pol,
                "direction": direction_name,
                "step": step,
                "AD_A": ad,
                "FD_A": fd,
                "absolute_error_A": abs(ad - fd),
                "relative_error": abs(ad - fd) / max(abs(fd), 1e-30),
                "plus_A": float(plus["objective_A"]),
                "minus_A": float(minus["objective_A"]),
            }
            rows.append(row)
            print(
                f"[{pol}] h={step:.4g} FD={fd:.6e} rel={row['relative_error']:.3e}",
                flush=True,
            )

    output = base["optical_output"]
    eta0 = float(runner.model["fdtdx"].constants.eta0)
    p_closed = scale * eta0 * float(
        np.mean(
            np.asarray(output.detector_states["material_flux_td"]["poynting_flux"])[
                :, 0
            ]
        )
    )
    p_q = float(np.sum(base["source_power_W"]))
    closure = abs(p_q - p_closed) / max(abs(p_closed), np.finfo(float).tiny)
    strong_rows = [row for row in rows if row["direction"] == "adjoint_aligned"]
    scale_directional = max(abs(float(row["FD_A"])) for row in rows)
    normalized_error = max(float(row["absolute_error_A"]) for row in rows) / max(
        scale_directional, 1e-30
    )
    angle = directional_angle(rows)
    q_finite_nonnegative = all(
        np.all(np.isfinite(value)) and np.min(value) >= 0.0
        for value in base["q_fields_W_m3"].values()
    )
    gates = {
        "strong_direction_lt_1pct": max(float(row["relative_error"]) for row in strong_rows) < 0.01,
        "multi_direction_normalized_lt_1pct": normalized_error < 0.01,
        "directional_angle_lt_1deg": angle < 1.0,
        "mapping_transpose_lt_1e-12": float(base["weighted_contraction_relative_error"]) < 1e-12,
        "optical_closure_lt_0p5pct": closure < 0.005,
        "thermal_energy_balance_lt_1pct": float(base["thermal_audit"]["energy_balance_relative"]) < 0.01,
        "thermal_residual_lt_1e-8": float(base["thermal_audit"]["relative_residual"]) < 1e-8,
        "thermal_adjoint_residual_lt_1e-8": float(base["thermal_adjoint_audit"]["relative_residual"]) < 1e-8,
        "electrical_residual_lt_1e-8": float(base["electrical_audit"]["relative_residual"]) < 1e-8,
        "electrical_adjoint_residual_lt_1e-8": float(base["electrical_adjoint_audit"]["relative_residual"]) < 1e-8,
        "finite_nonnegative_Q": bool(q_finite_nonnegative),
        "no_clipping": True,
    }
    RAW.mkdir(parents=True, exist_ok=True)
    raw = RAW / f"combined_4um_rho0p5_{pol}.npz"
    np.savez_compressed(
        raw,
        rho=rho,
        gradient_total_A=gradient,
        gradient_optical_A=np.asarray(base["gradient_optical_A"]),
        gradient_optical_field_A=np.asarray(base["gradient_optical_field_A"]),
        gradient_optical_direct_loss_A=np.asarray(base["gradient_optical_direct_loss_A"]),
        gradient_thermal_A=np.asarray(base["gradient_thermal_A"]),
        gradient_electrical_A=np.asarray(base["gradient_electrical_A"]),
        source_power_W=np.asarray(base["source_power_W"]),
        temperature_K=np.asarray(base["temperature"]),
        tairte4_temperature_K=np.asarray(base["ta_temperature"]),
    )
    result = {
        "polarization": pol,
        "status": "VALIDATED_4UM_COMBINED_PHYSICAL_RHO_ADFD" if all(gates.values()) else "FAILED_4UM_COMBINED_PHYSICAL_RHO_ADFD",
        "objective_A": float(base["objective_A"]),
        "objective_nA": float(base["objective_A"]) * 1e9,
        "P_Q_W": p_q,
        "P_six_W": p_closed,
        "closure_relative": closure,
        "gradient_norms_A": {
            "total": float(np.linalg.norm(gradient)),
            "optical_total": float(np.linalg.norm(base["gradient_optical_A"])),
            "optical_field": float(np.linalg.norm(base["gradient_optical_field_A"])),
            "optical_direct_loss": float(np.linalg.norm(base["gradient_optical_direct_loss_A"])),
            "thermal_contact": float(np.linalg.norm(base["gradient_thermal_A"])),
            "electrical_weighting": float(np.linalg.norm(base["gradient_electrical_A"])),
        },
        "multi_direction_normalized_error": normalized_error,
        "directional_angle_deg": angle,
        "mapping_transpose_relative_error": float(base["weighted_contraction_relative_error"]),
        "compile_forward_s": runner.compile_forward_s,
        "compile_adjoint_s": runner.compile_adjoint_s,
        "baseline_forward_s": float(base["forward_s"]),
        "baseline_adjoint_s": float(base["adjoint_s"]),
        "case_runtime_s": time.perf_counter() - start,
        "gates": gates,
        "rows": rows,
        "raw": {"path": str(raw), "bytes": raw.stat().st_size, "sha256": sha256(raw)},
    }
    del base, runner
    gc.collect()
    return result


def main() -> int:
    if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
        raise RuntimeError("GPU-only combined gate requires CUDA_VISIBLE_DEVICES")
    OUT.mkdir(parents=True, exist_ok=True)
    calibration = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    if calibration["status"] != "VALIDATED_FDTDX_4UM_SOURCE_POWER_CALIBRATION":
        raise RuntimeError("fail-closed: source calibration not validated")
    scale = CONTRACT.reporting_incident_power_W / float(
        calibration["common_reference_incident_power_W"]
    )
    cases = [run_case(pol, scale) for pol in ("Ea", "Eb")]
    status = (
        "VALIDATED_4UM_DUALPOL_COMBINED_PHYSICAL_RHO_ADFD"
        if all(case["status"].startswith("VALIDATED_") for case in cases)
        else "FAILED_4UM_DUALPOL_COMBINED_PHYSICAL_RHO_ADFD"
    )
    summary = {
        "status": status,
        "scope": "rho=0.5 full Maxwell redistribution + direct loss + thermal/contact + electrical/weighting-field physical-density gradient",
        "source_power_scale": scale,
        "steps": STEPS,
        "no_clipping_smoothing_gain_rescaling": True,
        "cases": cases,
    }
    (OUT / "combined_4um_adfd.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    rows = [row for case in cases for row in case["rows"]]
    with (OUT / "combined_4um_adfd.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, case in zip(axes, cases, strict=True):
        for direction_name in sorted({str(row["direction"]) for row in case["rows"]}):
            subset = [row for row in case["rows"] if row["direction"] == direction_name]
            ax.plot(
                [float(row["FD_A"]) * 1e9 for row in subset],
                [float(row["AD_A"]) * 1e9 for row in subset],
                "o-",
                label=direction_name,
            )
        values = np.asarray(
            [(float(row["FD_A"]), float(row["AD_A"])) for row in case["rows"]]
        ) * 1e9
        lo, hi = float(np.min(values)), float(np.max(values))
        ax.plot((lo, hi), (lo, hi), "k--", label="ideal AD=FD")
        ax.set(
            title=f"{case['polarization']}: angle={case['directional_angle_deg']:.3f} deg",
            xlabel="central FD directional derivative (nA)",
            ylabel="combined adjoint directional derivative (nA)",
        )
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)
    fig.savefig(OUT / "COMBINED_4UM_ADFD_SCATTER.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 5, figsize=(19, 8), constrained_layout=True)
    names = (
        "gradient_total_A",
        "gradient_optical_field_A",
        "gradient_optical_direct_loss_A",
        "gradient_thermal_A",
        "gradient_electrical_A",
    )
    titles = ("total", "Maxwell field", "direct Au loss", "thermal/contact", "electrical/weighting")
    for row, case in enumerate(cases):
        with np.load(case["raw"]["path"], allow_pickle=False) as raw:
            arrays = [np.asarray(raw[name]) for name in names]
        for col, (array, title) in enumerate(zip(arrays, titles, strict=True)):
            vmax = float(np.max(np.abs(array)))
            image = axes[row, col].imshow(
                array.T * 1e12,
                origin="lower",
                extent=(-4, 4, -4, 4),
                cmap="coolwarm",
                vmin=-vmax * 1e12,
                vmax=vmax * 1e12,
            )
            axes[row, col].set(
                title=f"{case['polarization']}: {title} (pA/rho)",
                xlabel="x=b (um)",
                ylabel="y=a (um)",
                aspect="equal",
            )
            fig.colorbar(image, ax=axes[row, col], shrink=0.75)
    fig.savefig(OUT / "COMBINED_4UM_GRADIENT_COMPONENTS.png", dpi=180)
    plt.close(fig)

    lines = [
        "# Combined 4 um dual-polarization physical-density AD-FD",
        "",
        f"Status: **{status}**",
        "",
        "The gradient includes Maxwell field redistribution, direct Au loss, thermal/contact, and electrical/weighting-field paths.",
        "No time-history checkpointing, clipping, smoothing, empirical gradient normalization, or rescaling is used.",
        "",
        "| pol | I (nA) | P_Q (W) | closure | ||g|| (A) | normalized error | angle (deg) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for case in cases:
        lines.append(
            f"| {case['polarization']} | {case['objective_nA']:.9f} | {case['P_Q_W']:.8e} | "
            f"{case['closure_relative']:.3e} | {case['gradient_norms_A']['total']:.6e} | "
            f"{case['multi_direction_normalized_error']:.3e} | {case['directional_angle_deg']:.4f} |"
        )
    (OUT / "COMBINED_4UM_ADFD_REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    manifest = {
        "status": status,
        "raw_artifacts_committed": False,
        "artifacts": [case["raw"] for case in cases],
        "generation_command": "CUDA_VISIBLE_DEVICES=<gpu> ./run_combined_gpu_python.sh 07_validate_4um_combined_adfd.py",
    }
    (OUT / "RAW_ARTIFACT_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(status, flush=True)
    return 0 if status.startswith("VALIDATED_") else 2


if __name__ == "__main__":
    raise SystemExit(main())
