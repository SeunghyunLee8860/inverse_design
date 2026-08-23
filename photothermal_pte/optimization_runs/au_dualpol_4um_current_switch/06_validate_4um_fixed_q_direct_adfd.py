#!/usr/bin/env python3
"""Validate fixed-Q thermal/material and electrical/weighting AD--FD terms.

This stage intentionally freezes the Maxwell heat source.  It validates only
the two direct density paths that remain after Q is fixed:

  rho -> thermal operator/contact -> T -> I
  rho -> electrical operator/contact -> psi -> I

The Maxwell density path is certified separately by the combined stage.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.contract import (
    CONTRACT,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.multiphysics_4um import (
    evaluate_fixed_source,
)


HERE = Path(__file__).resolve().parent
FORWARD = HERE / "results_4um_multiphysics_forward/multiphysics_4um_forward.json"
OUT = HERE / "results_4um_fixed_q_direct_adfd"
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
    random = rng.normal(size=CONTRACT.design_shape)
    # Short solver-free smoothing avoids a direction dominated by one pixel.
    random = (
        random
        + np.roll(random, 1, 0)
        + np.roll(random, -1, 0)
        + np.roll(random, 1, 1)
        + np.roll(random, -1, 1)
    ) / 5.0
    candidates = {
        "adjoint_aligned": np.asarray(gradient, dtype=np.float64),
        "central_localized": np.exp(-(xx**2 + yy**2) / (2.0 * (0.75e-6) ** 2)),
        "asymmetric_smooth": np.sin(0.61e6 * xx + 0.37) * np.cos(0.43e6 * yy - 0.21),
        "fixed_seed_random": random,
    }
    result = {}
    for name, value in candidates.items():
        norm = float(np.max(np.abs(value)))
        if not np.isfinite(norm) or norm == 0.0:
            raise RuntimeError(f"invalid direction {name}")
        result[name] = value / norm
    return result


def main() -> int:
    if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
        raise RuntimeError("GPU-only stage requires CUDA_VISIBLE_DEVICES")
    OUT.mkdir(parents=True, exist_ok=True)
    forward = json.loads(FORWARD.read_text(encoding="utf-8"))
    if forward["status"] != "VALIDATED_4UM_DUALPOL_MULTIPHYSICS_RHO0P5_FORWARD":
        raise RuntimeError("fail-closed: forward checkpoint is not validated")

    all_rows: list[dict[str, object]] = []
    cases: list[dict[str, object]] = []
    gradient_images: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for case in forward["cases"]:
        pol = case["polarization"]
        raw_path = Path(case["raw"]["path"])
        if sha256(raw_path) != case["raw"]["sha256"]:
            raise RuntimeError(f"fail-closed: {pol} multiphysics raw SHA mismatch")
        with np.load(raw_path, allow_pickle=False) as raw:
            rho = np.asarray(raw["rho"], dtype=np.float64)
            source = np.asarray(raw["source_power_W"], dtype=np.float64)

        start = time.perf_counter()
        base = evaluate_fixed_source(rho, source, 0, need_gradient=True)
        gradient = np.asarray(base["gradient_direct_A"], dtype=np.float64)
        gradient_thermal = np.asarray(base["gradient_thermal_A"], dtype=np.float64)
        gradient_electrical = np.asarray(base["gradient_electrical_A"], dtype=np.float64)
        gradient_images[pol] = (gradient, gradient_thermal, gradient_electrical)
        rows = []
        for direction_name, direction in directions(gradient).items():
            ad = float(np.vdot(gradient, direction))
            for step in STEPS:
                if np.min(rho - step * direction) <= 0.0 or np.max(rho + step * direction) >= 1.0:
                    raise RuntimeError("AD-FD perturbation would clip a density")
                plus = float(
                    evaluate_fixed_source(
                        rho + step * direction, source, 0, need_gradient=False
                    )["objective_A"]
                )
                minus = float(
                    evaluate_fixed_source(
                        rho - step * direction, source, 0, need_gradient=False
                    )["objective_A"]
                )
                fd = (plus - minus) / (2.0 * step)
                row = {
                    "polarization": pol,
                    "direction": direction_name,
                    "step": step,
                    "AD_A": ad,
                    "FD_A": fd,
                    "absolute_error_A": abs(ad - fd),
                    "relative_error": abs(ad - fd) / max(abs(fd), 1e-30),
                    "plus_A": plus,
                    "minus_A": minus,
                }
                rows.append(row)
                all_rows.append(row)
        strong = [row for row in rows if row["direction"] == "adjoint_aligned"]
        scale = max(abs(float(row["FD_A"])) for row in rows)
        normalized_error = max(float(row["absolute_error_A"]) for row in rows) / max(
            scale, 1e-30
        )
        gates = {
            "strong_direction_lt_1pct": max(float(row["relative_error"]) for row in strong) < 0.01,
            "multi_direction_normalized_lt_1pct": normalized_error < 0.01,
            "thermal_adjoint_residual_lt_1e-8": float(base["thermal_adjoint_audit"]["relative_residual"]) < 1e-8,
            "electrical_adjoint_residual_lt_1e-8": float(base["electrical_adjoint_audit"]["relative_residual"]) < 1e-8,
            "no_clipping": True,
            "finite": bool(
                np.all(np.isfinite(gradient))
                and np.all(np.isfinite(gradient_thermal))
                and np.all(np.isfinite(gradient_electrical))
            ),
        }
        cases.append(
            {
                "polarization": pol,
                "status": "VALIDATED_4UM_FIXED_Q_DIRECT_ADFD" if all(gates.values()) else "FAILED_4UM_FIXED_Q_DIRECT_ADFD",
                "objective_A": float(base["objective_A"]),
                "gradient_norm_A": float(np.linalg.norm(gradient)),
                "thermal_gradient_norm_A": float(np.linalg.norm(gradient_thermal)),
                "electrical_gradient_norm_A": float(np.linalg.norm(gradient_electrical)),
                "multi_direction_normalized_error": normalized_error,
                "runtime_s": time.perf_counter() - start,
                "gates": gates,
                "rows": rows,
            }
        )
        print(
            f"[{pol}] I={base['objective_A']*1e9:.9f} nA, "
            f"max normalized error={normalized_error:.3e}",
            flush=True,
        )

    status = (
        "VALIDATED_4UM_DUALPOL_FIXED_Q_DIRECT_ADFD"
        if all(case["status"].startswith("VALIDATED_") for case in cases)
        else "FAILED_4UM_DUALPOL_FIXED_Q_DIRECT_ADFD"
    )
    summary = {
        "status": status,
        "scope": "fixed optical Q only: Au thermal/contact and electrical/weighting-field density derivatives",
        "explicitly_excluded": "Maxwell density derivative; certified in the next combined stage",
        "steps": STEPS,
        "no_clipping_rescaling": True,
        "cases": cases,
    }
    (OUT / "fixed_q_direct_adfd.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    with (OUT / "fixed_q_direct_adfd.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    for row, pol in enumerate(("Ea", "Eb")):
        for col, (data, title) in enumerate(
            zip(
                gradient_images[pol],
                ("total direct", "thermal/contact", "electrical/weighting"),
                strict=True,
            )
        ):
            vmax = float(np.max(np.abs(data)))
            image = axes[row, col].imshow(
                data.T * 1e12,
                origin="lower",
                extent=(-4, 4, -4, 4),
                cmap="coolwarm",
                vmin=-vmax * 1e12,
                vmax=vmax * 1e12,
            )
            axes[row, col].set(
                title=f"{pol}: {title} gradient (pA/rho)",
                xlabel="x=b (um)",
                ylabel="y=a (um)",
                aspect="equal",
            )
            fig.colorbar(image, ax=axes[row, col], shrink=0.8)
    fig.savefig(OUT / "FIXED_Q_DIRECT_GRADIENT_COMPONENTS.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, pol in zip(axes, ("Ea", "Eb"), strict=True):
        rows = [row for row in all_rows if row["polarization"] == pol]
        for direction_name in sorted({str(row["direction"]) for row in rows}):
            subset = [row for row in rows if row["direction"] == direction_name]
            ax.plot(
                [float(row["FD_A"]) * 1e12 for row in subset],
                [float(row["AD_A"]) * 1e12 for row in subset],
                "o-",
                label=direction_name,
            )
        values = np.asarray(
            [(float(row["FD_A"]), float(row["AD_A"])) for row in rows]
        ) * 1e12
        lower = float(np.min(values))
        upper = float(np.max(values))
        ax.plot((lower, upper), (lower, upper), "k--", label="ideal AD=FD")
        ax.set(
            title=pol,
            xlabel="central FD directional derivative (pA)",
            ylabel="adjoint directional derivative (pA)",
        )
        ax.legend(fontsize=7)
        ax.grid(alpha=0.25)
    fig.savefig(OUT / "FIXED_Q_DIRECT_ADFD_SCATTER.png", dpi=180)
    plt.close(fig)

    lines = [
        "# Fixed-Q direct AD-FD validation",
        "",
        f"Status: **{status}**",
        "",
        "This certificate freezes Maxwell Q. It validates Au thermal/contact and electrical/weighting-field derivatives only.",
        "No clipping, empirical normalization, or gradient rescaling is used.",
        "",
        "| pol | I (nA) | ||g direct|| (A) | ||g thermal|| (A) | ||g electrical|| (A) | normalized error |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for case in cases:
        lines.append(
            f"| {case['polarization']} | {case['objective_A']*1e9:.9f} | "
            f"{case['gradient_norm_A']:.6e} | {case['thermal_gradient_norm_A']:.6e} | "
            f"{case['electrical_gradient_norm_A']:.6e} | {case['multi_direction_normalized_error']:.3e} |"
        )
    (OUT / "FIXED_Q_DIRECT_ADFD_REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(status, flush=True)
    return 0 if status.startswith("VALIDATED_") else 2


if __name__ == "__main__":
    raise SystemExit(main())
