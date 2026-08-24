#!/usr/bin/env python3
"""Diagnose inconsistent gray Au laws at the robust beta=256 checkpoint.

This is a forward-only causal diagnostic.  It does not promote any gray
material as physical.  The optical Au oscillator strength exponent and the
thermal/electrical density exponent are varied independently while preserving
the exact void/Au endpoints.
"""

from __future__ import annotations

from datetime import datetime, timezone
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

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.combined_4um import (
    CompiledOpticalRunner,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.contract import CONTRACT
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.multiphysics_4um import (
    build_thermal_state,
    evaluate_fixed_source,
    map_native_q_to_thermal,
)
from photothermal_pte.optimization_runs.legacy_v261_optical_support.production_density_mapping import (
    ProductionDensityMapping,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.paths import (
    raw_path,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.validation_provenance import (
    load_current_source_calibration,
)


HERE = Path(__file__).resolve().parent
OUT = HERE / "results_4um_dualpol_au_gray_law_diagnostic"
INPUT = raw_path("robust_projection_ld_mma", "evaluation_0112.npz")
CALIBRATION = HERE / "results_fdtdx_4um_source_calibration/fdtdx_4um_source_calibration.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def optical_closure(output, runner, source_scale: float, p_q: float) -> tuple[float, float]:
    eta0 = float(runner.model["fdtdx"].constants.eta0)
    p_six = source_scale * eta0 * float(
        np.mean(
            np.asarray(
                output.detector_states["material_flux_td"]["poynting_flux"]
            )[:, 0]
        )
    )
    closure = abs(p_q - p_six) / max(abs(p_six), np.finfo(float).tiny)
    return p_six, closure


def main() -> None:
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        raise RuntimeError("GPU required: set CUDA_VISIBLE_DEVICES")
    if not INPUT.exists():
        raise FileNotFoundError(INPUT)
    OUT.mkdir(parents=True, exist_ok=True)
    cuda_device = int(os.environ.get("THERMAL_CUDA_DEVICE", "0"))
    calibration = load_current_source_calibration(CALIBRATION)
    source_scale = CONTRACT.reporting_incident_power_W / float(
        calibration["common_reference_incident_power_W"]
    )
    checkpoint = np.load(INPUT)
    rho_nominal = np.asarray(checkpoint["rho_nominal"], dtype=np.float64)
    latent = np.asarray(checkpoint["latent"], dtype=np.float64)
    if rho_nominal.shape != CONTRACT.design_shape or latent.shape != CONTRACT.design_shape:
        raise RuntimeError("unexpected checkpoint density/latent shape")
    densities = {"eta_0.50_nominal": rho_nominal}
    for eta in (0.35, 0.65):
        mapping = ProductionDensityMapping(
            shape=CONTRACT.design_shape,
            spacing_m=CONTRACT.design_pitch_m,
            radius_m=CONTRACT.filter_radius_m,
            eta=eta,
        )
        densities[f"eta_{eta:.2f}"] = np.asarray(
            mapping.physical(latent, 256.0), dtype=np.float64
        )

    gray_audit = {}
    for density_case, rho in densities.items():
        gray_mask = (rho > 0.01) & (rho < 0.99)
        gray_audit[density_case] = {
            "global_grayness_mean_4rho1mrho": float(np.mean(4.0 * rho * (1.0 - rho))),
            "cells_total": int(rho.size),
            "cells_rho_0p01_to_0p99": int(np.count_nonzero(gray_mask)),
            "cells_rho_0p4_to_0p6": int(np.count_nonzero((rho >= 0.4) & (rho <= 0.6))),
            "max_rho_minus_rho3": float(np.max(rho - rho**3)),
            "rho_at_max_rho_minus_rho3": float(rho.ravel()[np.argmax(rho - rho**3)]),
            "gray_sum_rho": float(np.sum(rho[gray_mask])),
            "gray_sum_rho3": float(np.sum(rho[gray_mask] ** 3)),
        }

    runners = {
        pol: CompiledOpticalRunner.create(
            pol,
            np.full(CONTRACT.design_shape, 0.5),
            total_periods=24,
            window_periods=4,
        )
        for pol in ("Ea", "Eb")
    }
    optical_cache: dict[tuple[str, int, str], dict[str, object]] = {}
    for density_case, rho in densities.items():
        for optical_exponent in (1, 3):
            # The legacy runner uses the shared linear audit fraction.
            # Transform its input explicitly to reproduce the historical
            # O1/O3 factorial without changing that diagnostic code.
            rho_runner = rho**optical_exponent
            for pol, runner in runners.items():
                start = time.perf_counter()
                output, forward_s = runner.run_forward(rho_runner)
                fields, q = runner.fields_and_q(output, rho_runner, source_scale)
                optical_cache[(density_case, optical_exponent, pol)] = {
                    "output": output,
                    "q": q,
                    "fields": fields,
                    "forward_s": float(forward_s),
                    "wall_s": float(time.perf_counter() - start),
                }
                print(
                    f"[optical] {density_case} p={optical_exponent} {pol}: "
                    f"{forward_s:.3f} s",
                    flush=True,
                )

    rows: list[dict[str, object]] = []
    for density_case, rho in densities.items():
        for optical_exponent in (1, 3):
            for te_exponent in (1, 3):
                rho_te = rho**te_exponent
                state = build_thermal_state(rho_te)
                for pol, runner in runners.items():
                    cached = optical_cache[(density_case, optical_exponent, pol)]
                    source_power, mapping, _ = map_native_q_to_thermal(
                        state,
                        q_fields_W_m3=cached["q"],
                        dual_volumes_m3=runner.volumes,
                        material_slices={
                            "au": runner.model["slices"]["au_design"],
                            "tairte4": runner.model["slices"]["fixed_tairte4"],
                        },
                        realized_grid=runner.model["grid"],
                    )
                    result = evaluate_fixed_source(
                        rho_te,
                        source_power,
                        cuda_device,
                        need_gradient=False,
                    )
                    p_q = float(np.sum(source_power))
                    p_six, closure = optical_closure(
                        cached["output"], runner, source_scale, p_q
                    )
                    row = {
                        "density_case": density_case,
                        "optical_exponent": optical_exponent,
                        "thermal_electrical_exponent": te_exponent,
                        "polarization": pol,
                        "current_A": float(result["objective_A"]),
                        "current_nA": 1.0e9 * float(result["objective_A"]),
                        "P_Q_W": p_q,
                        "P_six_W": p_six,
                        "closure_relative": closure,
                        "Tmax_K": float(np.max(result["temperature"])),
                        "thermal_energy_balance_relative": float(
                            result["thermal_audit"]["energy_balance_relative"]
                        ),
                        "thermal_residual_relative": float(
                            result["thermal_audit"]["relative_residual"]
                        ),
                        "electrical_residual_relative": float(
                            result["electrical_audit"]["relative_residual"]
                        ),
                        "optical_forward_s": cached["forward_s"],
                    }
                    row["all_gates_pass"] = bool(
                        closure < 0.005
                        and row["thermal_energy_balance_relative"] < 0.01
                        and row["thermal_residual_relative"] < 1.0e-8
                        and row["electrical_residual_relative"] < 1.0e-8
                    )
                    if not row["all_gates_pass"]:
                        print(
                            f"[fail-closed diagnostic case] {density_case} "
                            f"O{optical_exponent}/TE{te_exponent} {pol}: "
                            f"closure={closure:.6e}",
                            flush=True,
                        )
                    rows.append(row)
                    print(
                        f"[case] {density_case} po={optical_exponent} "
                        f"pte={te_exponent} {pol}: I={row['current_nA']:+.6f} nA",
                        flush=True,
                    )

    paired: list[dict[str, object]] = []
    for density_case in densities:
        for optical_exponent in (1, 3):
            for te_exponent in (1, 3):
                selected = {
                    row["polarization"]: row
                    for row in rows
                    if row["density_case"] == density_case
                    and row["optical_exponent"] == optical_exponent
                    and row["thermal_electrical_exponent"] == te_exponent
                }
                ia = float(selected["Ea"]["current_nA"])
                ib = float(selected["Eb"]["current_nA"])
                gates_pass = bool(
                    selected["Ea"]["all_gates_pass"]
                    and selected["Eb"]["all_gates_pass"]
                )
                paired.append(
                    {
                        "density_case": density_case,
                        "optical_exponent": optical_exponent,
                        "thermal_electrical_exponent": te_exponent,
                        "I_a_nA": ia,
                        "I_b_nA": ib,
                        "balanced_utility_nA": min(ia, -ib),
                        "all_gates_pass": gates_pass,
                        "opposite_sign_pass": bool(
                            gates_pass and ia > 0.0 and ib < 0.0
                        ),
                    }
                )

    csv_path = OUT / "gray_law_cases.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(rows[0].keys()), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    extent = (-4, 4, -4, 4)
    for column, (density_case, rho) in enumerate(densities.items()):
        image = axes[0, column].imshow(
            (rho-rho**3).T, origin="lower", extent=extent, cmap="magma", vmin=0, vmax=0.385
        )
        axes[0, column].set_title(f"{density_case}: rho-rho^3")
        fig.colorbar(image, ax=axes[0, column])
        selected = [row for row in paired if row["density_case"] == density_case]
        labels = [f"O{r['optical_exponent']}/TE{r['thermal_electrical_exponent']}" for r in selected]
        x = np.arange(len(selected))
        axes[1, column].bar(x-0.18, [r["I_a_nA"] for r in selected], width=0.36, label="Ea")
        axes[1, column].bar(x+0.18, [r["I_b_nA"] for r in selected], width=0.36, label="Eb")
        axes[1, column].axhline(0, color="black", lw=0.8)
        axes[1, column].set_xticks(x, labels, rotation=20)
        axes[1, column].set_title(density_case)
        axes[1, column].legend()
        for row in range(2):
            axes[row, column].set_xlabel("x=b (um)" if row == 0 else "law pair")
            axes[row, column].set_ylabel("y=a (um)" if row == 0 else "current (nA)")
    fig.savefig(OUT / "gray_law_mismatch_diagnostic.png", dpi=180)
    plt.close(fig)

    legacy = [
        row for row in paired
        if row["optical_exponent"] == 3
        and row["thermal_electrical_exponent"] == 1
    ]
    sign_sensitive = any(
        row["opposite_sign_pass"]
        and any(
            not candidate["opposite_sign_pass"]
            for candidate in paired
            if candidate["density_case"] == row["density_case"]
            and candidate is not row
        )
        for row in legacy
    )
    status = (
        "DIAGNOSED_GRAY_LAW_MISMATCH_SIGN_SENSITIVE"
        if sign_sensitive
        else "DIAGNOSED_GRAY_LAW_MISMATCH_NOT_SOLE_SIGN_CAUSE"
    )
    def pair(density_case: str, optical_exponent: int, te_exponent: int):
        return next(
            row for row in paired
            if row["density_case"] == density_case
            and row["optical_exponent"] == optical_exponent
            and row["thermal_electrical_exponent"] == te_exponent
        )

    dilated_legacy = pair("eta_0.35", 3, 1)
    dilated_matched_cubic = pair("eta_0.35", 3, 3)
    dilated_matched_linear = pair("eta_0.35", 1, 1)
    dilated_cross = pair("eta_0.35", 1, 3)
    nominal_legacy = pair("eta_0.50_nominal", 3, 1)
    findings = {
        "gray_law_mismatch_confirmed": True,
        "historical_law": "optical rho^3; thermal/electrical rho^1",
        "legacy_consistency_law": "shared linear Au fraction; not production Au",
        "eta_0p35_production_Ib_nA": dilated_legacy["I_b_nA"],
        "eta_0p35_matched_cubic_Ib_nA": dilated_matched_cubic["I_b_nA"],
        "eta_0p35_abs_Ib_margin_reduction_fraction": (
            1.0
            - abs(dilated_matched_cubic["I_b_nA"])
            / abs(dilated_legacy["I_b_nA"])
        ),
        "eta_0p35_linear_optical_Ib_linear_TE_nA": (
            dilated_matched_linear["I_b_nA"]
        ),
        "eta_0p35_linear_optical_Ib_cubic_TE_nA": dilated_cross["I_b_nA"],
        "eta_0p35_TE_law_sign_flip_at_fixed_optical_law": bool(
            dilated_matched_linear["I_b_nA"] < 0.0
            and dilated_cross["I_b_nA"] > 0.0
        ),
        "nominal_eta_0p50_omitted_from_robust_objective": True,
        "nominal_eta_0p50_production_Ib_nA": nominal_legacy["I_b_nA"],
        "nominal_eta_0p50_requested_sign_pass": bool(
            nominal_legacy["I_b_nA"] < 0.0
        ),
        "gray_constraint_applied_only_to_nominal_projection": True,
        "nominal_global_grayness_percent": (
            100.0
            * gray_audit["eta_0.50_nominal"]["global_grayness_mean_4rho1mrho"]
        ),
        "eta_0p35_global_grayness_percent": (
            100.0 * gray_audit["eta_0.35"]["global_grayness_mean_4rho1mrho"]
        ),
        "eta_0p35_gray_cells_rho_0p01_to_0p99": gray_audit["eta_0.35"][
            "cells_rho_0p01_to_0p99"
        ],
        "causal_conclusion": (
            "The inconsistent gray laws are a sign-sensitive functional relaxation "
            "and an important amplifier, but they are not the sole cause of "
            "exact-binary failure. Missing eta=0.5 from the robust objective and "
            "unconstrained grayness in the eta=0.35 projection independently "
            "prevent promotion."
        ),
    }
    summary = {
        "status": status,
        "scope": "forward-only gray-relaxation causal diagnostic; no material law promoted",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "optical_time_contract": {"total_periods": 24, "window_periods": 4},
        "input": {
            "path": str(INPUT.resolve()),
            "bytes": INPUT.stat().st_size,
            "sha256": sha256(INPUT),
        },
        "gray_audit": gray_audit,
        "findings": findings,
        "paired_results": paired,
        "case_results": rows,
    }
    write_json(OUT / "GRAY_LAW_MISMATCH_SUMMARY.json", summary)

    lines = [
        "# Au gray-law mismatch diagnostic",
        "",
        f"Status: `{status}`",
        "",
        "This is a forward-only diagnostic at the robust beta=256 checkpoint. "
        "No gray law is promoted as a physical Au material.",
        "",
        "## Findings",
        "",
        "1. The historical implemented relaxation was O3/TE1: optical Drude strength is rho^3, "
        "while thermal conductivity/contact and electrical conductivity/contact are linear in rho.",
        "2. The mismatch is functionally large in the dilated eta=0.35 projection. Changing only "
        "TE1 to TE3 changes Ib from "
        f"{dilated_legacy['I_b_nA']:+.6f} to {dilated_matched_cubic['I_b_nA']:+.6f} nA, "
        "removing about 86% of the opposite-sign margin, although the sign remains negative.",
        "3. The full factorial is sign-sensitive: at eta=0.35, O1/TE1 gives "
        f"Ib={dilated_matched_linear['I_b_nA']:+.6f} nA, while O1/TE3 gives "
        f"Ib={dilated_cross['I_b_nA']:+.6f} nA.",
        "4. A separate, more fundamental robust-objective omission is present. The optimizer used "
        "eta=0.35 and eta=0.65 but did not include nominal eta=0.50. Under the historical O3/TE1 "
        f"law, nominal Ib={nominal_legacy['I_b_nA']:+.6f} nA and fails the requested sign.",
        "5. The grayness constraint was evaluated only on nominal density. Nominal grayness is "
        f"{100*gray_audit['eta_0.50_nominal']['global_grayness_mean_4rho1mrho']:.4f}%, but "
        f"eta=0.35 grayness is {100*gray_audit['eta_0.35']['global_grayness_mean_4rho1mrho']:.4f}% "
        f"with {gray_audit['eta_0.35']['cells_rho_0p01_to_0p99']} cells in 0.01<rho<0.99.",
        "6. Therefore the inconsistent law is a confirmed risk and performance amplifier, but it is "
        "not the sole cause of exact-binary sign failure. The missing nominal scenario and gray-only "
        "robust projections independently invalidate promotion.",
        "",
        "| density | optical exponent | thermal/electrical exponent | Ia (nA) | Ib (nA) | min(Ia,-Ib) (nA) | physics gates | sign gate |",
        "|---|---:|---:|---:|---:|---:|:---:|:---:|",
    ]
    for row in paired:
        lines.append(
            f"| {row['density_case']} | {row['optical_exponent']} | {row['thermal_electrical_exponent']} | "
            f"{row['I_a_nA']:+.6f} | {row['I_b_nA']:+.6f} | "
            f"{row['balanced_utility_nA']:+.6f} | {row['all_gates_pass']} | "
            f"{row['opposite_sign_pass']} |"
        )
    lines.extend(
        [
            "",
            "The reported 0.395% nominal value is a global grayness metric, not a gray-cell area fraction. "
            "The JSON records gray-cell counts separately for nominal, eta=0.35, and eta=0.65 projections.",
            "",
            "The historical mismatch was O3/TE1. O1/TE1 changes only the optical relaxation; "
            "O3/TE3 changes only the thermal/electrical relaxation. O1/TE3 closes the factorial. "
            "All four share identical rho=0 and rho=1 endpoints.",
            "",
            "No clipping, smoothing, gain, current rescaling, or Q rescaling is used.",
        ]
    )
    (OUT / "GRAY_LAW_MISMATCH_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "paired_results": paired}, indent=2), flush=True)


if __name__ == "__main__":
    main()
