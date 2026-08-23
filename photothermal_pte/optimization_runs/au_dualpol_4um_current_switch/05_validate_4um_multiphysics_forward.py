#!/usr/bin/env python3
"""Validate rho=0.5 explicit thermal and left/right PTE forward operators."""

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

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.contract import CONTRACT
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_4um_model import build_model
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.material_fraction import (
    audit as material_fraction_audit,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.multiphysics_4um import (
    N_TA,
    build_electrical_system,
    build_thermal_state,
    current_integrand,
    map_native_q_to_thermal,
    solve_electrical,
    solve_thermal,
    tairte4_temperature,
    thermal_edges,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.paths import (
    raw_root,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.validation_provenance import (
    load_current_source_calibration,
    sha256,
)


HERE = Path(__file__).resolve().parent
OUT = HERE / "results_4um_multiphysics_forward"
RAW_OPTICAL = raw_root()
RAW = RAW_OPTICAL / "multiphysics"
CALIBRATION = HERE / "results_fdtdx_4um_source_calibration/fdtdx_4um_source_calibration.json"
FORWARD_SUMMARY = HERE / "results_fdtdx_4um_dualpol_forward/fdtdx_4um_dualpol_forward.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_case(
    polarization: str,
    cuda_device: int,
    scale: float,
    expected_sha256: str,
) -> dict[str, object]:
    model = build_model(polarization, include_adjoint_source=False)
    raw_path = RAW_OPTICAL / f"fdtdx_4um_rho0p5_{polarization}.npz"
    if _sha256(raw_path) != expected_sha256:
        raise RuntimeError(f"fail-closed: {polarization} optical raw SHA mismatch")
    with np.load(raw_path, allow_pickle=False) as raw:
        rho = np.asarray(raw["rho"], dtype=np.float64)
        q_fields = {
            "au": np.asarray(raw["q_au_W_m3"], dtype=np.float64) * scale,
            "tairte4": np.asarray(raw["q_tairte4_W_m3"], dtype=np.float64) * scale,
        }
        volumes = {
            "au": np.asarray(raw["volume_au_m3"], dtype=np.float64),
            "tairte4": np.asarray(raw["volume_tairte4_m3"], dtype=np.float64),
        }
    state = build_thermal_state(rho)
    source_power, mapping, _ = map_native_q_to_thermal(
        state,
        q_fields_W_m3=q_fields,
        dual_volumes_m3=volumes,
        material_slices={
            "au": model["slices"]["au_design"],
            "tairte4": model["slices"]["fixed_tairte4"],
        },
        realized_grid=model["grid"],
    )
    start = time.perf_counter()
    temperature, thermal_audit = solve_thermal(state, source_power, cuda_device)
    ta_temperature = tairte4_temperature(state, temperature)
    electrical = build_electrical_system(rho, ta_temperature)
    psi, current, electrical_audit = solve_electrical(electrical, cuda_device)
    runtime_s = time.perf_counter() - start
    psi_ta = psi[: N_TA * N_TA].reshape(N_TA, N_TA)
    integrand = current_integrand(ta_temperature, psi)
    integrand_current = float(np.sum(integrand) * CONTRACT.design_pitch_m**2)
    current_consistency = abs(integrand_current - current) / max(abs(current), 1e-30)
    gates = {
        "mapping_lt_1e-12": max(row["relative_error"] for row in mapping.values()) < 1e-12,
        "thermal_residual_lt_1e-8": thermal_audit["relative_residual"] < 1e-8,
        "thermal_energy_balance_lt_1pct": thermal_audit["energy_balance_relative"] < 0.01,
        "electrical_residual_lt_1e-8": electrical_audit["relative_residual"] < 1e-8,
        "electrical_terminal_balance_lt_1pct": electrical_audit["terminal_balance_relative"] < 0.01,
        "current_integrand_consistency_lt_1e-12": current_consistency < 1e-12,
        "finite": bool(
            np.all(np.isfinite(temperature))
            and np.all(np.isfinite(psi))
            and np.all(np.isfinite(integrand))
        ),
    }
    RAW.mkdir(parents=True, exist_ok=True)
    output_raw = RAW / f"multiphysics_4um_rho0p5_{polarization}.npz"
    np.savez_compressed(
        output_raw,
        rho=rho,
        source_power_W=source_power,
        temperature_K=temperature,
        tairte4_temperature_K=ta_temperature,
        weighting_potential_tairte4=psi_ta,
        current_integrand_A_m2=integrand,
    )
    return {
        "polarization": polarization,
        "status": "VALIDATED_4UM_MULTIPHYSICS_FORWARD" if all(gates.values()) else "FAILED_4UM_MULTIPHYSICS_FORWARD",
        "runtime_s": runtime_s,
        "source_power_W": float(np.sum(source_power)),
        "mapping": mapping,
        "Tmax_K": float(np.max(temperature)),
        "TaIrTe4_Tmax_K": float(np.max(ta_temperature)),
        "current_A": current,
        "current_nA": current * 1e9,
        "current_sign_convention": (
            "+I is internal conventional current along solver +x, from x_min "
            "to x_max; target is Ia>0 and Ib<0"
        ),
        "current_from_integrand_A": integrand_current,
        "current_integrand_consistency_relative": current_consistency,
        "thermal": thermal_audit,
        "electrical": electrical_audit,
        "gates": gates,
        "raw": {
            "path": str(output_raw),
            "bytes": output_raw.stat().st_size,
            "sha256": _sha256(output_raw),
        },
    }


def main() -> int:
    if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
        raise RuntimeError("GPU-only multiphysics gate requires CUDA_VISIBLE_DEVICES")
    cuda_device = 0
    OUT.mkdir(parents=True, exist_ok=True)
    calibration = load_current_source_calibration(CALIBRATION)
    scale = CONTRACT.reporting_incident_power_W / float(
        calibration["common_reference_incident_power_W"]
    )
    forward = json.loads(FORWARD_SUMMARY.read_text(encoding="utf-8"))
    if forward["status"] != "VALIDATED_FDTDX_4UM_DUALPOL_RHO0P5_FORWARD":
        raise RuntimeError("optical rho=0.5 forward checkpoint is not validated")
    if forward.get("au_material_fraction") != material_fraction_audit():
        raise RuntimeError(
            "optical checkpoint uses a different or undocumented Au material fraction"
        )
    expected_shas = {
        case["polarization"]: case["raw"]["sha256"] for case in forward["cases"]
    }
    cases = []
    for pol in ("Ea", "Eb"):
        print(f"[{pol}] explicit 3-D thermal + left/right electrical", flush=True)
        case = run_case(pol, cuda_device, scale, expected_shas[pol])
        cases.append(case)
        print(
            f"[{pol}] {case['status']} I={case['current_nA']:.6f} nA "
            f"Tmax={case['Tmax_K']:.6g} K runtime={case['runtime_s']:.2f}s",
            flush=True,
        )
    status = (
        "VALIDATED_4UM_DUALPOL_MULTIPHYSICS_RHO0P5_FORWARD"
        if all(case["status"].startswith("VALIDATED_") for case in cases)
        else "FAILED_4UM_DUALPOL_MULTIPHYSICS_RHO0P5_FORWARD"
    )
    summary = {
        "status": status,
        "scope": "rho=0.5 optical Q -> explicit 3D thermal -> floating-Au left/right weighting/PTE forward; no adjoint or optimization",
        "normalization": {
            "common_source_only_scale": scale,
            "incident_power_W": CONTRACT.reporting_incident_power_W,
            "polarization_matching": False,
        },
        "geometry": CONTRACT.audit(),
        "au_material_fraction": material_fraction_audit(),
        "source_calibration_sha256": sha256(CALIBRATION),
        "cases": cases,
    }
    (OUT / "multiphysics_4um_forward.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    with (OUT / "multiphysics_4um_forward.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("polarization", "status", "P_Q_W", "Tmax_K", "current_nA", "runtime_s"))
        for case in cases:
            writer.writerow((case["polarization"], case["status"], case["source_power_W"], case["Tmax_K"], case["current_nA"], case["runtime_s"]))

    fig, axes = plt.subplots(2, 4, figsize=(18, 9), constrained_layout=True)
    x_edges, y_edges, _ = thermal_edges()
    area_xy = np.diff(x_edges)[:, None] * np.diff(y_edges)[None, :]
    for row, case in enumerate(cases):
        with np.load(case["raw"]["path"], allow_pickle=False) as raw:
            source = np.asarray(raw["source_power_W"]).sum(axis=2) / area_xy
            temp = np.asarray(raw["tairte4_temperature_K"])
            psi = np.asarray(raw["weighting_potential_tairte4"])
            current = np.asarray(raw["current_integrand_A_m2"])
        image = axes[row, 0].pcolormesh(
            x_edges * 1e6,
            y_edges * 1e6,
            source.T,
            shading="flat",
            cmap="inferno",
        )
        axes[row, 0].set(
            title=f"{case['polarization']}: depth-integrated Q (W/m²)",
            xlabel="x=b (µm)",
            ylabel="y=a (µm)",
            aspect="equal",
            xlim=(-10, 10),
            ylim=(-10, 10),
        )
        fig.colorbar(image, ax=axes[row, 0], shrink=0.8)
        for ax, data, title, cmap in zip(
            axes[row, 1:],
            (temp, psi, current),
            ("TaIrTe4 thickness-avg ΔT (K)", "weighting potential ψ", "PTE current integrand (A/m²)"),
            ("inferno", "viridis", "coolwarm"),
            strict=True,
        ):
            image = ax.imshow(data.T, origin="lower", extent=(-8, 8, -8, 8), cmap=cmap, aspect="equal")
            ax.set(title=f"{case['polarization']}: {title}", xlabel="x=b (µm)", ylabel="y=a (µm)")
            fig.colorbar(image, ax=ax, shrink=0.8)
    fig.suptitle("4 µm rho=0.5 Au design: explicit 3-D thermal and left/right PTE forward")
    fig.savefig(OUT / "MULTIPHYSICS_4UM_RHO0P5_FIELDS.png", dpi=170)
    plt.close(fig)

    lines = [
        "# 4 µm dual-polarization multiphysics rho=0.5 forward",
        "",
        f"Status: **{status}**",
        "",
        "The same source-only incident-power calibration is applied to both polarizations.",
        "Positive current is internal conventional current along solver +x, from x_min to x_max.",
        "The switching target is Ia>0 and Ib<0; the uniform design already has the requested signs, but is not optimized.",
        "",
        "| polarization | P_Q (W) | Tmax (K) | current (nA) | runtime (s) |",
        "|---|---:|---:|---:|---:|",
    ]
    for case in cases:
        lines.append(
            f"| {case['polarization']} | {case['source_power_W']:.8e} | {case['Tmax_K']:.8e} | {case['current_nA']:.8f} | {case['runtime_s']:.2f} |"
        )
    (OUT / "MULTIPHYSICS_4UM_RHO0P5_FORWARD.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if status.startswith("VALIDATED_") else 2


if __name__ == "__main__":
    raise SystemExit(main())
