#!/usr/bin/env python3
"""Publish plots and provenance for the finite-187T large-sheet diagnostic."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np


HERE = Path(__file__).resolve().parent
RAW_DIR = Path(
    "/home/seunghyun/tairte4/raw_artifacts/"
    "paper_tairte4_finite_187T_w12_large_sheet_thermal_pte"
)
RAW_NPZ = RAW_DIR / "finite_187T_large_sheet_thermal_pte.npz"
RAW_JSON = RAW_DIR / "FINITE_187T_LARGE_SHEET_THERMAL_PTE.json"
OUT = HERE / "results_finite_187T_large_sheet_thermal_pte"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_limit(values: np.ndarray, percentile: float = 99.5) -> float:
    finite = np.abs(np.asarray(values)[np.isfinite(values)])
    limit = float(np.percentile(finite, percentile)) if finite.size else 1.0
    return limit if limit > np.finfo(float).tiny else 1.0


def add_map(
    axis: plt.Axes,
    x_edges_um: np.ndarray,
    y_edges_um: np.ndarray,
    values: np.ndarray,
    title: str,
    label: str,
    *,
    signed: bool = False,
    cmap: str | None = None,
) -> None:
    if signed:
        limit = finite_limit(values)
        norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
        image = axis.pcolormesh(
            x_edges_um, y_edges_um, values.T, shading="flat", cmap=cmap or "coolwarm", norm=norm
        )
    else:
        image = axis.pcolormesh(
            x_edges_um, y_edges_um, values.T, shading="flat", cmap=cmap or "inferno"
        )
    axis.set_title(title)
    axis.set_xlabel("Lumerical x = TaIrTe4 b (um)")
    axis.set_ylabel("Lumerical y = TaIrTe4 a (um)")
    axis.set_aspect("equal")
    colorbar = plt.colorbar(image, ax=axis, shrink=0.82)
    colorbar.set_label(label)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    source = json.loads(RAW_JSON.read_text())
    with np.load(RAW_NPZ, allow_pickle=False) as data:
        arrays = {key: np.asarray(data[key]) for key in data.files}

    x, y, z = arrays["x_m"], arrays["y_m"], arrays["z_m"]
    xe, ye, ze = arrays["x_edges_m"], arrays["y_edges_m"], arrays["z_edges_m"]
    xum, yum, zum = x * 1e6, y * 1e6, z * 1e6
    xeum, yeum, zeum = xe * 1e6, ye * 1e6, ze * 1e6
    dz = np.diff(ze)
    q = arrays["Q_285uW_W_m3"]
    qxy = np.sum(q * dz[None, None, :], axis=2)
    temperature = arrays["temperature_3d_K"]
    tflake = arrays["temperature_flake_K"]
    grad_b = arrays["grad_b_K_m"]
    grad_a = arrays["grad_a_K_m"]
    grad_mag = arrays["gradient_magnitude_K_m"]
    psi = arrays["weighting_potential"]
    ew_b = arrays["weighting_field_b_m_inv"]
    ew_a = arrays["weighting_field_a_m_inv"]
    jb = arrays["J_PTE_b_A_m2"]
    ja = arrays["J_PTE_a_A_m2"]
    integrand = arrays["terminal_current_integrand_A_m2"]

    # Complete field overview: optical -> thermal -> electrical collection.
    fig, axes = plt.subplots(3, 4, figsize=(23, 17), constrained_layout=True)
    add_map(axes[0, 0], xeum, yeum, qxy, "Depth-integrated all-material Q", "W/m2")
    add_map(axes[0, 1], xeum, yeum, tflake, "TaIrTe4 thickness-averaged DeltaT", "K")
    add_map(axes[0, 2], xeum, yeum, grad_b, "strict centered dT/db", "K/m", signed=True)
    add_map(axes[0, 3], xeum, yeum, grad_a, "strict centered dT/da", "K/m", signed=True)
    add_map(axes[1, 0], xeum, yeum, grad_mag, "strict centered |grad T|", "K/m", cmap="viridis")
    add_map(axes[1, 1], xeum, yeum, psi, "weighting potential psi", "1", cmap="viridis")
    add_map(axes[1, 2], xeum, yeum, jb, "local PTE source J_b", "A/m2", signed=True)
    add_map(axes[1, 3], xeum, yeum, ja, "local PTE source J_a", "A/m2", signed=True)
    add_map(axes[2, 0], xeum, yeum, integrand, "terminal-current integrand", "A/m2", signed=True)
    add_map(axes[2, 1], xeum, yeum, ew_b, "weighting field E_W,b = -dpsi/db", "1/m", signed=True)
    add_map(axes[2, 2], xeum, yeum, ew_a, "weighting field E_W,a = -dpsi/da", "1/m", signed=True)
    stride = max(1, x.size // 25)
    axes[2, 3].pcolormesh(xeum, yeum, psi.T, shading="flat", cmap="viridis")
    axes[2, 3].quiver(
        xum[::stride], yum[::stride], ew_b[::stride, ::stride].T,
        ew_a[::stride, ::stride].T, color="white", pivot="mid", scale=None
    )
    axes[2, 3].set_title("weighting-field direction on psi")
    axes[2, 3].set_xlabel("x=b (um)")
    axes[2, 3].set_ylabel("y=a (um)")
    axes[2, 3].set_aspect("equal")
    fig.suptitle(
        "Finite 187 inverse-T array: validated optical Q -> diagnostic large-sheet thermal/PTE",
        fontsize=18,
    )
    overview = OUT / "finite_187T_large_sheet_all_fields.png"
    fig.savefig(overview, dpi=180)
    plt.close(fig)

    # Vertical Q and temperature cross-sections at the optical axis.
    ix0, iy0 = int(np.argmin(np.abs(x))), int(np.argmin(np.abs(y)))
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    qlim = finite_limit(q, 99.8)
    im = axes[0, 0].pcolormesh(xeum, zeum, q[:, iy0, :].T, shading="flat", cmap="inferno", vmax=qlim)
    axes[0, 0].set_title("Q(x,z), y closest to 0")
    axes[0, 0].set_xlabel("x=b (um)"); axes[0, 0].set_ylabel("z (um)")
    axes[0, 0].set_ylim(-0.9, 0.6); plt.colorbar(im, ax=axes[0, 0], label="W/m3")
    im = axes[0, 1].pcolormesh(yeum, zeum, q[ix0, :, :].T, shading="flat", cmap="inferno", vmax=qlim)
    axes[0, 1].set_title("Q(y,z), x closest to 0")
    axes[0, 1].set_xlabel("y=a (um)"); axes[0, 1].set_ylabel("z (um)")
    axes[0, 1].set_ylim(-0.9, 0.6); plt.colorbar(im, ax=axes[0, 1], label="W/m3")
    im = axes[1, 0].pcolormesh(xeum, zeum, temperature[:, iy0, :].T, shading="flat", cmap="magma")
    axes[1, 0].set_title("DeltaT(x,z), y closest to 0")
    axes[1, 0].set_xlabel("x=b (um)"); axes[1, 0].set_ylabel("z (um)")
    axes[1, 0].set_ylim(-2.0, 0.6); plt.colorbar(im, ax=axes[1, 0], label="K")
    im = axes[1, 1].pcolormesh(yeum, zeum, temperature[ix0, :, :].T, shading="flat", cmap="magma")
    axes[1, 1].set_title("DeltaT(y,z), x closest to 0")
    axes[1, 1].set_xlabel("y=a (um)"); axes[1, 1].set_ylabel("z (um)")
    axes[1, 1].set_ylim(-2.0, 0.6); plt.colorbar(im, ax=axes[1, 1], label="K")
    cross_sections = OUT / "finite_187T_Q_temperature_cross_sections.png"
    fig.savefig(cross_sections, dpi=180)
    plt.close(fig)

    # Central profiles preserve sign and make cancellation explicit.
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    axes[0, 0].plot(xum, tflake[:, iy0], label="along b at a=0")
    axes[0, 0].plot(yum, tflake[ix0, :], label="along a at b=0")
    axes[0, 0].set_title("central DeltaT profiles"); axes[0, 0].set_ylabel("K"); axes[0, 0].legend()
    axes[0, 1].plot(xum, grad_b[:, iy0], label="dT/db")
    axes[0, 1].plot(yum, grad_a[ix0, :], label="dT/da")
    axes[0, 1].set_title("signed central gradients"); axes[0, 1].set_ylabel("K/m"); axes[0, 1].legend()
    axes[1, 0].plot(xum, jb[:, iy0], label="J_b")
    axes[1, 0].plot(yum, ja[ix0, :], label="J_a")
    axes[1, 0].set_title("local thermoelectric source"); axes[1, 0].set_ylabel("A/m2"); axes[1, 0].legend()
    axes[1, 1].plot(xum, integrand[:, iy0], label="current integrand along b")
    axes[1, 1].plot(yum, integrand[ix0, :], label="current integrand along a")
    axes[1, 1].axhline(0.0, color="black", linewidth=0.8)
    axes[1, 1].set_title("signed terminal-current integrand"); axes[1, 1].set_ylabel("A/m2"); axes[1, 1].legend()
    for axis in axes.ravel(): axis.set_xlabel("coordinate (um)"); axis.grid(alpha=0.25)
    profiles = OUT / "finite_187T_signed_central_profiles.png"
    fig.savefig(profiles, dpi=180)
    plt.close(fig)

    summary = {
        "status": source["status"],
        "classification": source["classification"],
        "axis_mapping": source["axis_mapping"],
        "input_Q": source["input_Q"],
        "geometry": source["geometry"],
        "boundaries": source["boundaries"],
        "thermal_materials_W_mK": source["thermal_materials_W_mK"],
        "interface_model": source["interface_model"],
        "illumination": source["illumination"],
        "Q": source["Q"],
        "thermal": source["thermal"],
        "electrical": source["electrical"],
        "gates": source["gates"],
        "interpretation": [
            "All component-specific Yee Q is conservatively remapped; no crop or material masking is applied.",
            "The electrical collection model is an ideal full-width y-edge contact diagnostic, not measured electrode CAD.",
            "The near-zero terminal current reflects signed spatial cancellation in this symmetric large-sheet diagnostic.",
            "The 285 nm SiO2 optical-closure thickness is used, so this is not the paper's physical finite device stack.",
        ],
    }
    summary_path = OUT / "finite_187T_large_sheet_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    cases_path = OUT / "finite_187T_large_sheet_cases.csv"
    with cases_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["case", "value", "unit"], lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(
            [
                {"case": "mapped_absorbed_power_at_285uW", "value": source["Q"]["absorbed_power_at_285uW_W"], "unit": "W"},
                {"case": "Tmax", "value": source["thermal"]["Tmax_K"], "unit": "K"},
                {"case": "Tavg_flake", "value": source["thermal"]["Tavg_flake_K"], "unit": "K"},
                {"case": "max_gradient", "value": source["thermal"]["max_gradient_K_m"], "unit": "K/m"},
                {"case": "short_circuit_current", "value": source["electrical"]["short_circuit_current_A"], "unit": "A"},
                {"case": "open_circuit_voltage", "value": source["electrical"]["open_circuit_voltage_V"], "unit": "V"},
            ]
        )

    report = OUT / "FINITE_187T_LARGE_SHEET_THERMAL_PTE_REPORT.md"
    report.write_text(
        f"""# Finite 187-T large-sheet thermal/PTE diagnostic

Status: `{source['status']}`

## Scope

This is a **large finite computational TaIrTe4 sheet with ideal full-width y-edge contacts**.  It retains the entire validated component-specific Yee heat source and is not an experimental finite-contact prediction.  Axis mapping is `x=b, y=a, z=c`.

The optical-closure stack is air / finite 187 Au inverse-Ts / TaIrTe4 100 nm / Al2O3 35 nm / Au mirror 200 nm / SiO2 285 nm / Si.  Lateral and top thermal faces are adiabatic; the Si bottom is fixed at DeltaT=0.  SiO2/Si uses G=1.1e9 W/(m2 K); the other interfaces are explicitly perfect-contact diagnostic assumptions.

## Certified results at 285 uW incident power

- mapped absorbed power: `{source['Q']['absorbed_power_at_285uW_W']:.9e} W`
- Q mapping error: `{source['Q']['mapping_relative_error']:.3e}`
- flake Tmax: `{source['thermal']['Tmax_K']:.9e} K`
- flake area-average DeltaT: `{source['thermal']['Tavg_flake_K']:.9e} K`
- max strict-centered |grad T|: `{source['thermal']['max_gradient_K_m']:.9e} K/m`
- short-circuit terminal current: `{source['electrical']['short_circuit_current_A']:.9e} A`
- open-circuit voltage: `{source['electrical']['open_circuit_voltage_V']:.9e} V`
- CUDA solve: `{source['thermal']['CUDA_solve_seconds']:.3f} s`, `{source['thermal']['CUDA_PCG_iterations']} iterations`
- residual: `{source['thermal']['linear_residual_relative']:.3e}`
- energy-balance error: `{source['thermal']['energy_balance_relative_error']:.3e}`

The terminal value is small because positive and negative current-integrand regions largely cancel under the symmetric ideal-contact diagnostic. It must not be interpreted as the finite experimental device current.

## Fields

- [all optical/thermal/electrical fields](finite_187T_large_sheet_all_fields.png)
- [Q and temperature cross-sections](finite_187T_Q_temperature_cross_sections.png)
- [signed central profiles](finite_187T_signed_central_profiles.png)
- [summary JSON](finite_187T_large_sheet_summary.json)
- [cases CSV](finite_187T_large_sheet_cases.csv)
- [artifact manifest](RAW_ARTIFACT_MANIFEST.json)

No Q clipping, smoothing, gain, global shape rescaling, or source deletion was used. The only scaling is certified linear scaling from the source-only incident power to 285 uW.
"""
    )

    artifacts = [RAW_NPZ, RAW_JSON, overview, cross_sections, profiles, summary_path, cases_path, report]
    manifest = {
        "status": source["status"],
        "raw_not_committed": True,
        "generation_commands": [
            "/home/eidl/miniconda3/envs/EIDL-Lumapi/bin/python -u 39_run_finite_187t_large_sheet_thermal_pte.py",
            "/home/eidl/miniconda3/envs/EIDL-Lumapi/bin/python 40_summarize_finite_187t_large_sheet_thermal_pte.py",
        ],
        "artifacts": [
            {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in artifacts
        ],
    }
    manifest_path = OUT / "RAW_ARTIFACT_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": source["status"], "output": str(OUT)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
