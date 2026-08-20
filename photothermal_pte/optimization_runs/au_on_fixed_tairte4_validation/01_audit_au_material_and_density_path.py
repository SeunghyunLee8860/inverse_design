#!/usr/bin/env python3
"""Offline audit of the 10-um Au endpoint and metal density path."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from material_model import (
    AU_BULK_ELECTRICAL_CONDUCTIVITY_S_M,
    AU_BULK_SEEBECK_V_K,
    AU_BULK_THERMAL_CONDUCTIVITY_W_MK,
    EPSILON_AIR,
    EPSILON_AU_ORDAL_10UM,
    N_AIR,
    N_AU_ORDAL_10UM,
    TEMPERATURE_K,
    WAVELENGTH_M,
    linear_epsilon_path,
    nonlinear_index_path,
    passive_index,
    wiedemann_franz_thermal_conductivity,
)


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "results"
SAMPLE_RHO = np.asarray(
    [0.0, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0]
)


def complex_record(value: complex) -> dict[str, float]:
    number = complex(value)
    return {"real": float(number.real), "imag": float(number.imag)}


def first_real_zero(rho: np.ndarray, values: np.ndarray) -> float | None:
    real = np.real(values)
    indices = np.flatnonzero(real[:-1] * real[1:] <= 0.0)
    if indices.size == 0:
        return None
    index = int(indices[0])
    x0, x1 = rho[index : index + 2]
    y0, y1 = real[index : index + 2]
    if y1 == y0:
        return float(x0)
    return float(x0 - y0 * (x1 - x0) / (y1 - y0))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_plot(output: Path, rho: np.ndarray, paths: list) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    colors = ("tab:red", "tab:blue")
    for path, color in zip(paths, colors):
        label = path.name.replace("_", " ")
        axes[0, 0].plot(rho, path.epsilon.real, color=color, label=label)
        axes[0, 1].plot(rho, path.epsilon.imag, color=color, label=label)
        index = passive_index(path.epsilon)
        axes[1, 0].plot(rho, index.real, color=color, label=label)
        axes[1, 0].plot(rho, index.imag, color=color, linestyle="--")
        axes[1, 1].plot(path.epsilon.real, path.epsilon.imag, color=color, label=label)
    axes[0, 0].axhline(0.0, color="black", linewidth=0.8)
    axes[0, 0].set(xlabel=r"physical density $\rho$", ylabel=r"Re $\epsilon_r$", title="Real permittivity")
    axes[0, 1].set(xlabel=r"physical density $\rho$", ylabel=r"Im $\epsilon_r$", title="Loss / passivity")
    axes[1, 0].set(xlabel=r"physical density $\rho$", ylabel="n or k", title="Passive square-root index (solid n, dashed k)")
    axes[1, 1].set(xlabel=r"Re $\epsilon_r$", ylabel=r"Im $\epsilon_r$", title="Complex-permittivity path")
    for axis in axes.ravel():
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.suptitle("Au/air interpolation audit at 10 µm — Ordal Au endpoint")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    dense_rho = np.linspace(0.0, 1.0, 20001)
    dense_paths = [linear_epsilon_path(dense_rho), nonlinear_index_path(dense_rho)]
    sampled_paths = [linear_epsilon_path(SAMPLE_RHO), nonlinear_index_path(SAMPLE_RHO)]

    rows: list[dict[str, object]] = []
    for path in sampled_paths:
        index = passive_index(path.epsilon)
        for density, epsilon, derivative, n_value in zip(
            SAMPLE_RHO, path.epsilon, path.derivative, index
        ):
            rows.append(
                {
                    "law": path.name,
                    "rho": float(density),
                    "epsilon_real": float(epsilon.real),
                    "epsilon_imag": float(epsilon.imag),
                    "d_epsilon_d_rho_real": float(derivative.real),
                    "d_epsilon_d_rho_imag": float(derivative.imag),
                    "passive_n": float(n_value.real),
                    "passive_k": float(n_value.imag),
                }
            )
    csv_path = output / "au_density_paths.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    plot_path = output / "au_density_interpolation_audit.png"
    write_plot(plot_path, dense_rho, dense_paths)

    law_summaries = []
    all_passive = True
    endpoints_exact = True
    for path in dense_paths:
        passive = bool(np.min(path.epsilon.imag) >= -1e-12)
        endpoint_error = max(
            abs(path.epsilon[0] - EPSILON_AIR),
            abs(path.epsilon[-1] - EPSILON_AU_ORDAL_10UM),
        )
        all_passive = bool(all_passive and passive)
        endpoints_exact = bool(endpoints_exact and endpoint_error < 1e-12)
        law_summaries.append(
            {
                "law": path.name,
                "passive_no_negative_imaginary_epsilon": passive,
                "minimum_imaginary_epsilon": float(np.min(path.epsilon.imag)),
                "maximum_imaginary_epsilon": float(np.max(path.epsilon.imag)),
                "first_Re_epsilon_zero_rho": first_real_zero(dense_rho, path.epsilon),
                "maximum_endpoint_absolute_error": float(endpoint_error),
                "maximum_abs_d_epsilon_d_rho": float(np.max(abs(path.derivative))),
            }
        )

    sigma = AU_BULK_ELECTRICAL_CONDUCTIVITY_S_M
    k_wf = wiedemann_franz_thermal_conductivity(sigma)
    payload = {
        "status": (
            "OFFLINE_AU_MATERIAL_CONTRACT_READY_FDTD_READBACK_PENDING"
            if all_passive and endpoints_exact
            else "FAILED_AU_MATERIAL_OR_INTERPOLATION_AUDIT"
        ),
        "scope": "offline material and differentiable density-path audit; no FDTD, thermal, electrical, adjoint, or optimization solve",
        "wavelength_m": WAVELENGTH_M,
        "gold_optical_endpoint": {
            "n_plus_ik": complex_record(N_AU_ORDAL_10UM),
            "epsilon_r": complex_record(EPSILON_AU_ORDAL_10UM),
            "source": "Ordal et al., Applied Optics 26, 744-752 (1987), exact tabulated 10.0-um row",
            "source_url": "https://doi.org/10.1364/AO.26.000744",
            "data_url": "https://raw.githubusercontent.com/polyanskiy/refractiveindex.info-database/main/database/data/main/Au/nk/Ordal.yml",
        },
        "transport_references_at_approximately_300K": {
            "electrical_conductivity_S_m": sigma,
            "electrical_resistivity_ohm_m": 1.0 / sigma,
            "electrical_source": "Ordal 1985 optical resistivity 2.43 micro-ohm cm; used only as a bulk reference until film data are fixed",
            "thermal_conductivity_W_mK": AU_BULK_THERMAL_CONDUCTIVITY_W_MK,
            "thermal_source": "NIST compilation value for well-annealed high-purity bulk Au at 300 K",
            "wiedemann_franz_k_W_mK": k_wf,
            "wiedemann_franz_relative_difference_from_317": abs(k_wf - AU_BULK_THERMAL_CONDUCTIVITY_W_MK) / AU_BULK_THERMAL_CONDUCTIVITY_W_MK,
            "seebeck_V_K_sensitivity_only": AU_BULK_SEEBECK_V_K,
            "seebeck_baseline_for_first_electrical_control_V_K": 0.0,
            "thin_film_warning": "sigma, kappa, and Seebeck are fabrication- and thickness-dependent; bulk values are references, not promoted film values",
        },
        "skin_depth_estimates": {
            "field_amplitude_1_over_e_m": WAVELENGTH_M / (2.0 * np.pi * N_AU_ORDAL_10UM.imag),
            "intensity_1_over_e_m": WAVELENGTH_M / (4.0 * np.pi * N_AU_ORDAL_10UM.imag),
        },
        "production_candidate": {
            "law": "linear complex refractive index followed by epsilon=n^2",
            "meaning": "numerical topology relaxation only; gray rho is not claimed to be a physical Au/air effective medium",
            "fallback": "sharp-interface level-set/shape optimization if solver readback, binary equivalence, or AD-FD fails",
        },
        "laws": law_summaries,
        "gates": {
            "endpoints_exact": endpoints_exact,
            "all_paths_passive": all_passive,
            "FDTD_readback_complete": False,
            "Maxwell_AD_FD_complete": False,
            "combined_AD_FD_complete": False,
        },
    }
    summary_path = output / "au_material_contract.json"
    summary_path.write_text(json.dumps(payload, indent=2) + "\n")

    report_path = output / "AU_MATERIAL_AND_INTERPOLATION_AUDIT.md"
    report_path.write_text(
        "# Au material and density-interpolation audit\n\n"
        f"Status: `{payload['status']}`\n\n"
        "This checkpoint performs no Maxwell, thermal, electrical, adjoint, or optimization solve.\n\n"
        "## Frozen 10 µm optical endpoint\n\n"
        f"- Ordal Au: `n + ik = {N_AU_ORDAL_10UM.real:.6g} + {N_AU_ORDAL_10UM.imag:.6g}i`\n"
        f"- Relative permittivity: `epsilon = {EPSILON_AU_ORDAL_10UM.real:.6g} + {EPSILON_AU_ORDAL_10UM.imag:.6g}i`\n"
        "- The earlier Lumerical CRC value at 11 µm is not reused as the 10 µm production endpoint.\n\n"
        "Source: [Ordal et al. (1987)](https://doi.org/10.1364/AO.26.000744); "
        "[CC0 tabulation](https://raw.githubusercontent.com/polyanskiy/refractiveindex.info-database/main/database/data/main/Au/nk/Ordal.yml).\n\n"
        "## Interpolation decision\n\n"
        "The legacy linear-complex-epsilon law is retained only as a diagnostic. "
        "The production candidate interpolates complex refractive index and then squares it. "
        "This is the nonlinear law described for plasmonic FDTD topology optimization by "
        "[Zeng et al.](https://doi.org/10.1021/acsphotonics.1c00260). Gray density is explicitly not a physical effective medium.\n\n"
        "The nonlinear candidate preserves both endpoints and remains passive in this offline audit. "
        "It still crosses `Re(epsilon)=0`, so only a Lumerical material-fit/readback and binary-control campaign can certify it.\n\n"
        "## Transport references\n\n"
        f"- Bulk-reference electrical conductivity: `{sigma:.8e} S/m`\n"
        f"- Bulk-reference thermal conductivity: `{AU_BULK_THERMAL_CONDUCTIVITY_W_MK:.6g} W/(m K)`\n"
        f"- Wiedemann-Franz check at 300 K: `{k_wf:.6g} W/(m K)`\n"
        "- Initial electrical control uses `S_Au=0`; `+1.94 µV/K` is reserved for sensitivity.\n\n"
        "These are bulk references, not certified properties of the fabricated Au film. Film thickness, deposition, grain size, and Au/TaIrTe4 contacts remain named physical uncertainties.\n\n"
        "## Next fail-closed gate\n\n"
        "Open a new v261 session, import the Ordal sampled material, and compare requested, fitted (`getfdtdindex`), and native index-monitor values. No optimization is permitted before that gate and binary Au/air controls pass.\n"
    )

    manifest = {
        "status": payload["status"],
        "files": [
            {"path": str(path.relative_to(HERE)), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in (summary_path, csv_path, plot_path, report_path)
        ],
        "raw_FSP_or_NPZ_committed": False,
    }
    (output / "RAW_ARTIFACT_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 0 if payload["status"].startswith("OFFLINE_") else 2


if __name__ == "__main__":
    raise SystemExit(main())
