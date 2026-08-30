#!/usr/bin/env python3
"""Publish paired Ea/Eb Q, T, gradient, weighting and PTE figures."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
RAW = Path("/home/seunghyun/tairte4/raw_artifacts/paper_tairte4_finite_187T_w12_ea_eb_thermal_pair")
OUT = HERE / "results_finite_187T_large_sheet_ea_eb_thermal_pte"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def centers_to_edges(values: np.ndarray) -> np.ndarray:
    edges = np.empty(values.size + 1, float)
    edges[1:-1] = 0.5 * (values[:-1] + values[1:])
    edges[0] = values[0] - 0.5 * (values[1] - values[0])
    edges[-1] = values[-1] + 0.5 * (values[-1] - values[-2])
    return edges


def symmetric_limit(*arrays: np.ndarray) -> float:
    return max(float(np.nanmax(np.abs(item))) for item in arrays)


def main() -> int:
    summary_path = RAW / "FINITE_187T_EA_EB_OPTICAL_THERMAL_PAIR.json"
    summary = json.loads(summary_path.read_text())
    if summary.get("status") != "VALIDATED_FINITE_187T_EA_EB_OPTICAL_THERMAL_PAIR":
        raise RuntimeError(f"pair is not validated: {summary.get('status')}")
    cases = {}
    for pol in ("Ea", "Eb"):
        path = RAW / pol / f"finite_187T_large_sheet_thermal_pte_{pol}.npz"
        with np.load(path, allow_pickle=False) as data:
            cases[pol] = {key: np.asarray(data[key]) for key in data.files}
        cases[pol]["_path"] = path

    OUT.mkdir(parents=True, exist_ok=True)
    x = cases["Ea"]["x_m"] * 1e6
    y = cases["Ea"]["y_m"] * 1e6
    x_edges = cases["Ea"]["x_edges_m"] * 1e6
    y_edges = cases["Ea"]["y_edges_m"] * 1e6
    dz = np.diff(cases["Ea"]["z_edges_m"])
    extent = (x_edges[0], x_edges[-1], y_edges[0], y_edges[-1])
    fields: dict[str, dict[str, np.ndarray]] = {}
    for pol in ("Ea", "Eb"):
        item = cases[pol]
        fields[pol] = {
            "q_sheet": np.sum(item["Q_285uW_W_m3"] * dz[None, None, :], axis=2),
            "temperature": item["temperature_flake_K"],
            "grad_b": item["grad_b_K_m"],
            "grad_a": item["grad_a_K_m"],
            "grad_mag": item["gradient_magnitude_K_m"],
            "psi": item["weighting_potential"],
            "current": item["terminal_current_integrand_A_m2"],
        }

    signed_specs = (("grad_b", "strict-centered dT/db (K/m)"),
                    ("grad_a", "strict-centered dT/da (K/m)"),
                    ("current", "terminal-current integrand (A/m2)"))
    positive_specs = (("q_sheet", "depth-integrated Q (W/m2)"),
                      ("temperature", "TaIrTe4 thickness-avg dT (K)"),
                      ("grad_mag", "strict-centered |grad T| (K/m)"),
                      ("psi", "weighting potential psi"))
    limits = {key: symmetric_limit(fields["Ea"][key], fields["Eb"][key]) for key, _ in signed_specs}
    positive_limits = {
        key: max(float(np.nanmax(fields[pol][key])) for pol in ("Ea", "Eb"))
        for key, _ in positive_specs
    }
    fig, axes = plt.subplots(2, 7, figsize=(31, 9), constrained_layout=True)
    specs = list(positive_specs[:2]) + list(signed_specs[:2]) + [positive_specs[2], positive_specs[3], signed_specs[2]]
    for row, pol in enumerate(("Ea", "Eb")):
        for col, (key, title) in enumerate(specs):
            array = fields[pol][key].T
            if key in limits:
                vmax = limits[key]
                image = axes[row, col].imshow(array, origin="lower", extent=extent,
                                               cmap="coolwarm", vmin=-vmax, vmax=vmax,
                                               interpolation="nearest", aspect="equal")
            else:
                image = axes[row, col].imshow(array, origin="lower", extent=extent,
                                               cmap="inferno" if key != "psi" else "viridis",
                                               vmin=0.0, vmax=positive_limits[key],
                                               interpolation="nearest", aspect="equal")
            axes[row, col].set_title(f"{pol}: {title}")
            axes[row, col].set_xlabel("Lumerical x=b (um)")
            axes[row, col].set_ylabel("Lumerical y=a (um)")
            fig.colorbar(image, ax=axes[row, col], shrink=0.78)
    fig.suptitle("Finite 187 inverse-T: paired E||a / E||b optical Q -> identical thermal, weighting and PTE")
    all_fields = OUT / "finite_187T_Ea_Eb_all_fields.png"
    fig.savefig(all_fields, dpi=180)
    plt.close(fig)

    labels = ["P_Q (uW)", "Tmax (K)", "max |gradT| (1e5 K/m)", "I (nA)"]
    values = {"Ea": [], "Eb": []}
    for pol in ("Ea", "Eb"):
        record = summary["cases"][pol]
        thermal_json = json.loads(Path(record["thermal_json"]).read_text())
        values[pol] = [
            float(thermal_json["Q"]["absorbed_power_at_285uW_W"]) * 1e6,
            float(record["Tmax_K_at_285uW"]),
            float(record["max_gradient_K_m"]) / 1e5,
            float(record["short_circuit_current_A"]) * 1e9,
        ]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4), constrained_layout=True)
    for index, label in enumerate(labels):
        axes[index].bar(["E||a", "E||b"], [values["Ea"][index], values["Eb"][index]], color=["#2878b5", "#e67e22"])
        axes[index].set_title(label)
        axes[index].grid(axis="y", alpha=0.25)
    comparison = OUT / "finite_187T_Ea_Eb_scalar_comparison.png"
    fig.savefig(comparison, dpi=220)
    plt.close(fig)

    published = {
        **summary,
        "plots": [str(all_fields), str(comparison)],
        "raw_pair_summary": {"path": str(summary_path), "sha256": sha256(summary_path)},
        "raw_npz": {
            pol: {"path": str(cases[pol]["_path"]), "sha256": sha256(cases[pol]["_path"])}
            for pol in ("Ea", "Eb")
        },
    }
    json_out = OUT / "FINITE_187T_EA_EB_PUBLISHED_SUMMARY.json"
    json_out.write_text(json.dumps(published, indent=2) + "\n")
    report = OUT / "FINITE_187T_EA_EB_OPTICAL_THERMAL_REPORT.md"
    report.write_text(
        "# Finite 187 inverse-T paired optical/thermal result\n\n"
        f"Status: `{summary['status']}`\n\n"
        "Both `E||a` and `E||b` use the same finite geometry, Gaussian source contract, "
        "incident power, conservative remap, explicit 3-D thermal operator, boundary "
        "conditions, and electrical weighting definition. A single-polarization result "
        "cannot promote this report. This remains a named large-sheet diagnostic, not an "
        "experimental finite-contact prediction.\n\n"
        "![all fields](finite_187T_Ea_Eb_all_fields.png)\n\n"
        "![scalar comparison](finite_187T_Ea_Eb_scalar_comparison.png)\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
