#!/usr/bin/env python3
"""Publish the paper-like 11 um Lumerical→thermal→PTE sanity result."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path as PolygonPath
import numpy as np


TMM = {"a": 0.17673296, "b": 0.26328721}
PAPER_ABSORPTION_APPROX = {"a": 0.18, "b": 0.26}
PAPER_EDGE_CURRENT_RATIO_APPROX = 0.80
FLAKE_VERTICES_UM = np.asarray(
    [[-11, -10], [12, -10], [12, 10], [-6, 10], [-6, 6], [-11, 1]],
    float,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def artifact_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def optical_metrics(case_dir: Path) -> dict[str, Any]:
    payload = load_json(case_dir / "case_result.json")
    run = payload["run_result"]
    incident = run["normalization"]["incident_power_W_at_1_W_m2"]
    return {
        "case_dir": str(case_dir.resolve()),
        "case_result_sha256": sha256(case_dir / "case_result.json"),
        "Q_sha256": sha256(case_dir / "finite_q_on_artifact.npz"),
        "P_Q_W_at_1_W_m2_central": run["P_Q_W"],
        "P_six_W_at_1_W_m2_central": run["P_six_face_W"],
        "six_face_closure": run["six_face_relative_closure"],
        "incident_power_W_at_1_W_m2_central": incident,
        "absorbed_fraction_of_total_Gaussian": run["P_Q_W"] / incident,
        "hotspot": run["Q_hotspot"],
        "component_power_W": run["component_power_W"],
        "domain_um": payload["domain_um"],
        "pml_layers": payload["pml_layers"],
        "flake_dz_nm": payload["flake_dz_nm"],
        "gpu_device": payload["pre_run_contract"]["solver"]["resources"]["2"][
            "device type"
        ],
        "version": payload["pre_run_contract"]["solver"]["version"],
    }


def read_q_areal(case_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(case_dir / "finite_q_on_artifact.npz") as raw:
        x = raw["x_m"] * 1e6
        y = raw["y_m"] * 1e6
        z = raw["z_m"]
        q = raw["Q_on_W_m3"]
    wz = np.empty_like(z)
    wz[0] = 0.5 * (z[1] - z[0])
    wz[-1] = 0.5 * (z[-1] - z[-2])
    wz[1:-1] = 0.5 * (z[2:] - z[:-2])
    return x, y, np.einsum("ijk,k->ij", q, wz)


def read_thermal(case_dir: Path) -> dict[str, Any]:
    summary = load_json(case_dir / "summary.json")
    with np.load(case_dir / "thermal_pte_fields.npz") as raw:
        x = 0.5 * (raw["x_edges_m"][:-1] + raw["x_edges_m"][1:]) * 1e6
        y = 0.5 * (raw["y_edges_m"][:-1] + raw["y_edges_m"][1:]) * 1e6
        fields = {
            key: np.asarray(raw[key])
            for key in (
                "temperature_flake_average_K",
                "weighting_potential",
                "shockley_ramo_integrand_A_m2",
                "grad_T_x_K_m",
                "grad_T_y_K_m",
            )
        }
    mask = np.isfinite(fields["temperature_flake_average_K"])
    gradient = np.hypot(fields["grad_T_x_K_m"], fields["grad_T_y_K_m"])
    summary["max_inplane_gradient_K_m"] = float(np.max(gradient[mask]))
    return {"summary": summary, "x_um": x, "y_um": y, **fields}


def flake_limits(ax: Any) -> None:
    vertices = np.vstack((FLAKE_VERTICES_UM, FLAKE_VERTICES_UM[0]))
    ax.plot(vertices[:, 0], vertices[:, 1], "w--", lw=1.2)
    ax.set(xlim=(-13, 14), ylim=(-12, 12), xlabel="lab x = b (µm)", ylabel="lab y = a (µm)")


def plot_summary(
    output: Path,
    optical: dict[str, dict[str, Any]],
    edge_dirs: dict[str, Path],
    expanded: dict[str, dict[str, Any]],
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True)
    ax = axes[0, 0]
    labels = ["paper Fig.3D", "TMM", "Lumerical finite"]
    positions = np.arange(3)
    width = 0.34
    for offset, axis, color in ((-0.5, "a", "tab:blue"), (0.5, "b", "tab:red")):
        values = [
            PAPER_ABSORPTION_APPROX[axis],
            TMM[axis],
            optical[f"center_{axis}"]["absorbed_fraction_of_total_Gaussian"],
        ]
        ax.bar(positions + offset * width, np.asarray(values) * 100, width, label=f"E∥{axis}", color=color)
    ax.set_xticks(positions, labels, rotation=12)
    ax.set_ylabel("absorption / total Gaussian power (%)")
    ax.set_title("11 µm absorption sanity")
    ax.legend()

    for column, axis in enumerate(("a", "b"), start=1):
        x, y, q = read_q_areal(edge_dirs[axis])
        ax = axes[0, column]
        image = ax.imshow(
            q.T,
            origin="lower",
            extent=[x[0], x[-1], y[0], y[-1]],
            cmap="inferno",
            aspect="equal",
        )
        flake_limits(ax)
        ax.plot(-8.5, 3.5, "cx", ms=8, mew=2)
        ax.set_title(f"Lumerical edge Q, E∥{axis}")
        fig.colorbar(image, ax=ax, label="absorbed areal power at unit I (W/m²)")

    for column, axis in enumerate(("a", "b")):
        data = expanded[axis]
        ax = axes[1, column]
        x, y = data["x_um"], data["y_um"]
        image = ax.imshow(
            data["temperature_flake_average_K"].T,
            origin="lower",
            extent=[x[0], x[-1], y[0], y[-1]],
            cmap="inferno",
            aspect="equal",
        )
        flake_limits(ax)
        ax.set_title(f"Expanded FVM ΔT, E∥{axis}")
        fig.colorbar(image, ax=ax, label="ΔT at 285 µW incident (K)")

    ax = axes[1, 2]
    ratios = [
        optical["edge_a"]["P_Q_W_at_1_W_m2_central"]
        / optical["edge_b"]["P_Q_W_at_1_W_m2_central"],
        expanded["a"]["summary"]["thermal"]["Tmax_rise_K"]
        / expanded["b"]["summary"]["thermal"]["Tmax_rise_K"],
        abs(expanded["a"]["summary"]["PTE_current_A_at_285uW_incident"])
        / abs(expanded["b"]["summary"]["PTE_current_A_at_285uW_incident"]),
    ]
    ax.bar(["edge P_Q", "edge Tmax", "edge |I|"], ratios, color=["0.5", "tab:orange", "tab:purple"])
    ax.axhline(1.0, color="k", ls="--", label="equal")
    ax.axhline(PAPER_EDGE_CURRENT_RATIO_APPROX, color="tab:green", ls=":", label="paper |Ia|/|Ib|≈0.8")
    ax.set_ylabel("E∥a / E∥b ratio")
    ax.set_title("Coupled trend diagnostic")
    ax.legend(fontsize=8)
    fig.suptitle("Paper-like Device A IR sanity check: actual Lumerical Q → thermal FVM → PTE", fontsize=15)
    fig.savefig(output / "DEVICE_A_IR_COUPLED_SANITY.png", dpi=210)
    plt.close(fig)

    # Weighting and local collection diagnostics.
    data = expanded["b"]
    x, y = data["x_um"], data["y_um"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    for ax, key, title, cmap in (
        (axes[0], "weighting_potential", "Approximate electrode weighting ψ", "viridis"),
        (axes[1], "shockley_ramo_integrand_A_m2", "Shockley–Ramo integrand, E∥b", "RdBu_r"),
        (axes[2], "temperature_flake_average_K", "Edge temperature, E∥b", "inferno"),
    ):
        image = ax.imshow(
            data[key].T,
            origin="lower",
            extent=[x[0], x[-1], y[0], y[-1]],
            aspect="equal",
            cmap=cmap,
        )
        flake_limits(ax)
        ax.set_title(title)
        fig.colorbar(image, ax=ax)
    fig.savefig(output / "DEVICE_A_WEIGHTING_AND_PTE_DIAGNOSTICS.png", dpi=210)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    root = args.artifact_root
    optical_dirs = {
        "center_a": root / "finite_center_a_w6p5_dz10_gpu2_20260730",
        "center_b": root / "finite_center_b_w6p5_dz10_gpu3_20260730",
        "edge_a": root / "finite_edge_a_w6p5_dz10_gpu4_20260730",
        "edge_b": root / "finite_edge_b_w6p5_dz10_gpu5_20260730",
    }
    expanded_dirs = {
        "a": root / "thermal_edge_a_core200_20260730",
        "b": root / "thermal_edge_b_core200_20260730",
    }
    reduced_dirs = {
        "a": root / "reduced_edge_a_core200_20260730",
        "b": root / "reduced_edge_b_core200_20260730",
    }
    optical = {name: optical_metrics(path) for name, path in optical_dirs.items()}
    expanded = {axis: read_thermal(path) for axis, path in expanded_dirs.items()}
    reduced = {axis: read_thermal(path) for axis, path in reduced_dirs.items()}
    optical_center_ratio = (
        optical["center_a"]["absorbed_fraction_of_total_Gaussian"]
        / optical["center_b"]["absorbed_fraction_of_total_Gaussian"]
    )
    optical_edge_ratio = (
        optical["edge_a"]["P_Q_W_at_1_W_m2_central"]
        / optical["edge_b"]["P_Q_W_at_1_W_m2_central"]
    )
    expanded_current_ratio = abs(
        expanded["a"]["summary"]["PTE_current_A_at_285uW_incident"]
    ) / abs(expanded["b"]["summary"]["PTE_current_A_at_285uW_incident"])
    reduced_current_ratio = abs(
        reduced["a"]["summary"]["PTE_current_A_at_285uW_incident"]
    ) / abs(reduced["b"]["summary"]["PTE_current_A_at_285uW_incident"])
    optical_pass = (
        all(value["six_face_closure"] < 0.005 for value in optical.values())
        and abs(optical["center_a"]["absorbed_fraction_of_total_Gaussian"] - TMM["a"]) / TMM["a"] < 0.05
        and abs(optical["center_b"]["absorbed_fraction_of_total_Gaussian"] - TMM["b"]) / TMM["b"] < 0.05
    )
    thermal_numerical_pass = all(
        item[axis]["summary"]["thermal"]["energy_balance_relative_error"] < 0.01
        and item[axis]["summary"]["thermal"]["linear_residual_relative"] < 1e-8
        and item[axis]["summary"]["mapping"]["mapping_relative_power_error"] < 0.005
        for item in (expanded, reduced)
        for axis in ("a", "b")
    )
    pte_trend_pass = expanded_current_ratio < 1.0 and reduced_current_ratio < 1.0
    status = (
        "VALIDATED_DEVICE_A_IR_COUPLED_SANITY"
        if optical_pass and thermal_numerical_pass and pte_trend_pass
        else "FAILED_COUPLED_DEVICE_A_IR_PTE_SANITY_GEOMETRY_UNRESOLVED"
    )
    summary = {
        "status": status,
        "optical_sanity_passed": optical_pass,
        "thermal_numerical_gates_passed": thermal_numerical_pass,
        "paper_pte_polarization_trend_passed": pte_trend_pass,
        "central_absorption_ratio_a_over_b": optical_center_ratio,
        "edge_absorbed_power_ratio_a_over_b": optical_edge_ratio,
        "expanded_edge_current_ratio_abs_a_over_b": expanded_current_ratio,
        "paper_reduced_edge_current_ratio_abs_a_over_b": reduced_current_ratio,
        "paper_measured_edge_current_ratio_approx": PAPER_EDGE_CURRENT_RATIO_APPROX,
        "interpretation": (
            "The material/thickness/polarization optical certificate agrees "
            "with paper Fig. 3D. The approximate full-device polygon creates "
            "an E||a corner hotspot and reverses the paper PTE trend in both "
            "thermal boundary models; exact CAD/beam metrology or a paper-like "
            "local half-plane geometry is required before promotion."
        ),
        "optical": optical,
        "expanded": {axis: value["summary"] for axis, value in expanded.items()},
        "paper_reduced": {axis: value["summary"] for axis, value in reduced.items()},
        "assumptions_not_published": {
            "flake_polygon": FLAKE_VERTICES_UM.tolist(),
            "beam_waist_radius_um": 6.5,
            "beam_waist_note": "named midpoint scenario inferred from a roughly 9-16 um diffraction-limited spot",
            "beam_edge_center_um": [-8.5, 3.5],
            "electrode_masks": "approximated from paper Figure 2A",
            "Au_Ti_in_off_axis_optical_model": False,
        },
        "no_optimization": True,
        "no_adjoint_gradient": True,
    }
    (args.output_dir / "device_a_ir_sanity_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    rows = []
    for name, values in optical.items():
        rows.append(
            {
                "stage": "optical",
                "case": name,
                "P_Q_W": values["P_Q_W_at_1_W_m2_central"],
                "P_six_W": values["P_six_W_at_1_W_m2_central"],
                "closure": values["six_face_closure"],
                "absorbed_fraction": values["absorbed_fraction_of_total_Gaussian"],
                "Tmax_K": "",
                "PTE_A": "",
            }
        )
    for model, values in (("expanded", expanded), ("paper_reduced", reduced)):
        for axis in ("a", "b"):
            item = values[axis]["summary"]
            rows.append(
                {
                    "stage": model,
                    "case": f"edge_{axis}",
                    "P_Q_W": item["mapping"]["P_Q_target_W"],
                    "P_six_W": "",
                    "closure": item["thermal"]["energy_balance_relative_error"],
                    "absorbed_fraction": "",
                    "Tmax_K": item["thermal"]["Tmax_rise_K"],
                    "PTE_A": item["PTE_current_A_at_285uW_incident"],
                }
            )
    with (args.output_dir / "device_a_ir_sanity_cases.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=rows[0].keys(),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    plot_summary(args.output_dir, optical, {"a": optical_dirs["edge_a"], "b": optical_dirs["edge_b"]}, expanded)

    raw_paths: list[Path] = []
    for path in optical_dirs.values():
        raw_paths += [
            path / "finite_q_on_artifact.npz",
            path / "finite_2um_optical_q.fsp",
            path / "case_result.json",
        ]
    for path in (*expanded_dirs.values(), *reduced_dirs.values()):
        raw_paths += [path / "thermal_pte_fields.npz", path / "summary.json"]
    manifest = {
        "policy": "Raw NPZ/FSP artifacts remain outside Git; only paths, sizes, and SHA-256 are published.",
        "artifacts": [artifact_record(path) for path in raw_paths],
    }
    (args.output_dir / "RAW_ARTIFACT_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )

    report = f"""# Device A paper-like 11 µm coupled sanity check

Status: `{status}`

## Outcome

This was an actual coupled run:

`Lumerical v261 GPU Gaussian Q → conservative support remap → current expanded
thermal-cell FVM → solved approximate electrode weighting potential → PTE`.

The optical sanity check passed, but the full coupled experimental sanity
check did **not**. No empirical gain, current rescaling, or parameter fitting
was used.

## Published inputs held fixed

- Device-A TaIrTe4 thickness: 130 nm; substrate: 285 nm SiO2/Si.
- `kappa_a,b,c = 14.4, 3.8, 1.0 W/(m K)`.
- `sigma_a,b = 4.91e5, 1.10e5 S/m`.
- `S_a,b = -6, +27 µV/K`.
- `G_TaIrTe4/air = 1 W/(m² K)`;
  `G_TaIrTe4/thermally-grown-SiO2 = 7.37e6 W/(m² K)`.
- Normal-incidence 11 µm Gaussian illumination, 285 µW incident power,
  both `E||a` and `E||b`.

Density and heat capacity were not used because this is steady state.

## Optical result

| metric | E||a | E||b | a/b |
|---|---:|---:|---:|
| TMM absorption | {TMM['a']:.3%} | {TMM['b']:.3%} | {TMM['a']/TMM['b']:.3f} |
| finite central Lumerical absorption | {optical['center_a']['absorbed_fraction_of_total_Gaussian']:.3%} | {optical['center_b']['absorbed_fraction_of_total_Gaussian']:.3%} | {optical_center_ratio:.3f} |
| off-axis-edge absorbed power at 285 µW | {expanded['a']['summary']['mapping']['P_Q_target_W']*1e6:.3f} µW | {expanded['b']['summary']['mapping']['P_Q_target_W']*1e6:.3f} µW | {optical_edge_ratio:.3f} |
| six-face closure, central | {optical['center_a']['six_face_closure']:.3%} | {optical['center_b']['six_face_closure']:.3%} | — |
| six-face closure, edge | {optical['edge_a']['six_face_closure']:.3%} | {optical['edge_b']['six_face_closure']:.3%} | — |

The central values agree with the paper Fig. 3D (approximately 18% and 26%)
and the independent TMM. This validates the 130-nm material-axis optical
contract at 11 µm.

## Thermal/PTE result

| model | Tmax E||a | Tmax E||b | |Ia|/|Ib| |
|---|---:|---:|---:|
| current expanded FVM | {expanded['a']['summary']['thermal']['Tmax_rise_K']:.4f} K | {expanded['b']['summary']['thermal']['Tmax_rise_K']:.4f} K | {expanded_current_ratio:.3f} |
| paper Eq. S4 reduced Robin reference | {reduced['a']['summary']['thermal']['Tmax_rise_K']:.4f} K | {reduced['b']['summary']['thermal']['Tmax_rise_K']:.4f} K | {reduced_current_ratio:.3f} |
| paper experiment | — | — | approximately {PAPER_EDGE_CURRENT_RATIO_APPROX:.2f} |

Both numerical models conserve energy below 1%, have residual below 1e-8,
and preserve Q mapping power. Nevertheless both predict `|Ia|/|Ib| > 1` at
the chosen edge point, opposite to the paper.

The immediate cause is visible in the raw Lumerical Q: `E||a` creates a
strong hotspot at the upper concave corner of the approximate polygon. The
same reversal in the expanded and reduced thermal models shows that the
production boundary expansion is not the primary cause.

## What remains unresolved

The exact Device-A CAD, electrode mask, beam location, and wavelength-specific
beam radius are not published numerically. We used a named polygon digitized
from Fig. 2A, `w0=6.5 µm`, and edge centre `(-8.5, 3.5) µm`. The off-axis
optical model excludes the 5-nm Ti/50-nm Au contacts because the selected
spot is away from the contacts. These approximations are sufficient for an
optical material check, but not for promotion as a quantitative experimental
PTE reproduction.

The calculated absolute current is also much larger than the paper's
order-100-pA map. Therefore it is not called an experimental prediction.
The next discriminating checks are exact CAD/spot metrology if available,
or a separate local half-plane edge geometry matching the paper's Fig. 3F
thermal idealization, plus optical/thermal mesh refinement of the `E||a`
corner hotspot.

## Model separation

The expanded case uses explicit bulk Si, SiO2 and air, `G_SiO2/Si=1.1e9`,
top `h=10`, far-x/y fixed DeltaT=0, and bottom fixed DeltaT=0. Those are the
current inverse-design production assumptions and are **not** claimed as
paper-supplied values.

The paper-reduced reference instead solves the flake with top/bottom Eq. S4
Robin conductances. The previous 2-D analytic/FEM work remains only
`PASSED_PAPER_EQUATION_MECHANISM_CONTROL`; this report supersedes any wording
that implied it was an actual Lumerical experimental reproduction.

## Files

- `DEVICE_A_IR_COUPLED_SANITY.png`
- `DEVICE_A_WEIGHTING_AND_PTE_DIAGNOSTICS.png`
- `device_a_ir_sanity_summary.json`
- `device_a_ir_sanity_cases.csv`
- `RAW_ARTIFACT_MANIFEST.json`
"""
    (args.output_dir / "DEVICE_A_IR_COUPLED_SANITY_REPORT.md").write_text(report)
    print(json.dumps(summary, indent=2))
    return 0 if status.startswith("VALIDATED") else 2


if __name__ == "__main__":
    raise SystemExit(main())
