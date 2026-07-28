#!/usr/bin/env python3
"""Publish compact reports and figures from immutable gray/latent raw results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gray-result", required=True)
    parser.add_argument("--latent-result", required=True)
    parser.add_argument("--report-dir", required=True)
    return parser.parse_args()


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def artifact(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "byte_size": path.stat().st_size,
        "sha256": digest(path),
    }


def percent(value: float) -> float:
    return 100.0 * float(value)


def main() -> int:
    args = parse_args()
    gray_path = Path(args.gray_result).expanduser().resolve()
    latent_path = Path(args.latent_result).expanduser().resolve()
    gray = json.loads(gray_path.read_text())
    latent = json.loads(latent_path.read_text())
    if gray["status"] != "COMPLETED_COUPLED_GRAY_LAW_SENSITIVITY":
        raise RuntimeError("gray result is not complete")
    expected = (
        "VALIDATED_FULL_LATENT_COMBINED_PTE_ADFD_"
        "WITH_USER_ACCEPTED_FD_NOISE"
    )
    if latent["status"] != expected or not latent["passed"]:
        raise RuntimeError("full latent result is not validated")
    report_dir = Path(args.report_dir).expanduser().resolve()
    figures = report_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    gray_npz = Path(gray["arrays"]["path"]).resolve()
    latent_npz = Path(latent["arrays"]["path"]).resolve()
    arrays = np.load(latent_npz)

    directions = list(latent["directions"])
    colors = {"4um": "#1f77b4", "6um": "#ff7f0e"}
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
    all_values = []
    for scenario, color in colors.items():
        ad = np.asarray(
            [
                latent["scenarios"][scenario]["directions"][name][
                    "analytic_directional_A"
                ]
                for name in directions
            ]
        )
        fd = np.asarray(
            [
                next(
                    row["finite_difference_directional_A"]
                    for row in latent["scenarios"][scenario]["directions"][
                        name
                    ]["steps"]
                    if np.isclose(row["step"], 0.005)
                )
                for name in directions
            ]
        )
        axes[0].scatter(fd, ad, s=58, label=scenario, color=color)
        for index, name in enumerate(directions):
            axes[0].annotate(str(index + 1), (fd[index], ad[index]))
        all_values.extend(ad.tolist() + fd.tolist())
        errors = [
            percent(
                latent["scenarios"][scenario]["directions"][name][
                    "selected_relative_error"
                ]
            )
            for name in directions
        ]
        offset = -0.18 if scenario == "4um" else 0.18
        axes[1].bar(
            np.arange(len(directions)) + offset,
            errors,
            width=0.36,
            color=color,
            label=scenario,
        )
    lower, upper = min(all_values), max(all_values)
    padding = 0.04 * (upper - lower)
    axes[0].plot(
        [lower - padding, upper + padding],
        [lower - padding, upper + padding],
        "k--",
        label="ideal AD = FD",
    )
    axes[0].set_xlabel("Finite-difference directional derivative [A]")
    axes[0].set_ylabel("Adjoint directional derivative [A]")
    axes[0].set_title("Full latent AD–FD parity, h=0.005")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].axhline(1.0, color="k", linestyle="--", label="1% gate")
    axes[1].set_xticks(
        np.arange(len(directions)),
        [name.replace("_", "\n") for name in directions],
        rotation=18,
        ha="right",
    )
    axes[1].set_yscale("log")
    axes[1].set_ylabel("AD–FD relative error [%]")
    axes[1].set_title("Five-direction selected-step errors")
    axes[1].legend()
    axes[1].grid(alpha=0.25, axis="y")
    fig.tight_layout()
    parity_path = figures / "18_full_latent_adfd_parity.png"
    fig.savefig(parity_path, dpi=190)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.2))
    entries = [
        ("latent", "latent"),
        ("physical_rho", "projected physical density"),
        ("latent_gradient_4um_A", "latent gradient, 4 µm"),
        ("latent_gradient_6um_A", "latent gradient, 6 µm"),
        ("physical_gradient_4um_A", "physical gradient, 4 µm"),
        ("physical_gradient_6um_A", "physical gradient, 6 µm"),
    ]
    for axis, (key, title) in zip(axes.flat, entries):
        image = axis.imshow(
            arrays[key].T,
            origin="lower",
            extent=(-1, 1, -1, 1),
            aspect="equal",
            cmap="viridis" if "gradient" not in key else "coolwarm",
        )
        axis.set_title(title)
        axis.set_xlabel("x [µm]")
        axis.set_ylabel("y [µm]")
        fig.colorbar(image, ax=axis, shrink=0.78)
    fig.tight_layout()
    maps_path = figures / "19_full_latent_gradient_maps.png"
    fig.savefig(maps_path, dpi=190)
    plt.close(fig)

    exponents = [1, 2, 3]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.4))
    axes[0].plot(
        exponents,
        [gray["cases"][f"p{p}"]["P_Q_W"] for p in exponents],
        "o-",
    )
    axes[0].set_title("Absorbed power")
    axes[0].set_xlabel("gray exponent p")
    axes[0].set_ylabel("$P_Q$ [W]")
    for scenario, color in colors.items():
        axes[1].plot(
            exponents,
            [
                gray["cases"][f"p{p}"]["scenarios"][scenario][
                    "objective_A"
                ]
                for p in exponents
            ],
            "o-",
            color=color,
            label=scenario,
        )
    axes[1].set_title("PTE objective")
    axes[1].set_xlabel("gray exponent p")
    axes[1].set_ylabel("objective [A]")
    axes[1].legend()
    for label, marker in (("p2_vs_p1", "o"), ("p3_vs_p1", "s")):
        axes[2].scatter(
            [4, 6],
            [
                gray["comparisons"][label]["scenarios"][scenario][
                    "nominal_gradient_angle_deg"
                ]
                for scenario in ("4um", "6um")
            ],
            marker=marker,
            s=65,
            label=label.replace("_", " "),
        )
    axes[2].set_title("Gradient-direction sensitivity")
    axes[2].set_xlabel("thermal flake span [µm]")
    axes[2].set_ylabel("angle from p=1 [deg]")
    axes[2].legend()
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.tight_layout()
    gray_path_figure = figures / "20_gray_law_sensitivity.png"
    fig.savefig(gray_path_figure, dpi=190)
    plt.close(fig)

    rows = []
    for scenario in ("4um", "6um"):
        for name in directions:
            item = latent["scenarios"][scenario]["directions"][name]
            selected = next(
                row
                for row in item["steps"]
                if np.isclose(row["step"], 0.005)
            )
            rows.append(
                {
                    "scenario": scenario,
                    "direction": name,
                    "step": 0.005,
                    "AD_A": item["analytic_directional_A"],
                    "FD_A": selected["finite_difference_directional_A"],
                    "relative_error": selected["relative_error"],
                }
            )
    csv_path = report_dir / "final_full_latent_pte_adfd_cases.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "status": latent["status"],
        "passed": True,
        "accepted_exception": latent["accepted_exception"],
        "latent_contract": latent["latent_contract"],
        "gates": latent["gates"],
        "scenario_metrics": {
            scenario: {
                key: value
                for key, value in latent["scenarios"][scenario].items()
                if key
                in {
                    "base_objective_A",
                    "physical_gradient_L2_A",
                    "latent_gradient_L2_A",
                    "optical_gradient_L2_A",
                    "thermal_gradient_L2_A",
                    "directional_subspace_normalized_error",
                    "directional_subspace_gradient_angle_deg",
                }
            }
            for scenario in ("4um", "6um")
        },
        "gray_law": {
            "status": gray["status"],
            "definition": gray["gray_laws"],
            "comparisons": gray["comparisons"],
        },
        "optimization_run": False,
    }
    summary_path = report_dir / "final_full_latent_pte_adfd_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    manifest = {
        "status": latent["status"],
        "generation_command": (
            "python -m photothermal_pte.finite_inverse_design."
            "publish_final_latent_gray_adfd --gray-result <external> "
            "--latent-result <external> --report-dir "
            "photothermal_pte/reports/inverse_design_pte_adfd"
        ),
        "raw_artifacts_not_committed": [
            artifact(gray_path),
            artifact(gray_npz),
            artifact(latent_path),
            artifact(latent_npz),
        ],
        "published_artifacts": [
            str(summary_path),
            str(csv_path),
            str(parity_path),
            str(maps_path),
            str(gray_path_figure),
        ],
    }
    manifest_path = (
        report_dir / "FINAL_FULL_LATENT_PTE_ADFD_RAW_ARTIFACT_MANIFEST.json"
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    gates = latent["gates"]
    report = f"""# Final full-latent combined PTE AD–FD

**Status: `{latent["status"]}`**

This is the end-to-end finite, nonperiodic certificate:

`81x81 latent -> 500 nm finite conic filter -> beta=8 tanh projection ->
component-specific Yee material Jacobian -> Maxwell Q -> conservative thermal
remap -> explicit anisotropic/material/interface thermal solve -> uniform-45°
PTE objective`.

No clipping, periodic wrapping, empirical normalization, gradient rescaling,
gain, Q smoothing, or Q rescaling was used. Optimization was not run.

## User-approved exception

The earlier physical-rho near-null strict plateau failure remains immutable.
The user explicitly accepted that solver-noise-level miss and authorized the
next stages. The present certificate therefore does not relabel the earlier
checkpoint; it uses the core 1% parity/angle/conservation gates without making
strict h-to-h/2 plateau convergence a gate.

## Full latent result

| metric | result | gate |
|---|---:|---:|
| worst strong-direction error | {percent(gates["worst_strong_direction_relative_error"]):.6f}% | <1% |
| global five-direction normalized error | {percent(gates["global_multidirection_normalized_error"]):.6f}% | <1% |
| global directional gradient angle | {gates["global_directional_gradient_angle_deg"]:.6f}° | <1° |
| finite filter/projection transpose error | {gates["finite_mapping_transpose_relative_error"]:.3e} | <1e-12 |
| component-Yee transpose error | {gates["component_yee_mapping_transpose_relative_error"]:.3e} | <1e-12 |
| worst optical six-face closure | {percent(gates["worst_optical_closure_relative_error"]):.6f}% | <0.5% |
| worst Q mapping error | {gates["worst_Q_mapping_relative_error"]:.3e} | <0.5% |
| worst thermal energy-balance error | {gates["worst_thermal_energy_balance_relative_error"]:.3e} | <1% |
| worst linear residual | {gates["worst_linear_residual_relative"]:.3e} | <1e-8 |

At `h=0.005`, all five directions (adjoint-aligned, central-localized,
design-edge-localized, smooth-asymmetric, and fixed-seed random) pass for both
4 and 6 µm named thermal flake scenarios. The maximum individual selected
error is {percent(max(row["relative_error"] for row in rows)):.6f}%.

## Gray-law sensitivity

`phi_p(rho)=rho^p`, for `p=1,2,3`, was applied consistently to optical
permittivity, bulk thermal conductivity, and interface conductance via the
same effective fraction. These are numerical scenarios, not a confidence
interval. Relative to `p=1`, `p=2` changes P_Q by
{percent(gray["comparisons"]["p2_vs_p1"]["P_Q_relative_change"]):.3f}% and
the PTE gradients by about 7.6–7.9°. `p=3` changes P_Q by
{percent(gray["comparisons"]["p3_vs_p1"]["P_Q_relative_change"]):.3f}% and
the gradients by about 14.4–14.8°. Gray-law choice is therefore a material
model uncertainty and must remain explicit during optimization.

## Figures

- [Full latent parity](figures/18_full_latent_adfd_parity.png)
- [Latent and gradient maps](figures/19_full_latent_gradient_maps.png)
- [Gray-law sensitivity](figures/20_gray_law_sensitivity.png)

Raw NPZ/FSP files remain outside Git and are SHA-pinned by the manifest.
"""
    report_path = report_dir / "FINAL_FULL_LATENT_PTE_ADFD_REPORT.md"
    report_path.write_text(report)
    latest = report_dir / "LATEST_STATUS.md"
    previous = latest.read_text() if latest.is_file() else ""
    marker = "# Final full-latent combined PTE AD–FD\n"
    if not previous.startswith(marker):
        latest.write_text(
            marker
            + "\n"
            + f"- Status: `{latent['status']}`\n"
            + "- Gray-law sensitivity: completed; materially non-negligible.\n"
            + "- Optimization: not run; requires a separate user decision on "
            + "the production gray-law scenario.\n"
            + "- Report: `FINAL_FULL_LATENT_PTE_ADFD_REPORT.md`\n\n"
            + previous
        )
    print(json.dumps({"status": latent["status"], "report": str(report_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
