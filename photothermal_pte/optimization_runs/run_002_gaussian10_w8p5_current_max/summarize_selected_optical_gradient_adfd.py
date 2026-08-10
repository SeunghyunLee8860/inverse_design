#!/usr/bin/env python3
"""Publish the selected-grid optical-gradient normalization correction."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, object]:
    value = path.expanduser().resolve()
    return {
        "path": str(value),
        "size_bytes": value.stat().st_size,
        "sha256": sha256(value),
    }


def relative(left: float, right: float) -> float:
    return abs(left - right) / max(abs(left), abs(right), np.finfo(float).tiny)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--optical-diagnostic", type=Path, required=True)
    parser.add_argument("--scalar-diagnostic", type=Path, required=True)
    parser.add_argument("--combined-decomposition", type=Path, required=True)
    parser.add_argument("--results-directory", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    args = parser.parse_args()

    optical_path = args.optical_diagnostic.expanduser().resolve()
    scalar_path = args.scalar_diagnostic.expanduser().resolve()
    decomposition_path = args.combined_decomposition.expanduser().resolve()
    optical = json.loads(optical_path.read_text())
    scalar = json.loads(scalar_path.read_text())
    decomposition = json.loads(decomposition_path.read_text())
    gradient_path = Path(optical["artifacts"]["corrected_gradient_NPZ"]["path"])
    if sha256(gradient_path) != optical["artifacts"]["corrected_gradient_NPZ"]["sha256"]:
        raise RuntimeError("corrected gradient NPZ SHA mismatch")
    gradient = np.load(gradient_path)
    optical_gradient = np.asarray(gradient["gradient_total_optical_A"], float)

    optical_ad = float(optical["directional_derivatives_A"]["total_optical_AD"])
    optical_fd = float(optical["directional_derivatives_A"]["total_optical_FD"])
    thermal_ad = float(
        decomposition["directional_derivatives_A"]["AD_thermal_material"]
    )
    combined_fd = float(decomposition["directional_derivatives_A"]["FD_combined"])
    combined_ad = optical_ad + thermal_ad
    optical_error = relative(optical_ad, optical_fd)
    combined_error = relative(combined_ad, combined_fd)
    scalar_error = float(
        scalar["relative_errors"]["named_source_corrected_total_vs_FD_total"]
    )
    normalization_residual = float(
        optical["named_source_normalization"][
            "two_normalization_state_spatial_residual"
        ]
    )
    passed = bool(
        optical_error < 0.01
        and combined_error < 0.01
        and scalar_error < 0.01
        and normalization_residual < 1.0e-12
        and float(optical["coordinate_mismatch_m"]) < 2.0e-18
    )
    if not passed:
        raise RuntimeError("selected optical-gradient publication gates failed")

    output = args.results_directory.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "selected_optical_gradient_adfd_summary.json"
    csv_path = output / "selected_optical_gradient_adfd_cases.csv"
    report_path = output / "SELECTED_OPTICAL_GRADIENT_ADFD_REPORT.md"
    plot_path = output / "selected_optical_gradient_adfd.png"
    generated = datetime.now(timezone.utc).isoformat()

    rows = [
        {
            "case": "scalar_P_Q_named_source_corrected",
            "AD": scalar["named_source_normalization"][
                "corrected_total_directional_W"
            ],
            "FD": scalar["directional_derivatives_W"]["FD_total"],
            "relative_error": scalar_error,
            "unit": "W",
            "gate": 0.01,
            "passed": scalar_error < 0.01,
        },
        {
            "case": "spatially_weighted_PTE_optical",
            "AD": optical_ad,
            "FD": optical_fd,
            "relative_error": optical_error,
            "unit": "A",
            "gate": 0.01,
            "passed": optical_error < 0.01,
        },
        {
            "case": "combined_physical_rho_one_direction_recomputed",
            "AD": combined_ad,
            "FD": combined_fd,
            "relative_error": combined_error,
            "unit": "A",
            "gate": 0.01,
            "passed": combined_error < 0.01,
        },
    ]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)

    diagnostic_errors = {
        "original_first_source_CW": float(decomposition["relative_errors"]["optical"]),
        "exact_inverse_first_source_CW": 0.03208284171239694,
        "multiplier_common_first_source_CW": 0.251269,
        "FieldRegion_only_CW": optical_error,
    }
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    labels = ["optical", "combined"]
    ad_values = np.asarray([optical_ad, combined_ad]) / 1.0e-20
    fd_values = np.asarray([optical_fd, combined_fd]) / 1.0e-20
    x = np.arange(len(labels))
    axes[0].bar(x - 0.18, ad_values, 0.36, label="AD")
    axes[0].bar(x + 0.18, fd_values, 0.36, label="FD")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel(r"directional derivative ($10^{-20}$ A)")
    axes[0].set_title("Selected-grid directional AD–FD")
    axes[0].legend()

    names = list(diagnostic_errors)
    errors = [100.0 * diagnostic_errors[name] for name in names]
    axes[1].barh(np.arange(len(names)), errors)
    axes[1].axvline(1.0, color="black", linestyle="--", label="1% gate")
    axes[1].set_yticks(
        np.arange(len(names)),
        [
            "old CW(first)",
            "inverse collocation",
            "multiplier common",
            "FieldRegion-only CW",
        ],
    )
    axes[1].set_xscale("log")
    axes[1].set_xlabel("relative error (%)")
    axes[1].set_title("Normalization/source diagnostics")
    axes[1].legend()

    image = axes[2].imshow(
        optical_gradient.T,
        origin="lower",
        cmap="coolwarm",
        aspect="equal",
    )
    axes[2].set_title("Corrected optical gradient (373×373)")
    axes[2].set_xlabel("design node x")
    axes[2].set_ylabel("design node y")
    fig.colorbar(image, ax=axes[2], label="A per physical-density node")
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    summary = {
        "status": "VALIDATED_SELECTED_OPTICAL_GRADIENT_ADFD",
        "passed": True,
        "generated_at_utc": generated,
        "scope": (
            "one selected 373x373 nonuniform physical-density direction at "
            "h=0.005; optical gradient plus corrected one-direction combined smoke"
        ),
        "optimizer_started": False,
        "new_Maxwell_solves_for_final_correction": 0,
        "empirical_normalization": False,
        "gradient_rescaling": False,
        "root_cause": (
            "the zero-amplitude forward Gaussian remained active as a mesh anchor, "
            "so default cwnorm(1) normalized the FieldRegion adjoint monitor to the "
            "wrong active source spectrum"
        ),
        "correction": optical["named_source_normalization"],
        "directional_derivatives_A": {
            "optical_indirect": optical["directional_derivatives_A"][
                "indirect_field_mediated"
            ],
            "optical_direct": optical["directional_derivatives_A"][
                "direct_material_loss"
            ],
            "optical_AD": optical_ad,
            "optical_FD": optical_fd,
            "thermal_material_AD_reused": thermal_ad,
            "combined_AD_recomputed": combined_ad,
            "combined_FD_reused": combined_fd,
        },
        "gates": {
            "scalar_P_Q_AD_FD_relative_error": scalar_error,
            "optical_AD_FD_relative_error": optical_error,
            "combined_AD_FD_relative_error": combined_error,
            "limit": 0.01,
            "coordinate_mismatch_m": optical["coordinate_mismatch_m"],
            "normalization_spatial_residual": normalization_residual,
            "normalization_residual_limit": 1.0e-12,
            "mapping_transpose_error": optical["operator"][
                "fresh_transpose_dot_error"
            ],
        },
        "diagnostic_errors": diagnostic_errors,
        "remaining_before_optimization": [
            "selected-grid broader combined physical-rho directions and FD steps",
            "coupled optical gray-law sensitivity",
            "exact-binary DRC fixtures",
            "full latent/filter/projection AD-FD",
        ],
        "raw_artifacts": {
            "optical_diagnostic": artifact(optical_path),
            "corrected_gradient_NPZ": artifact(gradient_path),
            "scalar_decomposition": artifact(scalar_path),
            "combined_decomposition": artifact(decomposition_path),
            "adjoint_FSP": artifact(
                Path(optical["artifacts"]["adjoint_FSP"]["path"])
            ),
            "base_FSP": artifact(Path(optical["artifacts"]["base_FSP"]["path"])),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    report_path.write_text(
        f"# Selected production optical-gradient AD–FD\n\n"
        f"Status: `VALIDATED_SELECTED_OPTICAL_GRADIENT_ADFD`\n\n"
        f"The selected 373×373 physical-density optical gradient now passes. "
        f"No new Maxwell solve, thermal solve, or optimization iteration was used "
        f"for the final correction.\n\n"
        f"## Result\n\n"
        f"| quantity | AD | FD | relative error | gate |\n"
        f"|---|---:|---:|---:|---:|\n"
        f"| scalar $P_Q$ control | {float(scalar['named_source_normalization']['corrected_total_directional_W']):.12e} W | "
        f"{float(scalar['directional_derivatives_W']['FD_total']):.12e} W | {scalar_error:.6e} | <1% |\n"
        f"| spatially weighted optical PTE | {optical_ad:.12e} A | {optical_fd:.12e} A | {optical_error:.6e} | <1% |\n"
        f"| corrected one-direction combined smoke | {combined_ad:.12e} A | {combined_fd:.12e} A | {combined_error:.6e} | <1% |\n\n"
        f"The optical terms are indirect `{optical['directional_derivatives_A']['indirect_field_mediated']:.12e} A` "
        f"and direct material loss `{optical['directional_derivatives_A']['direct_material_loss']:.12e} A`.\n\n"
        f"## Root cause and correction\n\n"
        f"The forward Gaussian must remain active with zero amplitude to preserve "
        f"the exact forward auto-nonuniform mesh. That left two active sources in "
        f"the adjoint project, while default `cwnorm(1)` normalized monitor fields "
        f"to the zero-amplitude Gaussian source spectrum instead of the FieldRegion "
        f"spectrum. The FieldRegion-only CW field is reconstructed from the same raw "
        f"monitor data under official `cwnorm(1)` and `cwnorm(2)` states. The two-state "
        f"spatial residual is `{normalization_residual:.6e}`. No FD-derived scale, "
        f"empirical normalization, or gradient rescaling is used.\n\n"
        f"## Scope\n\n"
        f"This validates one selected-grid physical-density direction at `h=0.005`. "
        f"It does not yet validate broader combined directions, optical gray-law "
        f"sensitivity, exact-binary DRC, full latent AD–FD, or optimization.\n"
    )

    manifest_path = args.manifest.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text())
    manifest["selected_optical_gradient_adfd"] = {
        "status": summary["status"],
        "raw_artifacts_committed_to_git": False,
        **summary["raw_artifacts"],
    }
    manifest["current_promoted_status"] = summary["status"]
    manifest["current_promoted_at_utc"] = generated
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    status_path = args.status.expanduser().resolve()
    status = json.loads(status_path.read_text())
    status.update(
        {
            "status": summary["status"],
            "last_updated_utc": generated,
            "optimization_started": False,
            "message": (
                "Selected-grid optical AD-FD is validated after reconstructing "
                "the FieldRegion-only CW adjoint from official cwnorm(1)/cwnorm(2) "
                f"states. Optical error={optical_error:.3e}, corrected one-direction "
                f"combined error={combined_error:.3e}, scalar P_Q error={scalar_error:.3e}. "
                "No empirical rescaling was used. Broader combined directions, "
                "coupled optical gray-law sensitivity, exact-binary DRC, and full "
                "latent AD-FD still block optimization."
            ),
        }
    )
    status_path.write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
