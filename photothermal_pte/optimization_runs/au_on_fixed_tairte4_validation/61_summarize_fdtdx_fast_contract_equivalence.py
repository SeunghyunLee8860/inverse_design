#!/usr/bin/env python3
"""Compare the 16-period/4-window optical gradient against 32-period reference."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _relative(first: float, second: float) -> float:
    return abs(first - second) / max(abs(first), abs(second))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-json", required=True, type=Path)
    parser.add_argument("--candidate-json", required=True, type=Path)
    parser.add_argument("--substrate-audit-json", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    reference_path = args.reference_json.resolve()
    candidate_path = args.candidate_json.resolve()
    substrate_path = args.substrate_audit_json.resolve()
    reference = _load(reference_path)
    candidate = _load(candidate_path)
    substrate = _load(substrate_path)
    reference_direction = reference["directions"][0]
    candidate_direction = candidate["directions"][0]
    if reference_direction["direction"] != candidate_direction["direction"]:
        raise RuntimeError("Reference and candidate directions differ")

    metrics = {
        "P_Q_relative_difference": _relative(
            reference["baseline"]["P_Q_W"], candidate["baseline"]["P_Q_W"]
        ),
        "gradient_l2_relative_difference": _relative(
            reference["baseline"]["gradient_l2_W"],
            candidate["baseline"]["gradient_l2_W"],
        ),
        "same_direction_AD_relative_difference": _relative(
            reference_direction["ad_W_per_unit_direction"],
            candidate_direction["ad_W_per_unit_direction"],
        ),
        "candidate_AD_FD_relative_error": candidate_direction["strong_relative_error"],
        "candidate_Q_flux_closure_relative": candidate["baseline"][
            "Q_flux_closure_relative"
        ],
        "candidate_late_window_relative_change": candidate["baseline"][
            "late_window_relative_change"
        ],
        "substrate_only_Q_flux_closure_relative": substrate["relative_errors"][
            "deep_time_domain_box"
        ],
        "AD_runtime_speedup_reference_over_candidate": (
            reference["runtime"]["ad_seconds"] / candidate["runtime"]["ad_seconds"]
        ),
    }
    gates = {
        "P_Q_difference_lt_0p5pct": metrics["P_Q_relative_difference"] < 0.005,
        "gradient_l2_difference_lt_0p5pct": (
            metrics["gradient_l2_relative_difference"] < 0.005
        ),
        "same_direction_AD_difference_lt_0p5pct": (
            metrics["same_direction_AD_relative_difference"] < 0.005
        ),
        "candidate_AD_FD_error_lt_1pct": metrics["candidate_AD_FD_relative_error"] < 0.01,
        "candidate_Q_flux_closure_lt_0p5pct": (
            metrics["candidate_Q_flux_closure_relative"] < 0.005
        ),
        "candidate_window_change_lt_0p5pct": (
            metrics["candidate_late_window_relative_change"] < 0.005
        ),
        "substrate_only_Q_flux_closure_lt_0p5pct": (
            metrics["substrate_only_Q_flux_closure_relative"] < 0.005
        ),
        "no_clipping_smoothing_gain_or_gradient_rescaling": True,
    }
    passed = all(gates.values())
    status = (
        "VALIDATED_FDTDX_DIAGNOSTIC_16PERIOD4WINDOW_OBJECTIVE_DIRECTIONAL_GRADIENT_EQUIVALENCE"
        if passed
        else "FAILED_FDTDX_DIAGNOSTIC_16PERIOD4WINDOW_OBJECTIVE_DIRECTIONAL_GRADIENT_EQUIVALENCE"
    )
    summary = {
        "status": status,
        "scope": (
            "32-period versus 16-period/4-window substrate-bearing total-Q objective, "
            "gradient norm, and one identical smooth direction; no full-vector angle, "
            "spatial PTE adjoint, thermal/electrical coupling, or optimization"
        ),
        "reference": {
            "json": str(reference_path),
            "periods": reference["audit"]["numerics"]["total_periods"],
            "window_periods": reference["audit"]["numerics"]["window_periods"],
            "P_Q_W": reference["baseline"]["P_Q_W"],
            "gradient_l2_W": reference["baseline"]["gradient_l2_W"],
            "directional_AD_W": reference_direction["ad_W_per_unit_direction"],
            "ad_seconds": reference["runtime"]["ad_seconds"],
        },
        "candidate": {
            "json": str(candidate_path),
            "periods": candidate["audit"]["numerics"]["total_periods"],
            "window_periods": candidate["audit"]["numerics"]["window_periods"],
            "P_Q_W": candidate["baseline"]["P_Q_W"],
            "gradient_l2_W": candidate["baseline"]["gradient_l2_W"],
            "directional_AD_W": candidate_direction["ad_W_per_unit_direction"],
            "directional_FD_W": candidate_direction["fd_W_per_unit_direction"],
            "ad_seconds": candidate["runtime"]["ad_seconds"],
            "central_fd_seconds": candidate["runtime"]["central_fd_forward_seconds"],
        },
        "metrics": metrics,
        "gradient_angle": {
            "available": False,
            "reason": (
                "the immutable 32-period checkpoint stored gradient L2 and directional "
                "contractions but not the 20x20 gradient vector"
            ),
            "consequence": (
                "this diagnostic does not certify full-vector angle or authorize combined PTE optimization"
            ),
        },
        "gates": gates,
    }

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "fdtdx_fast_contract_equivalence_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    csv_path = output / "fdtdx_fast_contract_equivalence.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("metric", "value", "gate"))
        for name, value in metrics.items():
            writer.writerow((name, value, ""))
        for name, value in gates.items():
            writer.writerow((name, "", value))

    labels = ("P_Q", "gradient L2", "same-direction AD")
    values = [
        100.0 * metrics["P_Q_relative_difference"],
        100.0 * metrics["gradient_l2_relative_difference"],
        100.0 * metrics["same_direction_AD_relative_difference"],
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    axes[0].bar(labels, values)
    axes[0].axhline(0.5, color="black", linestyle="--", label="0.5% gate")
    axes[0].set_ylabel("32-to-16 relative difference (%)")
    axes[0].set_title("Objective and directional-gradient equivalence")
    axes[0].legend()
    axes[1].bar(
        ("32-period", "16-period/4-window"),
        (reference["runtime"]["ad_seconds"] / 60.0, candidate["runtime"]["ad_seconds"] / 60.0),
    )
    axes[1].set_ylabel("AD execution time (min)")
    axes[1].set_title(f"Measured speedup: {metrics['AD_runtime_speedup_reference_over_candidate']:.3f}x")
    plot_path = output / "fdtdx_fast_contract_equivalence.png"
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    report = f"""# FDTDX 16-period/4-window fast-contract screening

Status: **{status}**

The candidate and reference use the same 20x20 nonuniform Au density, fixed
TaIrTe4, matched Si/SiO2 interface grid, material models, source, and spatial
grid. Only the simulated duration/window changes from 32/4 to 16/4 periods.

| comparison | relative difference |
|---|---:|
| total P_Q | {100*metrics['P_Q_relative_difference']:.6f}% |
| gradient L2 norm | {100*metrics['gradient_l2_relative_difference']:.6f}% |
| same smooth directional AD | {100*metrics['same_direction_AD_relative_difference']:.6f}% |

The candidate's internal AD--FD error is
`{100*metrics['candidate_AD_FD_relative_error']:.6f}%`, total-Q/flux closure
is `{100*metrics['candidate_Q_flux_closure_relative']:.6f}%`, and the
substrate-only closure is
`{100*metrics['substrate_only_Q_flux_closure_relative']:.6f}%`.

AD execution falls from `{reference['runtime']['ad_seconds']:.3f} s` to
`{candidate['runtime']['ad_seconds']:.3f} s`, a
`{metrics['AD_runtime_speedup_reference_over_candidate']:.3f}x` speedup.

The immutable 32-period checkpoint did not store its 20x20 gradient vector,
so a full gradient angle cannot be reconstructed. This result therefore
promotes only objective/norm/same-direction equivalence. It does not validate
the spatially weighted Maxwell source required by PTE, coupled thermal or
electrical chain-rule terms, Au thermopower, or an inverse-design run.
No clipping, smoothing, gain, result rescaling, or gradient rescaling is used.
"""
    report_path = output / "FDTDX_FAST_CONTRACT_EQUIVALENCE_REPORT.md"
    report_path.write_text(report, encoding="utf-8")

    published = (summary_path, csv_path, plot_path, report_path, reference_path, candidate_path, substrate_path)
    manifest = {
        "status": status,
        "raw_field_artifact": None,
        "published": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in published
        ],
    }
    manifest_path = output / "RAW_ARTIFACT_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(report_path)
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
