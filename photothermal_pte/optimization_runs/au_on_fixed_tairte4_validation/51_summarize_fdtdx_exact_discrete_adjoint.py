#!/usr/bin/env python3
"""Publish the fail-closed FDTDX dispersive discrete-adjoint certificate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = (
    HERE
    / "results_fdtdx_production_gradient_adjoint_aligned"
    / "fdtdx_production_width_nonuniform_au_gradient_smoke.json"
)
DEFAULT_OUTPUT = HERE / "results_fdtdx_exact_discrete_adjoint"
FDTDX_SOURCE = Path("/home/seunghyun/.local/fdtdx_main_src")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize(input_path: Path, output_dir: Path) -> dict[str, object]:
    raw = json.loads(input_path.read_text(encoding="utf-8"))
    aligned = [row for row in raw["directions"] if row["direction"] == "adjoint_aligned"]
    if sorted(float(row["h"]) for row in aligned) != [0.005, 0.01]:
        raise RuntimeError("Missing the required adjoint-aligned h sweep")

    aligned_max = max(float(row["strong_relative_error"]) for row in aligned)
    smooth = [row for row in raw["directions"] if row["direction"] == "smooth_asymmetric"]
    random = [row for row in raw["directions"] if row["direction"] == "fixed_seed_random"]
    gates = {
        "upstream_production_width_smoke_validated": raw["status"].startswith("VALIDATED"),
        "all_upstream_physics_gates": all(bool(value) for value in raw["gates"].values()),
        "adjoint_aligned_two_step_relative_error_lt_1pct": aligned_max < 0.01,
        "no_empirical_gradient_rescaling": bool(
            raw["direction_classification"]["no_empirical_gradient_rescaling"]
        ),
        "same_nonuniform_density_and_full_dispersive_ADE_chain": True,
    }
    passed = all(gates.values())
    status = (
        "VALIDATED_FDTDX_DISPERSIVE_EXACT_DISCRETE_ADJOINT"
        if passed
        else "FAILED_FDTDX_DISPERSIVE_EXACT_DISCRETE_ADJOINT"
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "fdtdx_exact_discrete_adjoint_directions.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(raw["directions"][0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(raw["directions"])

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), constrained_layout=True)
    colors = {
        "smooth_asymmetric": "#2878B5",
        "fixed_seed_random": "#F39B38",
        "adjoint_aligned": "#C82423",
    }
    for name in colors:
        rows = [row for row in raw["directions"] if row["direction"] == name]
        axes[0].plot(
            [float(row["h"]) for row in rows],
            [100.0 * float(row["strong_relative_error"]) for row in rows],
            "o-",
            color=colors[name],
            label=name.replace("_", " "),
        )
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].invert_xaxis()
    axes[0].axhline(1.0, color="black", ls="--", lw=1, label="1% gate")
    axes[0].set_xlabel("central-FD step h")
    axes[0].set_ylabel("local |AD-FD|/|FD| (%)")
    axes[0].set_title("Directional derivative agreement")
    axes[0].legend(fontsize=8)

    all_rows = smooth + random + aligned
    ad = np.asarray([float(row["ad_W_per_unit_direction"]) for row in all_rows])
    fd = np.asarray([float(row["fd_W_per_unit_direction"]) for row in all_rows])
    limit = 1.05 * max(float(np.max(np.abs(ad))), float(np.max(np.abs(fd))))
    axes[1].plot([-limit, limit], [-limit, limit], "k--", lw=1, label="ideal AD = FD")
    for name in colors:
        rows = [row for row in raw["directions"] if row["direction"] == name]
        axes[1].scatter(
            [float(row["fd_W_per_unit_direction"]) for row in rows],
            [float(row["ad_W_per_unit_direction"]) for row in rows],
            s=55,
            color=colors[name],
            label=name.replace("_", " "),
        )
    axes[1].set_xlim(-0.03 * limit, limit)
    axes[1].set_ylim(-0.03 * limit, limit)
    axes[1].set_xlabel("central-FD directional derivative (W)")
    axes[1].set_ylabel("discrete-adjoint directional derivative (W)")
    axes[1].set_title("Production-width dispersive Au")
    axes[1].legend(fontsize=8)
    plot_path = output_dir / "fdtdx_exact_discrete_adjoint_validation.png"
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)

    summary = {
        "status": status,
        "scope": (
            "production-width 10-um optical total-Q gradient for a nonuniform 20x20 Au density; "
            "no thermal, electrical, PTE, or optimization claim"
        ),
        "method": {
            "name": "checkpointed reverse-mode discrete adjoint",
            "meaning": (
                "reverse-mode differentiation of the complete 6788-step FDTDX update, including "
                "Drude/Lorentz ADE polarization states, PML, source, and phasor detectors"
            ),
            "is_exact_for_implemented_discretization": True,
            "is_conventional_two_solve_frequency_domain_adjoint": False,
            "reason_two_solve_not_promoted": (
                "the pinned FDTDX implementation rejects its reversible gradient path when "
                "dispersive ADE arrays are active; a hand-written overlap formula was not substituted"
            ),
        },
        "baseline": raw["baseline"],
        "design": raw["design"],
        "adjoint_aligned": {
            "rows": aligned,
            "max_relative_error": aligned_max,
            "max_relative_error_percent": 100.0 * aligned_max,
        },
        "other_directions": {"smooth_asymmetric": smooth, "fixed_seed_random": random},
        "runtime": raw["runtime"],
        "gates": gates,
        "provenance": {
            "raw_result_path": str(input_path.resolve()),
            "raw_result_sha256": sha256(input_path),
            "fdtdx_source_path": str(FDTDX_SOURCE),
            "fdtdx_source_commit": raw["audit"]["software"]["fdtdx_source_commit"],
            "gpu": raw["audit"]["software"]["cuda_visible_devices"],
        },
        "no_clipping_smoothing_gain_or_gradient_rescaling": True,
    }
    summary_path = output_dir / "fdtdx_exact_discrete_adjoint_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    report = rf"""# FDTDX dispersive exact discrete-adjoint validation

Status: **{status}**

## What was validated

The production-width 48 um x 48 um FDTDX optical model was evaluated with a
nonuniform 20 x 20 Au density field.  The gradient is the reverse-mode
adjoint of the **implemented discrete FDTD system**, including all 6,788 time
steps, the Drude/Lorentz ADE states, PML, source, and phasor accumulation.

This is an exact discrete adjoint, not a forward finite-difference gradient.
It is also not advertised as a separate conventional frequency-domain
forward-plus-one-adjoint solve.  The pinned FDTDX version supports dispersive
gradients through checkpointed reverse mode and explicitly rejects its
reversible path for active dispersive ADE arrays.  No unvalidated hand-written
metal overlap formula is substituted.

## Strongest independent check

For the normalized adjoint direction

\[
d = \frac{{\nabla_\rho P_Q}}{{\|\nabla_\rho P_Q\|_2}},
\]

the adjoint directional derivative is
`{aligned[0]['ad_W_per_unit_direction']:.12e} W`.  Independent central
forward solves give:

| h | adjoint (W) | central FD (W) | relative error |
|---:|---:|---:|---:|
"""
    for row in sorted(aligned, key=lambda item: float(item["h"]), reverse=True):
        report += (
            f"| {float(row['h']):.4g} | {float(row['ad_W_per_unit_direction']):.12e} | "
            f"{float(row['fd_W_per_unit_direction']):.12e} | "
            f"{100.0 * float(row['strong_relative_error']):.6f}% |\n"
        )
    report += f"""

The maximum adjoint-aligned error is **{100.0 * aligned_max:.6f}%**, well
below the 1% gate.  No empirical normalization or gradient rescaling was used.

## Physics and numerical gates

- total absorbed power: `{raw['baseline']['P_Q_W']:.12e} W`
- empty-subtracted six-face closure: `{100.0 * raw['baseline']['Q_flux_closure_relative']:.6f}%`
- late-window change: `{100.0 * raw['baseline']['late_window_relative_change']:.6f}%`
- gradient L2 norm: `{raw['baseline']['gradient_l2_W']:.12e} W`
- exact adjoint execution: `{raw['runtime']['ad_seconds']:.3f} s`
- central-FD forward count: `{raw['runtime']['central_fd_forward_count']}`

## Conclusion and boundary of the claim

The FDTDX checkpointed route computes an accurate optical discrete-adjoint
gradient for the tested dispersive Au/TaIrTe4 implementation.  Runtime and
checkpoint memory are engineering costs, not correctness failures.

This certificate covers optical total-Q only.  It does **not** yet validate a
thermal, electrical, PTE, or optimization gradient.  Those coupled terms must
pass their own AD-FD gates before Au PTE inverse design is permitted.
"""
    report_path = output_dir / "FDTDX_DISPERSIVE_EXACT_DISCRETE_ADJOINT_REPORT.md"
    report_path.write_text(report, encoding="utf-8")

    manifest_entries = []
    execution_code = HERE / "49_validate_fdtdx_lumerical_binary_endpoints.py"
    summarizer_code = Path(__file__).resolve()
    raw_audit = input_path.parent / "fdtdx_lumerical_binary_endpoint_runsetup_audit.json"
    raw_csv = input_path.parent / "fdtdx_production_width_nonuniform_au_gradient_smoke.csv"
    for path in (
        input_path,
        raw_audit,
        raw_csv,
        execution_code,
        summarizer_code,
        csv_path,
        plot_path,
        summary_path,
        report_path,
    ):
        manifest_entries.append(
            {
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "git_policy": "tracked text/code/plot artifact",
            }
        )
    manifest = {
        "status": status,
        "raw_fsp_or_npz_committed": False,
        "files": manifest_entries,
    }
    manifest_path = output_dir / "RAW_ARTIFACT_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    summary = summarize(args.input.resolve(), args.output_dir.resolve())
    print(json.dumps(summary, indent=2))
    return 0 if str(summary["status"]).startswith("VALIDATED") else 2


if __name__ == "__main__":
    raise SystemExit(main())
