#!/usr/bin/env python3
"""Publish the fixed-external-field Au boundary-kernel AD--FD diagnostic."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
DEFAULT_RAW = Path("/home/seunghyun/tairte4/raw_artifacts/au_topology_validation")
DEFAULT_CASE = "sharp_width_8p0_edge25_external_field_adjoint_gpu0"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def raw_entry(role: str, path: Path) -> dict[str, object]:
    return {
        "role": role,
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--case", default=DEFAULT_CASE)
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    args = parser.parse_args()

    case = args.raw_root / args.case
    raw_result = case / "au_sharp_interface_external_field_result.json"
    source = json.loads(raw_result.read_text())
    if source.get("passed", False):
        raise RuntimeError("failure summarizer received a passing result")
    if source.get("status") != (
        "FAILED_AU_SHARP_INTERFACE_EXTERNAL_FIELD_BOUNDARY_KERNEL_ADFD"
    ):
        raise RuntimeError(f"unexpected source status: {source.get('status')}")

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    status = "BLOCKED_AU_SHARP_INTERFACE_BOUNDARY_QUADRATURE_UNRESOLVED"
    fd = source["finite_difference"]
    comparisons = source["AD_FD_comparison"]
    strong_fd = float(fd["h_0.05_um"]["derivative_J_proxy_per_um"])
    ad = float(comparisons["h_0.05_um"]["AD_J_proxy_per_um"])
    strong_error = float(comparisons["h_0.05_um"]["relative_error"])
    quadrature = source["boundary_quadrature"]

    summary = {
        "status": status,
        "diagnostic_status": source["status"],
        "scope": source["scope"],
        "objective": source["objective"],
        "finite_difference": fd,
        "FD_step_plateau_relative_change": source[
            "FD_step_plateau_relative_change"
        ],
        "strong_h0p05_step_fraction_of_baseline": source[
            "strong_h0p05_step_fraction_of_baseline"
        ],
        "AD_FD_comparison": comparisons,
        "boundary_quadrature": quadrature,
        "boundary_quadrature_final_relative_change": source[
            "boundary_quadrature_final_relative_change"
        ],
        "material_readback": source["material_readback"],
        "source": source["source"],
        "adjoint": source["adjoint"],
        "gates": source["gates"],
        "interpretation": {
            "validated": [
                "fixed external objective has no moving-domain or direct material-loss term",
                "central-FD h=0.10 to 0.05 um plateau",
                "GPU FieldRegion source exact round trip",
                "forward/adjoint component-grid coordinate equality",
                "Au fitted-epsilon readback",
                "adjoint auto-shutoff",
                "AD and FD sign agreement",
            ],
            "unresolved": [
                "strong-direction AD-FD error remains above 1 percent",
                "center-plane boundary quadrature is not converged",
                "the current three-point quadrature record does not independently prove which edge/corner sample causes the drift",
            ],
            "consequence": (
                "the v261 sharp-interface Au boundary kernel is not promoted; "
                "the P_Q direct discrete-loss derivative and all Au topology "
                "thermal/electrical/PTE optimization remain blocked"
            ),
        },
        "production_Au_optimization_permitted": False,
        "empirical_normalization": False,
        "gradient_rescaling": False,
        "finite_difference_used_to_fit_AD": False,
    }
    summary_path = output / "au_sharp_interface_external_field_adjoint_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    rows: list[dict[str, object]] = []
    for key, row in fd.items():
        rows.append(
            {
                "kind": "central_FD",
                "name": key,
                "resolution_or_step": row["h_um"],
                "derivative_J_proxy_per_um": row["derivative_J_proxy_per_um"],
                "relative_error_to_strong_FD": abs(
                    float(row["derivative_J_proxy_per_um"]) - strong_fd
                )
                / abs(strong_fd),
            }
        )
    rows.append(
        {
            "kind": "boundary_AD",
            "name": "selected_801_points_per_edge",
            "resolution_or_step": quadrature[-1]["dy_m"],
            "derivative_J_proxy_per_um": ad,
            "relative_error_to_strong_FD": strong_error,
        }
    )
    csv_path = output / "au_sharp_interface_external_field_adjoint_terms.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.5))
    labels = ["FD h=0.10", "FD h=0.05", "AD 801-point"]
    values = [
        float(fd["h_0.1_um"]["derivative_J_proxy_per_um"]),
        strong_fd,
        ad,
    ]
    axes[0].bar(labels, values, color=["#2878B5", "#2878B5", "#C82423"])
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_ylabel("derivative (J-proxy/um)")
    axes[0].set_title("Sign agrees; 1% magnitude gate fails")
    axes[0].tick_params(axis="x", rotation=18)
    axes[0].grid(axis="y", alpha=0.25)

    points = [int(row["n_points_per_edge"]) for row in quadrature]
    q_values = [1e-6 * float(row["total_J_proxy_per_m"]) for row in quadrature]
    axes[1].plot(points, q_values, "o-", color="#C82423")
    axes[1].axhline(strong_fd, color="#2878B5", linestyle="--", label="strong FD")
    axes[1].set_xlabel("boundary samples per vertical edge")
    axes[1].set_ylabel("derivative (J-proxy/um)")
    axes[1].set_title("Boundary quadrature remains unresolved")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    figure.tight_layout()
    plot_path = output / "au_sharp_interface_external_field_adjoint_failure.png"
    figure.savefig(plot_path, dpi=180)
    plt.close(figure)

    report = f"""# Au sharp-interface external-field boundary-kernel diagnostic

Status: `{status}`

This diagnostic removes the explicit moving-Au loss term. Its objective is a
fixed smooth electric-field-energy proxy in air, at least 150 nm below the Au
film. Therefore the derivative contains only the field-mediated sharp-interface
boundary kernel; it is not a `P_Q`, thermal, electrical, PTE, or optimization
result.

The independent central differences are `{float(fd['h_0.1_um']['derivative_J_proxy_per_um']):.12e}`
and `{strong_fd:.12e} J-proxy/um` at `h=0.10` and `0.05 um`. Their relative
change is `{100.0*float(source['FD_step_plateau_relative_change']):.6f}%`, and
the strong perturbation changes the baseline observable by
`{100.0*float(source['strong_h0p05_step_fraction_of_baseline']):.6f}%`. The FD
signal is therefore neither near-null nor step-size dominated.

The GPU adjoint completed in `{float(source['adjoint']['wall_s']):.3f} s` on
`{source['adjoint']['resources']['2']['device type']}` with final auto-shutoff
`{float(source['adjoint']['log_audit']['final_auto_shutoff']):.6e}`. The source
round trip and forward/adjoint coordinate mismatch are exactly zero. The Au
fitted-epsilon readback relative error is
`{float(source['material_readback']['relative_error']):.6e}`.

The selected 801-point boundary result is `{ad:.12e} J-proxy/um`. Its sign
agrees with FD, but its strong-direction relative error is
`{100.0*strong_error:.6f}%`, above the 1% gate. More importantly, increasing
the vertical-edge quadrature from 201 to 401 to 801 samples changes the
derivative from `{1e-6*float(quadrature[0]['total_J_proxy_per_m']):.12e}` to
`{1e-6*float(quadrature[1]['total_J_proxy_per_m']):.12e}` to
`{1e-6*float(quadrature[2]['total_J_proxy_per_m']):.12e} J-proxy/um`; the final
change is `{100.0*float(source['boundary_quadrature_final_relative_change']):.6f}%`.
The near-halving pattern is consistent with a grid-point/corner-dominated
sample, but the present artifact did not store the kernel profile, so that
mechanism is explicitly a hypothesis rather than a validated conclusion.

This is a useful improvement over the rejected `P_Q` trace: the sign is now
correct and the error is 6.77%, not five orders of magnitude. It is not a
certificate. No empirical normalization, FD fitting, sign change, or gradient
rescaling is used. Au topology optimization remains prohibited until the
boundary integral converges and the discrete conformal-Yee `P_Q` material-loss
derivative is separately certified.
"""
    report_path = output / "AU_SHARP_INTERFACE_EXTERNAL_FIELD_ADJOINT_REPORT.md"
    report_path.write_text(report)

    raw_files = []
    for role, path in (
        ("adjoint_template_fsp", case / "au_external_field_adjoint_template.fsp"),
        ("adjoint_gpu_fsp", case / "au_external_field_adjoint_gpu.fsp"),
        ("adjoint_log", case / "au_external_field_adjoint_gpu_p0.log"),
        ("raw_result", raw_result),
    ):
        raw_files.append(raw_entry(role, path))
    for name in sorted(
        {
            "sharp_width_7p9_edge25_forward",
            "sharp_width_7p95_edge25_forward",
            "sharp_width_8p0_edge25_forward",
            "sharp_width_8p05_edge25_forward",
            "sharp_width_8p1_edge25_forward",
        }
    ):
        raw_files.append(
            raw_entry(
                f"forward_fsp_{name}",
                args.raw_root / name / "complex_material_control.fsp",
            )
        )
    manifest = {
        "status": status,
        "raw_files_committed": False,
        "generation_command": (
            "python 12_run_au_sharp_interface_external_field_adjoint.py "
            "--output-dir <raw_case> --gpu-device 'GPU 0'"
        ),
        "Maxwell_forward_solves": 0,
        "Maxwell_adjoint_solves": 1,
        "raw_files": raw_files,
    }
    manifest_path = output / "AU_SHARP_INTERFACE_EXTERNAL_FIELD_RAW_ARTIFACT_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": status,
                "strong_AD_FD_relative_error": strong_error,
                "boundary_quadrature_final_relative_change": source[
                    "boundary_quadrature_final_relative_change"
                ],
                "raw_artifacts": len(raw_files),
            },
            indent=2,
        )
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
