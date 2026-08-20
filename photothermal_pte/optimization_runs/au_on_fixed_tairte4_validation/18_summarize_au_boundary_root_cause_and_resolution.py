#!/usr/bin/env python3
"""Publish the Au optical-boundary root-cause and smooth-shape control."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
DEFAULT_RAW = Path("/home/seunghyun/tairte4/raw_artifacts/au_topology_validation")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(role: str, path: Path, stored: dict | None = None) -> dict[str, object]:
    row = {
        "role": role,
        "path": str(path.resolve()),
        "bytes": int(path.stat().st_size),
    }
    if stored and Path(stored["path"]).resolve() == path.resolve():
        row["sha256"] = stored["sha256"]
    else:
        row["sha256"] = sha256(path)
    return row


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    args = parser.parse_args()
    raw = args.raw_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    legacy = read(
        raw
        / "sharp_width_8p0_edge25_external_field_adjoint_gpu0"
        / "au_sharp_interface_external_field_result.json"
    )
    corner_free = read(
        raw
        / "corner_free_y18_external_field_adjoint_gpu0"
        / "au_sharp_interface_external_field_result.json"
    )
    deps = read(
        raw
        / "corner_free_y18_solver_discrete_deps"
        / "au_solver_discrete_deps_result.json"
    )
    localization = read(output / "au_boundary_corner_localization_summary.json")
    smooth_root = raw / "smooth_ellipse_external_field_adjoint_gpu0"
    smooth_result_path = smooth_root / "au_sharp_interface_external_field_result.json"
    smooth = read(smooth_result_path)

    smooth_passed = bool(smooth.get("passed", False))
    status = (
        "VALIDATED_SMOOTH_AU_FIELD_MEDIATED_BOUNDARY_KERNEL"
        if smooth_passed
        else "BLOCKED_AU_BOUNDARY_ADJOINT_UNRESOLVED_AFTER_SMOOTH_CONTROL"
    )
    production_permitted = False
    summary = {
        "status": status,
        "root_cause": (
            "sharp extruded-Au boundary singularities make the continuous and "
            "solver-discrete shape derivatives non-convergent"
        ),
        "not_root_cause": [
            "PML closure",
            "GPU execution",
            "forward/adjoint source round trip",
            "component-specific Yee coordinate mismatch",
            "thermal or Robin boundary conditions",
        ],
        "sharp_rectangle": {
            "FD_step_plateau_relative_change": legacy[
                "FD_step_plateau_relative_change"
            ],
            "strong_AD_FD_relative_error": legacy["AD_FD_comparison"][
                "h_0.05_um"
            ]["relative_error"],
            "boundary_quadrature_final_relative_change": legacy[
                "boundary_quadrature_final_relative_change"
            ],
        },
        "corner_localization": {
            "status": localization["status"],
            "endpoint_fraction_at_801": localization[
                "combined_endpoint_fraction_at_801_points"
            ],
            "smooth_interior_201_to_6401_relative_change": localization[
                "combined_interior_201_to_6401_relative_change"
            ],
        },
        "moved_y_corners_outside_active_support": {
            "FD_step_plateau_relative_change": corner_free[
                "FD_step_plateau_relative_change"
            ],
            "center_z_final_relative_change": corner_free[
                "official_center_depth_final_relative_change"
            ],
            "full_yz_surface_final_relative_change": corner_free[
                "full_surface_midpoint_final_relative_change"
            ],
            "strong_AD_FD_relative_error": corner_free["AD_FD_comparison"][
                "h_0.05_um"
            ]["relative_error"],
            "interpretation": (
                "moving lateral corners away is insufficient because the "
                "extruded metal retains top/bottom rim singularities"
            ),
        },
        "solver_discrete_depsilon": {
            "status": deps["status"],
            "runsetup_only_remeshes": deps["runsetup_only_remeshes"],
            "maximum_field_index_coordinate_mismatch_m": max(
                row["field_index_maximum_coordinate_mismatch_m"]
                for row in deps["coordinate_audits"]
            ),
            "final_step_relative_change": deps["d_epsilon_final_relative_change"],
            "strong_AD_FD_relative_error": deps["AD_FD_comparison"][
                "h_0.05_um"
            ]["relative_error"],
        },
        "smooth_closed_ellipse": {
            "status": smooth["status"],
            "geometry": smooth["geometry_control"],
            "FD_step_plateau_relative_change": smooth[
                "FD_step_plateau_relative_change"
            ],
            "strong_AD_FD_relative_error": smooth["AD_FD_comparison"][
                "h_0.05_um"
            ]["relative_error"],
            "boundary_quadrature_final_relative_change": smooth[
                "boundary_quadrature_final_relative_change"
            ],
            "gates": smooth["gates"],
        },
        "validated_scope": (
            "field-mediated fixed-external-objective optical shape derivative"
            if smooth_passed
            else "no optical Au shape derivative promoted"
        ),
        "production_Au_optimization_permitted": production_permitted,
        "remaining_blocker": (
            "direct moving-Au spatial absorption/P_Q contribution required by "
            "thermal-PTE objective is not yet AD-FD certified"
        ),
        "empirical_normalization": False,
        "gradient_rescaling": False,
        "thermal_PTE_or_optimization_executed": False,
    }
    summary_path = output / "au_boundary_root_cause_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    rows = []
    for name, item, method in (
        ("sharp_rectangle", legacy, "endpoint center-z"),
        ("corners_moved_y18", corner_free, "full y-z midpoint"),
        ("smooth_ellipse", smooth, "endpoint-free Gauss-Legendre"),
    ):
        rows.append(
            {
                "case": name,
                "method": method,
                "fd_step_change_percent": 100.0
                * float(item["FD_step_plateau_relative_change"]),
                "quadrature_change_percent": 100.0
                * float(item["boundary_quadrature_final_relative_change"]),
                "ad_fd_error_percent": 100.0
                * float(item["AD_FD_comparison"]["h_0.05_um"]["relative_error"]),
                "sign_agrees": item["AD_FD_comparison"]["h_0.05_um"][
                    "sign_agrees"
                ],
                "passed": bool(item.get("passed", False)),
            }
        )
    rows.append(
        {
            "case": "solver_discrete_depsilon",
            "method": "runsetup-only conformal d-epsilon",
            "fd_step_change_percent": 100.0
            * float(corner_free["FD_step_plateau_relative_change"]),
            "quadrature_change_percent": 100.0
            * float(deps["d_epsilon_final_relative_change"]),
            "ad_fd_error_percent": 100.0
            * float(deps["AD_FD_comparison"]["h_0.05_um"]["relative_error"]),
            "sign_agrees": deps["AD_FD_comparison"]["h_0.05_um"]["sign_agrees"],
            "passed": bool(deps.get("passed", False)),
        }
    )
    csv_path = output / "au_boundary_root_cause_cases.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.7))
    names = [row["case"] for row in rows]
    x = np.arange(len(names))
    axes[0].bar(x - 0.18, [row["ad_fd_error_percent"] for row in rows], 0.36, label="AD-FD")
    axes[0].bar(x + 0.18, [row["quadrature_change_percent"] for row in rows], 0.36, label="quadrature/step")
    axes[0].axhline(1.0, color="black", linestyle="--", linewidth=1.0, label="1% AD-FD gate")
    axes[0].axhline(0.5, color="gray", linestyle=":", linewidth=1.0, label="0.5% quadrature gate")
    axes[0].set_yscale("log")
    axes[0].set_xticks(x, names, rotation=18, ha="right")
    axes[0].set_ylabel("relative difference (%)")
    axes[0].set_title("Boundary derivative diagnostics")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(fontsize=8)

    smooth_fd = smooth["finite_difference"]
    strong_fd = float(smooth_fd["h_0.05_um"]["derivative_J_proxy_per_um"])
    ad = float(smooth["AD_FD_comparison"]["h_0.05_um"]["AD_J_proxy_per_um"])
    axes[1].bar(
        ["FD h=.10", "FD h=.05", "smooth AD"],
        [
            float(smooth_fd["h_0.1_um"]["derivative_J_proxy_per_um"]),
            strong_fd,
            ad,
        ],
        color=["#2878B5", "#2878B5", "#C82423"],
    )
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_ylabel("derivative (J-proxy/um)")
    axes[1].set_title("Smooth closed-boundary AD vs FD")
    axes[1].grid(axis="y", alpha=0.25)
    figure.tight_layout()
    plot_path = output / "au_boundary_root_cause_and_resolution.png"
    figure.savefig(plot_path, dpi=180)
    plt.close(figure)

    report = f"""# Au optical-boundary root cause and resolution

Status: `{status}`

## Conclusion

The failing gate is not a PML, GPU, source-normalization, Yee-coordinate, or
thermal-boundary problem. It is localized to the sharp boundary of the
extruded Au geometry. The original rectangular control had a converged central
FD plateau (`{100*legacy['FD_step_plateau_relative_change']:.6f}%`) but its
continuous boundary quadrature changed by
`{100*legacy['boundary_quadrature_final_relative_change']:.6f}%` and missed the
strong FD by `{100*legacy['AD_FD_comparison']['h_0.05_um']['relative_error']:.6f}%`.

At 801 vertical-face samples, the two `y=+-10 um` endpoints supplied
`{100*localization['combined_endpoint_fraction_at_801_points']:.6f}%` of the stored
tangential-E proxy. Removing those endpoints leaves a smooth-face interior
whose 201-to-6401 change is only
`{100*localization['combined_interior_201_to_6401_relative_change']:.8f}%`.
Moving the y corners out to `+-18 um` did not solve the full 3D problem: the
extruded film still has top/bottom rims, and the full y-z surface rule changed
by `{100*corner_free['full_surface_midpoint_final_relative_change']:.6f}%`.

The independent solver-discrete conformal `d-epsilon` route was also tested.
All component coordinates match the electric-field grid to
`{summary['solver_discrete_depsilon']['maximum_field_index_coordinate_mismatch_m']:.6e} m`,
but its final step change is
`{100*deps['d_epsilon_final_relative_change']:.6f}%` and its strong FD error is
`{100*deps['AD_FD_comparison']['h_0.05_um']['relative_error']:.6f}%`. Therefore
coordinate repair alone does not make a sharp metal boundary differentiable
on this conformal mesh.

## Controlled remedy

The replacement control uses exact binary scalar Au (`n=12.1+69.2i`) with a
smooth closed in-plane ellipse represented by 512 counter-clockwise vertices.
No gray Au, clipping, fitting, normalization, or gradient rescaling is used.
The boundary is integrated with endpoint-free Gauss-Legendre nodes, and an
analytic geometry test independently verifies that its normal shape velocity
recovers `dA/da`.

For this control, the FD step change is
`{100*smooth['FD_step_plateau_relative_change']:.6f}%`, the final quadrature
change is `{100*smooth['boundary_quadrature_final_relative_change']:.6f}%`, and
the strong-direction AD-FD error is
`{100*smooth['AD_FD_comparison']['h_0.05_um']['relative_error']:.6f}%`.
The result is `{smooth['status']}`.

This resolves only the field-mediated fixed-external-objective boundary
kernel. Production Au PTE optimization is still prohibited because the direct
moving-Au spatial absorption (`P_Q`) contribution has not passed AD-FD. No
thermal, electrical, PTE, or optimization solve is included in this report.
"""
    report_path = output / "AU_BOUNDARY_ROOT_CAUSE_AND_RESOLUTION_REPORT.md"
    report_path.write_text(report)

    raw_files = [
        artifact("smooth_raw_result", smooth_result_path),
        artifact(
            "solver_discrete_depsilon_result",
            raw
            / "corner_free_y18_solver_discrete_deps"
            / "au_solver_discrete_deps_result.json",
        ),
    ]
    for case_name in (
        "smooth_ellipse_a7p9_b10_edge25_forward",
        "smooth_ellipse_a7p95_b10_edge25_forward",
        "smooth_ellipse_a8p0_b10_edge25_forward",
        "smooth_ellipse_a8p05_b10_edge25_forward",
        "smooth_ellipse_a8p1_b10_edge25_forward",
    ):
        case_result = read(raw / case_name / "case_result.json")
        stored_by_path = {
            str(Path(row["path"]).resolve()): row
            for row in case_result["raw_artifacts"]
        }
        for filename, role in (
            ("complex_material_control.fsp", "forward_FSP"),
            ("complex_material_control_q.npz", "forward_Q_NPZ"),
        ):
            path = raw / case_name / filename
            raw_files.append(
                artifact(
                    f"{role}_{case_name}",
                    path,
                    stored_by_path.get(str(path.resolve())),
                )
            )
    for filename, role in (
        ("au_external_field_adjoint_template.fsp", "adjoint_template_FSP"),
        ("au_external_field_adjoint_gpu.fsp", "adjoint_GPU_FSP"),
        ("au_external_field_adjoint_gpu_p0.log", "adjoint_GPU_log"),
    ):
        raw_files.append(artifact(role, smooth_root / filename))
    manifest = {
        "status": status,
        "raw_files_committed": False,
        "generation_commands": [
            "python 16_run_au_smooth_ellipse_width_control.py --au-half-x-um <7.9|7.95|8.0|8.05|8.1> --au-half-y-um 10 --gpu-device 'GPU 0' --output-dir <raw_case>",
            "python 17_run_au_smooth_ellipse_external_field_adjoint.py --gpu-device 'GPU 0' --output-dir <raw_adjoint_case>",
            "python 18_summarize_au_boundary_root_cause_and_resolution.py",
        ],
        "new_Maxwell_forward_solves": 4,
        "reused_baseline_forward_solves": 1,
        "new_Maxwell_adjoint_solves": 1,
        "raw_files": raw_files,
    }
    manifest_path = output / "AU_BOUNDARY_ROOT_CAUSE_RAW_ARTIFACT_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    print(
        json.dumps(
            {
                "status": status,
                "smooth_passed": smooth_passed,
                "smooth_AD_FD_error": smooth["AD_FD_comparison"]["h_0.05_um"][
                    "relative_error"
                ],
                "smooth_quadrature_change": smooth[
                    "boundary_quadrature_final_relative_change"
                ],
                "production_Au_optimization_permitted": production_permitted,
            },
            indent=2,
        )
    )
    return 0 if smooth_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
