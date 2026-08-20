#!/usr/bin/env python3
"""Publish the fixed-material, moving-boundary, and Au-rim diagnosis."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
HOME_RAW = Path("/home/seunghyun/tairte4/raw_artifacts/au_topology_validation")
DATA_RAW = Path("/data/seunghyun/tairte4/raw_artifacts/au_topology_validation")


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(role: str, path: Path, stored: dict | None = None) -> dict[str, object]:
    row = {"role": role, "path": str(path.resolve()), "size_bytes": path.stat().st_size}
    if stored and Path(stored["path"]).resolve() == path.resolve():
        row["sha256"] = str(stored["sha256"])
    else:
        row["sha256"] = sha256(path)
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--home-raw", type=Path, default=HOME_RAW)
    parser.add_argument("--data-raw", type=Path, default=DATA_RAW)
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    args = parser.parse_args()
    home = args.home_raw.resolve()
    data = args.data_raw.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    material_path = (
        home
        / "pva5_fixedgrid_material_adjoint_control"
        / "au_fixed_geometry_material_adjoint_result.json"
    )
    moving_path = (
        home
        / "pva5_fixedgrid_smooth_ellipse_external_field_adjoint_gpu0"
        / "au_sharp_interface_external_field_result.json"
    )
    deps_path = (
        home
        / "pva5_fixedgrid_native_yee_geometry_deps"
        / "au_pva_native_yee_geometry_deps_result.json"
    )
    traces_path = (
        home
        / "pva5_fixedgrid_boundary_one_sided_traces"
        / "au_pva_boundary_one_sided_trace_result.json"
    )
    sheet_layout_path = (
        data / "pva5_fixedgrid_sampled2d_sheet_endpoint_runsetup" / "case_result.json"
    )
    sheet_gpu_path = (
        data / "pva5_fixedgrid_sampled2d_sheet_endpoint_forward" / "case_result.json"
    )
    ellipsoid_path = (
        data
        / "pva5_smooth3d_ellipsoid_boundary_adjoint_gpu0"
        / "au_smooth3d_ellipsoid_boundary_adjoint_result.json"
    )
    material = read(material_path)
    moving = read(moving_path)
    deps = read(deps_path)
    traces = read(traces_path)
    sheet_layout = read(sheet_layout_path)
    sheet_gpu = read(sheet_gpu_path)
    ellipsoid = read(ellipsoid_path) if ellipsoid_path.is_file() else None

    fixed_strong = material["material_steps"][-1]
    moving_strong = moving["AD_FD_comparison"]["h_0.05_um"]
    native_strong = deps["AD_FD_comparison"]["h_0.05_um"]
    zero_trace = next(
        row for row in traces["traces"] if float(row["normal_offset_m"]) == 0.0
    )
    sheet_fit = sheet_layout["material"]["surface_conductivity"]
    if ellipsoid is None:
        status = "IN_PROGRESS_AU_SMOOTH3D_RIM_REMOVAL_CONTROL"
        conclusion = (
            "fixed-geometry material adjoint is validated, while every moving "
            "50-nm-film boundary route remains invalid; a fully smooth 3-D "
            "ellipsoid control is still running"
        )
    elif bool(ellipsoid.get("passed", False)):
        status = "VALIDATED_AU_SMOOTH3D_CONTROL_PRODUCTION_STILL_BLOCKED"
        conclusion = (
            "removing every top/bottom rim restores the field-mediated Au "
            "shape derivative; the production finite-thickness sharp-rim "
            "electrode remains uncertified"
        )
    else:
        status = "BLOCKED_AU_MOVING_BOUNDARY_ADJOINT_AFTER_SMOOTH3D_CONTROL"
        conclusion = (
            "even a fully smooth 3-D boundary does not restore AD-FD, so the "
            "remaining incompatibility is broader than the thin-film rim alone"
        )
    summary = {
        "status": status,
        "conclusion": conclusion,
        "fixed_geometry_material_control": {
            "status": material["status"],
            "FD_step_change": material["FD_step_relative_change"],
            "AD_step_change": material["AD_step_relative_change"],
            "strong_AD_FD_relative_error": fixed_strong["official_relative_error"],
            "sign_agrees": fixed_strong["official_sign_agrees"],
            "interpretation": (
                "FieldRegion source normalization, unconjugated Ef*Ea convention, "
                "component Yee pairing, and fixed-domain volume integration pass"
            ),
        },
        "moving_50nm_film_boundary": {
            "status": moving["status"],
            "FD_step_change": moving["FD_step_plateau_relative_change"],
            "strong_AD_FD_relative_error": moving_strong["relative_error"],
            "sign_agrees": moving_strong["sign_agrees"],
            "quadrature_change": moving["boundary_quadrature_final_relative_change"],
        },
        "native_Yee_geometry_depsilon": {
            "status": deps["status"],
            "maximum_coordinate_mismatch_m": deps["maximum_coordinate_mismatch_m"],
            "final_step_change": deps["d_epsilon_final_relative_change"],
            "strong_AD_FD_relative_error": native_strong["relative_error"],
            "sign_agrees": native_strong["sign_agrees"],
        },
        "one_sided_trace_and_rim_decomposition": {
            "status": traces["status"],
            "all_sampled_sides_have_correct_sign": all(
                float(row["total_J_proxy_per_um"])
                * float(traces["FD_target_h0p05_J_proxy_per_um"])
                > 0.0
                for row in traces["traces"]
            ),
            "geometric_trace_total_J_proxy_per_um": zero_trace[
                "total_J_proxy_per_um"
            ],
            "central_region_excluding_top_bottom_10nm_J_proxy_per_m": zero_trace[
                "middle_10nm_trimmed_J_proxy_per_m"
            ],
            "top_bottom_10nm_rims_J_proxy_per_m": zero_trace[
                "top_bottom_10nm_rims_J_proxy_per_m"
            ],
            "rim_fraction_of_absolute_total": zero_trace[
                "rim_fraction_of_absolute_total"
            ],
            "interpretation": (
                "the central film-depth contribution has the FD sign, but the "
                "top/bottom 10-nm rims dominate and reverse the integral"
            ),
        },
        "sampled_2D_sheet_GPU_control": {
            "layout_status": sheet_layout["status"],
            "surface_conductivity_fit_relative_error": sheet_fit["fit_relative_error"],
            "GPU_status": "BLOCKED_AU_SAMPLED_2D_GPU_UNSUPPORTED",
            "engine_error": sheet_gpu.get("error"),
            "CPU_fallback_run": False,
            "official_GPU_limitation": (
                "v261 GPU FDTD supports no 2-D standard optical conductivity "
                "material except PEC"
            ),
        },
        "fully_smooth_3D_control": ellipsoid,
        "production_Au_optimization_permitted": False,
        "empirical_normalization": False,
        "gradient_rescaling": False,
        "thermal_PTE_or_optimization_executed": False,
    }
    summary_path = output / "au_pva_rim_resolution_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str) + "\n")

    rows = [
        {
            "case": "fixed_geometry_material",
            "status": material["status"],
            "AD_FD_error_percent": 100.0 * fixed_strong["official_relative_error"],
            "sign_agrees": fixed_strong["official_sign_agrees"],
            "numerical_change_percent": 100.0 * material["FD_step_relative_change"],
            "passed": material["passed"],
        },
        {
            "case": "moving_PVA_50nm_film",
            "status": moving["status"],
            "AD_FD_error_percent": 100.0 * moving_strong["relative_error"],
            "sign_agrees": moving_strong["sign_agrees"],
            "numerical_change_percent": 100.0
            * moving["boundary_quadrature_final_relative_change"],
            "passed": moving["passed"],
        },
        {
            "case": "native_Yee_geometry_depsilon",
            "status": deps["status"],
            "AD_FD_error_percent": 100.0 * native_strong["relative_error"],
            "sign_agrees": native_strong["sign_agrees"],
            "numerical_change_percent": 100.0 * deps["d_epsilon_final_relative_change"],
            "passed": deps["passed"],
        },
    ]
    if ellipsoid is not None and "AD_FD_comparison" in ellipsoid:
        strong = ellipsoid["AD_FD_comparison"]["h_0.05_um"]
        rows.append(
            {
                "case": "fully_smooth_3D_ellipsoid",
                "status": ellipsoid["status"],
                "AD_FD_error_percent": 100.0 * strong["relative_error"],
                "sign_agrees": strong["sign_agrees"],
                "numerical_change_percent": 100.0
                * ellipsoid["boundary_quadrature_final_relative_change"],
                "passed": ellipsoid["passed"],
            }
        )
    csv_path = output / "au_pva_rim_resolution_cases.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.5))
    labels = [row["case"] for row in rows]
    values = [row["AD_FD_error_percent"] for row in rows]
    colors = ["#2A9D8F" if row["passed"] else "#D1495B" for row in rows]
    axes[0].bar(np.arange(len(rows)), values, color=colors)
    axes[0].axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    axes[0].set_yscale("log")
    axes[0].set_xticks(np.arange(len(rows)), labels, rotation=20, ha="right")
    axes[0].set_ylabel("AD-FD relative error (%)")
    axes[0].set_title("Fixed material passes; moving boundary fails")
    axes[0].grid(axis="y", alpha=0.25)

    offsets = np.asarray([row["normal_offset_m"] for row in traces["traces"]]) * 1e9
    trace_values = np.asarray([row["total_J_proxy_per_um"] for row in traces["traces"]]) * 1e30
    fd_value = float(traces["FD_target_h0p05_J_proxy_per_um"]) * 1e30
    axes[1].plot(offsets, trace_values, "o-", label="boundary AD trace")
    axes[1].axhline(fd_value, color="black", linestyle="--", label="central FD")
    axes[1].axhline(0.0, color="gray", linewidth=0.8)
    axes[1].set_xlabel("normal trace offset (nm); negative=inside Au")
    axes[1].set_ylabel(r"derivative ($10^{-30}$ J-proxy/um)")
    axes[1].set_title("Changing trace side does not recover sign")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)

    z_nm = np.asarray(zero_trace["z_center_m"]) * 1e9
    tangent = np.asarray(zero_trace["tangential_E_term_J_proxy_per_m_by_z"]) * 1e24
    normal = np.asarray(zero_trace["normal_D_term_J_proxy_per_m_by_z"]) * 1e24
    total = np.asarray(zero_trace["total_J_proxy_per_m_by_z"]) * 1e24
    axes[2].plot(z_nm, tangent, label="tangential E")
    axes[2].plot(z_nm, normal, label="normal D")
    axes[2].plot(z_nm, total, color="black", linewidth=1.5, label="total")
    axes[2].axvspan(z_nm.min(), z_nm.min() + 10.0, color="#F4A261", alpha=0.22)
    axes[2].axvspan(z_nm.max() - 10.0, z_nm.max(), color="#F4A261", alpha=0.22)
    axes[2].axhline(0.0, color="gray", linewidth=0.8)
    axes[2].set_xlabel("Au depth coordinate z (nm)")
    axes[2].set_ylabel(r"per-bin contribution ($10^{-24}$ J-proxy/m)")
    axes[2].set_title("Top/bottom 10-nm rims flip the integral")
    axes[2].grid(alpha=0.25)
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    plot_path = output / "au_pva_rim_resolution.png"
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    ellipsoid_sentence = (
        "The fully smooth 3-D control is still running."
        if ellipsoid is None
        else (
            f"The fully smooth 3-D result is `{ellipsoid['status']}` with "
            f"{100*ellipsoid['AD_FD_comparison']['h_0.05_um']['relative_error']:.6f}% "
            "strong-direction error."
        )
    )
    report = f"""# Au PVA boundary-rim diagnosis

Status: `{status}`

## Result

The fixed-geometry material derivative passes AD--FD with
`{100*fixed_strong['official_relative_error']:.6f}%` error. This independently
validates the FieldRegion source normalization, the official unconjugated
`E_f E_a` convention, component-specific Yee coordinates, and volume
integration. The same chain fails only when the Au boundary moves.

For the fixed-grid 50-nm PVA film, the central FD steps agree to
`{100*moving['FD_step_plateau_relative_change']:.6f}%`, but the continuous
boundary AD has the wrong sign and `{100*moving_strong['relative_error']:.6f}%`
error. Native component-grid `d-epsilon` also has the wrong sign and changes by
`{100*deps['d_epsilon_final_relative_change']:.6f}%` at the final step, despite
a maximum E/index coordinate mismatch of only
`{deps['maximum_coordinate_mismatch_m']:.6e} m`.

Sampling 50 nm inside Au, exactly on the boundary, or 50 nm outside air never
recovers the FD sign. At the geometric trace, the film-depth region after
removing the top and bottom 10 nm contributes
`{zero_trace['middle_10nm_trimmed_J_proxy_per_m']:.12e} J-proxy/m`, which has
the FD sign. The two 10-nm rims contribute
`{zero_trace['top_bottom_10nm_rims_J_proxy_per_m']:.12e} J-proxy/m` and reverse
the total. The rim magnitude is
`{100*zero_trace['rim_fraction_of_absolute_total']:.3f}%` of the absolute final
integral.

The exact Au 50-nm sheet-conductivity endpoint was also constructed. Its fitted
surface conductivity differs from the requested value by only
`{100*sheet_fit['fit_relative_error']:.6f}%`, but v261 GPU FDTD explicitly
rejects the sampled-2D material. No CPU fallback was run. The official GPU
limitation states that 2-D optical-conductivity materials are unsupported
except PEC; PEC cannot supply lossy Au absorption.

{ellipsoid_sentence}

Production Au PTE optimization remains prohibited. Passing a mathematical
smooth-3-D control would isolate the defect to the non-smooth finite-film rim,
but a realistic rounded thin-Au endpoint and the direct moving-material `P_Q`
term would still require separate AD--FD certification. No thermal, electrical,
PTE, or optimization solve is part of this checkpoint.

Official GPU limitation: https://optics.ansys.com/hc/en-us/articles/17518942465811-Getting-started-with-running-FDTD-on-GPU
"""
    report_path = output / "AU_PVA_RIM_RESOLUTION_REPORT.md"
    report_path.write_text(report)

    raw_paths = [material_path, moving_path, deps_path, traces_path, sheet_layout_path, sheet_gpu_path]
    if ellipsoid is not None:
        raw_paths.append(ellipsoid_path)
    manifest = {
        "status": status,
        "raw_files_committed": False,
        "raw_files": [artifact(path.stem, path) for path in raw_paths],
        "generation_commands": [
            "python 21_validate_fixed_geometry_au_material_adjoint.py --output-dir <raw>",
            "python 22_validate_pva_native_yee_geometry_deps.py --output-dir <raw>",
            "python 23_analyze_pva_boundary_one_sided_traces.py --output-dir <raw>",
            "python 24_run_au_sampled_2d_sheet_endpoint.py --output-dir <raw>",
            "python 25_run_au_smooth_3d_ellipsoid_width_control.py --au-half-x-um <...> --output-dir <raw>",
            "python 26_validate_au_smooth_3d_ellipsoid_boundary_adjoint.py --output-dir <raw>",
            "python 28_summarize_au_pva_rim_resolution.py",
        ],
        "CPU_FDTD_fallback": False,
        "thermal_PTE_or_optimization_executed": False,
    }
    manifest_path = output / "AU_PVA_RIM_RESOLUTION_RAW_ARTIFACT_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": status, "production_Au_optimization_permitted": False}, indent=2))
    return 0 if ellipsoid is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
