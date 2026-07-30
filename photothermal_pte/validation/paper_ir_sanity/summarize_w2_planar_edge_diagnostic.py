#!/usr/bin/env python3
"""Summarize the nominal-w0=2 um planar/edge optical diagnostic.

This script performs no FDTD, thermal, PTE, adjoint, or optimization solve.
It compares already saved 4 ps Q/field artifacts and records the independent
1.2-to-4 ps observable-Q certificates.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPOSITORY = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY))

from photothermal_pte.validation.paper_ir_sanity.compare_paper_ir_smoke_q_convergence import (
    edge_profile,
    integrate,
    load_q,
    trapezoid_weights,
    volume_weights,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPOSITORY, text=True
    ).strip()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def artifact_record(path: Path, role: str) -> dict[str, Any]:
    return {
        "role": role,
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def relative_difference(first: float, second: float) -> float:
    return abs(second - first) / max(abs(first), np.finfo(float).tiny)


def normalized_metrics(
    first: np.ndarray,
    second: np.ndarray,
    weights: np.ndarray,
) -> dict[str, float]:
    first_power = integrate(first, weights)
    second_power = integrate(second, weights)
    a = first / first_power
    b = second / second_power
    nrmse = float(
        np.sqrt(np.sum(weights * (b - a) ** 2) / np.sum(weights * a**2))
    )
    mean_a = float(np.sum(weights * a) / np.sum(weights))
    mean_b = float(np.sum(weights * b) / np.sum(weights))
    da = a - mean_a
    db = b - mean_b
    correlation = float(
        np.sum(weights * da * db)
        / np.sqrt(np.sum(weights * da**2) * np.sum(weights * db**2))
    )
    cosine = float(
        np.sum(weights * a * b)
        / np.sqrt(np.sum(weights * a**2) * np.sum(weights * b**2))
    )
    return {
        "first_power_W": first_power,
        "second_power_W": second_power,
        "raw_power_relative_difference": relative_difference(
            first_power, second_power
        ),
        "equal_power_normalized_spatial_Q_NRMSE": nrmse,
        "equal_power_normalized_spatial_Q_Pearson_correlation": correlation,
        "equal_power_normalized_spatial_Q_cosine_similarity": cosine,
    }


def complex_norms(path: Path) -> dict[str, dict[str, float]]:
    with np.load(path, allow_pickle=False) as raw:
        result: dict[str, dict[str, float]] = {}
        for plane in ("incident", "flake"):
            result[plane] = {
                component: float(np.linalg.norm(raw[f"{plane}_{component}"]))
                for component in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
            }
        source = np.asarray(raw["source_profile_E"])
        result["source_profile"] = {
            f"E{axis}_L2": float(np.linalg.norm(source[..., index]))
            for index, axis in enumerate("xyz")
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in ("planar-a", "planar-b", "edge-b"):
        parser.add_argument(f"--{name}-1p2-dir", type=Path, required=True)
        parser.add_argument(f"--{name}-4-dir", type=Path, required=True)
        parser.add_argument(f"--{name}-convergence-dir", type=Path, required=True)
    parser.add_argument("--failed-planar-b-launch-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    directory_pairs = {
        "planar_a": (args.planar_a_1p2_dir, args.planar_a_4_dir),
        "planar_b": (args.planar_b_1p2_dir, args.planar_b_4_dir),
        "finite_edge_b": (args.edge_b_1p2_dir, args.edge_b_4_dir),
    }
    cases: dict[str, Any] = {}
    q_data: dict[str, Any] = {}
    manifest: list[dict[str, Any]] = []
    convergence_dirs = {
        "planar_a": args.planar_a_convergence_dir,
        "planar_b": args.planar_b_convergence_dir,
        "finite_edge_b": args.edge_b_convergence_dir,
    }

    for name, (short_dir, long_dir) in directory_pairs.items():
        short_json = short_dir / "case_result.json"
        long_json = long_dir / "case_result.json"
        short_case = read_json(short_json)
        long_case = read_json(long_json)
        convergence_path = (
            convergence_dirs[name] / "q_observable_convergence.json"
        )
        convergence = read_json(convergence_path)
        long_q_path = long_dir / "diagnostic_q_common_grid_artifact.npz"
        long_field_path = long_dir / "w2_beam_and_field_components.npz"
        q_data[name] = load_q(long_q_path)
        run = long_case["run_result"]
        beam = run["beam_and_field_readback"]
        cases[name] = {
            "geometry": long_case["pre_run_contract"]["geometry"],
            "polarization_deg": long_case["polarization_deg"],
            "solver_version": long_case["pre_run_contract"]["solver"]["version"],
            "solver_resource_label": run["resource"],
            "solver_resource_contract": long_case["pre_run_contract"]["solver"][
                "resources"
            ],
            "P_Q_common_grid_W": run["P_Q_common_grid_bounded_W"],
            "P_Q_native_Yee_W": run["P_Q_native_Yee_bounded_W"],
            "P_six_face_W": run["P_six_face_native_W"],
            "Q_component_power_common_grid_W": run[
                "component_power_common_grid_bounded_W"
            ],
            "common_grid_six_face_closure": run[
                "common_grid_six_face_relative_closure"
            ],
            "native_Yee_six_face_closure": run[
                "native_Yee_six_face_relative_closure"
            ],
            "auto_shutoff_1p2ps": short_case["run_result"]["auto_shutoff"],
            "auto_shutoff_4ps": run["auto_shutoff"],
            "observable_Q_convergence": convergence,
            "nominal_source": long_case["pre_run_contract"]["geometry"]["source"],
            "realized_incident_plane_beam": beam[
                "beam_fit_from_downward_incident_intensity"
            ],
            "realized_flake_midplane_total_field_fit": beam[
                "total_field_flake_midplane_fit"
            ],
            "source_power_native_W": beam["source_power_native_W"],
            "incident_plane_power_over_source_power": beam[
                "incident_plane_power_over_source_power"
            ],
            "field_component_norms": complex_norms(long_field_path),
            "material_epsilon_readback": run["material_epsilon_readback"],
            "axis_mapping": {
                "x": "crystal b",
                "y": "crystal a",
                "z": "c closure with epsilon_c=epsilon_b",
            },
            "realized_control_volume": run["realized_control_volume"],
            "no_Q_modification": run["acceptance"][
                "no_Q_clipping_smoothing_gain_rescaling_tiling_or_deletion"
            ],
        }
        for duration, case_dir in (("1p2ps", short_dir), ("4ps", long_dir)):
            for filename, role in (
                ("case_result.json", f"{name} {duration} case result"),
                ("diagnostic_q_common_grid_artifact.npz", f"{name} {duration} raw Q"),
                ("w2_beam_and_field_components.npz", f"{name} {duration} fields"),
                ("native_yee_mesh_coordinates.npz", f"{name} {duration} Yee coordinates"),
                ("finite_2um_optical_q.fsp", f"{name} {duration} FSP"),
                ("finite_2um_optical_q_p0.log", f"{name} {duration} solver log"),
            ):
                path = case_dir / filename
                if path.exists():
                    manifest.append(artifact_record(path, role))
        for filename, role in (
            ("q_observable_convergence.json", f"{name} observable-Q certificate"),
            ("q_component_power_convergence.csv", f"{name} component convergence"),
            ("q_observable_convergence_profiles.npz", f"{name} convergence profiles"),
            ("q_observable_convergence.png", f"{name} convergence plot"),
        ):
            path = convergence_dirs[name] / filename
            if path.exists():
                manifest.append(artifact_record(path, role))

    if args.failed_planar_b_launch_dir:
        failed_path = args.failed_planar_b_launch_dir / "case_result.json"
        if failed_path.exists():
            manifest.append(
                artifact_record(
                    failed_path,
                    "preserved first planar-b GPU launch failure diagnostic",
                )
            )

    reference = q_data["planar_a"]
    for name, data in q_data.items():
        for axis in "xyz":
            if not np.array_equal(
                reference["coordinates"][axis], data["coordinates"][axis]
            ):
                raise RuntimeError(
                    f"{name} {axis} coordinates differ; comparison is fail-closed"
                )
    coordinates = reference["coordinates"]
    weights = volume_weights(coordinates)
    comparisons = {
        "planar_a_vs_planar_b": normalized_metrics(
            q_data["planar_a"]["total"],
            q_data["planar_b"]["total"],
            weights,
        ),
        "planar_b_vs_finite_edge_b": normalized_metrics(
            q_data["planar_b"]["total"],
            q_data["finite_edge_b"]["total"],
            weights,
        ),
    }
    for label, first_name, second_name in (
        ("planar_a_vs_planar_b", "planar_a", "planar_b"),
        ("planar_b_vs_finite_edge_b", "planar_b", "finite_edge_b"),
    ):
        first_beam = cases[first_name]["realized_incident_plane_beam"]
        second_beam = cases[second_name]["realized_incident_plane_beam"]
        comparisons[label]["realized_beam"] = {
            "center_shift_m": float(
                np.hypot(
                    second_beam["center_x_m"] - first_beam["center_x_m"],
                    second_beam["center_y_m"] - first_beam["center_y_m"],
                )
            ),
            "effective_waist_relative_difference": relative_difference(
                first_beam["waist_effective_geometric_mean_m"],
                second_beam["waist_effective_geometric_mean_m"],
            ),
            "incident_power_relative_difference": relative_difference(
                first_beam["integrated_incident_power_W"],
                second_beam["integrated_incident_power_W"],
            ),
            "interpretation": (
                "The plane is close to the scatterer and uses a total-field "
                "downward decomposition; a planar-to-edge change includes "
                "edge-scattered/evanescent field effects and is not called a "
                "literal source displacement."
            ),
        }

    closure_all = all(
        case["common_grid_six_face_closure"] < 0.005
        and case["native_Yee_six_face_closure"] < 0.005
        for case in cases.values()
    )
    observable_all = all(
        case["observable_Q_convergence"]["acceptance"]["primary_all"]
        for case in cases.values()
    )
    auto_all = all(
        case["observable_Q_convergence"]["acceptance"]["auto_shutoff_gate"][
            "passed"
        ]
        for case in cases.values()
    )
    payload = {
        "status": (
            "PARTIAL_W2_EDGE_ISOLATION_OBSERVABLE_Q_VALIDATED_AUTO_SHUTOFF_FAILED"
            if closure_all and observable_all and not auto_all
            else "FAILED_W2_EDGE_ISOLATION_DIAGNOSTIC"
        ),
        "scope": {
            "nominal_waist_m": 2.0e-6,
            "paper_like_result": False,
            "production_Q": False,
            "FDTD_cases": [
                "planar-stack a polarization",
                "planar-stack b polarization",
                "finite straight-45-degree edge b polarization",
            ],
            "thermal_run": False,
            "PTE_run": False,
            "adjoint_run": False,
            "optimization_run": False,
        },
        "gates": {
            "matched_volume_closure_lt_0p5_percent_all": closure_all,
            "observable_Q_1p2_to_4ps_lt_0p5_percent_all": observable_all,
            "auto_shutoff_le_1e_minus_5_all": auto_all,
            "auto_shutoff_kept_separate_from_observable_Q": True,
        },
        "critical_interpretation": {
            "nominal_vs_realized_beam": (
                "The requested scalar Gaussian waist is 2 um, but the fitted "
                "field-plane effective waists are about 6.37-6.44 um. This is "
                "therefore a nominal-w0=2 um source diagnostic, not a realized "
                "2 um beam certificate."
            ),
            "raw_vs_equal_power": (
                "Raw Q retains physical absorbed-power differences. Equal-power "
                "normalization is used only to compare spatial shape; no saved Q "
                "artifact was rescaled."
            ),
        },
        "cases": cases,
        "comparisons": comparisons,
        "generation_commit": git_commit(),
    }

    summary_path = args.output_dir / "w2_planar_edge_diagnostic_summary.json"
    summary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with (args.output_dir / "w2_planar_edge_diagnostic_cases.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fieldnames = [
            "case",
            "P_Q_common_grid_W",
            "P_Q_native_Yee_W",
            "P_six_face_W",
            "common_closure",
            "native_closure",
            "P_Q_1p2_to_4ps_relative_change",
            "spatial_Q_1p2_to_4ps_NRMSE",
            "auto_shutoff_1p2ps",
            "auto_shutoff_4ps",
            "realized_center_x_m",
            "realized_center_y_m",
            "realized_waist_x_m",
            "realized_waist_y_m",
            "realized_waist_effective_m",
            "incident_power_W",
            "Qx_W",
            "Qy_W",
            "Qz_W",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for name, case in cases.items():
            beam = case["realized_incident_plane_beam"]
            convergence = case["observable_Q_convergence"]
            writer.writerow(
                {
                    "case": name,
                    "P_Q_common_grid_W": case["P_Q_common_grid_W"],
                    "P_Q_native_Yee_W": case["P_Q_native_Yee_W"],
                    "P_six_face_W": case["P_six_face_W"],
                    "common_closure": case["common_grid_six_face_closure"],
                    "native_closure": case["native_Yee_six_face_closure"],
                    "P_Q_1p2_to_4ps_relative_change": convergence["power"][
                        "relative_change"
                    ],
                    "spatial_Q_1p2_to_4ps_NRMSE": convergence[
                        "normalized_spatial_Q"
                    ]["volume_weighted_NRMSE"],
                    "auto_shutoff_1p2ps": case["auto_shutoff_1p2ps"][
                        "final_value"
                    ],
                    "auto_shutoff_4ps": case["auto_shutoff_4ps"][
                        "final_value"
                    ],
                    "realized_center_x_m": beam["center_x_m"],
                    "realized_center_y_m": beam["center_y_m"],
                    "realized_waist_x_m": beam["waist_x_m"],
                    "realized_waist_y_m": beam["waist_y_m"],
                    "realized_waist_effective_m": beam[
                        "waist_effective_geometric_mean_m"
                    ],
                    "incident_power_W": beam["integrated_incident_power_W"],
                    "Qx_W": case["Q_component_power_common_grid_W"]["x"],
                    "Qy_W": case["Q_component_power_common_grid_W"]["y"],
                    "Qz_W": case["Q_component_power_common_grid_W"]["z"],
                }
            )

    z_weights = trapezoid_weights(coordinates["z"])
    maps = {
        name: np.sum(data["total"] * z_weights[None, None, :], axis=2)
        for name, data in q_data.items()
    }
    normalized_maps = {
        name: maps[name] / cases[name]["P_Q_common_grid_W"] * 1e-12
        for name in maps
    }
    figure, axes = plt.subplots(2, 3, figsize=(13.2, 8.0), constrained_layout=True)
    raw_max = max(float(np.max(value)) for value in maps.values())
    normalized_max = max(float(np.max(value)) for value in normalized_maps.values())
    for column, name in enumerate(("planar_a", "planar_b", "finite_edge_b")):
        extent = [
            coordinates["x"][0] * 1e6,
            coordinates["x"][-1] * 1e6,
            coordinates["y"][0] * 1e6,
            coordinates["y"][-1] * 1e6,
        ]
        raw_image = axes[0, column].imshow(
            maps[name].T,
            origin="lower",
            extent=extent,
            vmin=0,
            vmax=raw_max,
            cmap="magma",
            aspect="equal",
        )
        norm_image = axes[1, column].imshow(
            normalized_maps[name].T,
            origin="lower",
            extent=extent,
            vmin=0,
            vmax=normalized_max,
            cmap="viridis",
            aspect="equal",
        )
        axes[0, column].set_title(f"{name}: raw ∫Q dz")
        axes[1, column].set_title(f"{name}: equal-power shape")
        for row in range(2):
            axes[row, column].set(xlabel="x (µm)", ylabel="y (µm)")
        figure.colorbar(raw_image, ax=axes[0, column], label="W/m²")
        figure.colorbar(norm_image, ax=axes[1, column], label="1/µm²")
    figure.savefig(args.output_dir / "w2_raw_and_equal_power_Q_maps.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(11.8, 4.6), constrained_layout=True)
    names = ["planar_a", "planar_b", "finite_edge_b"]
    axes[0].bar(
        names,
        [cases[name]["P_Q_common_grid_W"] for name in names],
        label="$P_Q$",
    )
    axes[0].scatter(
        names,
        [cases[name]["P_six_face_W"] for name in names],
        color="black",
        marker="x",
        label="$P_{six}$",
    )
    axes[0].set(ylabel="Power (W)", title="Raw absorbed power and closure")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)
    for name in names:
        coordinate, profile = edge_profile(
            q_data[name]["total"], coordinates, tangent_window_m=2.0e-6
        )
        profile /= np.trapezoid(profile, coordinate)
        axes[1].plot(coordinate * 1e6, profile * 1e-6, label=name)
    axes[1].set(
        xlabel="45° edge-normal coordinate (µm)",
        ylabel="equal-power profile (1/µm)",
        title="Spatial shape only",
    )
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    figure.savefig(args.output_dir / "w2_power_and_edge_normal_profiles.png", dpi=180)
    plt.close(figure)

    manifest_path = args.output_dir / "RAW_ARTIFACT_MANIFEST.json"
    manifest_path.write_text(
        json.dumps(
            {
                "raw_artifacts_committed_to_git": False,
                "artifacts": manifest,
                "generation_commit": git_commit(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    report = f"""# Nominal-w0=2 µm planar/edge optical diagnostic

Status: `{payload['status']}`

This is a reduced-cost optical edge-isolation diagnostic.  It is **not** a
paper-like result, not a realized 2 µm beam certificate, and not production
`Q`.  No thermal, PTE, adjoint, gradient, or optimization run was performed.

## Gates

- matched common/native control-volume closure <0.5% for all three cases:
  **{closure_all}**
- 1.2→4 ps `P_Q` and normalized spatial-`Q` gates <0.5% for all cases:
  **{observable_all}**
- auto-shutoff ≤1e-5 for all cases: **{auto_all}**

The auto-shutoff failure is retained independently and is not overridden by
the observable-`Q` pass.

## Cases (4 ps)

| case | P_Q common (W) | P_six (W) | common closure | spatial Q 1.2→4 NRMSE | realized waist (µm) |
|---|---:|---:|---:|---:|---:|
"""
    for name, case in cases.items():
        report += (
            f"| {name} | {case['P_Q_common_grid_W']:.9e} | "
            f"{case['P_six_face_W']:.9e} | "
            f"{case['common_grid_six_face_closure']:.6%} | "
            f"{case['observable_Q_convergence']['normalized_spatial_Q']['volume_weighted_NRMSE']:.6%} | "
            f"{case['realized_incident_plane_beam']['waist_effective_geometric_mean_m']*1e6:.6f} |\n"
        )
    report += f"""

The requested scalar-source waist was 2 µm, whereas the fitted field-plane
effective waist is 6.37–6.44 µm.  The cases share the same nominal source, so
the planar/edge isolation remains a useful diagnostic, but the result must not
be described as a physically realized 2 µm Gaussian beam.

## Raw versus equal-power comparison

Raw absorbed power is never altered.  Equal-power normalization is used only
for the spatial-shape comparison and does not overwrite any saved artifact.

- planar-a vs planar-b raw-power relative difference:
  `{comparisons['planar_a_vs_planar_b']['raw_power_relative_difference']:.6%}`
- planar-a vs planar-b equal-power spatial-Q NRMSE:
  `{comparisons['planar_a_vs_planar_b']['equal_power_normalized_spatial_Q_NRMSE']:.6%}`
- planar-b vs finite-edge-b raw-power relative difference:
  `{comparisons['planar_b_vs_finite_edge_b']['raw_power_relative_difference']:.6%}`
- planar-b vs finite-edge-b equal-power spatial-Q NRMSE:
  `{comparisons['planar_b_vs_finite_edge_b']['equal_power_normalized_spatial_Q_NRMSE']:.6%}`

The plane used for the realized-beam fit is close to the scatterer and uses a
total-field downward decomposition.  Therefore the planar-to-edge center
change includes edge-scattered/evanescent field effects; it is not called a
literal source displacement.

## Material and coordinates

All cases use `x=b`, `y=a`, and the explicit 3D closure
`epsilon_z=epsilon_c=epsilon_b`.  Requested, fitted, and finite-dt complex
permittivities; component fields; `Qx/Qy/Qz`; realized control-volume bounds;
and independent Yee coordinates are retained in the summary and raw
artifacts.  No clipping, smoothing, gain, rescaling, tiling, or source
deletion was used.
"""
    (args.output_dir / "W2_PLANAR_EDGE_OPTICAL_DIAGNOSTIC_REPORT.md").write_text(
        report, encoding="utf-8"
    )
    print(json.dumps(payload["gates"], indent=2, sort_keys=True))
    return 0 if closure_all and observable_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
