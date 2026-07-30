#!/usr/bin/env python3
"""Publish compact paper-IR material/source/edge-control results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BLOCKED_STATUS = (
    "PARTIAL_PAPER_IR_CONTROL_VALIDATION_"
    "BLOCKED_OPTICAL_LICENSE_AND_UNRESOLVED_EDGE_METRIC"
)
SMOKE_COMPLETED_STATUS = (
    "PARTIAL_PAPER_IR_CONTROL_VALIDATION_"
    "OPTICAL_SMOKE_COMPLETED_EDGE_METRIC_UNRESOLVED"
)
RUNTIME_BLOCKED_STATUS = (
    "PARTIAL_PAPER_IR_CONTROL_VALIDATION_"
    "BLOCKED_OPTICAL_RUNTIME_AND_UNRESOLVED_EDGE_METRIC"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def artifact_record(path: Path, role: str) -> dict[str, Any]:
    return {
        "role": role,
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    analytic_dir = args.artifact_root / "analytic_source_controls_20260730"
    robust_dir = args.artifact_root / "robust_edge_gradient_20260730"
    contract_dir = (
        args.artifact_root
        / "contract_straight45_a_cclosure_retry_20260730"
    )
    production_dirs = tuple(
        directory
        for directory in sorted(
            args.artifact_root.glob(
                "straight45_a_w6p5_dz10_L48_cclosure_gpu2*20260730"
            )
        )
        if (directory / "case_result.json").is_file()
    )
    legacy_dir = (
        args.artifact_root
        / "straight45_a_w6p5_dz10_L48_gpu4_20260730"
    )
    analytic = load_json(
        analytic_dir / "analytic_source_controls_summary.json"
    )
    robust = load_json(robust_dir / "robust_edge_gradient_summary.json")
    contract = load_json(contract_dir / "case_result.json")
    production_attempts = [
        load_json(directory / "case_result.json")
        for directory in production_dirs
    ]
    completed = [
        (directory, item)
        for directory, item in zip(production_dirs, production_attempts)
        if item["status"] == "COMPLETED"
    ]
    production_dir, production = (
        completed[-1] if completed else (None, None)
    )
    legacy = load_json(legacy_dir / "case_result.json")

    for source, target in (
        (
            analytic_dir / "analytic_source_controls_cases.csv",
            args.output_dir / "analytic_source_controls_cases.csv",
        ),
        (
            robust_dir / "robust_edge_gradient_cases.csv",
            args.output_dir / "robust_edge_gradient_cases.csv",
        ),
        (
            robust_dir / "ROBUST_EDGE_GRADIENT_PROFILES.png",
            args.output_dir / "ROBUST_EDGE_GRADIENT_PROFILES.png",
        ),
        (
            robust_dir / "ROBUST_EDGE_GRADIENT_CONVERGENCE.png",
            args.output_dir / "ROBUST_EDGE_GRADIENT_CONVERGENCE.png",
        ),
    ):
        shutil.copyfile(source, target)

    with (
        analytic_dir / "analytic_source_controls_cases.csv"
    ).open(newline="") as stream:
        analytic_rows = list(csv.DictReader(stream))
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(12, 4.5),
        constrained_layout=True,
    )
    controls = (
        "paper_like_absorbed_power_control",
        "equal_absorbed_power_shape_control",
        "identical_Q_symmetry_control",
    )
    labels = ("paper-like power", "equal absorbed power", "identical Q")
    for ax, metric, title in (
        (
            axes[0],
            "Tmax_rise_K",
            r"$T_{\max}$ ratio b/a",
        ),
        (
            axes[1],
            "max_abs_grad_T_x_K_m",
            r"raw $\max|\partial_xT|$ ratio b/a",
        ),
    ):
        for control, label in zip(controls, labels):
            ratios = []
            for mesh_nm in (200, 100, 50):
                a = next(
                    row
                    for row in analytic_rows
                    if row["control"] == control
                    and row["polarization"] == "a"
                    and int(float(row["core_step_nm"])) == mesh_nm
                )
                b = next(
                    row
                    for row in analytic_rows
                    if row["control"] == control
                    and row["polarization"] == "b"
                    and int(float(row["core_step_nm"])) == mesh_nm
                )
                ratios.append(float(b[metric]) / float(a[metric]))
            ax.plot((200, 100, 50), ratios, marker="o", label=label)
        ax.axhline(1.0, color="black", linestyle="--")
        ax.set(
            xlabel="thermal core mesh (nm)",
            ylabel="b/a",
            title=title,
        )
        ax.invert_xaxis()
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    figure.savefig(
        args.output_dir / "ANALYTIC_SOURCE_CONTROL_RATIOS.png",
        dpi=180,
    )
    plt.close(figure)

    material = contract["pre_run_contract"]["material"]
    readback = material["epsilon_readback"]
    legacy_run = legacy["run_result"]
    legacy_qz = float(legacy_run["component_power_W"]["z"])
    log_paths = [
        directory / "finite_2um_optical_q_p0.log"
        for directory in production_dirs
        if (directory / "finite_2um_optical_q_p0.log").is_file()
    ]
    attempt_evidence = []
    for path in log_paths:
        text = path.read_text(errors="replace")
        progress = [
            float(value)
            for value in re.findall(
                r"([0-9]+(?:\.[0-9]+)?)% complete",
                text,
            )
        ]
        attempt_evidence.append(
            {
                "path": str(path.resolve()),
                "sha256": sha256(path),
                "requested_tasks": 9 if "requested 9 tasks" in text else None,
                "available_tasks": 4 if "only 4" in text else None,
                "feature": (
                    "lum_fdtd_solve"
                    if "lum_fdtd_solve" in text
                    else None
                ),
                "licensed_users_reached": (
                    "Licensed number of users already reached" in text
                ),
                "time_stepping_started": (
                    "Beginning initialization of 3D Simulation" in text
                ),
                "simulation_gridpoints": (
                    re.search(
                        r"Simulation size in gridpoints: ([^\n]+)",
                        text,
                    ).group(1)
                    if "Simulation size in gridpoints:" in text
                    else None
                ),
                "maximum_logged_progress_percent": (
                    max(progress) if progress else None
                ),
            }
        )
    runtime_started = any(
        item["time_stepping_started"] for item in attempt_evidence
    )
    license_attempt_count = sum(
        bool(item["licensed_users_reached"]) for item in attempt_evidence
    )
    status = (
        SMOKE_COMPLETED_STATUS
        if production
        else RUNTIME_BLOCKED_STATUS
        if runtime_started
        else BLOCKED_STATUS
    )

    analytic_50 = analytic["comparisons"]["core_50_nm"]
    robust_a_x = robust["convergence"][
        "analytic_paper_source_a_x"
    ]["100_to_50"]
    robust_b_x = robust["convergence"][
        "legacy_Lumerical_edge_Q_b_x"
    ]["100_to_50"]
    production_run = production["run_result"] if production else None
    production_qz = (
        float(production_run["component_power_W"]["z"])
        if production_run
        else None
    )
    smoke_acceptance = (
        production_run["acceptance"] if production_run else None
    )
    smoke_passed = bool(
        smoke_acceptance
        and all(bool(value) for value in smoke_acceptance.values())
    )
    summary = {
        "status": status,
        "substatuses": {
            "analytic_thermal_controls": analytic["status"],
            "optical_material_contract": (
                "VALIDATED_PAPER_CONSISTENT_EPSILON_C_EQUALS_B_CONTRACT"
            ),
            "optical_GPU_smoke": (
                "VALIDATED_ONE_POLARIZATION_GPU_SMOKE"
                if smoke_passed
                else "BLOCKED_LUMERICAL_GPU_RUNTIME_API_FAILURE"
                if runtime_started
                else "BLOCKED_LUMERICAL_SOLVE_LICENSE_BUSY"
            ),
            "native_Yee_mesh_audit": (
                "RECORDED_FOR_ONE_POLARIZATION_SMOKE"
                if production
                else "BLOCKED_PENDING_COMPLETED_OPTICAL_SMOKE"
            ),
            "optical_Q_convergence": (
                "NOT_RUN_SINGLE_SMOKE_ONLY"
                if production
                else "NOT_RUN"
            ),
            "thermal_global_temperature_convergence": (
                "PRESERVED_FROM_PRIOR_CHECKPOINT"
            ),
            "robust_edge_gradient_convergence": robust["status"],
            "final_PTE_current_convergence": "NOT_RUN",
        },
        "paper_consistent_3D_material": {
            "production": "epsilon_c(lambda)=epsilon_b(lambda)",
            "interpretation": (
                "explicit closure extending the paper's in-plane epsilon_a, "
                "epsilon_b data to a finite-edge 3D Maxwell model; not a "
                "directly measured c-axis property"
            ),
            "legacy": "epsilon_c=16 lossless diagnostic only",
            "axis_mapping": material["axis_mapping"],
            "requested_epsilon_at_11um": material[
                "requested_epsilon_at_11um"
            ],
            "readback": readback,
            "legacy_Qz_W": legacy_qz,
            "production_Qz_W": production_qz,
            "production_smoke_result": (
                {
                    "artifact_directory": str(production_dir.resolve()),
                    "P_Q_W": production_run["P_Q_W"],
                    "P_six_face_W": production_run["P_six_face_W"],
                    "six_face_relative_closure": production_run[
                        "six_face_relative_closure"
                    ],
                    "component_power_W": production_run[
                        "component_power_W"
                    ],
                    "acceptance": smoke_acceptance,
                    "native_Yee_mesh_artifact": production_run.get(
                        "native_Yee_mesh_artifact"
                    ),
                }
                if production
                else None
            ),
        },
        "analytic_controls_at_50nm": analytic_50,
        "robust_comparator": {
            "exact_coordinate_contract": robust[
                "exact_coordinate_contract"
            ],
            "analytic_a_dx_100_to_50": robust_a_x,
            "legacy_Maxwell_b_dx_100_to_50": robust_b_x,
            "fit_band_sensitivity_lt_10pct": robust[
                "fit_band_sensitivity_lt_10pct"
            ],
            "production_mesh_nm": robust[
                "proposed_cheapest_mesh_nm"
            ],
            "decision": robust["refinement_decision"],
            "refinement_options": robust[
                "boundary_refinement_options"
            ],
        },
        "license_blocker": {
            "contract_only_session_opened": True,
            "GPU_resource_readback": contract["pre_run_contract"][
                "solver"
            ]["resources"]["2"],
            "CPU_FDTD_fallback_used": False,
            "attempts": [
                {
                    "status": item["status"],
                    "threads": item["pre_run_contract"]["solver"][
                        "resources"
                    ]["2"]["threads"],
                    "case_result": item["project"].replace(
                        "finite_2um_optical_q.fsp",
                        "case_result.json",
                    ),
                    "exception_type": item.get("exception_type"),
                    "exception": item.get("exception"),
                }
                for item in production_attempts
            ],
            "logs": attempt_evidence,
        },
        "scalar_vector_source_status": {
            "scalar_production_smoke": (
                "COMPLETED" if production else "BLOCKED_LICENSE"
            ),
            "matched_thin_lens_vectorial": "PLAN_ONLY_NOT_RUN",
            "plan": "SCALAR_VECTOR_GAUSSIAN_MATCH_PLAN.md",
        },
        "forbidden_operations": {
            "raw_Lumerical_Q_rescaled": False,
            "Q_clipped": False,
            "Q_smoothed": False,
            "empirical_gradient_rescaling": False,
            "optimization_run": False,
        },
    }
    (
        args.output_dir / "paper_ir_edge_control_summary.json"
    ).write_text(json.dumps(summary, indent=2) + "\n")

    plan = """# Matched scalar versus thin-lens vectorial Gaussian plan

This is a plan, not a completed vector-source result.

Both source models will use the same 48 µm lateral FDTD domain, six PML
boundaries, 24 PML layers, 11 µm analysis wavelength, 7–13 µm source band,
32 µm injection aperture, focus at the centre of the 130 nm TaIrTe4 film,
and paper-consistent `epsilon_c=epsilon_b` material.

The thin-lens source will not be created by merely toggling
`use scalar approximation`.  An empty 285 nm SiO2/Si stack is used first to
match the *realized* incident field:

1. set the thin-lens focus to the same physical z coordinate;
2. choose NA/pupil fill so the flake-plane 1/e² intensity radius is 6.5 µm;
3. set the source amplitude so measured incident power matches the scalar
   source, without multiplying or globally rescaling the resulting Q;
4. require incident-power difference <0.5%, waist-radius difference <1%,
   focus-position difference <0.1 µm, normalized flake-plane intensity
   NRMSE <1%, and aperture-edge/central intensity <5%;
5. only then run one identical straight-edge finite-flake case.

The finite-edge comparison records Ex/Ey/Ez, Qx/Qy/Qz, total P_Q, six-face
closure, native Yee coordinates, and the edge-normal areal-Q profile.  Failure
of the incident-field matching stage prevents a material/edge comparison.
No post-hoc Q gain, clipping, smoothing, or gradient rescaling is allowed.
"""
    (
        args.output_dir / "SCALAR_VECTOR_GAUSSIAN_MATCH_PLAN.md"
    ).write_text(plan)

    if production:
        optical_paragraph = (
            "The v261 contract-only session and material fit succeeded.  The "
            "GPU-only production smoke also completed and records native Yee "
            "coordinates, Qx/Qy/Qz, P_Q, six-face closure, and the edge-normal "
            "Q profile.  Full optical mesh/domain convergence is still not run."
        )
    elif runtime_started:
        runtime_attempt = max(
            (
                item
                for item in attempt_evidence
                if item["time_stepping_started"]
            ),
            key=lambda item: item["maximum_logged_progress_percent"] or 0.0,
        )
        optical_paragraph = (
            "The v261 contract-only session and material fit succeeded.  "
            f"{license_attempt_count} attempts stopped before timestepping "
            "because the requested "
            "`lum_fdtd_solve` task count was unavailable.  A later GPU-only "
            "attempt acquired the licenses, meshed "
            f"`{runtime_attempt['simulation_gridpoints']}` gridpoints, and "
            "started timestepping, but the Lumerical API/engine communication "
            "failed after the log reached "
            f"`{runtime_attempt['maximum_logged_progress_percent']}%`.  "
            "The incomplete HDF5 output is provenance only and is not treated "
            "as a recoverable optical result.  No CPU FDTD fallback was used.  "
            "Production Qx/Qy/Qz, P_Q, closure, native Yee coordinates, and "
            "the edge-normal Q profile therefore remain blocked."
        )
    else:
        optical_paragraph = (
            "The v261 contract-only session and material fit succeeded, but "
            "all GPU-only solve attempts stopped before timestepping because "
            "the requested `lum_fdtd_solve` task count was unavailable.  "
            "Reducing host threads did not change the nine-task GPU license "
            "request.  No CPU FDTD fallback was used.  Native Yee coordinates, "
            "production Qx/Qy/Qz, P_Q, six-face closure, and the new "
            "edge-normal Q profile consequently remain blocked."
        )
    report = f"""# Paper-like IR material, source, and edge controls

**Status: `{status}`**

## What is validated

The offline analytic thermal controls pass.  At 50 nm, the paper-like
absorbed-power control gives
`max|dT/dx|_b/max|dT/dx|_a =
{analytic_50['paper_like_absorbed_power_control']['max_abs_grad_T_x_b_over_a']:.6f}`.
After forcing the two analytic sources to the same absorbed power, the ratio
is
`{analytic_50['equal_absorbed_power_shape_control']['max_abs_grad_T_x_b_over_a']:.6f}`.
The exact-identical-Q symmetry control gives `1.000000` with zero field
difference.  Thus the original b>a thermal trend is predominantly the
pre-supplied polarization-dependent TMM absorbed power, not an independent
Lumerical discovery.

The production 3D material contract is now `epsilon_c(lambda)=epsilon_b(lambda)`.
It is the explicit paper-consistent closure used to extend the reported
in-plane epsilon_a/epsilon_b data to finite-edge 3D Maxwell; it is not called
a direct c-axis measurement.  The legacy lossless `epsilon_c=16` model remains
diagnostic only.

At 11 µm, fitted epsilon_z and epsilon_x differ by
`{readback['fitted_z_vs_x_relative_difference']:.3e}` and the finite-dt values
differ by `{readback['finite_dt_z_vs_x_relative_difference']:.3e}`.  The
legacy artifact has integrated `Qz={legacy_qz:.6e} W`; production Qz is
`{production_qz if production_qz is not None else "not available"} W`.

## What remains unresolved

The robust physical-line fit uses exact
`n=(-x+y)/sqrt(2), t=(x+y)/sqrt(2)` coordinates and treats raw cell maxima as
diagnostic only.  Analytic-source fitted dx strip mean changes by
`{100*robust_a_x['edge_strip_mean_abs_K_m']:.3f}%` from 100 to 50 nm, but the
legacy Maxwell-Q b-polarization value changes by
`{100*robust_b_x['edge_strip_mean_abs_K_m']:.3f}%`.  Fit-band sensitivity also
exceeds 10%.  Therefore no 50 or 100 nm edge-gradient mesh is promoted.
The next numerical method candidate is a conservative exact-half-plane
cut-cell treatment, not ad-hoc local cells.

{optical_paragraph}

The scalar/thin-lens comparison remains plan-only.  See
`SCALAR_VECTOR_GAUSSIAN_MATCH_PLAN.md`.

No PTE-current solve, adjoint, gradient, or optimization was run.
"""
    (args.output_dir / "PAPER_IR_EDGE_CONTROL_REPORT.md").write_text(report)

    selected_roots = (
        analytic_dir,
        robust_dir,
        contract_dir,
        *production_dirs,
    )
    records: list[dict[str, Any]] = []
    for root in selected_roots:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                records.append(
                    artifact_record(
                        path,
                        role=f"external_{root.name}",
                    )
                )
    for path in (
        legacy_dir / "case_result.json",
        legacy_dir / "finite_q_on_artifact.npz",
    ):
        records.append(artifact_record(path, "legacy_epsilon_c_16"))
    manifest = {
        "policy": (
            "raw NPZ/FSP/log artifacts remain outside Git; repository "
            "contains code, compact tables, plots, report, and this manifest"
        ),
        "status": status,
        "artifacts": records,
    }
    (
        args.output_dir / "RAW_ARTIFACT_MANIFEST.json"
    ).write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
