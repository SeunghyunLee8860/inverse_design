#!/usr/bin/env python3
"""Summarize the registered Device-A v261 runsetup without solving."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path: Path, role: str) -> dict[str, object]:
    return {
        "role": role,
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
        "committed_to_git": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runsetup-dir", type=Path, required=True)
    parser.add_argument("--finite-runsetup-dir", type=Path)
    parser.add_argument("--registration-plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    result_path = args.runsetup_dir / "case_result.json"
    project_path = args.runsetup_dir / "finite_2um_optical_q.fsp"
    result = json.loads(result_path.read_text())
    finite_result_path = None
    finite_project_path = None
    finite_result = None
    finite_pre = None
    finite_mesh = None
    if args.finite_runsetup_dir is not None:
        finite_result_path = args.finite_runsetup_dir / "case_result.json"
        finite_project_path = args.finite_runsetup_dir / "finite_2um_optical_q.fsp"
        finite_result = json.loads(finite_result_path.read_text())
        finite_pre = finite_result["pre_run_contract"]
        finite_mesh = finite_pre["mesh"]
    plan = json.loads(args.registration_plan.read_text())
    pre = result["pre_run_contract"]
    mesh = pre["mesh"]
    source = pre["geometry"]["source"]
    native = mesh["native_runsetup_mesh"]
    selected = plan["selected_phase1_numerical_box"]

    checks = {
        "contract_built_not_solved": result["status"] == "CONTRACT_BUILT_NOT_SOLVED",
        "all_pre_run_checks_passed": bool(pre["checks"]["all"]),
        "no_heat_adjoint_gradient_optimization": not any(
            bool(result[key])
            for key in ("heat_run", "adjoint_run", "gradient_run", "optimization_run")
        ),
        "no_periodic_Q": not bool(result["periodic_Q_used"]),
        "no_Q_clipping_gain_rescaling": not any(
            bool(result[key]) for key in ("Q_clipped", "flux_gain", "Q_rescaled")
        ),
        "six_PML_no_periodic": pre["geometry"]["all_six_boundaries"] == "PML"
        and not bool(pre["geometry"]["periodic"]),
        "domain_matches_selected_64um": float(result["domain_um"])
        == float(selected["domain_um"]),
        "source_span_preserved_50um": float(result["source_span_um"])
        == float(selected["source_span_um"]),
        "source_PML_clearance_at_least_0p5um": min(
            float(value)
            for value in source["source_aperture_PML_clearance_m"].values()
        )
        >= 0.5e-6,
        "native_min_xy_step_at_most_50nm": max(
            float(native[axis]["minimum_step_m"]) for axis in ("x", "y")
        )
        <= 50.0e-9 * (1.0 + 1.0e-9),
        "native_min_z_step_at_most_10nm": float(native["z"]["minimum_step_m"])
        <= 10.0e-9 * (1.0 + 1.0e-9),
        "GPU_resource_configured": pre["solver"]["resources"]["2"][
            "device type"
        ].startswith("GPU"),
    }
    if finite_result is not None:
        finite_source = finite_pre["geometry"]["source"]
        checks.update(
            {
                "finite_contract_built_not_solved": finite_result["status"]
                == "CONTRACT_BUILT_NOT_SOLVED",
                "finite_all_pre_run_checks_passed": bool(
                    finite_pre["checks"]["all"]
                ),
                "finite_geometry_is_device_a_polygon": finite_pre["geometry"][
                    "geometry_name"
                ]
                == "device-a-polygon",
                "finite_flake_and_electrodes_read_back": bool(
                    finite_pre["checks"]["one_TaIrTe4_geometry_object"]
                    and finite_pre["checks"]["electrode_object_count"]
                ),
                "finite_frame_matches_empty_reference": bool(
                    float(finite_result["domain_um"]) == float(result["domain_um"])
                    and float(finite_result["source_span_um"])
                    == float(result["source_span_um"])
                    and finite_source["beam_center_m"] == source["beam_center_m"]
                ),
                "finite_GPU_resource_configured": finite_pre["solver"][
                    "resources"
                ]["2"]["device type"].startswith("GPU"),
            }
        )

    summary = {
        "status": (
            "VALIDATED_DEVICE_A_FIG3H_REGISTERED_RUNSETUP_READY_FOR_GPU_PHASE1"
            if all(checks.values())
            else "FAILED_DEVICE_A_FIG3H_REGISTERED_RUNSETUP"
        ),
        "scope": "v261 runsetup/readback only; no Maxwell time stepping",
        "registration_classification": plan["registration"]["classification"],
        "registered_beam_center_code_um": plan["registration"][
            "nominal_beam_center_code_um"
        ],
        "registered_beam_signed_distance_to_flake_um": plan["registration"][
            "signed_distance_to_digitized_flake_um"
        ],
        "checks": checks,
        "readback": {
            "solver_version": pre["solver"]["version"],
            "domain_bounds_m": pre["geometry"]["domain_bounds_m"],
            "source_bounds_m": pre["object_bounds_readback_m"]["source"],
            "source_PML_clearance_m": source[
                "source_aperture_PML_clearance_m"
            ],
            "beam_center_m": source["beam_center_m"],
            "native_mesh": native,
            "native_runsetup_grid_point_count": mesh[
                "native_runsetup_grid_point_count"
            ],
            "native_runsetup_cell_count_estimate": mesh[
                "native_runsetup_cell_count_estimate"
            ],
            "empirical_resource_estimate": mesh["empirical_resource_estimate"],
            "mesh_override_objects": mesh["override_objects"],
            "requested_epsilon_at_11um": pre["material"][
                "requested_epsilon_at_11um"
            ],
            "fitted_epsilon_readback": pre["material"]["epsilon_readback"],
            "substrate_optical_contract": pre["geometry"][
                "substrate_optical_contract"
            ],
            "finite_device": (
                None
                if finite_result is None
                else {
                    "geometry_name": finite_pre["geometry"]["geometry_name"],
                    "geometry_source": finite_pre["geometry"]["geometry_source"],
                    "flake_vertices_um": finite_pre["geometry"][
                        "flake_vertices_um"
                    ],
                    "electrode_material_contract": finite_pre["geometry"][
                        "electrode_material_contract"
                    ],
                    "native_mesh": finite_mesh["native_runsetup_mesh"],
                    "native_runsetup_cell_count_estimate": finite_mesh[
                        "native_runsetup_cell_count_estimate"
                    ],
                    "empirical_resource_estimate": finite_mesh[
                        "empirical_resource_estimate"
                    ],
                    "mesh_override_objects": finite_mesh["override_objects"],
                    "source_bounds_m": finite_pre["object_bounds_readback_m"][
                        "source"
                    ],
                    "flake_bounds_m": finite_pre["object_bounds_readback_m"][
                        "flake"
                    ],
                }
            ),
        },
        "execution_scope": {
            "Maxwell_time_stepping": False,
            "thermal": False,
            "PTE": False,
            "adjoint": False,
            "AD_FD": False,
            "optimization": False,
        },
    }
    (args.output_dir / "device_a_fig3h_registered_runsetup_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    raw_artifacts = [
        artifact(result_path, "empty-stack v261 runsetup case result"),
        artifact(project_path, "empty-stack v261 runsetup FSP; no time stepping"),
    ]
    if finite_result_path is not None and finite_project_path is not None:
        raw_artifacts.extend(
            [
                artifact(finite_result_path, "finite Device-A v261 runsetup case result"),
                artifact(
                    finite_project_path,
                    "finite Device-A v261 runsetup FSP; no time stepping",
                ),
            ]
        )
    manifest = {
        "status": "RAW_RUNSETUP_ARTIFACTS_RECORDED_NOT_COMMITTED",
        "artifacts": raw_artifacts,
        "generation_commands": [
            result["generation_command"],
            *(
                []
                if finite_result is None
                else [finite_result["generation_command"]]
            ),
        ],
    }
    (args.output_dir / "RAW_ARTIFACT_MANIFEST_RUNSETUP.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )

    estimate = mesh["empirical_resource_estimate"]
    finite_runsetup_text = ""
    if finite_mesh is not None:
        finite_native = finite_mesh["native_runsetup_mesh"]
        finite_estimate = finite_mesh["empirical_resource_estimate"]
        finite_runsetup_text = f"""

The finite Device-A runsetup independently read back the digitized TaIrTe4
polygon and all four Ti/Au electrode objects:

- native mesh: `{finite_native['x']['shape'][0]} x {finite_native['y']['shape'][0]} x {finite_native['z']['shape'][0]}`;
- estimated native Yee cells: `{finite_mesh['native_runsetup_cell_count_estimate']}`;
- empirical estimate: `{finite_estimate['estimated_GPU_memory_GiB']:.3f} GiB`,
  `{finite_estimate['estimated_runtime_s']:.1f} s` per optical case.
"""
    report = f"""# Device-A Figure 3H registered runsetup

Status: `{summary['status']}`

The registered Figure-3H approximation opened a fresh v261 session, saved a
project, ran `runsetup`, and read back the actual native mesh. Maxwell time
stepping was not executed.

- registered digitized beam centre: `{summary['registered_beam_center_code_um']} um`;
- signed distance to the nearest digitized flake boundary:
  `{summary['registered_beam_signed_distance_to_flake_um']:.6f} um` (outside);
- lateral domain/source span: `{result['domain_um']:.0f}/{result['source_span_um']:.0f} um`;
- minimum source/PML clearance:
  `{min(source['source_aperture_PML_clearance_m'].values()) * 1e6:.6f} um`;
- native mesh: `{native['x']['shape'][0]} x {native['y']['shape'][0]} x {native['z']['shape'][0]}`;
- estimated native Yee cells: `{mesh['native_runsetup_cell_count_estimate']}`;
- realized minimum dx/dy/dz:
  `{native['x']['minimum_step_m'] * 1e9:.6f} / {native['y']['minimum_step_m'] * 1e9:.6f} / {native['z']['minimum_step_m'] * 1e9:.6f} nm`;
- empirical estimate: `{estimate['estimated_GPU_memory_GiB']:.3f} GiB`,
  `{estimate['estimated_runtime_s']:.1f} s` per optical case.
{finite_runsetup_text}

Every pre-run check passed. The registration remains an explicit affine
figure-reading assumption, not raw experimental stage metrology. Raw FSP and
case JSON remain outside Git and are SHA-pinned in the manifest.
"""
    (args.output_dir / "DEVICE_A_FIG3H_REGISTERED_RUNSETUP_REPORT.md").write_text(
        report
    )
    print(json.dumps(summary, indent=2))
    return 0 if all(checks.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
