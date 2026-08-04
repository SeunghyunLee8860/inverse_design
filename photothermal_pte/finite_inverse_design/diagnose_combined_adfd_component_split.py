#!/usr/bin/env python3
"""Split the existing Stage-10 combined FD into optical and thermal paths."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import traceback

import numpy as np

from .explicit_thermal import build_explicit_geometry, solve_explicit_forward
from .probe_v261_gpu_plane_wave_roi import load_lumapi
from .run_combined_physical_rho_pte_adfd import (
    apply_existing_mapping,
    build_native_thermal_mapping,
    coupling_for_geometry,
    physical_state,
    run_forward_density,
)
from .run_v261_large_background_tfsf_forward import sha256


STATUS = "DIAGNOSTIC_COMBINED_ADFD_COMPONENT_SPLIT"
FLUX_SIGNS = {
    f"device_flux_{axis}_{side}": (-1.0 if side == "min" else 1.0)
    for axis in "xyz"
    for side in ("min", "max")
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-forward", required=True)
    parser.add_argument("--base-sha256", required=True)
    parser.add_argument("--stage10-result", required=True)
    parser.add_argument("--stage10-sha256", required=True)
    parser.add_argument("--gradient-4um", required=True)
    parser.add_argument("--gradient-4um-sha256", required=True)
    parser.add_argument("--gradient-6um", required=True)
    parser.add_argument("--gradient-6um-sha256", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--threads", type=int, default=16)
    return parser.parse_args()


def checked(path_text: str, expected: str) -> Path:
    path = Path(path_text).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if sha256(path) != expected:
        raise RuntimeError(f"SHA-256 mismatch: {path}")
    return path


def relative(value: float, reference: float) -> float:
    return abs(value - reference) / max(
        abs(value), abs(reference), np.finfo(float).tiny
    )


def thermal_kwargs(flake_um: float) -> dict[str, float]:
    return {
        "lateral_domain_m": 32.0e-6,
        "si_depth_m": 20.0e-6,
        "flake_span_m": flake_um * 1.0e-6,
        "core_xy_cell_size_m": 100.0e-9,
        "flake_dz_m": 25.0e-9,
        "design_dz_m": 100.0e-9,
    }


def main() -> int:
    args = parse_args()
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "combined_adfd_component_split_result.json"
    result: dict[str, object] = {
        "status": "BLOCKED_COMBINED_ADFD_COMPONENT_SPLIT_NOT_RUN",
        "passed": False,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "re-read existing SHA-pinned Stage-10 baseline and +/- FSPs; "
            "split combined objective FD into optical-Q-only and "
            "thermal-material-only paths"
        ),
        "new_Maxwell_solves": 0,
        "empirical_normalization": False,
        "gradient_rescaling": False,
        "optimization_run": False,
    }
    fdtd = None
    started = time.monotonic()
    try:
        base_path = checked(args.base_forward, args.base_sha256)
        stage10_path = checked(
            args.stage10_result, args.stage10_sha256
        )
        gradient_paths = {
            4.0: checked(args.gradient_4um, args.gradient_4um_sha256),
            6.0: checked(args.gradient_6um, args.gradient_6um_sha256),
        }
        stage10 = json.loads(stage10_path.read_text())
        if stage10["status"] != "FAILED_COMBINED_PHYSICAL_RHO_PTE_ADFD":
            raise RuntimeError("unexpected Stage-10 diagnostic status")
        rho, direction = physical_state()
        lumapi = load_lumapi()
        fdtd = lumapi.FDTD(hide=True, serverArgs={"platform": "offscreen"})
        base = run_forward_density(
            fdtd,
            rho=rho,
            project=base_path,
            threads=args.threads,
            flux_signs=FLUX_SIGNS,
            reuse_completed=True,
        )
        scenario_data = {}
        for flake_um in (4.0, 6.0):
            initial = build_explicit_geometry(
                np.full((20, 20), 0.5),
                **thermal_kwargs(flake_um),
            )
            coupling = coupling_for_geometry(initial)
            thermal_rho = coupling.thermal(rho)
            geometry = build_explicit_geometry(
                thermal_rho, **thermal_kwargs(flake_um)
            )
            mapping = build_native_thermal_mapping(
                base["native"], geometry
            )
            with np.load(gradient_paths[flake_um]) as gradient:
                optical_adjoint = float(
                    np.sum(gradient["optical_gradient_A"] * direction)
                )
                thermal_adjoint = float(
                    np.sum(gradient["thermal_gradient_A"] * direction)
                )
                combined_adjoint = float(
                    np.sum(gradient["combined_gradient_A"] * direction)
                )
            scenario_data[flake_um] = {
                "coupling": coupling,
                "thermal_rho": thermal_rho,
                "geometry": geometry,
                "mapping": mapping,
                "optical_adjoint_directional_A": optical_adjoint,
                "thermal_adjoint_directional_A": thermal_adjoint,
                "combined_adjoint_directional_A": combined_adjoint,
                "rows": [],
            }
        stage_rows = {
            float(row["step"]): row
            for row in stage10["scenarios"]["4um"]["fd_rows"]
        }
        for step in sorted(stage_rows, reverse=True):
            pair = {}
            for label in ("plus", "minus"):
                project_record = stage_rows[step][
                    f"{label}_forward"
                ]["project"]
                project = checked(
                    project_record["path"], project_record["sha256"]
                )
                fdtd.switchtolayout()
                perturbed = run_forward_density(
                    fdtd,
                    rho=(
                        rho
                        + (step if label == "plus" else -step)
                        * direction
                    ),
                    project=project,
                    threads=args.threads,
                    flux_signs=FLUX_SIGNS,
                    reuse_completed=True,
                )
                pair[label] = {"forward": perturbed, "objectives": {}}
                sign = 1.0 if label == "plus" else -1.0
                for flake_um, data in scenario_data.items():
                    mapped = apply_existing_mapping(
                        perturbed["native"],
                        data["mapping"],
                        data["geometry"],
                    )
                    optical_only = solve_explicit_forward(
                        rho=data["thermal_rho"],
                        source_W_m3=mapped["source_W_m3"],
                        **thermal_kwargs(flake_um),
                    )
                    thermal_only = solve_explicit_forward(
                        rho=data["coupling"].thermal(
                            rho + sign * step * direction
                        ),
                        source_W_m3=data["mapping"]["source_W_m3"],
                        **thermal_kwargs(flake_um),
                    )
                    pair[label]["objectives"][flake_um] = {
                        "optical_Q_only_A": float(
                            optical_only.objective_A
                        ),
                        "thermal_material_only_A": float(
                            thermal_only.objective_A
                        ),
                        "mapping_relative_power_error": mapped[
                            "relative_power_error"
                        ],
                        "optical_energy_balance_relative_error": float(
                            optical_only.solved.energy_balance_relative_error
                        ),
                        "thermal_energy_balance_relative_error": float(
                            thermal_only.solved.energy_balance_relative_error
                        ),
                        "worst_linear_residual_relative": max(
                            float(
                                optical_only.solved.linear_residual_relative
                            ),
                            float(
                                thermal_only.solved.linear_residual_relative
                            ),
                        ),
                    }
            for flake_um, data in scenario_data.items():
                plus = pair["plus"]["objectives"][flake_um]
                minus = pair["minus"]["objectives"][flake_um]
                optical_fd = (
                    plus["optical_Q_only_A"]
                    - minus["optical_Q_only_A"]
                ) / (2.0 * step)
                thermal_fd = (
                    plus["thermal_material_only_A"]
                    - minus["thermal_material_only_A"]
                ) / (2.0 * step)
                combined_stage_row = next(
                    row
                    for row in stage10["scenarios"][f"{flake_um:g}um"][
                        "fd_rows"
                    ]
                    if float(row["step"]) == step
                )
                combined_fd = float(
                    combined_stage_row[
                        "finite_difference_directional_A"
                    ]
                )
                split_sum = optical_fd + thermal_fd
                row = {
                    "step": step,
                    "optical_Q_only_FD_A": optical_fd,
                    "thermal_material_only_FD_A": thermal_fd,
                    "split_FD_sum_A": split_sum,
                    "combined_Stage10_FD_A": combined_fd,
                    "split_sum_vs_combined_FD_relative_error": relative(
                        split_sum, combined_fd
                    ),
                    "optical_adjoint_directional_A": data[
                        "optical_adjoint_directional_A"
                    ],
                    "thermal_adjoint_directional_A": data[
                        "thermal_adjoint_directional_A"
                    ],
                    "combined_adjoint_directional_A": data[
                        "combined_adjoint_directional_A"
                    ],
                    "optical_adjoint_vs_FD_relative_error": relative(
                        data["optical_adjoint_directional_A"], optical_fd
                    ),
                    "thermal_adjoint_vs_FD_relative_error": relative(
                        data["thermal_adjoint_directional_A"], thermal_fd
                    ),
                    "combined_adjoint_vs_Stage10_FD_relative_error": (
                        relative(
                            data["combined_adjoint_directional_A"],
                            combined_fd,
                        )
                    ),
                    "plus": plus,
                    "minus": minus,
                }
                data["rows"].append(row)
                print(
                    "COMBINED_SPLIT "
                    f"flake={flake_um:g}um h={step:g} "
                    f"optical_error="
                    f"{row['optical_adjoint_vs_FD_relative_error']:.6e} "
                    f"thermal_error="
                    f"{row['thermal_adjoint_vs_FD_relative_error']:.6e} "
                    f"combined_error="
                    f"{row['combined_adjoint_vs_Stage10_FD_relative_error']:.6e}",
                    flush=True,
                )
        published_scenarios = {}
        for flake_um, data in scenario_data.items():
            published_scenarios[f"{flake_um:g}um"] = {
                key: value
                for key, value in data.items()
                if key
                not in {
                    "coupling",
                    "thermal_rho",
                    "geometry",
                    "mapping",
                }
            }
        result.update(
            {
                "status": STATUS,
                "passed": True,
                "base_forward": {
                    "path": str(base_path),
                    "byte_size": base_path.stat().st_size,
                    "sha256": args.base_sha256,
                },
                "stage10_result": {
                    "path": str(stage10_path),
                    "byte_size": stage10_path.stat().st_size,
                    "sha256": args.stage10_sha256,
                },
                "gradient_artifacts": {
                    f"{flake_um:g}um": {
                        "path": str(path),
                        "byte_size": path.stat().st_size,
                        "sha256": (
                            args.gradient_4um_sha256
                            if flake_um == 4.0
                            else args.gradient_6um_sha256
                        ),
                    }
                    for flake_um, path in gradient_paths.items()
                },
                "scenarios": published_scenarios,
                "interpretation": (
                    "This diagnostic does not validate combined AD-FD. "
                    "It identifies which differentiated path remains "
                    "inconsistent before any new multi-direction Maxwell "
                    "FD sweep."
                ),
            }
        )
    except Exception as exc:
        result.update(
            {
                "status": (
                    "FAILED_COMBINED_ADFD_COMPONENT_SPLIT_DIAGNOSTIC"
                ),
                "passed": False,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        if fdtd is not None:
            try:
                fdtd.close()
            except Exception:
                pass
        result["wall_s"] = time.monotonic() - started
        result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": result["status"],
                "passed": result["passed"],
                "result_path": str(result_path),
            }
        )
    )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
