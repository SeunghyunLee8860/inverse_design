#!/usr/bin/env python3
"""Audit clipped versus full Yee dual-cell measure in combined AD--FD."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import traceback

import numpy as np
from scipy import sparse

from .contract import DESIGN_BOUNDS_M
from .native_yee_q import EPS0
from .nonperiodic_yee_metric import clipped_component_yee_volumes
from .probe_v261_cpu_tfsf_device import FREQUENCY_HZ, PABS_FIELD
from .probe_v261_gpu_plane_wave_roi import load_lumapi
from .run_combined_physical_rho_pte_adfd import (
    native_weight_and_source,
    physical_state,
    run_forward_density,
)
from .run_corrected_combined_physical_rho_pte_adfd import thermal_state
from .run_v261_large_background_mixed_optical_adfd import (
    component_volumes,
    fieldregion_profile,
    monitor_electric,
)
from .run_v261_large_background_tfsf_forward import sha256
from .yee_material_jacobian import SparseYeeMaterialJacobian


STATUS_PASS = "VALIDATED_FULL_YEE_DUAL_CELL_GRADIENT_MEASURE"
STATUS_FAIL = "FAILED_FULL_YEE_DUAL_CELL_GRADIENT_MEASURE"
FLUX_SIGNS = {
    f"device_flux_{axis}_{side}": (-1.0 if side == "min" else 1.0)
    for axis in "xyz"
    for side in ("min", "max")
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-forward", required=True)
    parser.add_argument("--base-sha256", required=True)
    parser.add_argument("--component-split", required=True)
    parser.add_argument("--component-split-sha256", required=True)
    parser.add_argument("--jacobian-dir", required=True)
    parser.add_argument("--adjoint-4um", required=True)
    parser.add_argument("--adjoint-4um-sha256", required=True)
    parser.add_argument("--adjoint-6um", required=True)
    parser.add_argument("--adjoint-6um-sha256", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--threads", type=int, default=16)
    return parser.parse_args()


def checked(path_text: str, expected: str) -> Path:
    path = Path(path_text).expanduser().resolve()
    if not path.is_file() or sha256(path) != expected:
        raise RuntimeError(f"missing or SHA-mismatched artifact: {path}")
    return path


def relative(value: float, reference: float) -> float:
    return abs(value - reference) / max(
        abs(value), abs(reference), np.finfo(float).tiny
    )


def optical_gradient(
    *,
    operator: SparseYeeMaterialJacobian,
    base: dict,
    adjoint: np.ndarray,
    profile_scale: float,
    base_amplitude: float,
    volumes: dict[int, np.ndarray],
) -> np.ndarray:
    cotangent = {}
    for index, component in enumerate("xyz"):
        cotangent[component] = (
            (2.0 * EPS0 / base_amplitude)
            * volumes[index]
            * base["electric"][..., 0, index]
            * (adjoint[..., 0, index] * profile_scale)
        )
    return operator.vjp(cotangent)


def main() -> int:
    args = parse_args()
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "yee_dual_cell_measure_correction.json"
    result = {
        "status": "BLOCKED_YEE_DUAL_CELL_MEASURE_NOT_RUN",
        "passed": False,
        "new_Maxwell_solves": 0,
        "empirical_normalization": False,
        "gradient_rescaling": False,
        "optimization_run": False,
    }
    fdtd = None
    started = time.monotonic()
    try:
        base_path = checked(args.base_forward, args.base_sha256)
        split_path = checked(
            args.component_split, args.component_split_sha256
        )
        adjoints = {
            4.0: checked(args.adjoint_4um, args.adjoint_4um_sha256),
            6.0: checked(args.adjoint_6um, args.adjoint_6um_sha256),
        }
        split = json.loads(split_path.read_text())
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
        shape = base["electric"].shape[:3]
        jacobian_dir = Path(args.jacobian_dir).expanduser().resolve()
        operator = SparseYeeMaterialJacobian(
            density_shape=(81, 81),
            component_shapes={component: shape for component in "xyz"},
            matrices={
                component: sparse.load_npz(
                    jacobian_dir / f"J_{component}.npz"
                )
                for component in "xyz"
            },
        )
        volume_sets = {
            "clipped_design_box": clipped_component_yee_volumes(
                base["grid"], DESIGN_BOUNDS_M
            ),
            "full_yee_dual_cell": component_volumes(base["grid"]),
        }
        support = {}
        for index, component in enumerate("xyz"):
            active = np.asarray(
                operator.matrices[component].getnnz(axis=1) > 0
            ).reshape(shape)
            ratio = (
                volume_sets["clipped_design_box"][index][active]
                / volume_sets["full_yee_dual_cell"][index][active]
            )
            support[component] = {
                "active_J_row_count": int(np.count_nonzero(active)),
                "clipped_to_full_volume_ratio_bounds": [
                    float(np.min(ratio)),
                    float(np.max(ratio)),
                ],
                "active_rows_changed_by_clipping": int(
                    np.count_nonzero(np.abs(ratio - 1.0) > 1.0e-12)
                ),
            }
        scenarios = {}
        all_full_errors = []
        all_clipped_reproduction_errors = []
        for flake_um in (4.0, 6.0):
            kwargs, coupling, geometry, mapping, evaluation = thermal_state(
                rho, flake_um, base["native"]
            )
            _, weighted_source, pullback = native_weight_and_source(
                evaluation=evaluation,
                native=base["native"],
                mapping=mapping,
                electric=base["electric"],
                epsilon=base["epsilon"],
                frequency_Hz=float(base["grid"]["f"][0]),
            )
            _, profile_scale = fieldregion_profile(weighted_source)
            fdtd.load(str(adjoints[flake_um]))
            adjoint, adjoint_grid = monitor_electric(fdtd, PABS_FIELD)
            coordinate_mismatch = max(
                float(
                    np.max(
                        np.abs(
                            np.asarray(base["grid"][key])
                            - np.asarray(adjoint_grid[key])
                        )
                    )
                )
                for key in (
                    "x",
                    "y",
                    "z",
                    "delta_x",
                    "delta_y",
                    "delta_z",
                )
            )
            base_amplitude = float(
                fdtd.getnamed(
                    "large_background_q_fieldregion", "base amplitude"
                )
            )
            thermal_gradient = coupling.thermal_vjp(
                evaluation.gradient_rho_A
            )
            gradients = {}
            for name, volumes in volume_sets.items():
                optical = optical_gradient(
                    operator=operator,
                    base=base,
                    adjoint=adjoint,
                    profile_scale=profile_scale,
                    base_amplitude=base_amplitude,
                    volumes=volumes,
                )
                gradients[name] = {
                    "optical": optical,
                    "combined": optical + thermal_gradient,
                }
            rows = []
            split_scenario = split["scenarios"][f"{flake_um:g}um"]
            for source_row in split_scenario["rows"]:
                optical_fd = float(source_row["optical_Q_only_FD_A"])
                combined_fd = float(
                    source_row["combined_Stage10_FD_A"]
                )
                row = {"step": float(source_row["step"])}
                for name in volume_sets:
                    optical_ad = float(
                        np.sum(gradients[name]["optical"] * direction)
                    )
                    combined_ad = float(
                        np.sum(gradients[name]["combined"] * direction)
                    )
                    row[name] = {
                        "optical_directional_A": optical_ad,
                        "combined_directional_A": combined_ad,
                        "optical_relative_error": relative(
                            optical_ad, optical_fd
                        ),
                        "combined_relative_error": relative(
                            combined_ad, combined_fd
                        ),
                    }
                all_full_errors.extend(
                    (
                        row["full_yee_dual_cell"][
                            "optical_relative_error"
                        ],
                        row["full_yee_dual_cell"][
                            "combined_relative_error"
                        ],
                    )
                )
                all_clipped_reproduction_errors.append(
                    relative(
                        row["clipped_design_box"][
                            "combined_directional_A"
                        ],
                        float(
                            source_row[
                                "combined_adjoint_directional_A"
                            ]
                        ),
                    )
                )
                row["finite_difference"] = {
                    "optical_Q_only_A": optical_fd,
                    "combined_A": combined_fd,
                }
                rows.append(row)
            scenarios[f"{flake_um:g}um"] = {
                "rows": rows,
                "coordinate_mismatch_m": coordinate_mismatch,
                "native_Q_pullback": pullback,
                "adjoint_FSP": {
                    "path": str(adjoints[flake_um]),
                    "byte_size": adjoints[flake_um].stat().st_size,
                    "sha256": sha256(adjoints[flake_um]),
                },
                "full_gradient_L2_A": float(
                    np.linalg.norm(
                        gradients["full_yee_dual_cell"]["combined"]
                    )
                ),
                "clipped_gradient_L2_A": float(
                    np.linalg.norm(
                        gradients["clipped_design_box"]["combined"]
                    )
                ),
            }
        worst_full = max(all_full_errors)
        worst_reproduction = max(all_clipped_reproduction_errors)
        passed = worst_full < 1.0e-2 and worst_reproduction < 1.0e-12
        result.update(
            {
                "status": STATUS_PASS if passed else STATUS_FAIL,
                "passed": passed,
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "root_cause": (
                    "The component J_c operators already encode conformal "
                    "fill and exact design support. Clipping Yee dual-cell "
                    "dV_c to the nominal design box applied support twice."
                ),
                "correct_measure": (
                    "complete component-specific Yee dual-cell volume"
                ),
                "component_J_support": support,
                "scenarios": scenarios,
                "gates": {
                    "worst_full_measure_ADFD_relative_error": worst_full,
                    "limit": 1.0e-2,
                    "clipped_path_reproduction_relative_error": (
                        worst_reproduction
                    ),
                    "clipped_reproduction_limit": 1.0e-12,
                },
                "base_forward": {
                    "path": str(base_path),
                    "byte_size": base_path.stat().st_size,
                    "sha256": sha256(base_path),
                },
                "component_split": {
                    "path": str(split_path),
                    "byte_size": split_path.stat().st_size,
                    "sha256": sha256(split_path),
                },
                "jacobian_dir": str(jacobian_dir),
            }
        )
    except Exception as exc:
        result.update(
            {
                "status": STATUS_FAIL,
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
    print(json.dumps({"status": result["status"], "path": str(result_path)}))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
