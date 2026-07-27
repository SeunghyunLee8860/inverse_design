#!/usr/bin/env python3
"""Diagnose nonuniform imported-density P_Q AD-FD on component Yee grids."""

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
    physical_state,
    run_forward_density,
)
from .run_v261_large_background_mixed_optical_adfd import (
    absorption_objective_and_source,
    component_volumes,
    fieldregion_profile,
    monitor_electric,
    prepare_adjoint_layout,
    run_adjoint,
)
from .run_v261_large_background_tfsf_forward import sha256
from .yee_material_jacobian import SparseYeeMaterialJacobian


STATUS = "DIAGNOSTIC_NONUNIFORM_IMPORTED_PQ_ADFD"
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
    parser.add_argument("--jacobian-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--gpu-device", default="GPU 1")
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


def main() -> int:
    args = parse_args()
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "nonuniform_optical_pq_adfd_result.json"
    result: dict[str, object] = {
        "status": "BLOCKED_NONUNIFORM_IMPORTED_PQ_ADFD_NOT_RUN",
        "passed": False,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "new_forward_Maxwell_solves": 0,
        "new_adjoint_Maxwell_solves": 1,
        "empirical_normalization": False,
        "gradient_rescaling": False,
        "optimization_run": False,
    }
    fdtd = None
    started = time.monotonic()
    try:
        base_path = checked(args.base_forward, args.base_sha256)
        stage_path = checked(
            args.stage10_result, args.stage10_sha256
        )
        stage = json.loads(stage_path.read_text())
        jdir = Path(args.jacobian_dir).expanduser().resolve()
        matrices = {
            component: sparse.load_npz(jdir / f"J_{component}.npz")
            for component in "xyz"
        }
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
        objective, source, components = absorption_objective_and_source(
            base["electric"], base["epsilon"], base["grid"]
        )
        profile, profile_scale = fieldregion_profile(source)
        fdtd.load(str(base_path))
        template = output / "pq_adjoint_template.fsp"
        source_meta = prepare_adjoint_layout(
            fdtd,
            grid=base["grid"],
            profile=profile,
            template=template,
        )
        adjoint_project = output / "pq_adjoint_gpu.fsp"
        adjoint = run_adjoint(
            fdtd,
            template=template,
            project=adjoint_project,
            engine="GPU",
            threads=args.threads,
            gpu_device=args.gpu_device,
        )
        fdtd.load(str(adjoint_project))
        adjoint_electric, adjoint_grid = monitor_electric(
            fdtd, PABS_FIELD
        )
        coordinate_error = max(
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
        shape = base["electric"].shape[:3]
        operator = SparseYeeMaterialJacobian(
            density_shape=(81, 81),
            component_shapes={component: shape for component in "xyz"},
            matrices=matrices,
        )
        clipped = clipped_component_yee_volumes(
            base["grid"], DESIGN_BOUNDS_M
        )
        full = component_volumes(base["grid"])
        base_amplitude = source_meta["fieldregion_base_amplitude"]
        cotangent_indirect = {}
        cotangent_direct = {}
        omega = 2.0 * np.pi * FREQUENCY_HZ
        for index, component in enumerate("xyz"):
            forward = base["electric"][..., 0, index]
            adjoint_field = adjoint_electric[..., 0, index]
            cotangent_indirect[component] = (
                (2.0 * EPS0 / base_amplitude)
                * clipped[index]
                * forward
                * (adjoint_field * profile_scale)
            )
            cotangent_direct[component] = (
                -1j
                * 0.5
                * EPS0
                * omega
                * full[index]
                * np.abs(forward) ** 2
            )
        indirect = operator.vjp(cotangent_indirect)
        direct = operator.vjp(cotangent_direct)
        total = indirect + direct
        analytic = {
            "indirect_directional_W": float(
                np.sum(indirect * direction)
            ),
            "direct_directional_W": float(np.sum(direct * direction)),
            "total_directional_W": float(np.sum(total * direction)),
            "indirect_gradient_L2_W": float(np.linalg.norm(indirect)),
            "direct_gradient_L2_W": float(np.linalg.norm(direct)),
            "total_gradient_L2_W": float(np.linalg.norm(total)),
        }
        rows = []
        for stage_row in stage["scenarios"]["4um"]["fd_rows"]:
            step = float(stage_row["step"])
            plus = float(stage_row["plus_forward"]["P_Q_W"])
            minus = float(stage_row["minus_forward"]["P_Q_W"])
            finite_difference = (plus - minus) / (2.0 * step)
            rows.append(
                {
                    "step": step,
                    "P_Q_plus_W": plus,
                    "P_Q_minus_W": minus,
                    "finite_difference_directional_W": finite_difference,
                    "adjoint_directional_W": analytic[
                        "total_directional_W"
                    ],
                    "relative_error": relative(
                        analytic["total_directional_W"],
                        finite_difference,
                    ),
                }
            )
        result.update(
            {
                "status": STATUS,
                "passed": True,
                "P_Q_objective_from_common_grid_W": objective,
                "P_Q_native_W": base["P_Q_W"],
                "objective_relative_difference": relative(
                    objective, base["P_Q_W"]
                ),
                "component_power_W": components,
                "coordinate_mismatch_m": coordinate_error,
                "source_profile_roundtrip_max_abs_error": source_meta[
                    "source_profile_roundtrip_max_abs_error"
                ],
                "analytic": analytic,
                "FD_rows": rows,
                "base_forward": {
                    "path": str(base_path),
                    "byte_size": base_path.stat().st_size,
                    "sha256": args.base_sha256,
                },
                "adjoint_FSP": {
                    "path": str(adjoint_project),
                    "byte_size": adjoint_project.stat().st_size,
                    "sha256": sha256(adjoint_project),
                },
                "interpretation": (
                    "This diagnostic isolates the Maxwell/material "
                    "gradient independently of thermal/PTE coupling."
                ),
            }
        )
    except Exception as exc:
        result.update(
            {
                "status": "FAILED_NONUNIFORM_IMPORTED_PQ_ADFD_DIAGNOSTIC",
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
