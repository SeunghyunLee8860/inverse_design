#!/usr/bin/env python3
"""Test component-collocated FieldRegion source against weighted optical FD."""

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
from .explicit_thermal import build_explicit_geometry, evaluate_explicit_thermal
from .native_yee_q import EPS0
from .nonperiodic_yee_metric import clipped_component_yee_volumes
from .probe_v261_cpu_tfsf_device import FREQUENCY_HZ, PABS_FIELD
from .probe_v261_gpu_plane_wave_roi import load_lumapi
from .run_combined_physical_rho_pte_adfd import (
    build_native_thermal_mapping,
    coupling_for_geometry,
    native_weight_and_source,
    physical_state,
    run_forward_density,
)
from .run_v261_large_background_mixed_optical_adfd import (
    fieldregion_profile,
    invert_fieldregion_linear_collocation,
    monitor_electric,
    prepare_adjoint_layout,
    run_adjoint,
)
from .run_v261_large_background_tfsf_forward import sha256
from .yee_material_jacobian import SparseYeeMaterialJacobian


STATUS = "DIAGNOSTIC_COLLOCATED_WEIGHTED_ADJOINT_SOURCE"
FLUX_SIGNS = {
    f"device_flux_{axis}_{side}": (-1.0 if side == "min" else 1.0)
    for axis in "xyz"
    for side in ("min", "max")
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-forward", required=True)
    parser.add_argument("--base-sha256", required=True)
    parser.add_argument("--split-result", required=True)
    parser.add_argument("--split-sha256", required=True)
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
    result_path = output / "collocated_weighted_adjoint_source_result.json"
    result: dict[str, object] = {
        "status": "BLOCKED_COLLOCATED_WEIGHTED_ADJOINT_NOT_RUN",
        "passed": False,
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
        split_path = checked(args.split_result, args.split_sha256)
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
        kwargs = {
            "lateral_domain_m": 32.0e-6,
            "si_depth_m": 20.0e-6,
            "flake_span_m": 4.0e-6,
            "core_xy_cell_size_m": 100.0e-9,
            "flake_dz_m": 25.0e-9,
            "design_dz_m": 100.0e-9,
        }
        initial = build_explicit_geometry(np.full((20, 20), 0.5), **kwargs)
        coupling = coupling_for_geometry(initial)
        thermal_rho = coupling.thermal(rho)
        geometry = build_explicit_geometry(thermal_rho, **kwargs)
        mapping = build_native_thermal_mapping(base["native"], geometry)
        evaluation = evaluate_explicit_thermal(
            rho=thermal_rho,
            source_W_m3=mapping["source_W_m3"],
            **kwargs,
        )
        coefficient, weighted_source, pullback = native_weight_and_source(
            evaluation=evaluation,
            native=base["native"],
            mapping=mapping,
            electric=base["electric"],
            epsilon=base["epsilon"],
            frequency_Hz=float(base["grid"]["f"][0]),
        )
        native_profile, profile_scale = fieldregion_profile(weighted_source)
        profile, source_grid, collocation = (
            invert_fieldregion_linear_collocation(
                base["grid"], native_profile
            )
        )
        fdtd.load(str(base_path))
        original_bounds = {
            axis: [
                float(fdtd.getnamed(
                    "large_background_q_fieldregion", f"{axis} min"
                )),
                float(fdtd.getnamed(
                    "large_background_q_fieldregion", f"{axis} max"
                )),
            ]
            for axis in "xyz"
        }
        fdtd.switchtolayout()
        for axis in "xyz":
            fdtd.setnamed(
                "large_background_q_fieldregion",
                f"{axis} max",
                float(source_grid[axis][-1]),
            )
        collocation["fieldregion_bounds_before_m"] = original_bounds
        collocation["fieldregion_bounds_after_m"] = {
            axis: [
                original_bounds[axis][0],
                float(source_grid[axis][-1]),
            ]
            for axis in "xyz"
        }
        template = output / "collocated_weighted_adjoint_template.fsp"
        source_meta = prepare_adjoint_layout(
            fdtd,
            grid=source_grid,
            profile=profile,
            template=template,
        )
        adjoint_project = output / "collocated_weighted_adjoint_gpu.fsp"
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
            matrices={
                component: sparse.load_npz(
                    Path(args.jacobian_dir) / f"J_{component}.npz"
                )
                for component in "xyz"
            },
        )
        volumes = clipped_component_yee_volumes(
            base["grid"], DESIGN_BOUNDS_M
        )
        base_amplitude = source_meta["fieldregion_base_amplitude"]
        cotangent = {}
        direct = {}
        omega = 2.0 * np.pi * FREQUENCY_HZ
        for index, component in enumerate("xyz"):
            forward = base["electric"][..., 0, index]
            cotangent[component] = (
                (2.0 * EPS0 / base_amplitude)
                * volumes[index]
                * forward
                * (adjoint_electric[..., 0, index] * profile_scale)
            )
            direct[component] = (
                -1j
                * 0.5
                * EPS0
                * omega
                * coefficient[..., index]
                * np.abs(forward) ** 2
            )
        indirect_gradient = operator.vjp(cotangent)
        direct_gradient = operator.vjp(direct)
        gradient = indirect_gradient + direct_gradient
        analytic = float(np.sum(gradient * direction))
        rows = []
        for row in split["scenarios"]["4um"]["rows"]:
            finite_difference = float(row["optical_Q_only_FD_A"])
            rows.append(
                {
                    "step": row["step"],
                    "finite_difference_directional_A": finite_difference,
                    "collocated_adjoint_directional_A": analytic,
                    "relative_error": relative(
                        analytic, finite_difference
                    ),
                }
            )
        result.update(
            {
                "status": STATUS,
                "passed": True,
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "source_collocation": source_meta,
                "native_to_common_grid_collocation": collocation,
                "coordinate_mismatch_m": coordinate_error,
                "native_Q_pullback": pullback,
                "indirect_directional_A": float(
                    np.sum(indirect_gradient * direction)
                ),
                "direct_directional_A": float(
                    np.sum(direct_gradient * direction)
                ),
                "total_directional_A": analytic,
                "FD_rows": rows,
                "adjoint_FSP": {
                    "path": str(adjoint_project),
                    "byte_size": adjoint_project.stat().st_size,
                    "sha256": sha256(adjoint_project),
                },
                "interpretation": (
                    "Each component is coordinate-resampled from its native "
                    "Yee positions onto the one common vector FieldRegion "
                    "grid required by v261 before a single adjoint solve."
                ),
            }
        )
    except Exception as exc:
        result.update(
            {
                "status": "FAILED_COLLOCATED_WEIGHTED_ADJOINT_DIAGNOSTIC",
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
