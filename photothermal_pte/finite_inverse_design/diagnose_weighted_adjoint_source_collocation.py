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

from .explicit_thermal import build_explicit_geometry, evaluate_explicit_thermal
from .native_yee_q import EPS0
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
    component_volumes,
    fieldregion_profile,
    invert_fieldregion_linear_collocation,
    monitor_electric,
    prepare_adjoint_layout,
    prepare_component_yee_adjoint_layout,
    prepare_single_component_yee_adjoint_layout,
    run_adjoint,
    weighted_fieldregion_source_from_native_multiplier,
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
    parser.add_argument(
        "--resume-component",
        action="append",
        default=[],
        metavar="COMPONENT=PATH@SHA256",
        help=(
            "Reuse a completed component GPU adjoint FSP after exact SHA "
            "verification. May be repeated for x/y/z."
        ),
    )
    parser.add_argument(
        "--source-mode",
        choices=(
            "inverse-product",
            "collocated-multiplier",
            "component-yee-sources",
            "component-yee-separate",
        ),
        default="inverse-product",
    )
    return parser.parse_args()


def checked(path_text: str, expected: str) -> Path:
    path = Path(path_text).expanduser().resolve()
    if not path.is_file() or sha256(path) != expected:
        raise RuntimeError(f"missing or SHA-mismatched artifact: {path}")
    return path


def component_resumes(values: list[str]) -> dict[str, Path]:
    result = {}
    for value in values:
        component, rest = value.split("=", 1)
        path_text, expected = rest.rsplit("@", 1)
        if component not in "xyz" or component in result:
            raise ValueError(f"invalid component resume {value!r}")
        result[component] = checked(path_text, expected)
    return result


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
        resumes = component_resumes(args.resume_component)
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
        if args.source_mode == "collocated-multiplier":
            source, collocation = (
                weighted_fieldregion_source_from_native_multiplier(
                    electric=base["electric"],
                    epsilon=base["epsilon"],
                    coefficient=coefficient,
                    grid=base["grid"],
                    frequency_hz=float(base["grid"]["f"][0]),
                )
            )
            profile, profile_scale = fieldregion_profile(source)
            source_grid = base["grid"]
        elif args.source_mode == "inverse-product":
            native_profile, profile_scale = fieldregion_profile(
                weighted_source
            )
            profile, source_grid, collocation = (
                invert_fieldregion_linear_collocation(
                    base["grid"], native_profile
                )
            )
        else:
            profile = None
            profile_scale = None
            source_grid = None
            collocation = {
                "method": (
                    "no common-grid collocation; component-specific native "
                    "Yee source coordinates"
                )
            }
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
        if args.source_mode == "inverse-product":
            for axis in "xyz":
                fdtd.setnamed(
                    "large_background_q_fieldregion",
                    f"{axis} max",
                    float(source_grid[axis][-1]),
                )
        collocation["fieldregion_bounds_before_m"] = original_bounds
        if args.source_mode not in (
            "component-yee-sources",
            "component-yee-separate",
        ):
            collocation["fieldregion_bounds_after_m"] = {
                axis: [
                    original_bounds[axis][0],
                    (
                        float(source_grid[axis][-1])
                        if args.source_mode == "inverse-product"
                        else original_bounds[axis][1]
                    ),
                ]
                for axis in "xyz"
            }
        template = output / "collocated_weighted_adjoint_template.fsp"
        component_adjoint_meta = None
        component_effective_adjoint = None
        if args.source_mode == "component-yee-sources":
            profile_scale, source_meta = (
                prepare_component_yee_adjoint_layout(
                    fdtd,
                    grid=base["grid"],
                    native_source=weighted_source,
                    template=template,
                )
            )
        elif args.source_mode == "component-yee-separate":
            component_adjoint_meta = {}
            component_effective_adjoint = np.zeros_like(
                base["electric"], dtype=np.complex128
            )
            for source_component in "xyz":
                source_index = "xyz".index(source_component)
                source_max = float(
                    np.max(
                        np.abs(
                            weighted_source[
                                ..., 0, source_index
                            ]
                        )
                    )
                )
                if source_max == 0.0:
                    component_adjoint_meta[source_component] = {
                        "source": {
                            "component": source_component,
                            "identically_zero": True,
                            "source_profile_scale": 0.0,
                        },
                        "run": None,
                        "PABS_grid_coordinate_max_mismatch": 0.0,
                    }
                    continue
                fdtd.load(str(base_path))
                component_template = (
                    output
                    / f"component_{source_component}_adjoint_template.fsp"
                )
                component_project = (
                    output
                    / f"component_{source_component}_adjoint_gpu.fsp"
                )
                if source_component in resumes:
                    component_project = resumes[source_component]
                    fdtd.load(str(component_project))
                    component_scale = source_max
                    component_source_meta = {
                        "method": (
                            "SHA-verified completed single-component "
                            "native-Yee FieldRegion source"
                        ),
                        "component": source_component,
                        "source_profile_scale": component_scale,
                        "base_amplitude": float(
                            fdtd.getnamed(
                                "large_background_q_fieldregion",
                                "base amplitude",
                            )
                        ),
                        "reused_completed_FSP": True,
                        "common_grid_interpolation": False,
                        "empirical_normalization": False,
                        "gradient_rescaling": False,
                    }
                    component_run = {
                        "engine": "GPU",
                        "reused_completed": True,
                        "project": {
                            "path": str(component_project),
                            "byte_size": component_project.stat().st_size,
                            "sha256": sha256(component_project),
                        },
                    }
                else:
                    component_scale, component_source_meta = (
                        prepare_single_component_yee_adjoint_layout(
                            fdtd,
                            grid=base["grid"],
                            native_source=weighted_source,
                            component=source_component,
                            template=component_template,
                        )
                    )
                    component_run = run_adjoint(
                        fdtd,
                        template=component_template,
                        project=component_project,
                        engine="GPU",
                        threads=args.threads,
                        gpu_device=args.gpu_device,
                    )
                fdtd.load(str(component_project))
                component_field, component_grid = monitor_electric(
                    fdtd, PABS_FIELD
                )
                component_effective_adjoint += (
                    component_field
                    * component_scale
                    / component_source_meta["base_amplitude"]
                )
                component_adjoint_meta[source_component] = {
                    "source": component_source_meta,
                    "run": component_run,
                    "PABS_grid_coordinate_max_mismatch": max(
                        float(
                            np.max(
                                np.abs(
                                    np.asarray(base["grid"][key])
                                    - np.asarray(component_grid[key])
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
                    ),
                }
            source_meta = {
                "method": (
                    "three sequential exact-component-Yee GPU adjoint "
                    "solves combined by linear superposition"
                ),
                "number_of_GPU_adjoint_solves": 3,
                "pixel_scaled_solve_count": False,
                "components": component_adjoint_meta,
                "empirical_normalization": False,
                "gradient_rescaling": False,
            }
            profile_scale = 1.0
        else:
            source_meta = prepare_adjoint_layout(
                fdtd,
                grid=source_grid,
                profile=profile,
                template=template,
            )
        if args.source_mode == "component-yee-separate":
            adjoint = {
                "engine": "GPU",
                "linear_superposition_of_components": True,
            }
            adjoint_project = None
            adjoint_electric = component_effective_adjoint
            coordinate_error = max(
                item["PABS_grid_coordinate_max_mismatch"]
                for item in component_adjoint_meta.values()
            )
        else:
            adjoint_project = (
                output / "collocated_weighted_adjoint_gpu.fsp"
            )
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
        volumes = component_volumes(base["grid"])
        base_amplitude = (
            1.0
            if args.source_mode == "component-yee-separate"
            else source_meta.get(
                "fieldregion_base_amplitude",
                source_meta.get("base_amplitude"),
            )
        )
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
                "adjoint_FSP": (
                    {
                        "path": str(adjoint_project),
                        "byte_size": adjoint_project.stat().st_size,
                        "sha256": sha256(adjoint_project),
                    }
                    if adjoint_project is not None
                    else {
                        component: (
                            meta["run"]["project"]
                            if meta["run"] is not None
                            else None
                        )
                        for component, meta in component_adjoint_meta.items()
                    }
                ),
                "interpretation": (
                    "The selected source mode is recorded verbatim. The "
                    "component-yee-sources mode uses three simultaneous "
                    "single-component sources at exact Yee coordinates in "
                    "one adjoint solve; other modes remain diagnostics."
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
