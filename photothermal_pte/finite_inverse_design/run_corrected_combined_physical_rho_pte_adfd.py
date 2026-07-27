#!/usr/bin/env python3
"""Corrected multi-direction physical-rho Maxwell/thermal/PTE AD--FD gate."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np
from scipy import sparse

from .contract import DESIGN_BOUNDS_M, design_nodal_coordinates_m
from .explicit_thermal import (
    build_explicit_geometry,
    evaluate_explicit_thermal,
    solve_explicit_forward,
)
from .native_yee_q import EPS0
from .nonperiodic_yee_metric import clipped_component_yee_volumes
from .probe_v261_cpu_tfsf_device import FREQUENCY_HZ, PABS_FIELD
from .probe_v261_gpu_plane_wave_roi import load_lumapi
from .run_combined_physical_rho_pte_adfd import (
    apply_existing_mapping,
    array_sha256,
    build_native_thermal_mapping,
    compact_forward,
    coupling_for_geometry,
    native_weight_and_source,
    physical_state,
    run_forward_density,
)
from .run_v261_large_background_mixed_optical_adfd import (
    FIELD_REGION,
    fieldregion_profile,
    invert_fieldregion_linear_collocation,
    monitor_electric,
    prepare_adjoint_layout,
    run_adjoint,
)
from .run_v261_large_background_tfsf_forward import sha256
from .yee_material_jacobian import SparseYeeMaterialJacobian


STATUS_PASS = "VALIDATED_CORRECTED_COMBINED_PHYSICAL_RHO_PTE_ADFD"
STATUS_FAIL = "FAILED_CORRECTED_COMBINED_PHYSICAL_RHO_PTE_ADFD"
FLUX_SIGNS = {
    f"device_flux_{axis}_{side}": (-1.0 if side == "min" else 1.0)
    for axis in "xyz"
    for side in ("min", "max")
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-forward", required=True)
    parser.add_argument("--base-sha256", required=True)
    parser.add_argument("--jacobian-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", default="0.01,0.005,0.0025")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--gpu-device", default="GPU 1")
    return parser.parse_args()


def checked_file(path_text: str, expected: str) -> Path:
    path = Path(path_text).expanduser().resolve()
    if not path.is_file() or sha256(path) != expected:
        raise RuntimeError(f"missing or SHA-mismatched artifact: {path}")
    return path


def normalized(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value, float)
    scale = float(np.max(np.abs(result)))
    if not np.isfinite(scale) or scale == 0.0:
        raise RuntimeError("invalid zero/nonfinite direction")
    return result / scale


def fixed_directions(
    gradients: dict[float, np.ndarray],
) -> dict[str, np.ndarray]:
    x, y = design_nodal_coordinates_m()
    xn = x[:, None] / 1.0e-6
    yn = y[None, :] / 1.0e-6
    _, asymmetric = physical_state()
    rng = np.random.default_rng(2026072711)
    aligned = sum(
        gradient / max(
            float(np.linalg.norm(gradient)), np.finfo(float).tiny
        )
        for gradient in gradients.values()
    )
    return {
        "adjoint_aligned": normalized(aligned),
        "central_localized": normalized(
            np.exp(-((xn / 0.24) ** 2 + (yn / 0.24) ** 2))
        ),
        "design_edge_localized": normalized(
            np.exp(-(((xn - 0.88) / 0.13) ** 2 + (yn / 0.28) ** 2))
        ),
        "smooth_asymmetric": normalized(asymmetric),
        "fixed_seed_random": normalized(rng.normal(size=(81, 81))),
    }


def load_operator(directory: Path, shape: tuple[int, ...]):
    result_path = directory / "component_yee_jacobian_result.json"
    if not result_path.is_file():
        raise FileNotFoundError(result_path)
    result = json.loads(result_path.read_text())
    if not result.get("passed"):
        raise RuntimeError("component Yee Jacobian input is not passed")
    operator = SparseYeeMaterialJacobian(
        density_shape=(81, 81),
        component_shapes={component: shape for component in "xyz"},
        matrices={
            component: sparse.load_npz(directory / f"J_{component}.npz")
            for component in "xyz"
        },
    )
    rng = np.random.default_rng(2026072712)
    direction = rng.normal(size=(81, 81))
    cotangent = {
        component: rng.normal(size=shape)
        for component in "xyz"
    }
    jvp = operator.jvp(direction)
    left = sum(
        float(np.vdot(cotangent[c], jvp[c]).real) for c in "xyz"
    )
    right = float(np.vdot(direction, operator.vjp(cotangent)).real)
    transpose_error = abs(left - right) / max(
        abs(left), abs(right), np.finfo(float).tiny
    )
    return operator, {
        "result": {
            "path": str(result_path),
            "byte_size": result_path.stat().st_size,
            "sha256": sha256(result_path),
        },
        "matrices": {
            c: {
                "path": str(directory / f"J_{c}.npz"),
                "byte_size": (directory / f"J_{c}.npz").stat().st_size,
                "sha256": sha256(directory / f"J_{c}.npz"),
            }
            for c in "xyz"
        },
        "fresh_JVP_VJP_transpose_relative_error": transpose_error,
    }


def thermal_state(rho: np.ndarray, flake_um: float, native: dict):
    kwargs = {
        "lateral_domain_m": 32.0e-6,
        "si_depth_m": 20.0e-6,
        "flake_span_m": flake_um * 1.0e-6,
        "core_xy_cell_size_m": 100.0e-9,
        "flake_dz_m": 25.0e-9,
        "design_dz_m": 100.0e-9,
    }
    initial = build_explicit_geometry(np.full((20, 20), 0.5), **kwargs)
    coupling = coupling_for_geometry(initial)
    thermal_rho = coupling.thermal(rho)
    geometry = build_explicit_geometry(thermal_rho, **kwargs)
    mapping = build_native_thermal_mapping(native, geometry)
    evaluation = evaluate_explicit_thermal(
        rho=thermal_rho,
        source_W_m3=mapping["source_W_m3"],
        **kwargs,
    )
    return kwargs, coupling, geometry, mapping, evaluation


def prepare_corrected_source(
    fdtd,
    *,
    base_project: Path,
    grid: dict,
    weighted_source: np.ndarray,
    output: Path,
    label: str,
):
    native_profile, profile_scale = fieldregion_profile(weighted_source)
    profile, source_grid, collocation = (
        invert_fieldregion_linear_collocation(grid, native_profile)
    )
    fdtd.load(str(base_project))
    bounds_before = {
        axis: [
            float(fdtd.getnamed(FIELD_REGION, f"{axis} min")),
            float(fdtd.getnamed(FIELD_REGION, f"{axis} max")),
        ]
        for axis in "xyz"
    }
    fdtd.switchtolayout()
    for axis in "xyz":
        fdtd.setnamed(
            FIELD_REGION, f"{axis} max", float(source_grid[axis][-1])
        )
    collocation["fieldregion_bounds_before_m"] = bounds_before
    collocation["fieldregion_bounds_after_m"] = {
        axis: [bounds_before[axis][0], float(source_grid[axis][-1])]
        for axis in "xyz"
    }
    template = output / f"{label}_adjoint_template.fsp"
    source_meta = prepare_adjoint_layout(
        fdtd, grid=source_grid, profile=profile, template=template
    )
    return template, profile_scale, collocation, source_meta


def component_gradient(
    *,
    operator,
    base: dict,
    adjoint_electric: np.ndarray,
    coefficient: np.ndarray,
    profile_scale: float,
    base_amplitude: float,
):
    volumes = clipped_component_yee_volumes(
        base["grid"], DESIGN_BOUNDS_M
    )
    indirect = {}
    direct = {}
    omega = 2.0 * np.pi * FREQUENCY_HZ
    for index, component in enumerate("xyz"):
        forward = base["electric"][..., 0, index]
        indirect[component] = (
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
    indirect_gradient = operator.vjp(indirect)
    direct_gradient = operator.vjp(direct)
    return indirect_gradient + direct_gradient, {
        "indirect_L2_A": float(np.linalg.norm(indirect_gradient)),
        "direct_L2_A": float(np.linalg.norm(direct_gradient)),
    }


def completed_forward(
    fdtd,
    *,
    rho: np.ndarray,
    base_project: Path,
    output: Path,
    role: str,
    threads: int,
):
    project = output / f"{role}.fsp"
    sidecar = output / f"{role}.json"
    rho_sha = array_sha256(rho)
    reuse = False
    if project.exists() or sidecar.exists():
        if not project.is_file() or not sidecar.is_file():
            raise RuntimeError(f"incomplete resume pair for {role}")
        metadata = json.loads(sidecar.read_text())
        if (
            metadata.get("rho_sha256") != rho_sha
            or metadata.get("fsp_sha256") != sha256(project)
        ):
            raise RuntimeError(f"resume metadata mismatch for {role}")
        reuse = True
    if not reuse:
        fdtd.load(str(base_project))
    forward = run_forward_density(
        fdtd,
        rho=rho,
        project=project,
        threads=threads,
        flux_signs=FLUX_SIGNS,
        reuse_completed=reuse,
    )
    if not reuse:
        sidecar.write_text(
            json.dumps(
                {
                    "role": role,
                    "rho_sha256": rho_sha,
                    "fsp_sha256": sha256(project),
                    "byte_size": project.stat().st_size,
                },
                indent=2,
            )
            + "\n"
        )
    return forward, {
        **compact_forward(forward),
        "reused_completed": reuse,
        "sidecar": str(sidecar),
    }


def objective_for_scenario(
    *,
    rho: np.ndarray,
    forward: dict,
    data: dict,
):
    thermal_rho = data["coupling"].thermal(rho)
    mapped = apply_existing_mapping(
        forward["native"], data["mapping"], data["geometry"]
    )
    evaluation = solve_explicit_forward(
        rho=thermal_rho,
        source_W_m3=mapped["source_W_m3"],
        **data["kwargs"],
    )
    return float(evaluation.objective_A), {
        "Q_mapping_relative_error": mapped["relative_power_error"],
        "outside_flake_nonzero_count": mapped[
            "outside_flake_nonzero_count"
        ],
        "energy_balance_relative_error": float(
            evaluation.solved.energy_balance_relative_error
        ),
        "linear_residual_relative": float(
            evaluation.solved.linear_residual_relative
        ),
    }


def relative(value: float, reference: float) -> float:
    return abs(value - reference) / max(
        abs(value), abs(reference), np.finfo(float).tiny
    )


def main() -> int:
    args = parse_args()
    steps = sorted(
        {float(item) for item in args.steps.split(",")}, reverse=True
    )
    if steps != [0.01, 0.005, 0.0025]:
        raise ValueError("required steps are exactly 0.01,0.005,0.0025")
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "corrected_combined_physical_rho_pte_adfd.json"
    base_project = checked_file(args.base_forward, args.base_sha256)
    result = {
        "status": "BLOCKED_CORRECTED_COMBINED_ADFD_NOT_RUN",
        "passed": False,
        "empirical_normalization": False,
        "gradient_rescaling": False,
        "clipping": False,
        "gray_law_sensitivity_run": False,
        "full_latent_adfd_run": False,
        "optimization_run": False,
    }
    fdtd = None
    started = time.monotonic()
    try:
        rho, _ = physical_state()
        lumapi = load_lumapi()
        fdtd = lumapi.FDTD(hide=True, serverArgs={"platform": "offscreen"})
        base = run_forward_density(
            fdtd,
            rho=rho,
            project=base_project,
            threads=args.threads,
            flux_signs=FLUX_SIGNS,
            reuse_completed=True,
        )
        shape = base["electric"].shape[:3]
        operator, operator_meta = load_operator(
            Path(args.jacobian_dir).expanduser().resolve(), shape
        )
        scenarios = {}
        gradients = {}
        for flake_um in (4.0, 6.0):
            kwargs, coupling, geometry, mapping, evaluation = thermal_state(
                rho, flake_um, base["native"]
            )
            coefficient, weighted_source, pullback = (
                native_weight_and_source(
                    evaluation=evaluation,
                    native=base["native"],
                    mapping=mapping,
                    electric=base["electric"],
                    epsilon=base["epsilon"],
                    frequency_Hz=float(base["grid"]["f"][0]),
                )
            )
            label = f"flake_{flake_um:g}um"
            template, profile_scale, collocation, source_meta = (
                prepare_corrected_source(
                    fdtd,
                    base_project=base_project,
                    grid=base["grid"],
                    weighted_source=weighted_source,
                    output=output,
                    label=label,
                )
            )
            adjoint_project = output / f"{label}_adjoint_gpu.fsp"
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
            optical_gradient, optical_parts = component_gradient(
                operator=operator,
                base=base,
                adjoint_electric=adjoint_electric,
                coefficient=coefficient,
                profile_scale=profile_scale,
                base_amplitude=source_meta["fieldregion_base_amplitude"],
            )
            thermal_gradient = coupling.thermal_vjp(
                evaluation.gradient_rho_A
            )
            total_gradient = optical_gradient + thermal_gradient
            gradients[flake_um] = total_gradient
            scenarios[flake_um] = {
                "kwargs": kwargs,
                "coupling": coupling,
                "geometry": geometry,
                "mapping": mapping,
                "base_evaluation": evaluation,
                "optical_gradient": optical_gradient,
                "thermal_gradient": thermal_gradient,
                "gradient": total_gradient,
                "pullback": pullback,
                "collocation": collocation,
                "source_meta": source_meta,
                "adjoint": {
                    key: value
                    for key, value in adjoint.items()
                    if key not in {"electric", "grid"}
                },
                "coordinate_mismatch_m": coordinate_mismatch,
                "optical_gradient_parts": optical_parts,
                "directions": {},
            }
        directions = fixed_directions(gradients)
        min_margin = min(float(np.min(rho)), float(np.min(1.0 - rho)))
        if max(steps) >= min_margin:
            raise RuntimeError("FD step would require clipping")
        for direction_name, direction in directions.items():
            for step in steps:
                pair = {}
                for sign, sign_name in ((1.0, "plus"), (-1.0, "minus")):
                    perturbed = rho + sign * step * direction
                    role = (
                        f"{direction_name}_h{step:g}_{sign_name}"
                        "_cpu_tfsf"
                    )
                    forward, forward_meta = completed_forward(
                        fdtd,
                        rho=perturbed,
                        base_project=base_project,
                        output=output,
                        role=role,
                        threads=args.threads,
                    )
                    pair[sign_name] = {
                        "forward": forward_meta,
                        "objectives": {},
                    }
                    for flake_um, data in scenarios.items():
                        objective, diagnostics = objective_for_scenario(
                            rho=perturbed,
                            forward=forward,
                            data=data,
                        )
                        pair[sign_name]["objectives"][flake_um] = {
                            "objective_A": objective,
                            **diagnostics,
                        }
                for flake_um, data in scenarios.items():
                    plus = pair["plus"]["objectives"][flake_um][
                        "objective_A"
                    ]
                    minus = pair["minus"]["objectives"][flake_um][
                        "objective_A"
                    ]
                    finite_difference = (plus - minus) / (2.0 * step)
                    analytic = float(
                        np.sum(data["gradient"] * direction)
                    )
                    data["directions"].setdefault(
                        direction_name,
                        {
                            "analytic_directional_A": analytic,
                            "steps": [],
                        },
                    )["steps"].append(
                        {
                            "step": step,
                            "finite_difference_directional_A": (
                                finite_difference
                            ),
                            "relative_error": relative(
                                analytic, finite_difference
                            ),
                            "plus": pair["plus"],
                            "minus": pair["minus"],
                        }
                    )
                    print(
                        "CORRECTED_COMBINED_ADFD "
                        f"flake={flake_um:g}um "
                        f"direction={direction_name} h={step:g} "
                        f"error={relative(analytic, finite_difference):.6e}",
                        flush=True,
                    )
        published_scenarios = {}
        all_mapping = []
        all_energy = []
        all_residual = []
        all_closure = [base["six_face_closure_relative_error"]]
        strong_errors = []
        multidirection_errors = []
        angles = []
        step_convergence = []
        for flake_um, data in scenarios.items():
            analytic_vector = []
            fd_vector = []
            directions_out = {}
            for name, direction_data in data["directions"].items():
                rows = sorted(
                    direction_data["steps"],
                    key=lambda item: item["step"],
                    reverse=True,
                )
                selected = rows[-1]
                analytic_vector.append(
                    direction_data["analytic_directional_A"]
                )
                fd_vector.append(
                    selected["finite_difference_directional_A"]
                )
                if name == "adjoint_aligned":
                    strong_errors.append(selected["relative_error"])
                differences = [
                    abs(
                        rows[i]["finite_difference_directional_A"]
                        - rows[i + 1][
                            "finite_difference_directional_A"
                        ]
                    )
                    for i in range(2)
                ]
                convergence_passed = differences[1] <= max(
                    1.2 * differences[0],
                    1.0e-8
                    * max(
                        abs(
                            selected[
                                "finite_difference_directional_A"
                            ]
                        ),
                        np.finfo(float).tiny,
                    ),
                )
                step_convergence.append(convergence_passed)
                directions_out[name] = {
                    **direction_data,
                    "h_to_h_over_2_difference_A": differences,
                    "step_convergence_passed": convergence_passed,
                }
                for row in rows:
                    all_closure.extend(
                        row[sign]["forward"][
                            "six_face_closure_relative_error"
                        ]
                        for sign in ("plus", "minus")
                    )
                    for sign in ("plus", "minus"):
                        diag = row[sign]["objectives"][flake_um]
                        all_mapping.append(
                            diag["Q_mapping_relative_error"]
                        )
                        all_energy.append(
                            diag["energy_balance_relative_error"]
                        )
                        all_residual.append(
                            diag["linear_residual_relative"]
                        )
            analytic_array = np.asarray(analytic_vector)
            fd_array = np.asarray(fd_vector)
            normalized_error = float(
                np.linalg.norm(analytic_array - fd_array)
                / max(np.linalg.norm(fd_array), np.finfo(float).tiny)
            )
            cosine = float(
                np.dot(analytic_array, fd_array)
                / max(
                    np.linalg.norm(analytic_array)
                    * np.linalg.norm(fd_array),
                    np.finfo(float).tiny,
                )
            )
            angle = float(
                np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
            )
            multidirection_errors.append(normalized_error)
            angles.append(angle)
            published_scenarios[f"{flake_um:g}um"] = {
                "base_objective_A": float(
                    data["base_evaluation"].objective_A
                ),
                "gradient_L2_A": float(
                    np.linalg.norm(data["gradient"])
                ),
                "optical_gradient_L2_A": float(
                    np.linalg.norm(data["optical_gradient"])
                ),
                "thermal_gradient_L2_A": float(
                    np.linalg.norm(data["thermal_gradient"])
                ),
                "coordinate_mismatch_m": data[
                    "coordinate_mismatch_m"
                ],
                "pullback": data["pullback"],
                "collocation": data["collocation"],
                "source_meta": data["source_meta"],
                "adjoint": data["adjoint"],
                "directions": directions_out,
                "directional_subspace_normalized_error": normalized_error,
                "directional_subspace_gradient_angle_deg": angle,
                "forward_energy_balance_relative_error": float(
                    data[
                        "base_evaluation"
                    ].solved.energy_balance_relative_error
                ),
                "forward_linear_residual_relative": float(
                    data[
                        "base_evaluation"
                    ].solved.linear_residual_relative
                ),
                "adjoint_linear_residual_relative": float(
                    data[
                        "base_evaluation"
                    ].adjoint_linear_residual_relative
                ),
            }
            all_energy.append(
                published_scenarios[f"{flake_um:g}um"][
                    "forward_energy_balance_relative_error"
                ]
            )
            all_residual.extend(
                [
                    published_scenarios[f"{flake_um:g}um"][
                        "forward_linear_residual_relative"
                    ],
                    published_scenarios[f"{flake_um:g}um"][
                        "adjoint_linear_residual_relative"
                    ],
                ]
            )
        gates = {
            "worst_strong_direction_relative_error": max(strong_errors),
            "strong_direction_limit": 1.0e-2,
            "worst_multidirection_normalized_error": max(
                multidirection_errors
            ),
            "multidirection_normalized_limit": 1.0e-2,
            "worst_directional_subspace_gradient_angle_deg": max(angles),
            "gradient_angle_limit_deg": 1.0,
            "mapping_transpose_relative_error": operator_meta[
                "fresh_JVP_VJP_transpose_relative_error"
            ],
            "mapping_transpose_limit": 1.0e-12,
            "worst_optical_closure_relative_error": max(all_closure),
            "optical_closure_limit": 5.0e-3,
            "worst_Q_mapping_relative_error": max(all_mapping),
            "Q_mapping_limit": 5.0e-3,
            "worst_thermal_energy_balance_relative_error": max(all_energy),
            "thermal_energy_balance_limit": 1.0e-2,
            "worst_linear_residual_relative": max(all_residual),
            "linear_residual_limit": 1.0e-8,
            "all_h_to_h_over_2_sequences_converged": all(
                step_convergence
            ),
        }
        passed = (
            gates["worst_strong_direction_relative_error"]
            < gates["strong_direction_limit"]
            and gates["worst_multidirection_normalized_error"]
            < gates["multidirection_normalized_limit"]
            and gates["worst_directional_subspace_gradient_angle_deg"]
            < gates["gradient_angle_limit_deg"]
            and gates["mapping_transpose_relative_error"]
            < gates["mapping_transpose_limit"]
            and gates["worst_optical_closure_relative_error"]
            < gates["optical_closure_limit"]
            and gates["worst_Q_mapping_relative_error"]
            < gates["Q_mapping_limit"]
            and gates["worst_thermal_energy_balance_relative_error"]
            < gates["thermal_energy_balance_limit"]
            and gates["worst_linear_residual_relative"]
            < gates["linear_residual_limit"]
            and gates["all_h_to_h_over_2_sequences_converged"]
        )
        result.update(
            {
                "status": STATUS_PASS if passed else STATUS_FAIL,
                "passed": passed,
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "base_forward": compact_forward(base),
                "base_forward_sha256_required": args.base_sha256,
                "physical_rho_sha256": array_sha256(rho),
                "directions": {
                    name: {
                        "sha256": array_sha256(direction),
                        "max_abs": float(np.max(np.abs(direction))),
                    }
                    for name, direction in directions.items()
                },
                "steps": steps,
                "operator": operator_meta,
                "scenarios": published_scenarios,
                "gates": gates,
                "gradient_angle_interpretation": (
                    "angle between five-component vectors of AD and "
                    "centered-FD directional derivatives; not a 6561-pixel "
                    "FD gradient reconstruction"
                ),
                "next_gate": (
                    "GRAY_LAW_SENSITIVITY"
                    if passed
                    else "CORRECT_COMBINED_PHYSICAL_RHO_PTE_ADFD"
                ),
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
