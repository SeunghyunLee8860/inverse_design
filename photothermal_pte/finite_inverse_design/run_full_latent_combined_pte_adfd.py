#!/usr/bin/env python3
"""Final finite 81x81 latent/filter/projection Maxwell/thermal/PTE AD--FD."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import traceback

import numpy as np

from .contract import CERTIFICATE_BETA, FILTER_RADIUS_M, PROJECTION_ETA
from .finite_mapping import FiniteDensityMapping
from .probe_v261_cpu_tfsf_device import PABS_FIELD
from .probe_v261_gpu_plane_wave_roi import load_lumapi
from .run_combined_physical_rho_pte_adfd import (
    apply_existing_mapping,
    array_sha256,
    compact_forward,
    native_weight_and_source,
    physical_state,
    run_forward_density,
)
from .run_corrected_combined_physical_rho_pte_adfd import (
    FLUX_SIGNS,
    completed_forward,
    component_gradient,
    fixed_directions,
    load_operator,
    objective_for_scenario,
    prepare_corrected_source,
    relative,
    thermal_state,
)
from .run_v261_large_background_mixed_optical_adfd import (
    monitor_electric,
    run_adjoint,
)
from .run_v261_large_background_tfsf_forward import sha256


STATUS_PASS = (
    "VALIDATED_FULL_LATENT_COMBINED_PTE_ADFD_WITH_USER_ACCEPTED_FD_NOISE"
)
STATUS_FAIL = "FAILED_FULL_LATENT_COMBINED_PTE_ADFD"
SELECTED_STEP = 0.005


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template-forward", required=True)
    parser.add_argument("--template-sha256", required=True)
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


def mapping_dot_test(
    mapping: FiniteDensityMapping,
    latent: np.ndarray,
) -> float:
    rng = np.random.default_rng(2026072817)
    direction = rng.normal(size=mapping.latent_shape)
    cotangent = rng.normal(size=mapping.latent_shape)
    left = float(
        np.vdot(cotangent, mapping.jvp_2d(latent, direction)).real
    )
    right = float(
        np.vdot(mapping.vjp_2d(latent, cotangent), direction).real
    )
    return abs(left - right) / max(
        abs(left), abs(right), np.finfo(float).tiny
    )


def main() -> int:
    args = parse_args()
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "full_latent_combined_pte_adfd.json"
    npz_path = output / "full_latent_combined_pte_adfd_arrays.npz"
    result: dict[str, object] = {
        "status": "BLOCKED_FULL_LATENT_COMBINED_PTE_ADFD_NOT_RUN",
        "passed": False,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "optimization_run": False,
        "empirical_normalization": False,
        "gradient_rescaling": False,
        "clipping": False,
        "accepted_exception": {
            "approved_by_user": True,
            "scope": (
                "continue past the preserved physical-rho near-null "
                "h-to-h/2 plateau miss"
            ),
            "does_not_relabel_prior_failure": True,
        },
    }
    fdtd = None
    started = time.monotonic()
    try:
        template_project = checked(
            args.template_forward, args.template_sha256
        )
        mapping = FiniteDensityMapping()
        latent, _ = physical_state()
        rho = mapping.physical_2d(latent, CERTIFICATE_BETA)
        if np.min(latent) <= 0.0 or np.max(latent) >= 1.0:
            raise RuntimeError("latent baseline lacks unclipped FD margin")
        lumapi = load_lumapi()
        fdtd = lumapi.FDTD(hide=True, serverArgs={"platform": "offscreen"})
        base, base_meta = completed_forward(
            fdtd,
            rho=rho,
            base_project=template_project,
            output=output,
            role="latent_base_forward_cpu_tfsf",
            threads=args.threads,
        )
        operator, operator_meta = load_operator(
            Path(args.jacobian_dir).expanduser().resolve(),
            base["electric"].shape[:3],
        )
        scenario_runtime: dict[float, dict[str, object]] = {}
        gradients: dict[float, np.ndarray] = {}
        arrays: dict[str, np.ndarray] = {
            "latent": latent,
            "filtered": mapping.filtered(latent),
            "physical_rho": rho,
        }
        for flake_um in (4.0, 6.0):
            kwargs, coupling, geometry, thermal_mapping, evaluation = (
                thermal_state(rho, flake_um, base["native"])
            )
            coefficient, weighted_source, pullback = (
                native_weight_and_source(
                    evaluation=evaluation,
                    native=base["native"],
                    mapping=thermal_mapping,
                    electric=base["electric"],
                    epsilon=base["epsilon"],
                    frequency_Hz=float(base["grid"]["f"][0]),
                )
            )
            label = f"latent_{flake_um:g}um"
            template, profile_scale, collocation, source_meta = (
                prepare_corrected_source(
                    fdtd,
                    base_project=base["project"]["path"],
                    grid=base["grid"],
                    weighted_source=weighted_source,
                    output=output,
                    label=label,
                )
            )
            adjoint_project = output / f"{label}_adjoint_gpu.fsp"
            if adjoint_project.is_file():
                adjoint_meta = {
                    "engine": "GPU",
                    "reused_completed": True,
                    "project": {
                        "path": str(adjoint_project),
                        "byte_size": adjoint_project.stat().st_size,
                        "sha256": sha256(adjoint_project),
                    },
                }
            else:
                adjoint_meta = run_adjoint(
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
            physical_gradient = optical_gradient + thermal_gradient
            latent_gradient = mapping.vjp_2d(
                latent, physical_gradient, CERTIFICATE_BETA
            )
            gradients[flake_um] = latent_gradient
            arrays[f"physical_gradient_{flake_um:g}um_A"] = (
                physical_gradient
            )
            arrays[f"latent_gradient_{flake_um:g}um_A"] = latent_gradient
            arrays[f"temperature_{flake_um:g}um_K"] = (
                evaluation.solved.temperature_K
            )
            scenario_runtime[flake_um] = {
                "kwargs": kwargs,
                "coupling": coupling,
                "geometry": geometry,
                "mapping": thermal_mapping,
                "evaluation": evaluation,
                "physical_gradient": physical_gradient,
                "latent_gradient": latent_gradient,
                "optical_gradient": optical_gradient,
                "thermal_gradient": thermal_gradient,
                "pullback": pullback,
                "collocation": collocation,
                "source_meta": source_meta,
                "adjoint": adjoint_meta,
                "coordinate_mismatch_m": coordinate_mismatch,
                "optical_gradient_parts": optical_parts,
                "directions": {},
            }
        directions = fixed_directions(gradients)
        # Five representative directions at h=0.005. The strong and
        # stochastic directions also receive h=0.01 to expose gross
        # nonlinearity without repeating the rejected strict plateau sweep.
        direction_steps = {
            name: (
                [0.01, SELECTED_STEP]
                if name in {"adjoint_aligned", "fixed_seed_random"}
                else [SELECTED_STEP]
            )
            for name in directions
        }
        margin = min(float(np.min(latent)), float(np.min(1.0 - latent)))
        if max(max(values) for values in direction_steps.values()) >= margin:
            raise RuntimeError("latent FD step would require clipping")
        for direction_name, direction in directions.items():
            arrays[f"direction_{direction_name}"] = direction
            for step in direction_steps[direction_name]:
                pair: dict[str, object] = {}
                for sign, sign_name in ((1.0, "plus"), (-1.0, "minus")):
                    perturbed_latent = latent + sign * step * direction
                    perturbed_rho = mapping.physical_2d(
                        perturbed_latent, CERTIFICATE_BETA
                    )
                    role = (
                        f"latent_{direction_name}_h{step:g}_{sign_name}"
                        "_cpu_tfsf"
                    )
                    forward, forward_meta = completed_forward(
                        fdtd,
                        rho=perturbed_rho,
                        base_project=template_project,
                        output=output,
                        role=role,
                        threads=args.threads,
                    )
                    pair[sign_name] = {
                        "forward": forward_meta,
                        "objectives": {},
                        "latent_sha256": array_sha256(perturbed_latent),
                        "rho_sha256": array_sha256(perturbed_rho),
                        "rho_bounds": [
                            float(np.min(perturbed_rho)),
                            float(np.max(perturbed_rho)),
                        ],
                    }
                    for flake_um, data in scenario_runtime.items():
                        objective, diagnostics = objective_for_scenario(
                            rho=perturbed_rho,
                            forward=forward,
                            data=data,
                        )
                        pair[sign_name]["objectives"][f"{flake_um:g}um"] = {
                            "objective_A": objective,
                            **diagnostics,
                        }
                for flake_um, data in scenario_runtime.items():
                    scenario_key = f"{flake_um:g}um"
                    plus = pair["plus"]["objectives"][scenario_key][
                        "objective_A"
                    ]
                    minus = pair["minus"]["objectives"][scenario_key][
                        "objective_A"
                    ]
                    finite_difference = (plus - minus) / (2.0 * step)
                    analytic = float(
                        np.sum(data["latent_gradient"] * direction)
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
                        "FULL_LATENT_ADFD "
                        f"flake={flake_um:g}um "
                        f"direction={direction_name} h={step:g} "
                        f"error={relative(analytic, finite_difference):.6e}",
                        flush=True,
                    )
        scenarios = {}
        strong_errors = []
        analytic_all = []
        fd_all = []
        all_closure = [base["six_face_closure_relative_error"]]
        all_mapping = []
        all_energy = []
        all_residual = []
        for flake_um, data in scenario_runtime.items():
            scenario_analytic = []
            scenario_fd = []
            directions_out = {}
            for name, direction_data in data["directions"].items():
                rows = sorted(
                    direction_data["steps"],
                    key=lambda item: item["step"],
                    reverse=True,
                )
                selected = next(
                    row
                    for row in rows
                    if np.isclose(row["step"], SELECTED_STEP)
                )
                scenario_analytic.append(
                    direction_data["analytic_directional_A"]
                )
                scenario_fd.append(
                    selected["finite_difference_directional_A"]
                )
                if name == "adjoint_aligned":
                    strong_errors.append(selected["relative_error"])
                for row in rows:
                    for sign in ("plus", "minus"):
                        all_closure.append(
                            row[sign]["forward"][
                                "six_face_closure_relative_error"
                            ]
                        )
                        diagnostics = row[sign]["objectives"][
                            f"{flake_um:g}um"
                        ]
                        all_mapping.append(
                            diagnostics["Q_mapping_relative_error"]
                        )
                        all_energy.append(
                            diagnostics["energy_balance_relative_error"]
                        )
                        all_residual.append(
                            diagnostics["linear_residual_relative"]
                        )
                directions_out[name] = {
                    **direction_data,
                    "selected_step": SELECTED_STEP,
                    "selected_relative_error": selected["relative_error"],
                }
            analytic_array = np.asarray(scenario_analytic)
            fd_array = np.asarray(scenario_fd)
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
            analytic_all.extend(analytic_array.tolist())
            fd_all.extend(fd_array.tolist())
            evaluation = data["evaluation"]
            all_energy.append(
                float(evaluation.solved.energy_balance_relative_error)
            )
            all_residual.extend(
                [
                    float(evaluation.solved.linear_residual_relative),
                    float(evaluation.adjoint_linear_residual_relative),
                ]
            )
            scenarios[f"{flake_um:g}um"] = {
                "base_objective_A": float(evaluation.objective_A),
                "physical_gradient_L2_A": float(
                    np.linalg.norm(data["physical_gradient"])
                ),
                "latent_gradient_L2_A": float(
                    np.linalg.norm(data["latent_gradient"])
                ),
                "optical_gradient_L2_A": float(
                    np.linalg.norm(data["optical_gradient"])
                ),
                "thermal_gradient_L2_A": float(
                    np.linalg.norm(data["thermal_gradient"])
                ),
                "coordinate_mismatch_m": data["coordinate_mismatch_m"],
                "pullback": data["pullback"],
                "collocation": data["collocation"],
                "source_meta": data["source_meta"],
                "adjoint": {
                    key: value
                    for key, value in data["adjoint"].items()
                    if key not in {"electric", "grid"}
                },
                "directions": directions_out,
                "directional_subspace_normalized_error": normalized_error,
                "directional_subspace_gradient_angle_deg": angle,
                "forward_energy_balance_relative_error": float(
                    evaluation.solved.energy_balance_relative_error
                ),
                "forward_linear_residual_relative": float(
                    evaluation.solved.linear_residual_relative
                ),
                "adjoint_linear_residual_relative": float(
                    evaluation.adjoint_linear_residual_relative
                ),
            }
        analytic_all_array = np.asarray(analytic_all)
        fd_all_array = np.asarray(fd_all)
        global_normalized_error = float(
            np.linalg.norm(analytic_all_array - fd_all_array)
            / max(np.linalg.norm(fd_all_array), np.finfo(float).tiny)
        )
        global_cosine = float(
            np.dot(analytic_all_array, fd_all_array)
            / max(
                np.linalg.norm(analytic_all_array)
                * np.linalg.norm(fd_all_array),
                np.finfo(float).tiny,
            )
        )
        global_angle = float(
            np.degrees(np.arccos(np.clip(global_cosine, -1.0, 1.0)))
        )
        mapping_transpose = mapping_dot_test(mapping, latent)
        gates = {
            "selected_step": SELECTED_STEP,
            "worst_strong_direction_relative_error": max(strong_errors),
            "strong_direction_limit": 1.0e-2,
            "global_multidirection_normalized_error": (
                global_normalized_error
            ),
            "multidirection_normalized_limit": 1.0e-2,
            "global_directional_gradient_angle_deg": global_angle,
            "gradient_angle_limit_deg": 1.0,
            "finite_mapping_transpose_relative_error": mapping_transpose,
            "component_yee_mapping_transpose_relative_error": operator_meta[
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
            "strict_h_to_h_over_2_plateau_required": False,
            "strict_plateau_waived_by_user": True,
        }
        passed = (
            gates["worst_strong_direction_relative_error"]
            < gates["strong_direction_limit"]
            and gates["global_multidirection_normalized_error"]
            < gates["multidirection_normalized_limit"]
            and gates["global_directional_gradient_angle_deg"]
            < gates["gradient_angle_limit_deg"]
            and gates["finite_mapping_transpose_relative_error"]
            < gates["mapping_transpose_limit"]
            and gates["component_yee_mapping_transpose_relative_error"]
            < gates["mapping_transpose_limit"]
            and gates["worst_optical_closure_relative_error"]
            < gates["optical_closure_limit"]
            and gates["worst_Q_mapping_relative_error"]
            < gates["Q_mapping_limit"]
            and gates["worst_thermal_energy_balance_relative_error"]
            < gates["thermal_energy_balance_limit"]
            and gates["worst_linear_residual_relative"]
            < gates["linear_residual_limit"]
        )
        np.savez_compressed(npz_path, **arrays)
        result.update(
            {
                "status": STATUS_PASS if passed else STATUS_FAIL,
                "passed": passed,
                "scope": (
                    "finite nonperiodic 81x81 latent -> 500 nm conic "
                    "filter -> beta=8 tanh projection -> component Yee "
                    "Maxwell Q -> explicit thermal material/interface -> "
                    "uniform-45-degree PTE"
                ),
                "latent_contract": {
                    "shape": list(mapping.latent_shape),
                    "node_spacing_m": [mapping.dx_m, mapping.dy_m],
                    "filter_radius_m": FILTER_RADIUS_M,
                    "filter_periodic_wrap": False,
                    "projection_beta": CERTIFICATE_BETA,
                    "projection_eta": PROJECTION_ETA,
                    "latent_bounds": [
                        float(np.min(latent)),
                        float(np.max(latent)),
                    ],
                    "physical_rho_bounds": [
                        float(np.min(rho)),
                        float(np.max(rho)),
                    ],
                    "latent_sha256": array_sha256(latent),
                    "physical_rho_sha256": array_sha256(rho),
                },
                "base_forward": base_meta,
                "operator": operator_meta,
                "directions": {
                    name: {
                        "sha256": array_sha256(direction),
                        "steps": direction_steps[name],
                    }
                    for name, direction in directions.items()
                },
                "scenarios": scenarios,
                "gates": gates,
                "arrays": {
                    "path": str(npz_path),
                    "byte_size": npz_path.stat().st_size,
                    "sha256": sha256(npz_path),
                },
                "next_gate": (
                    "OPTIMIZATION_SEPARATELY_REQUIRES_USER_APPROVAL"
                    if passed
                    else "CORRECT_FULL_LATENT_COMBINED_PTE_ADFD"
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
