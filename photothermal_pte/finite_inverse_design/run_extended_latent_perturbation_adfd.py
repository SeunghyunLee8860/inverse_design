#!/usr/bin/env python3
"""Add five independent latent perturbations to the final AD--FD plot set."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import traceback

import numpy as np

from .contract import CERTIFICATE_BETA, design_nodal_coordinates_m
from .finite_mapping import FiniteDensityMapping
from .probe_v261_gpu_plane_wave_roi import load_lumapi
from .run_combined_physical_rho_pte_adfd import (
    array_sha256,
    run_forward_density,
)
from .run_corrected_combined_physical_rho_pte_adfd import (
    FLUX_SIGNS,
    completed_forward,
    objective_for_scenario,
    relative,
    thermal_state,
)
from .run_v261_large_background_tfsf_forward import sha256


STATUS_PASS = "VALIDATED_EXTENDED_FULL_LATENT_PERTURBATION_ADFD"
STATUS_FAIL = "FAILED_EXTENDED_FULL_LATENT_PERTURBATION_ADFD"
STEP = 0.005


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-forward", required=True)
    parser.add_argument("--base-sha256", required=True)
    parser.add_argument("--latent-arrays", required=True)
    parser.add_argument("--latent-arrays-sha256", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--threads", type=int, default=16)
    return parser.parse_args()


def checked(path_text: str, expected: str) -> Path:
    path = Path(path_text).expanduser().resolve()
    if not path.is_file() or sha256(path) != expected:
        raise RuntimeError(f"missing or SHA-mismatched artifact: {path}")
    return path


def normalized(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value, float)
    scale = float(np.max(np.abs(result)))
    if not np.isfinite(scale) or scale == 0.0:
        raise RuntimeError("invalid perturbation direction")
    return result / scale


def extra_directions() -> dict[str, np.ndarray]:
    x, y = design_nodal_coordinates_m()
    xn = x[:, None] / 1.0e-6
    yn = y[None, :] / 1.0e-6
    radius = np.sqrt(xn**2 + yn**2)
    ring = np.exp(-((radius - 0.62) / 0.16) ** 2)
    ring -= np.mean(ring)
    return {
        "uniform": np.ones((81, 81)),
        "x_antisymmetric": normalized(np.broadcast_to(xn, (81, 81))),
        "y_antisymmetric": normalized(np.broadcast_to(yn, (81, 81))),
        "diagonal_quadrupole": normalized(xn * yn),
        "radial_ring": normalized(ring),
    }


def main() -> int:
    args = parse_args()
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "extended_latent_perturbation_adfd.json"
    array_path = output / "extended_latent_perturbation_directions.npz"
    result: dict[str, object] = {
        "status": "BLOCKED_EXTENDED_LATENT_PERTURBATION_ADFD_NOT_RUN",
        "passed": False,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "step": STEP,
        "clipping": False,
        "gradient_rescaling": False,
        "empirical_normalization": False,
        "optimization_run": False,
    }
    fdtd = None
    started = time.monotonic()
    try:
        base_project = checked(args.base_forward, args.base_sha256)
        latent_arrays_path = checked(
            args.latent_arrays, args.latent_arrays_sha256
        )
        stored = np.load(latent_arrays_path)
        latent = np.asarray(stored["latent"], float)
        mapping = FiniteDensityMapping()
        rho = mapping.physical_2d(latent, CERTIFICATE_BETA)
        if not np.array_equal(rho, stored["physical_rho"]):
            raise RuntimeError("stored latent does not reproduce physical rho")
        directions = extra_directions()
        margin = min(float(np.min(latent)), float(np.min(1.0 - latent)))
        if STEP >= margin:
            raise RuntimeError("FD step would require latent clipping")
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
        scenarios = {}
        for flake_um in (4.0, 6.0):
            kwargs, coupling, geometry, thermal_mapping, evaluation = (
                thermal_state(rho, flake_um, base["native"])
            )
            scenarios[flake_um] = {
                "kwargs": kwargs,
                "coupling": coupling,
                "geometry": geometry,
                "mapping": thermal_mapping,
                "base_evaluation": evaluation,
                "latent_gradient": np.asarray(
                    stored[f"latent_gradient_{flake_um:g}um_A"], float
                ),
                "directions": {},
            }
        for name, direction in directions.items():
            pair: dict[str, object] = {}
            for sign, sign_name in ((1.0, "plus"), (-1.0, "minus")):
                perturbed_latent = latent + sign * STEP * direction
                perturbed_rho = mapping.physical_2d(
                    perturbed_latent, CERTIFICATE_BETA
                )
                role = f"extended_{name}_{sign_name}_cpu_tfsf"
                forward, forward_meta = completed_forward(
                    fdtd,
                    rho=perturbed_rho,
                    base_project=base_project,
                    output=output,
                    role=role,
                    threads=args.threads,
                )
                pair[sign_name] = {
                    "forward": forward_meta,
                    "latent_sha256": array_sha256(perturbed_latent),
                    "rho_sha256": array_sha256(perturbed_rho),
                    "rho_bounds": [
                        float(np.min(perturbed_rho)),
                        float(np.max(perturbed_rho)),
                    ],
                    "objectives": {},
                }
                for flake_um, data in scenarios.items():
                    objective, diagnostics = objective_for_scenario(
                        rho=perturbed_rho,
                        forward=forward,
                        data=data,
                    )
                    pair[sign_name]["objectives"][f"{flake_um:g}um"] = {
                        "objective_A": objective,
                        **diagnostics,
                    }
            for flake_um, data in scenarios.items():
                key = f"{flake_um:g}um"
                plus = pair["plus"]["objectives"][key]["objective_A"]
                minus = pair["minus"]["objectives"][key]["objective_A"]
                finite_difference = (plus - minus) / (2.0 * STEP)
                analytic = float(
                    np.sum(data["latent_gradient"] * direction)
                )
                data["directions"][name] = {
                    "analytic_directional_A": analytic,
                    "finite_difference_directional_A": finite_difference,
                    "relative_error": relative(analytic, finite_difference),
                    "plus": pair["plus"],
                    "minus": pair["minus"],
                }
                print(
                    "EXTENDED_LATENT_ADFD "
                    f"flake={flake_um:g}um direction={name} "
                    f"error={relative(analytic, finite_difference):.6e}",
                    flush=True,
                )
        published = {}
        all_closure = []
        all_mapping = []
        all_energy = []
        all_residual = []
        errors = []
        for flake_um, data in scenarios.items():
            rows = data["directions"]
            analytic = np.asarray(
                [rows[name]["analytic_directional_A"] for name in directions]
            )
            finite_difference = np.asarray(
                [
                    rows[name]["finite_difference_directional_A"]
                    for name in directions
                ]
            )
            normalized_error = float(
                np.linalg.norm(analytic - finite_difference)
                / max(
                    np.linalg.norm(finite_difference),
                    np.finfo(float).tiny,
                )
            )
            cosine = float(
                np.dot(analytic, finite_difference)
                / max(
                    np.linalg.norm(analytic)
                    * np.linalg.norm(finite_difference),
                    np.finfo(float).tiny,
                )
            )
            angle = float(
                np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
            )
            for row in rows.values():
                errors.append(row["relative_error"])
                for sign in ("plus", "minus"):
                    all_closure.append(
                        row[sign]["forward"][
                            "six_face_closure_relative_error"
                        ]
                    )
                    diagnostic = row[sign]["objectives"][
                        f"{flake_um:g}um"
                    ]
                    all_mapping.append(diagnostic["Q_mapping_relative_error"])
                    all_energy.append(
                        diagnostic["energy_balance_relative_error"]
                    )
                    all_residual.append(
                        diagnostic["linear_residual_relative"]
                    )
            published[f"{flake_um:g}um"] = {
                "base_objective_A": float(
                    data["base_evaluation"].objective_A
                ),
                "directions": rows,
                "five_extra_direction_normalized_error": normalized_error,
                "five_extra_direction_gradient_angle_deg": angle,
            }
        gates = {
            "worst_individual_relative_error": max(errors),
            "individual_relative_error_limit": 1.0e-2,
            "worst_optical_closure_relative_error": max(all_closure),
            "optical_closure_limit": 5.0e-3,
            "worst_Q_mapping_relative_error": max(all_mapping),
            "Q_mapping_limit": 5.0e-3,
            "worst_thermal_energy_balance_relative_error": max(all_energy),
            "thermal_energy_balance_limit": 1.0e-2,
            "worst_linear_residual_relative": max(all_residual),
            "linear_residual_limit": 1.0e-8,
        }
        passed = (
            gates["worst_individual_relative_error"]
            < gates["individual_relative_error_limit"]
            and gates["worst_optical_closure_relative_error"]
            < gates["optical_closure_limit"]
            and gates["worst_Q_mapping_relative_error"]
            < gates["Q_mapping_limit"]
            and gates["worst_thermal_energy_balance_relative_error"]
            < gates["thermal_energy_balance_limit"]
            and gates["worst_linear_residual_relative"]
            < gates["linear_residual_limit"]
        )
        np.savez_compressed(array_path, **directions)
        result.update(
            {
                "status": STATUS_PASS if passed else STATUS_FAIL,
                "passed": passed,
                "base_forward": {
                    "path": str(base_project),
                    "byte_size": base_project.stat().st_size,
                    "sha256": sha256(base_project),
                },
                "latent_arrays": {
                    "path": str(latent_arrays_path),
                    "byte_size": latent_arrays_path.stat().st_size,
                    "sha256": sha256(latent_arrays_path),
                },
                "directions": {
                    name: {
                        "sha256": array_sha256(direction),
                        "maximum_absolute_value": float(
                            np.max(np.abs(direction))
                        ),
                    }
                    for name, direction in directions.items()
                },
                "scenarios": published,
                "gates": gates,
                "direction_arrays": {
                    "path": str(array_path),
                    "byte_size": array_path.stat().st_size,
                    "sha256": sha256(array_path),
                },
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
