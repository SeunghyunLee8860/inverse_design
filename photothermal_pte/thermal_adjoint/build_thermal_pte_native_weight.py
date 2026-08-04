#!/usr/bin/env python3
"""Pull the fixed-K thermal/PTE adjoint back to native Yee absorption."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.sparse import linalg as sparse_linalg


HERE = Path(__file__).resolve().parent
PHOTOTHERMAL = HERE.parent
REPOSITORY = PHOTOTHERMAL.parent
VOLUME_CURRENT = REPOSITORY / "volume_current_inverse_design"
FVM = PHOTOTHERMAL / "validation" / "photothermal_stage1"
for path in (HERE, VOLUME_CURRENT, VOLUME_CURRENT / "bundle", FVM):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from anisotropic_heat_fvm import (  # noqa: E402
    assemble_steady_diagonal_kappa,
    solve_assembled_thermal_system,
)
from conservative_remap import (  # noqa: E402
    build_conservative_density_remap,
    build_nodal_density_remap_1d,
    nodal_control_volume_edges,
)
from local_pte_functional import build_local_pte_functional  # noqa: E402
from maxwell_absorption_evaluator import (  # noqa: E402
    AbsorptionVolumeCurrentEvaluator,
)
from run_thermal_pte_adfd_certificate import (  # noqa: E402
    _grid_and_materials,
    _source_and_mask,
)
from yee_absorption_functional import (  # noqa: E402
    trapezoid_weights,
    weighted_yee_absorption_and_wirtinger,
)


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _common_volume(grid: dict[str, np.ndarray]) -> np.ndarray:
    weights = [trapezoid_weights(grid[axis]) for axis in "xyz"]
    return (
        weights[0][:, None, None]
        * weights[1][None, :, None]
        * weights[2][None, None, :]
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--forward-fsp", required=True)
    parser.add_argument("--solver-workdir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--polarization", choices=("x", "y"), default="x")
    args = parser.parse_args()
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    evaluator = AbsorptionVolumeCurrentEvaluator(
        workdir=Path(args.solver_workdir).expanduser().resolve(),
        incident_polarization=args.polarization,
    )
    evaluator.prepare(force_rebuild=False)
    optical = evaluator.postprocess_completed_forward(
        Path(args.forward_fsp).expanduser().resolve()
    )
    grid = optical.grid
    q_native = optical.observation.density_component_W_m3
    deltas = {
        axis: np.asarray(
            optical.field_result[f"fom_delta_{axis}"], float
        ).reshape(-1)
        for axis in "xyz"
    }
    component_remaps = {}
    q_common_components = np.zeros_like(q_native)
    for component, axis in enumerate("xy"):
        source_coordinates = np.asarray(grid[axis], float) + deltas[axis]
        remap = build_nodal_density_remap_1d(
            source_coordinates_m=source_coordinates,
            target_coordinates_m=np.asarray(grid[axis], float),
            periodic=True,
        )
        component_remaps[component] = remap
        q_common_components[..., component] = remap.apply_axis(
            q_native[..., component], axis=component
        )
    if optical.epsilon_imaginary[2] != 0.0:
        raise RuntimeError(
            "nonzero z loss needs an independently validated nonperiodic "
            "shifted-z remap"
        )
    q_common = np.sum(q_common_components, axis=-1)
    common_volume = _common_volume(grid)
    native_power = optical.observation.power_total_W
    common_power = float(np.sum(common_volume * q_common))

    (
        x_edges,
        y_edges,
        z_edges,
        material,
        kappa,
        interface,
        _,
        _,
    ) = _grid_and_materials()
    system = assemble_steady_diagonal_kappa(
        x_edges_m=x_edges,
        y_edges_m=y_edges,
        z_edges_m=z_edges,
        kappa_W_mK=kappa,
        dirichlet_temperature_K={"z_min": 0.0},
        interface_resistance_m2K_W=interface,
        periodic_axes=("x", "y"),
    )
    source_edges = tuple(
        nodal_control_volume_edges(np.asarray(grid[axis], float))
        for axis in "xyz"
    )
    target_flake_edges = (
        x_edges,
        y_edges,
        z_edges[-3:],
    )
    optical_to_thermal = build_conservative_density_remap(
        source_edges_m=source_edges,
        target_edges_m=target_flake_edges,
    )
    q_thermal_flake = optical_to_thermal.apply(q_common)
    q_thermal = np.zeros(system.shape, float)
    q_thermal[:, :, -2:] = q_thermal_flake
    thermal_power = float(
        np.sum(system.cell_volume_m3 * q_thermal)
    )
    solved = solve_assembled_thermal_system(
        system, source_W_m3=q_thermal
    )
    _, fom_mask = _source_and_mask(
        x_edges, y_edges, z_edges, material, system.cell_volume_m3
    )
    functional = build_local_pte_functional(
        x_edges_m=x_edges,
        y_edges_m=y_edges,
        z_edges_m=z_edges,
        active_mask=system.active_mask,
        active_ids=system.active_ids,
        fom_mask=fom_mask,
        periodic_axes=(),
    )
    temperature_active = solved.temperature_K[system.active_mask]
    pte_forward = functional.evaluate_active(temperature_active)
    lambda_t = sparse_linalg.spsolve(
        system.matrix_W_K.T.tocsc(),
        functional.temperature_source_A_m_K,
    )
    thermal_density_sensitivity_active = np.asarray(
        system.source_volume_operator_m3.T @ lambda_t
    ).reshape(-1)
    thermal_density_sensitivity = system.full_field(
        thermal_density_sensitivity_active
    )
    common_density_sensitivity = optical_to_thermal.transpose(
        thermal_density_sensitivity[:, :, -2:]
    )

    native_coefficient = np.zeros_like(q_native)
    native_coefficient[..., 0] = component_remaps[0].transpose_axis(
        common_density_sensitivity, axis=0
    )
    native_coefficient[..., 1] = component_remaps[1].transpose_axis(
        common_density_sensitivity, axis=1
    )
    native_density_weight = np.zeros_like(q_native)
    for component in range(3):
        volume = optical.component_volume_m3[component]
        if np.any(volume <= 0.0):
            raise RuntimeError("native observation quadrature has zero volume")
        native_density_weight[..., component] = (
            native_coefficient[..., component] / volume
        )
    weighted = weighted_yee_absorption_and_wirtinger(
        electric_field_V_m=optical.field_result["fom"]["E"],
        frequency_Hz=float(np.asarray(grid["f"]).reshape(-1)[0]),
        epsilon_imaginary=optical.epsilon_imaginary,
        component_volume_m3=optical.component_volume_m3,
        density_weight=native_density_weight,
        power_scale=1.0 / optical.incident_intensity_W_m2,
    )

    rng = np.random.default_rng(2026072609)
    transpose_errors = {}
    for component, axis in enumerate((0, 1)):
        perturbation = rng.normal(size=q_native[..., component].shape)
        left = float(
            np.sum(
                common_density_sensitivity
                * component_remaps[component].apply_axis(
                    perturbation, axis=axis
                )
            )
        )
        right = float(
            np.sum(native_coefficient[..., component] * perturbation)
        )
        transpose_errors[f"component_{'xy'[component]}"] = abs(
            left - right
        ) / max(abs(left), abs(right), 1e-300)
    common_perturbation = rng.normal(size=q_common.shape)
    left = float(
        np.sum(
            thermal_density_sensitivity[:, :, -2:]
            * optical_to_thermal.apply(common_perturbation)
        )
    )
    right = float(
        np.sum(common_density_sensitivity * common_perturbation)
    )
    transpose_errors["common_to_thermal"] = abs(left - right) / max(
        abs(left), abs(right), 1e-300
    )
    power_error_common = abs(common_power - native_power) / max(
        abs(native_power), 1e-300
    )
    power_error_thermal = abs(thermal_power - native_power) / max(
        abs(native_power), 1e-300
    )
    objective_error = abs(weighted.weighted_value - pte_forward) / max(
        abs(pte_forward), abs(weighted.weighted_value), 1e-300
    )
    gates = {
        "native_to_common_power_relative_error": power_error_common,
        "native_to_common_power_pass": power_error_common < 5e-13,
        "common_to_thermal_power_relative_error": power_error_thermal,
        "common_to_thermal_power_pass": power_error_thermal < 5e-13,
        "worst_transpose_relative_error": max(transpose_errors.values()),
        "transpose_pass": max(transpose_errors.values()) < 1e-12,
        "forward_objective_identity_relative_error": objective_error,
        "forward_objective_identity_pass": objective_error < 1e-11,
        "thermal_linear_residual_relative": solved.linear_residual_relative,
        "thermal_linear_residual_pass": solved.linear_residual_relative < 1e-8,
        "thermal_energy_balance_relative_error": (
            solved.energy_balance_relative_error
        ),
        "thermal_energy_balance_pass": (
            solved.energy_balance_relative_error < 0.01
        ),
    }
    passed = all(
        value for key, value in gates.items() if key.endswith("_pass")
    )
    raw = output / "thermal_pte_native_yee_pullback.npz"
    np.savez_compressed(
        raw,
        native_density_weight_A_m_W=native_density_weight,
        native_coefficient_A_m4_W=native_coefficient,
        common_density_sensitivity_A_m4_W=common_density_sensitivity,
        thermal_density_sensitivity_A_m4_W=(
            thermal_density_sensitivity[:, :, -2:]
        ),
        q_native_W_m3_per_W_m2=q_native,
        q_common_W_m3_per_W_m2=q_common,
        q_thermal_W_m3_per_W_m2=q_thermal,
        temperature_K_per_W_m2=solved.temperature_K,
        x_optical_m=grid["x"],
        y_optical_m=grid["y"],
        z_optical_m=grid["z"],
        x_thermal_edges_m=x_edges,
        y_thermal_edges_m=y_edges,
        z_thermal_edges_m=z_edges,
    )
    summary = {
        "schema_version": 1,
        "generated_at_utc": _utc(),
        "status": (
            "VALIDATED_THERMAL_PTE_TO_NATIVE_YEE_PULLBACK"
            if passed
            else "FAILED_THERMAL_PTE_TO_NATIVE_YEE_PULLBACK"
        ),
        "passed": passed,
        "scope": "fixed-K finite-local-mask thermal/PTE pullback",
        "power": {
            "native_W_per_W_m2": native_power,
            "common_W_per_W_m2": common_power,
            "thermal_W_per_W_m2": thermal_power,
        },
        "forward": {
            "pte_from_temperature_A_m_per_W_m2": pte_forward,
            "pte_from_native_weighted_absorption_A_m_per_W_m2": (
                weighted.weighted_value
            ),
            "temperature_max_K_per_W_m2": float(
                np.nanmax(solved.temperature_K)
            ),
        },
        "transpose_errors": transpose_errors,
        "gates": gates,
        "raw_artifact": {
            "path": str(raw),
            "bytes": raw.stat().st_size,
            "sha256": _sha256(raw),
            "committed_to_git": False,
        },
        "blockers": [
            "BLOCKED_FULL_RHO_DEPENDENT_THERMAL_MATERIAL_MODEL",
            "BLOCKED_PHYSICAL_WEIGHTING_POTENTIAL_OR_FINITE_FLAKE_MASK",
        ],
    }
    summary_path = output / "thermal_pte_native_yee_pullback_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
