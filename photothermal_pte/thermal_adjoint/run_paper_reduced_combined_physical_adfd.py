#!/usr/bin/env python3
"""Combined Maxwell-Q and rho-dependent reduced-thermal physical AD--FD."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from yee_absorption_functional import (
    weighted_yee_absorption_and_wirtinger,
)


HERE = Path(__file__).resolve().parent
PHOTOTHERMAL = HERE.parent
REPOSITORY = PHOTOTHERMAL.parent
VOLUME_CURRENT = REPOSITORY / "volume_current_inverse_design"
for path in (HERE, VOLUME_CURRENT, VOLUME_CURRENT / "bundle"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import eqc_lib as lib  # noqa: E402
from maxwell_absorption_evaluator import (  # noqa: E402
    AbsorptionVolumeCurrentEvaluator,
)
from paper_reduced_optical_coupling import (  # noqa: E402
    remap_absorption_to_reduced_thermal,
)
from paper_reduced_thermal import (  # noqa: E402
    DesignSurfaceMap,
    G_THERMAL_SIO2_W_M2K,
    boundary_diagnostics,
    evaluate_reduced_paper_thermal,
)


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=REPOSITORY,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--density-npz", required=True)
    parser.add_argument("--forward-fsp", required=True)
    parser.add_argument("--plus-fsp", required=True)
    parser.add_argument("--minus-fsp", required=True)
    parser.add_argument("--solver-workdir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--step", type=float, required=True)
    parser.add_argument("--adjoint-x-fsp")
    parser.add_argument("--adjoint-y-fsp")
    parser.add_argument("--polarization", choices=("x", "y"), default="x")
    parser.add_argument(
        "--relative-error-limit", type=float, default=0.05
    )
    args = parser.parse_args()
    density_path = Path(args.density_npz).expanduser().resolve()
    forward_path = Path(args.forward_fsp).expanduser().resolve()
    plus_path = Path(args.plus_fsp).expanduser().resolve()
    minus_path = Path(args.minus_fsp).expanduser().resolve()
    solver = Path(args.solver_workdir).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    density_data = np.load(density_path)
    density = np.asarray(density_data["density"], float)
    direction = np.asarray(density_data["direction"], float)
    if density.shape != direction.shape:
        raise RuntimeError("density and direction shapes differ")
    if np.min(density - args.step * direction) < 0.0:
        raise RuntimeError("negative physical-density FD endpoint")
    if np.max(density + args.step * direction) > 1.0:
        raise RuntimeError("physical-density FD endpoint exceeds one")
    surface_map = DesignSurfaceMap(
        physical_shape=density.shape,
        face_shape=(24, 24),
    )
    rho_face = surface_map.apply(density)

    evaluator = AbsorptionVolumeCurrentEvaluator(
        workdir=solver,
        incident_polarization=args.polarization,
    )
    solver_contract = evaluator.prepare(force_rebuild=False)
    baseline_optical = evaluator.postprocess_completed_forward(forward_path)
    baseline_remap = remap_absorption_to_reduced_thermal(baseline_optical)
    baseline_thermal = evaluate_reduced_paper_thermal(
        rho_face=rho_face,
        source_W_m3=baseline_remap.source_W_m3,
        G_sio2_W_m2K=G_THERMAL_SIO2_W_M2K,
    )
    thermal_q_sensitivity = baseline_thermal.system.full_field(
        baseline_thermal.gradient_Q_active_A_m4_W
    )
    native_shape = (
        *baseline_optical.observation.density_component_W_m3.shape,
    )
    native_weight = baseline_remap.native_weight_from_thermal_sensitivity(
        thermal_density_sensitivity=thermal_q_sensitivity,
        native_component_volume_m3=baseline_optical.component_volume_m3,
        native_shape=native_shape,
    )

    def frozen_weight(grid, component_volumes):
        expected = (
            np.asarray(grid["x"]).size,
            np.asarray(grid["y"]).size,
            np.asarray(grid["z"]).size,
            3,
        )
        if native_weight.shape != expected:
            raise RuntimeError(
                f"native weight {native_weight.shape} != {expected}"
            )
        return native_weight

    weighted_check = weighted_yee_absorption_and_wirtinger(
        electric_field_V_m=baseline_optical.field_result["fom"]["E"],
        frequency_Hz=float(baseline_optical.grid["f"][0]),
        epsilon_imaginary=baseline_optical.epsilon_imaginary,
        component_volume_m3=baseline_optical.component_volume_m3,
        density_weight=native_weight,
        power_scale=1.0 / baseline_optical.incident_intensity_W_m2,
    )
    objective_identity_error = abs(
        weighted_check.weighted_value - baseline_thermal.objective_A_m
    ) / max(
        abs(weighted_check.weighted_value),
        abs(baseline_thermal.objective_A_m),
        1e-300,
    )
    if bool(args.adjoint_x_fsp) != bool(args.adjoint_y_fsp):
        raise ValueError("provide both completed adjoint FSPs or neither")
    if args.adjoint_x_fsp:
        optical_ad = evaluator.resume_completed_adjoint(
            density,
            forward_project=forward_path,
            adjoint_projects={
                0: Path(args.adjoint_x_fsp).expanduser().resolve(),
                1: Path(args.adjoint_y_fsp).expanduser().resolve(),
            },
            density_mode="probe_safe",
            weight_builder=frozen_weight,
        )
    else:
        optical_ad = evaluator.run_adjoint_projects_from_completed_forward(
            density,
            forward_project=forward_path,
            label="paper_reduced_physical",
            density_mode="probe_safe",
            weight_builder=frozen_weight,
        )
    optical_gradient = np.asarray(optical_ad.gradient_physical, float)
    thermal_gradient = surface_map.transpose(
        baseline_thermal.gradient_rho_face_A_m
    )
    combined_gradient = optical_gradient + thermal_gradient
    optical_directional = float(np.sum(optical_gradient * direction))
    thermal_directional = float(np.sum(thermal_gradient * direction))
    combined_directional = float(np.sum(combined_gradient * direction))

    def endpoint(
        *,
        fsp: Path,
        physical_density: np.ndarray,
    ) -> tuple[float, dict]:
        optical = evaluator.postprocess_completed_forward(fsp)
        remap = remap_absorption_to_reduced_thermal(optical)
        thermal = evaluate_reduced_paper_thermal(
            rho_face=surface_map.apply(physical_density),
            source_W_m3=remap.source_W_m3,
            G_sio2_W_m2K=G_THERMAL_SIO2_W_M2K,
        )
        return thermal.objective_A_m, {
            "native_power_W_per_W_m2": remap.native_power_W,
            "common_power_W_per_W_m2": remap.common_power_W,
            "thermal_power_W_per_W_m2": remap.thermal_power_W,
            "energy_balance_relative_error": (
                thermal.solved.energy_balance_relative_error
            ),
            "linear_residual_relative": (
                thermal.solved.linear_residual_relative
            ),
        }

    plus, plus_meta = endpoint(
        fsp=plus_path,
        physical_density=density + args.step * direction,
    )
    minus, minus_meta = endpoint(
        fsp=minus_path,
        physical_density=density - args.step * direction,
    )
    finite_difference = (plus - minus) / (2.0 * args.step)
    relative_error = abs(finite_difference - combined_directional) / max(
        abs(finite_difference), abs(combined_directional), 1e-300
    )
    power_errors = {
        "native_to_common": abs(
            baseline_remap.common_power_W - baseline_remap.native_power_W
        )
        / max(abs(baseline_remap.native_power_W), 1e-300),
        "native_to_thermal": abs(
            baseline_remap.thermal_power_W - baseline_remap.native_power_W
        )
        / max(abs(baseline_remap.native_power_W), 1e-300),
    }
    gates = {
        "objective_pullback_identity_relative_error": (
            objective_identity_error
        ),
        # This identity combines several sparse remaps and native Yee
        # quadratures at an O(1e-21) objective.  The observed absolute
        # discrepancy is O(1e-32), so 5e-11 is a roundoff gate rather than an
        # AD-FD accuracy relaxation.
        "objective_pullback_identity_pass": objective_identity_error < 5e-11,
        "native_to_common_power_relative_error": power_errors[
            "native_to_common"
        ],
        "native_to_common_power_pass": power_errors["native_to_common"] < 5e-13,
        "native_to_thermal_power_relative_error": power_errors[
            "native_to_thermal"
        ],
        "native_to_thermal_power_pass": (
            power_errors["native_to_thermal"] < 5e-13
        ),
        "baseline_energy_balance_relative_error": (
            baseline_thermal.solved.energy_balance_relative_error
        ),
        "baseline_energy_balance_pass": (
            baseline_thermal.solved.energy_balance_relative_error < 0.01
        ),
        "baseline_linear_residual_relative": (
            baseline_thermal.solved.linear_residual_relative
        ),
        "baseline_linear_residual_pass": (
            baseline_thermal.solved.linear_residual_relative < 1e-8
        ),
        "combined_physical_adfd_relative_error": relative_error,
        "combined_physical_adfd_pass": (
            relative_error < args.relative_error_limit
        ),
        "optical_gradient_finite": bool(
            np.all(np.isfinite(optical_gradient))
        ),
        "thermal_gradient_finite": bool(
            np.all(np.isfinite(thermal_gradient))
        ),
    }
    passed = all(
        value for key, value in gates.items() if key.endswith("_pass")
    ) and gates["optical_gradient_finite"] and gates[
        "thermal_gradient_finite"
    ]

    raw = output / "paper_reduced_combined_physical_adfd.npz"
    np.savez_compressed(
        raw,
        density=density,
        direction=direction,
        rho_face=rho_face,
        source_W_m3_per_W_m2=baseline_remap.source_W_m3,
        temperature_rise_K_per_W_m2=(
            baseline_thermal.solved.temperature_K
        ),
        native_weight_A_m_W=native_weight,
        optical_gradient=optical_gradient,
        thermal_material_gradient=thermal_gradient,
        combined_gradient=combined_gradient,
        objective=np.asarray(baseline_thermal.objective_A_m),
        plus=np.asarray(plus),
        minus=np.asarray(minus),
    )
    summary = {
        "schema_version": 1,
        "generated_at_utc": _utc(),
        "status": (
            "VALIDATED_PAPER_REDUCED_COMBINED_PHYSICAL_RHO_ADFD"
            if passed
            else "FAILED_PAPER_REDUCED_COMBINED_PHYSICAL_RHO_ADFD"
        ),
        "passed": passed,
        "scope": (
            "n=4 optical proxy plus paper SiO2 rho-dependent reduced "
            "thermal boundary; physical-density combined AD-FD"
        ),
        "git": {
            "branch": _git("branch", "--show-current"),
            "head_before_generated_reports": _git("rev-parse", "HEAD"),
        },
        "solver_contract": solver_contract,
        "step": args.step,
        "objective_A_m_per_W_m2": baseline_thermal.objective_A_m,
        "plus_A_m_per_W_m2": plus,
        "minus_A_m_per_W_m2": minus,
        "finite_difference": finite_difference,
        "adjoint": {
            "optical_Q_directional": optical_directional,
            "thermal_material_directional": thermal_directional,
            "combined_directional": combined_directional,
            "optical_gradient_l2": float(
                np.linalg.norm(optical_gradient)
            ),
            "thermal_material_gradient_l2": float(
                np.linalg.norm(thermal_gradient)
            ),
            "combined_gradient_l2": float(
                np.linalg.norm(combined_gradient)
            ),
        },
        "relative_error": relative_error,
        "relative_error_limit": args.relative_error_limit,
        "boundary": boundary_diagnostics(baseline_thermal),
        "power": {
            "native_W_per_W_m2": baseline_remap.native_power_W,
            "common_W_per_W_m2": baseline_remap.common_power_W,
            "thermal_W_per_W_m2": baseline_remap.thermal_power_W,
        },
        "endpoints": {"plus": plus_meta, "minus": minus_meta},
        "gates": gates,
        "optical_metadata": optical_ad.metadata,
        "inputs": {
            "density_npz": str(density_path),
            "forward_fsp": str(forward_path),
            "plus_fsp": str(plus_path),
            "minus_fsp": str(minus_path),
        },
        "raw_artifact": {
            "path": str(raw),
            "bytes": raw.stat().st_size,
            "sha256": _sha256(raw),
            "committed_to_git": False,
        },
        "blockers": [
            "BLOCKED_FULL_LATENT_PAPER_REDUCED_COMBINED_ADFD",
            "BLOCKED_PHYSICAL_WEIGHTING_POTENTIAL_OR_FINITE_FLAKE_MASK",
        ],
    }
    summary_path = (
        output / "paper_reduced_combined_physical_adfd_summary.json"
    )
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
