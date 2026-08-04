#!/usr/bin/env python3
"""Full latent n=4 optical plus paper-reduced thermal-boundary AD--FD."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from autograd import tensor_jacobian_product


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
    boundary_diagnostics,
    evaluate_reduced_paper_thermal,
)
from periodic_constrained_mapping import MAPPING_VERSION  # noqa: E402


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
    parser.add_argument("--latent-npz", required=True)
    parser.add_argument("--forward-fsp", required=True)
    parser.add_argument("--plus-fsp", required=True)
    parser.add_argument("--minus-fsp", required=True)
    parser.add_argument("--solver-workdir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--step", type=float, default=0.005)
    parser.add_argument("--beta", type=float, default=8.0)
    parser.add_argument("--filter-radius-um", type=float, default=0.5)
    parser.add_argument("--polarization", choices=("x", "y"), default="x")
    parser.add_argument(
        "--relative-error-limit", type=float, default=0.05
    )
    parser.add_argument("--adjoint-x-fsp")
    parser.add_argument("--adjoint-y-fsp")
    args = parser.parse_args()

    latent_path = Path(args.latent_npz).expanduser().resolve()
    forward_path = Path(args.forward_fsp).expanduser().resolve()
    plus_path = Path(args.plus_fsp).expanduser().resolve()
    minus_path = Path(args.minus_fsp).expanduser().resolve()
    solver = Path(args.solver_workdir).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    data = np.load(latent_path)
    latent = np.asarray(data["latent"], float)
    direction = np.asarray(data["direction"], float)
    baseline_physical = np.asarray(data["baseline_physical"], float)
    plus_physical = np.asarray(data["plus_physical"], float)
    minus_physical = np.asarray(data["minus_physical"], float)

    # The immutable latent checkpoint records a 0.5 um production filter.
    # Set it before the lazily imported model module is loaded so replay is
    # fail-closed rather than silently using the repository's 0.2 um default.
    os.environ["MFS_UM"] = str(args.filter_radius_um)
    os.environ["MGS_UM"] = str(args.filter_radius_um)
    model = lib.load_model()
    mapping = model.mapping
    physical_shape = (model.Nx, model.Ny, model.Nz)
    if baseline_physical.shape != physical_shape:
        raise RuntimeError(
            f"stored physical {baseline_physical.shape} != {physical_shape}"
        )
    regenerated = np.asarray(
        mapping(latent, args.beta), float
    ).reshape(physical_shape)
    mapping_replay_error = float(
        np.max(np.abs(regenerated - baseline_physical))
    )
    if mapping_replay_error > 1e-13:
        raise RuntimeError(
            f"stored baseline does not replay mapping: {mapping_replay_error}"
        )
    surface_map = DesignSurfaceMap(
        physical_shape=physical_shape,
        face_shape=(24, 24),
    )

    evaluator = AbsorptionVolumeCurrentEvaluator(
        workdir=solver,
        incident_polarization=args.polarization,
    )
    solver_contract = evaluator.prepare(force_rebuild=False)
    optical = evaluator.postprocess_completed_forward(forward_path)
    remap = remap_absorption_to_reduced_thermal(optical)
    thermal = evaluate_reduced_paper_thermal(
        rho_face=surface_map.apply(baseline_physical),
        source_W_m3=remap.source_W_m3,
    )
    thermal_q_sensitivity = thermal.system.full_field(
        thermal.gradient_Q_active_A_m4_W
    )
    native_shape = (
        *optical.observation.density_component_W_m3.shape,
    )
    native_weight = remap.native_weight_from_thermal_sensitivity(
        thermal_density_sensitivity=thermal_q_sensitivity,
        native_component_volume_m3=optical.component_volume_m3,
        native_shape=native_shape,
    )

    def frozen_weight(grid, component_volumes):
        return native_weight

    if bool(args.adjoint_x_fsp) != bool(args.adjoint_y_fsp):
        raise ValueError("provide both completed adjoints or neither")
    if args.adjoint_x_fsp:
        optical_ad = evaluator.resume_completed_adjoint(
            baseline_physical,
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
            baseline_physical,
            forward_project=forward_path,
            label="paper_reduced_latent",
            density_mode="probe_safe",
            weight_builder=frozen_weight,
        )
    optical_gradient = np.asarray(optical_ad.gradient_physical, float)
    thermal_gradient = surface_map.transpose(
        thermal.gradient_rho_face_A_m
    )
    combined_physical_gradient = optical_gradient + thermal_gradient
    combined_latent_gradient = np.asarray(
        tensor_jacobian_product(
            lambda value: mapping(value, args.beta)
        )(latent, combined_physical_gradient.reshape(-1)),
        float,
    )
    optical_latent_gradient = np.asarray(
        tensor_jacobian_product(
            lambda value: mapping(value, args.beta)
        )(latent, optical_gradient.reshape(-1)),
        float,
    )
    thermal_latent_gradient = np.asarray(
        tensor_jacobian_product(
            lambda value: mapping(value, args.beta)
        )(latent, thermal_gradient.reshape(-1)),
        float,
    )
    optical_directional = float(
        np.dot(optical_latent_gradient, direction)
    )
    thermal_directional = float(
        np.dot(thermal_latent_gradient, direction)
    )
    combined_directional = float(
        np.dot(combined_latent_gradient, direction)
    )

    def endpoint(fsp: Path, density: np.ndarray) -> tuple[float, dict]:
        endpoint_optical = evaluator.postprocess_completed_forward(fsp)
        endpoint_remap = remap_absorption_to_reduced_thermal(
            endpoint_optical
        )
        endpoint_thermal = evaluate_reduced_paper_thermal(
            rho_face=surface_map.apply(density),
            source_W_m3=endpoint_remap.source_W_m3,
        )
        return endpoint_thermal.objective_A_m, {
            "source_power_W_per_W_m2": endpoint_remap.thermal_power_W,
            "energy_balance_relative_error": (
                endpoint_thermal.solved.energy_balance_relative_error
            ),
            "linear_residual_relative": (
                endpoint_thermal.solved.linear_residual_relative
            ),
        }

    plus, plus_meta = endpoint(plus_path, plus_physical)
    minus, minus_meta = endpoint(minus_path, minus_physical)
    finite_difference = (plus - minus) / (2.0 * args.step)
    relative_error = abs(finite_difference - combined_directional) / max(
        abs(finite_difference), abs(combined_directional), 1e-300
    )
    seam_errors = {
        "x": float(
            np.max(np.abs(baseline_physical[-1] - baseline_physical[0]))
        ),
        "y": float(
            np.max(
                np.abs(
                    baseline_physical[:, -1] - baseline_physical[:, 0]
                )
            )
        ),
        "z_extrusion": float(
            np.max(
                np.abs(
                    baseline_physical - baseline_physical[:, :, :1]
                )
            )
        ),
    }
    gates = {
        "mapping_replay_max_abs_error": mapping_replay_error,
        "mapping_replay_pass": mapping_replay_error < 1e-13,
        "periodic_x_fencepost_exact": seam_errors["x"] == 0.0,
        "periodic_y_fencepost_exact": seam_errors["y"] == 0.0,
        "z_extrusion_exact": seam_errors["z_extrusion"] == 0.0,
        "energy_balance_relative_error": (
            thermal.solved.energy_balance_relative_error
        ),
        "energy_balance_pass": (
            thermal.solved.energy_balance_relative_error < 0.01
        ),
        "linear_residual_relative": (
            thermal.solved.linear_residual_relative
        ),
        "linear_residual_pass": (
            thermal.solved.linear_residual_relative < 1e-8
        ),
        "combined_latent_adfd_relative_error": relative_error,
        "combined_latent_adfd_pass": (
            relative_error < args.relative_error_limit
        ),
    }
    passed = all(
        value
        for key, value in gates.items()
        if key.endswith("_pass") or key.endswith("_exact")
    )
    raw = output / "paper_reduced_combined_latent_adfd.npz"
    np.savez_compressed(
        raw,
        latent=latent,
        direction=direction,
        baseline_physical=baseline_physical,
        rho_face=surface_map.apply(baseline_physical),
        native_weight_A_m_W=native_weight,
        optical_physical_gradient=optical_gradient,
        thermal_physical_gradient=thermal_gradient,
        combined_physical_gradient=combined_physical_gradient,
        optical_latent_gradient=optical_latent_gradient,
        thermal_latent_gradient=thermal_latent_gradient,
        combined_latent_gradient=combined_latent_gradient,
        temperature_rise_K_per_W_m2=thermal.solved.temperature_K,
        plus=np.asarray(plus),
        minus=np.asarray(minus),
    )
    summary = {
        "schema_version": 1,
        "generated_at_utc": _utc(),
        "status": (
            "VALIDATED_PAPER_REDUCED_COMBINED_LATENT_ADFD"
            if passed
            else "FAILED_PAPER_REDUCED_COMBINED_LATENT_ADFD"
        ),
        "passed": passed,
        "scope": (
            "latent -> filter -> projection -> n=4 optical Q and "
            "paper-SiO2 rho-dependent Robin thermal boundary -> local PTE"
        ),
        "git": {
            "branch": _git("branch", "--show-current"),
            "head_before_generated_reports": _git("rev-parse", "HEAD"),
        },
        "solver_contract": solver_contract,
        "mapping": {
            "version": MAPPING_VERSION,
            "beta": args.beta,
            "step": args.step,
            "latent_shape": [mapping.Nux, mapping.Nuy],
            "physical_shape": list(physical_shape),
            "config": mapping.config.to_dict(),
            "replay_max_abs_error": mapping_replay_error,
        },
        "objective_A_m_per_W_m2": thermal.objective_A_m,
        "plus_A_m_per_W_m2": plus,
        "minus_A_m_per_W_m2": minus,
        "finite_difference": finite_difference,
        "adjoint": {
            "optical_Q_directional": optical_directional,
            "thermal_material_directional": thermal_directional,
            "combined_directional": combined_directional,
            "optical_latent_gradient_l2": float(
                np.linalg.norm(optical_latent_gradient)
            ),
            "thermal_latent_gradient_l2": float(
                np.linalg.norm(thermal_latent_gradient)
            ),
            "combined_latent_gradient_l2": float(
                np.linalg.norm(combined_latent_gradient)
            ),
        },
        "relative_error": relative_error,
        "relative_error_limit": args.relative_error_limit,
        "boundary": boundary_diagnostics(thermal),
        "source_power_W_per_W_m2": remap.thermal_power_W,
        "endpoints": {"plus": plus_meta, "minus": minus_meta},
        "seam_errors": seam_errors,
        "gates": gates,
        "optical_metadata": optical_ad.metadata,
        "inputs": {
            "latent_npz": str(latent_path),
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
            "BLOCKED_COMBINED_ADFD_STEP_SWEEP",
            "BLOCKED_PHYSICAL_WEIGHTING_POTENTIAL_OR_FINITE_FLAKE_MASK",
        ],
    }
    summary_path = (
        output / "paper_reduced_combined_latent_adfd_summary.json"
    )
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
