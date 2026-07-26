#!/usr/bin/env python3
"""Run the full fixed-K latent-to-thermal/PTE v261 AD--FD certificate."""

from __future__ import annotations

import argparse
import hashlib
import json
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
from periodic_constrained_mapping import MAPPING_VERSION  # noqa: E402


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=REPOSITORY,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()


def _latent_and_direction(nx: int, ny: int) -> tuple[np.ndarray, np.ndarray]:
    ix = np.arange(nx, dtype=float)
    iy = np.arange(ny, dtype=float)
    latent = (
        0.5
        + 0.15
        * np.sin(2.0 * np.pi * ix / nx)[:, None]
        * np.cos(2.0 * np.pi * iy / ny)[None, :]
    )
    # A uniform latent perturbation gives a strong FD signal.  At this
    # structured baseline, the tanh derivative remains spatially varying, so
    # this is not merely a constant physical-density perturbation.
    direction = np.ones((nx, ny), float)
    return latent.reshape(-1), direction.reshape(-1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight-npz", required=True)
    parser.add_argument("--solver-workdir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--polarization", choices=("x", "y"), default="x")
    parser.add_argument("--beta", type=float, default=8.0)
    parser.add_argument("--step", type=float, default=0.005)
    parser.add_argument("--relative-error-limit", type=float, default=0.05)
    parser.add_argument("--force-rebuild", action="store_true")
    args = parser.parse_args()
    if not np.isfinite(args.beta) or args.beta <= 0.0:
        raise ValueError("beta must be finite and positive")
    if not np.isfinite(args.step) or args.step <= 0.0:
        raise ValueError("step must be finite and positive")

    weight_path = Path(args.weight_npz).expanduser().resolve()
    weight_data = np.load(weight_path)
    native_weight = np.asarray(
        weight_data["native_density_weight_A_m_W"], float
    )
    if not np.all(np.isfinite(native_weight)):
        raise RuntimeError("native thermal/PTE weight contains NaN or Inf")

    def frozen_weight(grid, component_volumes):
        expected = (
            np.asarray(grid["x"]).size,
            np.asarray(grid["y"]).size,
            np.asarray(grid["z"]).size,
            3,
        )
        if native_weight.shape != expected:
            raise RuntimeError(
                f"frozen weight shape {native_weight.shape} != {expected}"
            )
        return native_weight

    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    solver = Path(args.solver_workdir).expanduser().resolve()
    evaluator = AbsorptionVolumeCurrentEvaluator(
        workdir=solver,
        incident_polarization=args.polarization,
    )
    solver_contract = evaluator.prepare(force_rebuild=args.force_rebuild)
    model = lib.load_model()
    mapping = model.mapping
    physical_shape = (model.Nx, model.Ny, model.Nz)
    if evaluator.physical_shape != physical_shape:
        raise RuntimeError(
            f"evaluator shape {evaluator.physical_shape} "
            f"!= mapping shape {physical_shape}"
        )
    latent, direction = _latent_and_direction(mapping.Nux, mapping.Nuy)
    if np.min(latent - args.step * direction) < 0.0:
        raise RuntimeError("negative latent FD endpoint")
    if np.max(latent + args.step * direction) > 1.0:
        raise RuntimeError("latent FD endpoint exceeds one")

    def physical(value):
        return np.asarray(mapping(value, args.beta), float).reshape(
            physical_shape
        )

    baseline_physical = physical(latent)
    evaluation = evaluator.value_and_gradient_absorption(
        baseline_physical,
        label="latent_thermal_pte",
        density_mode="probe_safe",
        weight_builder=frozen_weight,
    )
    physical_gradient = np.asarray(evaluation.gradient_physical, float)
    latent_gradient = np.asarray(
        tensor_jacobian_product(
            lambda value: mapping(value, args.beta)
        )(latent, physical_gradient.reshape(-1)),
        float,
    )
    adjoint_directional = float(np.dot(latent_gradient, direction))

    plus_physical = physical(latent + args.step * direction)
    minus_physical = physical(latent - args.step * direction)
    plus = evaluator.forward_absorption(
        plus_physical,
        label="latent_thermal_pte_fd_plus",
        density_mode="probe_safe",
        weight_builder=frozen_weight,
    )
    minus = evaluator.forward_absorption(
        minus_physical,
        label="latent_thermal_pte_fd_minus",
        density_mode="probe_safe",
        weight_builder=frozen_weight,
    )
    finite_difference = (plus - minus) / (2.0 * args.step)
    relative_error = abs(finite_difference - adjoint_directional) / max(
        abs(finite_difference), abs(adjoint_directional), 1e-300
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
        "gradient_all_finite": bool(
            np.all(np.isfinite(physical_gradient))
            and np.all(np.isfinite(latent_gradient))
        ),
        "gradient_nonzero": bool(np.any(latent_gradient != 0.0)),
        "periodic_x_fencepost_exact": seam_errors["x"] == 0.0,
        "periodic_y_fencepost_exact": seam_errors["y"] == 0.0,
        "z_extrusion_exact": seam_errors["z_extrusion"] == 0.0,
        "periodic_source_pairing_pass": (
            evaluation.metadata[
                "periodic_source_pairing_relative_error"
            ]
            < 1e-13
        ),
        "source_roundtrip_pass": all(
            item["roundtrip_max_abs_error"] == 0.0
            for item in evaluation.metadata["source"].values()
        ),
        "latent_adfd_relative_error": relative_error,
        "latent_adfd_pass": relative_error < args.relative_error_limit,
    }
    passed = all(
        value
        for key, value in gates.items()
        if key.endswith("_pass") or key.endswith("_exact")
    ) and gates["gradient_all_finite"] and gates["gradient_nonzero"]

    raw = output / "full_latent_thermal_pte_adfd.npz"
    np.savez_compressed(
        raw,
        latent=latent,
        direction=direction,
        baseline_physical=baseline_physical,
        plus_physical=plus_physical,
        minus_physical=minus_physical,
        physical_gradient=physical_gradient,
        latent_gradient=latent_gradient,
        objective=np.asarray(evaluation.fom),
        plus=np.asarray(plus),
        minus=np.asarray(minus),
    )
    summary = {
        "schema_version": 1,
        "generated_at_utc": _utc(),
        "status": (
            "VALIDATED_FIXED_K_LATENT_THERMAL_PTE_ADFD"
            if passed
            else "FAILED_FIXED_K_LATENT_THERMAL_PTE_ADFD"
        ),
        "passed": passed,
        "scope": (
            "actual periodic inverse-design Maxwell geometry; fixed-K "
            "thermal operator; finite-local-mask PTE functional"
        ),
        "git": {
            "branch": _git("branch", "--show-current"),
            "commit": _git("rev-parse", "HEAD"),
        },
        "solver_contract": solver_contract,
        "mapping": {
            "version": MAPPING_VERSION,
            "latent_shape": [mapping.Nux, mapping.Nuy],
            "physical_shape": list(physical_shape),
            "config": mapping.config.to_dict(),
            "beta": args.beta,
            "direction": "uniform latent perturbation",
            "step": args.step,
            "density_mode": "probe_safe for baseline, plus, minus, and AD",
        },
        "objective_A_m_per_W_m2": evaluation.fom,
        "adjoint_directional": adjoint_directional,
        "finite_difference": finite_difference,
        "plus": plus,
        "minus": minus,
        "relative_error": relative_error,
        "relative_error_limit": args.relative_error_limit,
        "gates": gates,
        "seam_errors": seam_errors,
        "gradient": {
            "physical_l2": float(np.linalg.norm(physical_gradient)),
            "latent_l2": float(np.linalg.norm(latent_gradient)),
            "latent_absolute_maximum": float(
                np.max(np.abs(latent_gradient))
            ),
        },
        "metadata": evaluation.metadata,
        "weight_artifact": {
            "path": str(weight_path),
            "bytes": weight_path.stat().st_size,
            "sha256": _sha256(weight_path),
        },
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
    summary_path = output / "full_latent_thermal_pte_adfd_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
