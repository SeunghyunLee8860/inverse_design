#!/usr/bin/env python3
"""End-to-end fixed-K thermal/PTE to physical-density Maxwell AD–FD."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
PHOTOTHERMAL = HERE.parent
REPOSITORY = PHOTOTHERMAL.parent
VOLUME_CURRENT = REPOSITORY / "volume_current_inverse_design"
for path in (HERE, VOLUME_CURRENT, VOLUME_CURRENT / "bundle"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from maxwell_absorption_evaluator import (  # noqa: E402
    AbsorptionVolumeCurrentEvaluator,
)
from run_maxwell_absorption_adfd_certificate import (  # noqa: E402
    _density_and_direction,
)


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight-npz", required=True)
    parser.add_argument("--forward-fsp", required=True)
    parser.add_argument("--plus-fsp", required=True)
    parser.add_argument("--minus-fsp", required=True)
    parser.add_argument("--solver-workdir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--polarization", choices=("x", "y"), default="x")
    parser.add_argument("--step", type=float, default=0.0025)
    args = parser.parse_args()
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
    evaluator = AbsorptionVolumeCurrentEvaluator(
        workdir=Path(args.solver_workdir).expanduser().resolve(),
        incident_polarization=args.polarization,
    )
    evaluator.prepare(force_rebuild=False)
    density, direction = _density_and_direction(
        evaluator.physical_shape, float(args.step), "uniform"
    )
    evaluation = evaluator.run_adjoint_projects_from_completed_forward(
        density,
        forward_project=Path(args.forward_fsp).expanduser().resolve(),
        label="thermal_pte",
        density_mode="probe_safe",
        weight_builder=frozen_weight,
    )
    plus = evaluator.postprocess_completed_forward(
        Path(args.plus_fsp).expanduser().resolve(),
        weight_builder=frozen_weight,
    ).value
    minus = evaluator.postprocess_completed_forward(
        Path(args.minus_fsp).expanduser().resolve(),
        weight_builder=frozen_weight,
    ).value
    gradient = np.asarray(evaluation.gradient_physical, float)
    adjoint_directional = float(np.sum(gradient * direction))
    finite_difference = (plus - minus) / (2.0 * args.step)
    relative_error = abs(finite_difference - adjoint_directional) / max(
        abs(finite_difference), abs(adjoint_directional), 1e-300
    )
    gates = {
        "gradient_all_finite": bool(np.all(np.isfinite(gradient))),
        "gradient_nonzero": bool(np.any(gradient != 0.0)),
        "periodic_source_pairing_pass": (
            evaluation.metadata["periodic_source_pairing_relative_error"]
            < 1e-13
        ),
        "source_roundtrip_pass": all(
            item["roundtrip_max_abs_error"] == 0.0
            for item in evaluation.metadata["source"].values()
        ),
        "physical_density_adfd_relative_error": relative_error,
        "physical_density_adfd_pass": relative_error < 0.05,
    }
    passed = all(
        value for key, value in gates.items() if key.endswith("_pass")
    )
    raw = output / "thermal_weighted_maxwell_adfd.npz"
    np.savez_compressed(
        raw,
        density=density,
        direction=direction,
        gradient=gradient,
        objective=np.asarray(evaluation.fom),
        plus=np.asarray(plus),
        minus=np.asarray(minus),
    )
    summary = {
        "schema_version": 1,
        "generated_at_utc": _utc(),
        "status": (
            "VALIDATED_FIXED_K_THERMAL_PTE_PHYSICAL_DENSITY_ADFD"
            if passed
            else "FAILED_FIXED_K_THERMAL_PTE_PHYSICAL_DENSITY_ADFD"
        ),
        "passed": passed,
        "scope": (
            "fixed-K finite-local-mask PTE; actual periodic inverse-design "
            "Maxwell geometry"
        ),
        "objective_A_m_per_W_m2": evaluation.fom,
        "step": args.step,
        "direction": "uniform physical density",
        "adjoint_directional": adjoint_directional,
        "finite_difference": finite_difference,
        "plus": plus,
        "minus": minus,
        "relative_error": relative_error,
        "gradient": {
            "l2": float(np.linalg.norm(gradient)),
            "absolute_maximum": float(np.max(np.abs(gradient))),
        },
        "gates": gates,
        "metadata": evaluation.metadata,
        "inputs": {
            "weight_npz": {
                "path": str(weight_path),
                "bytes": weight_path.stat().st_size,
                "sha256": _sha256(weight_path),
            },
            "forward_fsp": str(Path(args.forward_fsp).resolve()),
            "plus_fsp": str(Path(args.plus_fsp).resolve()),
            "minus_fsp": str(Path(args.minus_fsp).resolve()),
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
    summary_path = output / "thermal_weighted_maxwell_adfd_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
