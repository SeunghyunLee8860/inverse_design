#!/usr/bin/env python3
"""Run a single-direction v261 Maxwell absorption AD–FD certificate."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
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


STEPS = np.asarray([0.02, 0.01, 0.005])


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


def _density_and_direction(shape: tuple[int, int, int]):
    nx, ny, nz = shape
    x = np.linspace(0.0, 2.0 * np.pi, nx)
    y = np.linspace(0.0, 2.0 * np.pi, ny)
    z = np.linspace(0.0, 1.0, nz)
    density = (
        0.5
        + 0.12
        * np.sin(x)[:, None, None]
        * np.cos(y)[None, :, None]
        * (0.8 + 0.2 * z[None, None, :])
    )
    direction = (
        np.cos(2.0 * x + 0.21)[:, None, None]
        * np.sin(3.0 * y - 0.17)[None, :, None]
        * (0.65 + 0.35 * z[None, None, :])
    )
    direction /= np.max(np.abs(direction))
    # Enforce bitwise periodic fenceposts, rather than relying on trig roundoff.
    for value in (density, direction):
        value[-1, :, :] = value[0, :, :]
        value[:, -1, :] = value[:, 0, :]
    if np.min(density - STEPS[0] * direction) < 0.0:
        raise RuntimeError("negative lower perturbed density")
    if np.max(density + STEPS[0] * direction) > 1.0:
        raise RuntimeError("upper perturbed density exceeds one")
    return density, direction


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=("forward", "postprocess-forward", "adfd"),
        default="forward",
    )
    parser.add_argument("--polarization", choices=("x", "y"), default="x")
    parser.add_argument(
        "--output-dir",
        default=f"/tmp/tairte4-maxwell-absorption-{_utc()}",
    )
    parser.add_argument("--force-rebuild", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_dir).expanduser().resolve()
    solver = output / f"solver_{args.polarization}"
    output.mkdir(parents=True, exist_ok=True)
    evaluator = AbsorptionVolumeCurrentEvaluator(
        workdir=solver,
        incident_polarization=args.polarization,
    )
    contract = evaluator.prepare(force_rebuild=args.force_rebuild)
    density, direction = _density_and_direction(evaluator.physical_shape)
    common = {
        "schema_version": 1,
        "generated_at_utc": _utc(),
        "phase": args.phase,
        "polarization": args.polarization,
        "git": {
            "branch": _git("branch", "--show-current"),
            "commit": _git("rev-parse", "HEAD"),
        },
        "solver_contract": contract,
        "density": {
            "shape": list(density.shape),
            "minimum": float(np.min(density)),
            "maximum": float(np.max(density)),
            "periodic_fencepost": True,
            "kind": "deterministic smooth physical-density certificate field",
        },
        "direction": {
            "minimum": float(np.min(direction)),
            "maximum": float(np.max(direction)),
            "periodic_fencepost": True,
        },
        "objective": {
            "kind": "frozen periodic smooth native-Yee absorption weight",
            "scope": (
                "Maxwell/absorption transpose certificate only; not yet the "
                "thermal-adjoint weight"
            ),
            "normalization": "per incident plane-wave intensity",
        },
    }
    if args.phase == "postprocess-forward":
        completed = solver / "absorption_certificate_forward.fsp"
        forward = evaluator.postprocess_completed_forward(completed)
        raw = output / f"postprocess_forward_{args.polarization}.npz"
        np.savez_compressed(
            raw,
            density_component_W_m3_per_W_m2=(
                forward.observation.density_component_W_m3
            ),
            weight=forward.weight,
            q_observation=forward.q_observation,
        )
        summary = {
            **common,
            "status": "COMPLETED_MAXWELL_ABSORPTION_FORWARD_SMOKE",
            "value": forward.value,
            "incident_intensity_W_m2": (
                forward.incident_intensity_W_m2
            ),
            "epsilon_imaginary": forward.epsilon_imaginary.tolist(),
            "power_component_per_intensity_m2": (
                forward.observation.power_component_W.tolist()
            ),
            "power_total_per_intensity_m2": (
                forward.observation.power_total_W
            ),
            "completed_project": {
                "path": str(completed),
                "bytes": completed.stat().st_size,
                "sha256": _sha256(completed),
                "committed_to_git": False,
            },
            "raw_artifact": {
                "path": str(raw),
                "bytes": raw.stat().st_size,
                "sha256": _sha256(raw),
                "committed_to_git": False,
            },
        }
        exit_code = 0
    elif args.phase == "forward":
        value = evaluator.forward_absorption(
            density,
            label="absorption_certificate_forward",
            density_mode="probe_safe",
        )
        summary = {
            **common,
            "status": "COMPLETED_MAXWELL_ABSORPTION_FORWARD_SMOKE",
            "value": value,
        }
        exit_code = 0
    else:
        evaluation = evaluator.value_and_gradient_absorption(
            density,
            label="absorption_certificate",
            density_mode="probe_safe",
        )
        gradient = np.asarray(evaluation.gradient_physical, float)
        analytic = float(np.sum(gradient * direction))
        rows = []
        for step in STEPS:
            plus = evaluator.forward_absorption(
                density + step * direction,
                label=f"absorption_fd_plus_{step:g}",
                density_mode="probe_safe",
            )
            minus = evaluator.forward_absorption(
                density - step * direction,
                label=f"absorption_fd_minus_{step:g}",
                density_mode="probe_safe",
            )
            finite_difference = (plus - minus) / (2.0 * step)
            relative_error = abs(finite_difference - analytic) / max(
                abs(analytic), abs(finite_difference), 1e-300
            )
            rows.append(
                {
                    "step": float(step),
                    "plus": float(plus),
                    "minus": float(minus),
                    "finite_difference": float(finite_difference),
                    "adjoint_directional": analytic,
                    "relative_error": float(relative_error),
                }
            )
        best = min(rows, key=lambda item: item["relative_error"])
        gates = {
            "gradient_all_finite": bool(np.all(np.isfinite(gradient))),
            "gradient_nonzero": bool(np.any(gradient != 0.0)),
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
            "directional_adfd_pass": best["relative_error"] < 0.05,
        }
        passed = all(gates.values())
        raw = output / f"maxwell_absorption_adfd_{args.polarization}.npz"
        np.savez_compressed(
            raw,
            density=density,
            direction=direction,
            gradient=gradient,
            fom=np.asarray(evaluation.fom),
        )
        summary = {
            **common,
            "status": (
                "VALIDATED_MAXWELL_ABSORPTION_PHYSICAL_DENSITY_ADFD"
                if passed
                else "FAILED_MAXWELL_ABSORPTION_PHYSICAL_DENSITY_ADFD"
            ),
            "passed": passed,
            "value": evaluation.fom,
            "gradient": {
                "l2": float(np.linalg.norm(gradient)),
                "absolute_maximum": float(np.max(np.abs(gradient))),
                "directional_adjoint": analytic,
            },
            "metadata": evaluation.metadata,
            "finite_difference": rows,
            "best_step": best,
            "gates": gates,
            "raw_artifact": {
                "path": str(raw),
                "bytes": raw.stat().st_size,
                "sha256": _sha256(raw),
                "committed_to_git": False,
            },
        }
        exit_code = 0 if passed else 2
    summary_path = output / f"{args.phase}_{args.polarization}_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
