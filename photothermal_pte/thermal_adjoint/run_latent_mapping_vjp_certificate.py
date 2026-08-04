#!/usr/bin/env python3
"""Certify the production latent-to-physical mapping VJP without a solver.

The covector is taken from a completed thermal/PTE-to-Maxwell physical-density
adjoint.  It is normalized before the dot test because its physical units make
its entries extremely small.  This normalization does not change the mapping
Jacobian or the relative AD--FD error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import autograd.numpy as anp
import numpy as np
from autograd import tensor_jacobian_product


HERE = Path(__file__).resolve().parent
PHOTOTHERMAL = HERE.parent
REPOSITORY = PHOTOTHERMAL.parent
VOLUME_CURRENT = REPOSITORY / "volume_current_inverse_design"
for path in (VOLUME_CURRENT, VOLUME_CURRENT / "bundle"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import eqc_lib as lib  # noqa: E402
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
        + 0.12
        * np.sin(2.0 * np.pi * ix / nx)[:, None]
        * np.cos(4.0 * np.pi * iy / ny)[None, :]
    )
    rng = np.random.default_rng(20260726)
    direction = rng.standard_normal((nx, ny))
    direction /= np.linalg.norm(direction)
    return latent.reshape(-1), direction.reshape(-1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--physical-gradient-npz", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--betas", default="4,8,32")
    parser.add_argument("--step", type=float, default=1e-4)
    parser.add_argument("--relative-error-limit", type=float, default=1e-5)
    args = parser.parse_args()

    gradient_path = Path(args.physical_gradient_npz).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    betas = np.asarray(
        [float(value) for value in args.betas.split(",") if value.strip()],
        float,
    )
    if (
        betas.size == 0
        or np.any(~np.isfinite(betas))
        or np.any(betas <= 0.0)
    ):
        raise ValueError("betas must be finite and positive")
    if not np.isfinite(args.step) or args.step <= 0.0:
        raise ValueError("step must be finite and positive")

    data = np.load(gradient_path)
    physical_covector = np.asarray(data["gradient"], float)
    if not np.all(np.isfinite(physical_covector)):
        raise RuntimeError("physical gradient contains NaN or Inf")
    covector_l2 = float(np.linalg.norm(physical_covector))
    if covector_l2 == 0.0:
        raise RuntimeError("physical gradient is identically zero")
    normalized_covector = physical_covector.reshape(-1) / covector_l2

    model = lib.load_model()
    mapping = model.mapping
    expected_shape = (model.Nx, model.Ny, model.Nz)
    if physical_covector.shape != expected_shape:
        raise RuntimeError(
            f"physical gradient shape {physical_covector.shape} "
            f"!= production mapping shape {expected_shape}"
        )
    if (
        mapping.Nux != model.Nx - 1
        or mapping.Nuy != model.Ny - 1
        or mapping.Nz != model.Nz
    ):
        raise RuntimeError("production mapping grid contract is inconsistent")

    latent, direction = _latent_and_direction(mapping.Nux, mapping.Nuy)
    if np.min(latent - args.step * direction) < 0.0:
        raise RuntimeError("negative latent FD endpoint")
    if np.max(latent + args.step * direction) > 1.0:
        raise RuntimeError("latent FD endpoint exceeds one")

    cases = []
    gradients = {}
    for beta in betas:
        mapping_call = lambda value: mapping(value, float(beta))
        latent_gradient = np.asarray(
            tensor_jacobian_product(mapping_call)(
                latent, normalized_covector
            ),
            float,
        )
        adjoint_directional = float(np.dot(latent_gradient, direction))

        def scalar(value):
            return anp.sum(mapping_call(value) * normalized_covector)

        plus = float(scalar(latent + args.step * direction))
        minus = float(scalar(latent - args.step * direction))
        finite_difference = (plus - minus) / (2.0 * args.step)
        relative_error = abs(finite_difference - adjoint_directional) / max(
            abs(finite_difference), abs(adjoint_directional), 1e-300
        )
        physical = np.asarray(mapping_call(latent), float).reshape(
            expected_shape
        )
        seam_x = float(np.max(np.abs(physical[-1] - physical[0])))
        seam_y = float(np.max(np.abs(physical[:, -1] - physical[:, 0])))
        extrusion = float(
            np.max(np.abs(physical - physical[:, :, :1]))
        )
        cases.append(
            {
                "beta": float(beta),
                "step": float(args.step),
                "adjoint_directional": adjoint_directional,
                "finite_difference": finite_difference,
                "plus": plus,
                "minus": minus,
                "relative_error": relative_error,
                "pass": relative_error < args.relative_error_limit,
                "physical_minimum": float(np.min(physical)),
                "physical_maximum": float(np.max(physical)),
                "periodic_x_fencepost_max_abs_error": seam_x,
                "periodic_y_fencepost_max_abs_error": seam_y,
                "z_extrusion_max_abs_error": extrusion,
            }
        )
        gradients[f"latent_gradient_beta_{beta:g}"] = latent_gradient

    passed = all(case["pass"] for case in cases) and all(
        case["periodic_x_fencepost_max_abs_error"] == 0.0
        and case["periodic_y_fencepost_max_abs_error"] == 0.0
        and case["z_extrusion_max_abs_error"] == 0.0
        for case in cases
    )
    raw = output / "latent_mapping_vjp_certificate.npz"
    np.savez_compressed(
        raw,
        latent=latent,
        direction=direction,
        normalized_physical_covector=normalized_covector.reshape(
            expected_shape
        ),
        **gradients,
    )
    summary = {
        "schema_version": 1,
        "generated_at_utc": _utc(),
        "status": (
            "VALIDATED_LATENT_MAPPING_VJP"
            if passed
            else "FAILED_LATENT_MAPPING_VJP"
        ),
        "passed": passed,
        "scope": (
            "solver-free production mapping VJP only; not a latent-to-"
            "Maxwell-objective central finite difference"
        ),
        "git": {
            "branch": _git("branch", "--show-current"),
            "commit": _git("rev-parse", "HEAD"),
        },
        "mapping": {
            "version": MAPPING_VERSION,
            "latent_shape": [mapping.Nux, mapping.Nuy],
            "physical_shape": list(expected_shape),
            "config": mapping.config.to_dict(),
            "stages": [
                "periodic conic filter",
                "nominal tanh projection eta=0.5",
                "exact periodic fencepost",
                "z extrusion",
            ],
        },
        "covector": {
            "kind": "normalized completed thermal/PTE physical-density gradient",
            "unscaled_l2": covector_l2,
            "source": {
                "path": str(gradient_path),
                "bytes": gradient_path.stat().st_size,
                "sha256": _sha256(gradient_path),
            },
        },
        "relative_error_limit": args.relative_error_limit,
        "cases": cases,
        "raw_artifact": {
            "path": str(raw),
            "bytes": raw.stat().st_size,
            "sha256": _sha256(raw),
            "committed_to_git": False,
        },
        "next_gate": "LATENT_TO_MAXWELL_OBJECTIVE_V261_CENTRAL_FD",
    }
    summary_path = output / "latent_mapping_vjp_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
