#!/usr/bin/env python3
"""Generate auditable v261 physical- or latent-density FD endpoint FSPs."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[1]
VOLUME_CURRENT = REPOSITORY / "volume_current_inverse_design"
for path in (HERE, VOLUME_CURRENT, VOLUME_CURRENT / "bundle"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import eqc_lib as lib  # noqa: E402
from maxwell_absorption_evaluator import (  # noqa: E402
    AbsorptionVolumeCurrentEvaluator,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-npz", required=True)
    parser.add_argument("--mode", choices=("physical", "latent"), required=True)
    parser.add_argument("--step", type=float, required=True)
    parser.add_argument("--beta", type=float, default=8.0)
    parser.add_argument("--filter-radius-um", type=float, default=0.5)
    parser.add_argument("--solver-workdir", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--summary-json", required=True)
    args = parser.parse_args()
    data = np.load(Path(args.input_npz).expanduser().resolve())
    direction = np.asarray(data["direction"], float)
    if args.mode == "physical":
        baseline = np.asarray(data["density"], float)
        plus = baseline + args.step * direction
        minus = baseline - args.step * direction
    else:
        os.environ["MFS_UM"] = str(args.filter_radius_um)
        os.environ["MGS_UM"] = str(args.filter_radius_um)
        model = lib.load_model()
        mapping = model.mapping
        latent = np.asarray(data["latent"], float)
        physical_shape = (model.Nx, model.Ny, model.Nz)
        plus = np.asarray(
            mapping(latent + args.step * direction, args.beta), float
        ).reshape(physical_shape)
        minus = np.asarray(
            mapping(latent - args.step * direction, args.beta), float
        ).reshape(physical_shape)
    if np.min(plus) < 0.0 or np.max(plus) > 1.0:
        raise RuntimeError("plus endpoint is outside [0,1]")
    if np.min(minus) < 0.0 or np.max(minus) > 1.0:
        raise RuntimeError("minus endpoint is outside [0,1]")

    solver = Path(args.solver_workdir).expanduser().resolve()
    solver.mkdir(parents=True, exist_ok=True)
    evaluator = AbsorptionVolumeCurrentEvaluator(
        workdir=solver,
        incident_polarization="x",
    )
    contract = evaluator.prepare(force_rebuild=False)
    plus_label = f"{args.label}_plus"
    minus_label = f"{args.label}_minus"
    plus_check = evaluator.forward_absorption(
        plus,
        label=plus_label,
        density_mode="probe_safe",
    )
    minus_check = evaluator.forward_absorption(
        minus,
        label=minus_label,
        density_mode="probe_safe",
    )
    summary = {
        "mode": args.mode,
        "step": args.step,
        "beta": args.beta if args.mode == "latent" else None,
        "filter_radius_um": (
            args.filter_radius_um if args.mode == "latent" else None
        ),
        "input_npz": str(Path(args.input_npz).expanduser().resolve()),
        "plus_fsp": str(solver / f"{plus_label}.fsp"),
        "minus_fsp": str(solver / f"{minus_label}.fsp"),
        "plus_absorption_check": plus_check,
        "minus_absorption_check": minus_check,
        "solver_contract": contract,
    }
    summary_path = Path(args.summary_json).expanduser().resolve()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
