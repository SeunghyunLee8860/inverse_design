#!/usr/bin/env python3
"""Forward-only x/y evaluation of a saved candidate design."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("design", help="NPZ containing a 'physical' density array")
    parser.add_argument("--output", default="runs/forward_evaluation")
    parser.add_argument("--gpu", default=os.environ.get("CL_GPU_DEVICE", "GPU 1"))
    parser.add_argument("--rho-step", type=float, default=0.0025)
    args = parser.parse_args()
    os.environ["CL_GPU_DEVICE"] = args.gpu
    os.environ["LUMERICAL_SESSION_GPU_DEVICE"] = args.gpu

    from volume_current_evaluator import VolumeCurrentEvaluator

    source = Path(args.design).resolve()
    data = np.load(source)
    if "physical" not in data.files:
        raise KeyError(f"{source} has no 'physical' array")
    physical = np.asarray(data["physical"], float)
    output = (HERE / args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    values = {}
    for polarization in ("x", "y"):
        evaluator = VolumeCurrentEvaluator(
            output / f"solver_{polarization}", args.rho_step, polarization
        )
        evaluator.prepare(force_rebuild=False)
        values[f"F{polarization}"] = evaluator.forward_fom(
            physical, label="candidate_forward"
        )
    values["Fx_plus_Fy"] = values["Fx"] + values["Fy"]
    values["balance"] = min(values["Fx"], values["Fy"]) / max(values["Fx"], values["Fy"])
    report = {
        "design": str(source), "gpu": args.gpu,
        "source_type": "forward plane wave; no adjoint and no dipoles",
        **values,
    }
    (output / "forward_evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

