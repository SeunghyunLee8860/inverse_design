#!/usr/bin/env python3
"""Run a small literal-FVM CUDA forward/implicit-adjoint control."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


REPOSITORY = Path(__file__).resolve().parents[3]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from photothermal_pte.optimization_runs.cuda_thermal_adjoint import (  # noqa: E402
    solve_forward_adjoint_cuda,
)
from photothermal_pte.validation.photothermal_stage1.anisotropic_heat_fvm import (  # noqa: E402
    assemble_steady_diagonal_kappa,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda-device", type=int, default=4)
    args = parser.parse_args()
    edges = np.linspace(-0.5e-6, 0.5e-6, 9)
    shape = (8, 8, 8)
    kappa = np.empty((*shape, 3), float)
    kappa[..., 0] = 14.4
    kappa[..., 1] = 3.8
    kappa[..., 2] = 1.0
    system = assemble_steady_diagonal_kappa(
        x_edges_m=edges,
        y_edges_m=edges,
        z_edges_m=edges,
        kappa_W_mK=kappa,
        dirichlet_temperature_K={"z_min": 300.0, "z_max": 300.0},
    )
    source = np.zeros(shape, float)
    source[3:5, 3:5, 3:5] = 1.0e15
    forward_rhs = np.asarray(
        system.source_volume_operator_m3 @ system.active_source(source)
        + system.boundary_load_W
    )
    adjoint_rhs = np.linspace(0.1, 1.1, forward_rhs.size)
    result = solve_forward_adjoint_cuda(
        system.matrix_W_K,
        forward_rhs,
        adjoint_rhs,
        cuda_device=args.cuda_device,
        relative_tolerance=1.0e-10,
    )
    summary = {
        "status": "VALIDATED_SMALL_LITERAL_FVM_CUDA_FORWARD_ADJOINT_PAIR",
        "scope": "8x8x8 anisotropic control only; not production thermal scaling",
        "cuda_device_index": args.cuda_device,
        "shape_xyz": list(shape),
        "active_unknowns": int(system.matrix_W_K.shape[0]),
        "kappa_W_mK": [14.4, 3.8, 1.0],
        "forward": {
            "iterations": result.forward.iterations,
            "explicit_relative_residual": result.forward.explicit_relative_residual,
            "solve_seconds": result.forward.solve_seconds,
            "reliable_restarts": result.forward.reliable_restarts,
        },
        "implicit_adjoint": {
            "iterations": result.adjoint.iterations,
            "explicit_relative_residual": result.adjoint.explicit_relative_residual,
            "solve_seconds": result.adjoint.solve_seconds,
            "reliable_restarts": result.adjoint.reliable_restarts,
        },
        "forward_adjoint_reciprocity_relative_error": (
            result.reciprocity_relative_error
        ),
        "autograd_through_iterations": False,
        "CPU_linear_solve_fallback": False,
    }
    if (
        summary["forward"]["explicit_relative_residual"] >= 1.0e-10
        or summary["implicit_adjoint"]["explicit_relative_residual"] >= 1.0e-10
        or summary["forward_adjoint_reciprocity_relative_error"] >= 1.0e-9
    ):
        raise RuntimeError(f"CUDA thermal-pair gate failed: {summary}")
    output = Path(__file__).resolve().parent / "results" / "cuda_thermal_pair_summary.json"
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
