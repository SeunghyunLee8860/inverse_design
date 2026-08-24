#!/usr/bin/env python3
"""Prepare one independent smooth centered pair; perform no solver calls."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.lumerical_4um_adfd import (
    array_sha256,
    centered_density_pair,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.lumerical_4um_density import (
    density_state_audit,
    load_projected_density_file,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.validation_provenance import (
    sha256,
)


def _artifact(path: Path) -> dict[str, object]:
    value = path.resolve()
    return {"path": str(value), "size_bytes": value.stat().st_size, "sha256": sha256(value)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-density", required=True, type=Path)
    parser.add_argument("--density-key", default="rho")
    parser.add_argument("--step", type=float, default=0.0025)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    rho = load_projected_density_file(args.baseline_density, key=args.density_key)
    direction, plus, minus = centered_density_pair(rho, step=args.step)
    paths = {
        "baseline_density": output / "rho_baseline.npy",
        "direction": output / "direction.npy",
        "plus_density": output / "rho_plus.npy",
        "minus_density": output / "rho_minus.npy",
    }
    np.save(paths["baseline_density"], rho, allow_pickle=False)
    np.save(paths["direction"], direction, allow_pickle=False)
    np.save(paths["plus_density"], plus, allow_pickle=False)
    np.save(paths["minus_density"], minus, allow_pickle=False)
    result = {
        "status": "PREPARED_LUMERICAL_4UM_EA_COMBINED_ADFD_PAIR",
        "passed": True,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "independent smooth projected-density direction; no solver call",
        "step": args.step,
        "direction_definition": (
            "sin(pi*(0.73*x+0.41*y+0.17))*cos(pi*(0.31*x-0.67*y-0.09)); "
            "x,y are independent normalized nodal coordinates; L_inf normalized"
        ),
        "direction_selected_without_adjoint_gradient": True,
        "direction_sha256": array_sha256(direction, label="adfd-direction-v1"),
        "direction_range": [float(np.min(direction)), float(np.max(direction))],
        "direction_L2": float(np.linalg.norm(direction)),
        "baseline_density": density_state_audit(rho),
        "plus_density": density_state_audit(plus),
        "minus_density": density_state_audit(minus),
        "artifacts": {
            "input_baseline_density": _artifact(args.baseline_density.expanduser().resolve()),
            **{key: _artifact(path) for key, path in paths.items()},
        },
        "Maxwell_solves": 0,
        "custom_CUDA_solves": 0,
        "Lumerical_HEAT_or_CHARGE_solves": 0,
        "optimizer_iterations": 0,
        "wall_s": time.monotonic() - started,
    }
    result_path = output / "adfd_pair_manifest.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
