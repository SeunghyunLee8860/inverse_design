#!/usr/bin/env python3
"""Prepare one Ea or Eb latent centered pair; perform no solver calls."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.contract import (
    CONTRACT,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.lumerical_4um_adfd import (
    array_sha256,
    centered_density_pair,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.lumerical_4um_density import (
    density_state_audit,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.lumerical_4um_design_mapping import (
    NOMINAL_MAPPING,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.validation_provenance import (
    sha256,
)


def _artifact(path: Path) -> dict[str, object]:
    value = path.resolve()
    return {
        "path": str(value),
        "size_bytes": value.stat().st_size,
        "sha256": sha256(value),
    }


def independent_latent_baseline() -> np.ndarray:
    """Return a feasible smooth state selected without any field or gradient."""

    x = np.linspace(-1.0, 1.0, CONTRACT.design_node_shape[0])[:, None]
    y = np.linspace(-1.0, 1.0, CONTRACT.design_node_shape[1])[None, :]
    return np.ascontiguousarray(
        0.5 + 0.16 * np.sin(0.8 * np.pi * x) * np.cos(0.6 * np.pi * y)
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--polarization", choices=("Ea", "Eb"), default="Ea")
    parser.add_argument("--beta", type=float, default=4.0)
    parser.add_argument("--step", type=float, default=0.0025)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    latent = independent_latent_baseline()
    direction, latent_plus, latent_minus = centered_density_pair(
        latent, step=args.step
    )
    projected = NOMINAL_MAPPING.physical(latent, args.beta)
    projected_plus = NOMINAL_MAPPING.physical(latent_plus, args.beta)
    projected_minus = NOMINAL_MAPPING.physical(latent_minus, args.beta)
    arrays = {
        "latent_baseline": latent,
        "direction": direction,
        "latent_plus": latent_plus,
        "latent_minus": latent_minus,
        "baseline_density": projected,
        "plus_density": projected_plus,
        "minus_density": projected_minus,
    }
    paths = {name: output / f"{name}.npy" for name in arrays}
    for name, value in arrays.items():
        np.save(paths[name], value, allow_pickle=False)

    manifest = {
        "status": (
            f"PREPARED_LUMERICAL_4UM_{args.polarization.upper()}_LATENT_"
            "COMBINED_ADFD_PAIR"
        ),
        "passed": True,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            f"independent smooth 81x81 latent direction for {args.polarization}; "
            "no solver call"
        ),
        "polarization": args.polarization,
        "design_coordinate": "latent_81x81_before_filter_projection",
        "beta": args.beta,
        "step": args.step,
        "mapping": NOMINAL_MAPPING.audit(),
        "baseline_definition": (
            "0.5+0.16*sin(0.8*pi*x)*cos(0.6*pi*y); x,y are independent "
            "normalized nodal coordinates"
        ),
        "direction_definition": (
            "sin(pi*(0.73*x+0.41*y+0.17))*cos(pi*(0.31*x-0.67*y-0.09)); "
            "x,y are independent normalized nodal coordinates; L_inf normalized"
        ),
        "baseline_and_direction_selected_without_fields_or_gradient": True,
        "latent_baseline_sha256": array_sha256(
            latent, label="adfd-latent-baseline-v1"
        ),
        "direction_sha256": array_sha256(
            direction, label="adfd-latent-direction-v1"
        ),
        "latent_range": [float(np.min(latent)), float(np.max(latent))],
        "latent_plus_range": [
            float(np.min(latent_plus)),
            float(np.max(latent_plus)),
        ],
        "latent_minus_range": [
            float(np.min(latent_minus)),
            float(np.max(latent_minus)),
        ],
        "baseline_density": density_state_audit(projected),
        "plus_density": density_state_audit(projected_plus),
        "minus_density": density_state_audit(projected_minus),
        "artifacts": {name: _artifact(path) for name, path in paths.items()},
        "Maxwell_solves": 0,
        "custom_CUDA_solves": 0,
        "Lumerical_HEAT_or_CHARGE_solves": 0,
        "optimizer_iterations": 0,
        "wall_s": time.monotonic() - started,
    }
    result_path = output / "latent_adfd_pair_manifest.json"
    result_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
