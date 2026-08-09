#!/usr/bin/env python3
"""Create the immutable, exactly uniform Run009 latent input."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
RUN002 = HERE.parent / "run_002_gaussian10_w8p5_current_max"
if str(RUN002) not in sys.path:
    sys.path.insert(0, str(RUN002))

from production_density_mapping import ProductionDensityMapping  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    mapping = ProductionDensityMapping()
    latent = np.full(mapping.shape, 0.5, dtype=np.float64)
    filtered = mapping.filtered(latent)
    rho = mapping.physical(latent, beta=2.0)
    for name, values in (("latent", latent), ("filtered", filtered), ("rho", rho)):
        if not np.array_equal(values, np.full(mapping.shape, 0.5, dtype=np.float64)):
            raise RuntimeError(f"{name} is not exactly uniform 0.5")

    raw = output / "uniform_latent.npz"
    np.savez_compressed(raw, latent=latent)
    audit = {
        "status": "VALIDATED_EXACT_UNIFORM_INITIAL_DENSITY",
        "passed": True,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "shape": list(mapping.shape),
        "design_variable_count": int(latent.size),
        "value": 0.5,
        "latent_min": float(np.min(latent)),
        "latent_max": float(np.max(latent)),
        "latent_mean": float(np.mean(latent)),
        "latent_std": float(np.std(latent)),
        "filtered_min": float(np.min(filtered)),
        "filtered_max": float(np.max(filtered)),
        "rho_beta2_min": float(np.min(rho)),
        "rho_beta2_max": float(np.max(rho)),
        "random_perturbation": False,
        "seed_shape": "none",
        "fixed_internal_mask": False,
        "raw_artifact": {
            "path": str(raw),
            "size_bytes": raw.stat().st_size,
            "sha256": sha256(raw),
        },
    }
    (output / "uniform_initial_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
