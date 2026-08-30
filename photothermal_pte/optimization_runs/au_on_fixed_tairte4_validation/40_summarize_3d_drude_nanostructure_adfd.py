#!/usr/bin/env python3
"""Publish the validated 3-D causal-Drude Au AD-FD checkpoint."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[2]
RESULTS = ROOT / "results"
SUMMARY = RESULTS / "au_3d_drude_nanostructure_adfd_summary.json"
CSV_PATH = RESULTS / "au_3d_drude_nanostructure_adfd_directions.csv"
PLOT = RESULTS / "au_3d_drude_nanostructure_adfd.png"
REPORT = RESULTS / "AU_3D_CAUSAL_DRUDE_NANOSTRUCTURE_ADFD_REPORT.md"
MANIFEST = RESULTS / "AU_3D_CAUSAL_DRUDE_NANOSTRUCTURE_ADFD_MANIFEST.json"
SCRIPT = ROOT / "39_validate_3d_drude_nanostructure_adfd.py"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    if summary["status"] != "VALIDATED_3D_CAUSAL_DRUDE_AU_ADFD_CONTROL":
        raise SystemExit(f"Refusing to publish failed status: {summary['status']}")
    with CSV_PATH.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    finest = [row for row in rows if float(row["h"]) == 0.005]
    result = summary["results"]
    physical = summary["physical_contract"]
    numerics = summary["numerics"]
    gates = summary["gates"]

    direction_lines = []
    for row in finest:
        direction_lines.append(
            "| {direction} | {strong} | {ad:.6e} | {fd:.6e} | {rel:.6%} | {gnorm:.6%} |".format(
                direction=row["direction"],
                strong=row["strong_direction"],
                ad=float(row["ad_W_per_unit_direction"]),
                fd=float(row["fd_W_per_unit_direction"]),
                rel=float(row["strong_relative_error"]),
                gnorm=float(row["gradient_l2_normalized_error"]),
            )
        )

    report = f"""# 3-D causal-Drude Au nanostructure AD-FD control

Status: `{summary['status']}`

## What this validates

This checkpoint resolves the optical-gradient blocker for a **fixed-grid Au
nanoantenna density**, not for a v261 moving/conformal metal boundary.  The
design variable is a 2-D physical density extruded through a fixed Au
thickness.  It scales a passive Drude pole as `s(rho)=rho^3`; air and Au are
the exact endpoints.  The complete 3-D time-domain Maxwell trajectory and
Au absorption are differentiated with checkpointed reverse-mode AD on GPU.

This is an algorithmic control in air.  It is not yet the coupled TaIrTe4,
thermal, electrical, PTE, or production optimization result.

## Material and discretization

- wavelength: `{physical['wavelength_m']:.8e} m`
- frozen Au endpoint: `n={physical['au_n']}`, `k={physical['au_k']}`
- target epsilon: `{physical['epsilon_target'][0]:.8f} + {physical['epsilon_target'][1]:.8f}i`
- Drude omega_p: `{physical['omega_p_rad_s']:.12e} rad/s`
- Drude gamma: `{physical['gamma_rad_s']:.12e} rad/s`
- endpoint fit relative error: `{physical['endpoint_fit_relative_error']:.3e}`
- grid: `{numerics['domain_cells_xyz']}`, resolution `{numerics['resolution_m']:.3e} m`
- six PML boundaries: `{numerics['pml_cells_each_face']}` cells per face
- realized design cells: `{numerics['design_cells_xyz']}`
- Au time-resolution check: `omega_p*dt={numerics['omega_p_dt']:.6f}`
- total simulation: `{numerics['total_periods']}` optical periods, `{numerics['time_steps_total']}` steps

The initial Courant `0.95` debug run was rejected because the explicit Au ADE
became non-finite (`omega_p*dt` was about 2.5).  The promoted run uses Courant
`{numerics['courant_factor']}` and remains finite.  This is a physical
time-resolution correction, not gradient fitting or rescaling.

## Gates

- Au absorbed power: `{result['au_absorbed_power_W']:.12e} W` under the control-source normalization
- previous-to-late phasor-window change: `{result['observable_window_relative_change']:.6%}`
- gradient L2 norm: `{result['gradient_l2_W_per_rho']:.12e} W/rho`
- maximum strong-direction error at `h=0.005`: `{result['max_strong_relative_error_finest_step']:.6%}`
- maximum multi-direction gradient-normalized error: `{result['max_gradient_l2_normalized_error_finest_step']:.6%}`
- near-null direction retained as diagnostic: `{', '.join(result['near_null_directions_finest_step'])}`
- finite arrays: `{gates['finite']}`
- GPU-only: `{gates['gpu_only']}`

| direction | strong | AD (W) | central FD (W) | relative error | error / gradient L2 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(direction_lines)}

The central-localized direction is near-null, so its relative error with the
tiny FD denominator is not used as a strong-direction gate.  It is retained
and passes the global gradient-normalized gate.  No direction is deleted.

## Reproduction

```bash
env PYTHONPATH=/home/seunghyun/.local/au_fdtdx \\
  CUDA_VISIBLE_DEVICES=4 XLA_PYTHON_CLIENT_PREALLOCATE=false \\
  /home/eidl/miniconda3/envs/EIDL-Lumapi/bin/python \\
  {SCRIPT.relative_to(REPO_ROOT)} \\
  --output-dir {RESULTS.relative_to(REPO_ROOT)}

/home/eidl/miniconda3/envs/EIDL-Lumapi/bin/python \\
  {Path(__file__).resolve().relative_to(REPO_ROOT)}
```

The external FDTDX/JAX environment is not vendored.  Its pinned source commit
and targeted upstream tests are recorded in the manifest.  Raw compiler and
runtime caches are not committed.

## Next fail-closed gate

Add a fixed anisotropic TaIrTe4 layer and independently account for `Q_Au`
and `Q_TaIrTe4`; then cross-check exact-binary endpoints against Lumerical.
Thermal/PTE coupling and Au topology optimization remain blocked until that
combined optical checkpoint passes.
"""
    REPORT.write_text(report, encoding="utf-8")

    files = [SCRIPT, Path(__file__).resolve(), SUMMARY, CSV_PATH, PLOT, REPORT]
    manifest = {
        "status": summary["status"],
        "generation_command": (
            "env PYTHONPATH=/home/seunghyun/.local/au_fdtdx CUDA_VISIBLE_DEVICES=4 "
            "XLA_PYTHON_CLIENT_PREALLOCATE=false "
            "/home/eidl/miniconda3/envs/EIDL-Lumapi/bin/python "
            "39_validate_3d_drude_nanostructure_adfd.py"
        ),
        "external_dependency": {
            "fdtdx_version": "0.6.2",
            "fdtdx_source_commit": "bf7e45a406c8ee6026daa95bec6fbb57e4f595ca",
            "jax_version": summary["software"]["jax_version"],
            "targeted_upstream_tests": "20 passed",
            "raw_cache_committed": False,
        },
        "artifacts": [
            {
                "path": str(path.relative_to(REPO_ROOT)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in files
        ],
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(REPORT)
    print(MANIFEST)


if __name__ == "__main__":
    main()
