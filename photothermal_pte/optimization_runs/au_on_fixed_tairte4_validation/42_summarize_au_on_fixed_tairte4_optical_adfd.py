#!/usr/bin/env python3
"""Publish the Au-nanostructure-on-fixed-TaIrTe4 optical AD-FD gate."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[2]
RESULTS = ROOT / "results"
SUMMARY = RESULTS / "au_on_fixed_tairte4_optical_adfd_summary.json"
CSV_PATH = RESULTS / "au_on_fixed_tairte4_optical_adfd_directions.csv"
PLOT = RESULTS / "au_on_fixed_tairte4_optical_adfd.png"
REPORT = RESULTS / "AU_ON_FIXED_TAIRTE4_OPTICAL_ADFD_REPORT.md"
MANIFEST = RESULTS / "AU_ON_FIXED_TAIRTE4_OPTICAL_ADFD_MANIFEST.json"
SCRIPT = ROOT / "41_validate_au_on_fixed_tairte4_optical_adfd.py"
TEST = ROOT / "tests" / "test_au_on_tairte4_contract.py"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    expected = "VALIDATED_AU_ON_FIXED_TAIRTE4_OPTICAL_ADFD_CONTROL"
    if summary["status"] != expected:
        raise SystemExit(f"Refusing to publish {summary['status']}; expected {expected}")
    with CSV_PATH.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    finest = [row for row in rows if float(row["h"]) == 0.005]
    results = summary["results"]
    materials = summary["material_contract"]
    geometry = summary["geometry"]
    numerics = summary["numerics"]

    observable_summary: dict[str, dict[str, float | list[str]]] = {}
    for observable in ("au", "tairte4", "total"):
        selected = [row for row in finest if row["observable"] == observable]
        strong = [row for row in selected if row["strong_direction"] == "True"]
        observable_summary[observable] = {
            "max_strong_relative_error": max(float(row["strong_relative_error"]) for row in strong),
            "max_gradient_l2_normalized_error": max(
                float(row["gradient_l2_normalized_error"]) for row in selected
            ),
            "near_null_directions": [row["direction"] for row in selected if row["strong_direction"] != "True"],
        }

    direction_lines = []
    for row in [row for row in finest if row["observable"] == "total"]:
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

    eps = materials["tairte4_epsilon"]
    powers = results["powers_W"]
    report = f"""# Au nanostructure on fixed TaIrTe4: optical AD-FD control

Status: `{summary['status']}`

## Outcome

This checkpoint validates a differentiable **Au nanocube/nanoantenna design
material**, not an Au electrode.  A two-dimensional density is extruded
through a fixed Au thickness above a fixed TaIrTe4 slab.  The full 3-D causal
dispersive Maxwell trajectory is differentiated on GPU.  Au absorption,
TaIrTe4 absorption, and their sum are kept separate.

The production v261 moving/conformal-Au route remains blocked.  This result
instead establishes a working fixed-grid dispersive route whose total optical
gradient agrees with central finite differences.

## Materials and axes

- wavelength: `{materials['wavelength_m']:.8e} m`
- Au endpoint: `n={materials['au_n']}`, `k={materials['au_k']}`
- Au epsilon: `{materials['au_epsilon'][0]:.8f} + {materials['au_epsilon'][1]:.8f}i`
- TaIrTe4 epsilon_a: `{eps['a'][0]:.8f} + {eps['a'][1]:.8f}i`
- TaIrTe4 epsilon_b: `{eps['b'][0]:.8f} + {eps['b'][1]:.8f}i`
- TaIrTe4 epsilon_c: `{eps['c'][0]:.8f} + {eps['c'][1]:.8f}i`
- solver axes: `x=b`, `y=a`, `z=c=b closure`
- permittivity table SHA-256: `{materials['permittivity_table_sha256']}`

The TaIrTe4 `c=b` value is the repository's explicit 3-D closure, not a
directly measured independent c-axis response.  Each axis and Au use a
passive one-pole ADE fitted to the exact finite-time-step harmonic response at
10 um.  This is an exact single-frequency causal closure, not a measured
broadband pole fit.

The gray Au law is `pole strength = rho^3`.  It preserves exact air/Au
endpoints on a fixed Yee support.  It is a numerical topology relaxation and
is not called a physical gray effective medium.

## Geometry and numerics

- domain cells: `{geometry['domain_cells_xyz']}` at `{geometry['resolution_m']:.3e} m`
- six PML boundaries: `{geometry['pml_cells_each_face']}` cells each
- Au design cells: `{geometry['au_design_cells_xyz']}`
- fixed TaIrTe4 cells: `{geometry['tairte4_cells_xyz']}`
- direct optical Au/TaIrTe4 face contact: `{geometry['direct_optical_contact']}`
- optical periods: `{numerics['total_periods']}`; time steps: `{numerics['time_steps_total']}`
- two independent phasor windows: `{numerics['phasor_periods_per_window']}` periods each
- gradient: `{numerics['gradient_method']}`

This is a small optical algorithmic control in air.  It does not yet include
SiO2/Si, thermal contact conductance, electrode collection, PTE current, or a
production-size nanoantenna optimization.

## Absorption and settling

- `P_Au = {powers['au']:.12e} W`
- `P_TaIrTe4 = {powers['tairte4']:.12e} W`
- `P_total = {powers['total']:.12e} W`
- Au previous/late change: `{results['observable_window_relative_change']['au']:.6%}`
- TaIrTe4 previous/late change: `{results['observable_window_relative_change']['tairte4']:.6%}`
- total previous/late change: `{results['observable_window_relative_change']['total']:.6%}`

The powers use the control source normalization and are not scaled to 285 uW.
No clipping, smoothing, gain, or post-hoc power rescaling is applied.

## AD-FD certificate

- total gradient L2: `{results['gradient_l2_W_per_rho']['total']:.12e} W/rho`
- `g_total-(g_Au+g_TaIrTe4)` relative norm: `{results['gradient_sum_relative_error']:.3e}`
- maximum total strong-direction error at `h=0.005`: `{results['max_total_strong_relative_error_finest_step']:.6%}`
- maximum total multi-direction gradient-normalized error: `{results['max_total_gradient_l2_normalized_error_finest_step']:.6%}`

| total-power direction | strong | AD (W) | central FD (W) | relative error | error / gradient L2 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(direction_lines)}

Near-null directions are retained in the CSV and judged with the global
gradient-L2 normalization rather than a tiny directional denominator.

## Reproduction

```bash
env PYTHONPATH=/home/seunghyun/.local/au_fdtdx \
  CUDA_VISIBLE_DEVICES=4 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  /home/eidl/miniconda3/envs/EIDL-Lumapi/bin/python \
  {SCRIPT.relative_to(REPO_ROOT)}

/home/eidl/miniconda3/envs/EIDL-Lumapi/bin/python \
  -m pytest -q {TEST.relative_to(REPO_ROOT)}

/home/eidl/miniconda3/envs/EIDL-Lumapi/bin/python \
  {Path(__file__).resolve().relative_to(REPO_ROOT)}
```

## Next fail-closed gate

Cross-check exact-binary air/Au endpoints for the same fixed TaIrTe4 optical
stack against v261 Lumerical, including material readback, component powers,
PML/mesh convergence, and source normalization.  Thermal/PTE coupling and a
production Au topology optimization remain blocked until that endpoint check
is closed.
"""
    REPORT.write_text(report, encoding="utf-8")

    files = [SCRIPT, Path(__file__).resolve(), TEST, SUMMARY, CSV_PATH, PLOT, REPORT]
    manifest = {
        "status": summary["status"],
        "generation_command": (
            "env PYTHONPATH=/home/seunghyun/.local/au_fdtdx CUDA_VISIBLE_DEVICES=4 "
            "XLA_PYTHON_CLIENT_PREALLOCATE=false "
            "/home/eidl/miniconda3/envs/EIDL-Lumapi/bin/python "
            "41_validate_au_on_fixed_tairte4_optical_adfd.py"
        ),
        "external_dependency": {
            "fdtdx_version": "0.6.2",
            "fdtdx_source_commit": "bf7e45a406c8ee6026daa95bec6fbb57e4f595ca",
            "jax_version": summary["software"]["jax_version"],
            "raw_compiler_cache_committed": False,
        },
        "observable_metrics_at_h_0p005": observable_summary,
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
