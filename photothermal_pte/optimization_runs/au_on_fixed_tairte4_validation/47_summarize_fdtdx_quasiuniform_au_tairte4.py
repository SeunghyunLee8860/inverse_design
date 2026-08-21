#!/usr/bin/env python3
"""Publish the fail-closed FDTDX quasi-uniform Au/TaIrTe4 checkpoint."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
DEFAULT_BRIDGE = HERE / "results_fdtdx_quasiuniform_bridge"
DEFAULT_SOURCE = HERE / "results_fdtdx_quasiuniform_source"
DEFAULT_W8_SOURCE = HERE / "results_fdtdx_w8p5um_source_only"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bridge-dir", type=Path, default=DEFAULT_BRIDGE)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--w8-source-dir", type=Path, default=DEFAULT_W8_SOURCE)
    args = parser.parse_args()

    bridge_json = args.bridge_dir / "fdtdx_quasiuniform_au_tairte4_adfd_summary.json"
    source_json = args.source_dir / "fdtdx_quasiuniform_source_direction_summary.json"
    w8_source_json = args.w8_source_dir / "fdtdx_w8p5um_source_only_summary.json"
    w8_source_plot = args.w8_source_dir / "fdtdx_w8p5um_source_only.png"
    directions_csv = args.bridge_dir / "fdtdx_quasiuniform_au_tairte4_adfd_directions.csv"
    plot = args.bridge_dir / "fdtdx_quasiuniform_au_tairte4_adfd.png"
    bridge = load_json(bridge_json)
    source = load_json(source_json)
    w8_source = load_json(w8_source_json)

    with directions_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    finest = min(float(row["h"]) for row in rows)
    total_finest = [
        row for row in rows if row["observable"] == "total" and float(row["h"]) == finest
    ]
    errors = {
        row["direction"]: 100.0 * float(row["strong_relative_error"])
        for row in total_finest
    }

    source_cases = {(case["kind"], case["direction"]): case for case in source["cases"]}
    gaussian_compact = {
        direction: 100.0
        * abs(source_cases[("gaussian", direction)]["closed_surface_phasor_inward_W"])
        / abs(source_cases[("gaussian", direction)]["mean_downstream_power_W"])
        for direction in ("+", "-")
    }
    gaussian_large = {
        direction: 100.0
        * source_cases[("gaussian", direction)]["large_closed_surface_residual_over_downstream_power"]
        for direction in ("+", "-")
    }

    status = "PARTIAL_FDTDX_AU_GRADIENT_AND_W8P5_SOURCE_VALIDATED_PENDING_MATERIAL_CROSSCHECK"
    report_path = args.bridge_dir / "FDTDX_QUASIUNIFORM_AU_TAIRTE4_VALIDATION_REPORT.md"
    report = f"""# FDTDX quasi-uniform Au/TaIrTe4 validation

**Status: `{status}`**

This checkpoint tests whether the post-release FDTDX main branch can represent
physical 50-nm Au on physical 100-nm TaIrTe4 with a quasi-uniform Cartesian
grid and differentiate the dispersive material loss.  It is a compact optical
control.  It is **not** the production 8.5-um-waist beam, thermal/PTE model,
electrode model, or an optimization result.

## Reproducibility contract

- FDTDX source commit: `{bridge['software']['fdtdx_source_commit']}`
- imported module: `{bridge['software']['fdtdx_import_path']}`
- GPU requested through `CUDA_VISIBLE_DEVICES={bridge['software']['cuda_visible_devices']}`
- solver axes: `x=b`, `y=a`, `z=c=b` closure
- wavelength: `{1e6 * bridge['source']['wavelength_m']:.3f} um`
- grid: `{bridge['grid']['domain_cells_xyz']}` cells at
  `{[1e9*x for x in bridge['grid']['cell_size_m_xyz']]} nm`
- realized Au thickness: `{1e9 * bridge['grid']['realized_au_thickness_m']:.6f} nm`
- realized TaIrTe4 thickness: `{1e9 * bridge['grid']['realized_tairte4_thickness_m']:.6f} nm`
- no clipping, smoothing, gain, or result rescaling

The installed numbered release did not expose this nonuniform-grid route; this
checkpoint pins the exact post-release source commit above rather than silently
depending on a moving `main` branch.

## What passed

1. GPU execution and physical-thickness placement passed.
2. Source direction reciprocity passed: the `-z/+z` downstream-power ratios
   are `{source['minus_over_plus_power_magnitude_ratio']['uniform']:.8f}`
   (uniform source) and `{source['minus_over_plus_power_magnitude_ratio']['gaussian']:.8f}`
   (compact Gaussian).
3. Five independent total-loss directional AD--FD controls passed the 1% gate
   at `h={finest:g}`:

| direction | strong relative error |
|---|---:|
""" + "\n".join(
        f"| `{name}` | {value:.6f}% |" for name, value in errors.items()
    ) + f"""

The maximum multi-direction gradient-L2-normalized error is
`{100.0 * bridge['results']['max_total_gradient_l2_normalized_error_finest_step']:.6f}%`.
This validates reverse-mode differentiation of the fixed-support, dispersive Au
material relaxation for this compact grid.  It does not validate moving Au
boundaries in Lumerical.
4. A separate production-width empty-air source audit passed every gate.  For
   requested `w0=8.5 um`, primary `Ex` gives a realized mean waist of
   `{1e6 * w8_source['results']['beam_fit_primary_Ex']['mean_w0_m']:.6f} um`,
   `{100 * w8_source['results']['beam_fit_primary_Ex']['ellipticity']:.4f}%`
   ellipticity, and a closed-surface residual of
   `{100 * w8_source['results']['closed_surface_residual_over_incident_power']:.6f}%`
   of target-plane incident power.  Its GPU execution time after compilation
   was `{w8_source['results']['execution_seconds']:.4f} s` on this source-only grid.

## What failed and remains blocked

- The last two-window absorbed-power change is
  `{100.0 * bridge['results']['window_relative_change']['total']:.6f}%`, above
  the 0.5% gate in the eight-period quick run.
- The material-loss versus raw closed-surface flux mismatch is
  `{100.0 * bridge['results']['closed_surface_raw_closure_relative_error']:.6f}%`.
- The signed material-minus-empty closed-surface mismatch is
  `{100.0 * bridge['results']['closed_surface_background_subtracted_closure_relative_error']:.6f}%`.
- In the independent source-only control, compact-Gaussian closed-box residuals
  are `{gaussian_compact['+']:.4f}%` (`+z`) and
  `{gaussian_compact['-']:.4f}%` (`-z`); enlarging the box gives
  `{gaussian_large['+']:.4f}%` and `{gaussian_large['-']:.4f}%`.
  The corresponding uniform-source large-box residual is only about 0.08%.

Therefore the source direction is not the failure, and the closure problem is
specific to the deliberately subwavelength compact Gaussian (`w0≈0.594 um` at
`lambda=10 um`), not the production-width empty-air source.  The unresolved
item is the full material-bearing production-width cross-solver checkpoint.
The same-container zero-coupling ADE probe is **not** an
empirical correction: because every nominal material support has
`epsilon_inf=1` and all ADE field couplings are zeroed, it is an exact
empty-air optical control on the identical source/grid/PML layout.  Its flux
was already in watts; the earlier extra `1e-24` postprocessing factor has been
removed.  Both raw and background-subtracted closure still fail.

## Decision

FDTDX is usable now for algorithmic dispersive-material AD controls and for the
production-width source-only forward contract.  It is not yet promoted as the
production Au inverse-design solver.  Before thermal/PTE coupling or
optimization, the next optical checkpoint is a material-bearing
production-width FDTDX calculation against the already validated exact-binary
Lumerical endpoints, using local fine Au/TaIrTe4 mesh and coarse distant air.
"""
    report_path.write_text(report, encoding="utf-8")

    tracked = [
        HERE / "45_validate_fdtdx_quasiuniform_au_tairte4_adfd.py",
        HERE / "46_validate_fdtdx_quasiuniform_source_direction.py",
        Path(__file__).resolve(),
        HERE / "48_validate_fdtdx_w8p5um_source_only.py",
        bridge_json,
        directions_csv,
        plot,
        source_json,
        w8_source_json,
        w8_source_plot,
        report_path,
    ]
    manifest = {
        "status": status,
        "raw_fsp_or_npz_committed": False,
        "fdtdx_source": {
            "path": bridge["software"]["fdtdx_source_path"],
            "commit": bridge["software"]["fdtdx_source_commit"],
            "import_path": bridge["software"]["fdtdx_import_path"],
            "note": "exact post-release main snapshot; not a numbered release",
        },
        "generation_commands": [
            "CUDA_VISIBLE_DEVICES=5 PYTHONPATH=/home/seunghyun/.local/fdtdx_main_src/src:/home/seunghyun/.local/au_fdtdx JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false python 45_validate_fdtdx_quasiuniform_au_tairte4_adfd.py --quick --output-dir results_fdtdx_quasiuniform_bridge",
            "CUDA_VISIBLE_DEVICES=5 PYTHONPATH=/home/seunghyun/.local/fdtdx_main_src/src:/home/seunghyun/.local/au_fdtdx JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false python 46_validate_fdtdx_quasiuniform_source_direction.py --output-dir results_fdtdx_quasiuniform_source",
            "CUDA_VISIBLE_DEVICES=5 PYTHONPATH=/home/seunghyun/.local/fdtdx_main_src/src:/home/seunghyun/.local/au_fdtdx JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false python 48_validate_fdtdx_w8p5um_source_only.py --output-dir results_fdtdx_w8p5um_source_only",
            "python 47_summarize_fdtdx_quasiuniform_au_tairte4.py",
        ],
        "files": {
            str(path.relative_to(HERE)): {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in tracked
        },
    }
    manifest_path = args.bridge_dir / "RAW_ARTIFACT_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "report": str(report_path), "manifest": str(manifest_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
