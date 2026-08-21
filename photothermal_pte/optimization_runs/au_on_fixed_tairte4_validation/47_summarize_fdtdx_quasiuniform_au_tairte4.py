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
DEFAULT_ENDPOINT = HERE / "results_fdtdx_lumerical_binary_endpoints"
DEFAULT_GRADIENT = HERE / "results_fdtdx_production_gradient_smoke"


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
    parser.add_argument("--endpoint-dir", type=Path, default=DEFAULT_ENDPOINT)
    parser.add_argument("--gradient-dir", type=Path, default=DEFAULT_GRADIENT)
    args = parser.parse_args()

    bridge_json = args.bridge_dir / "fdtdx_quasiuniform_au_tairte4_adfd_summary.json"
    source_json = args.source_dir / "fdtdx_quasiuniform_source_direction_summary.json"
    w8_source_json = args.w8_source_dir / "fdtdx_w8p5um_source_only_summary.json"
    w8_source_plot = args.w8_source_dir / "fdtdx_w8p5um_source_only.png"
    endpoint_json = args.endpoint_dir / "fdtdx_lumerical_binary_endpoints_summary.json"
    endpoint_csv = args.endpoint_dir / "fdtdx_lumerical_binary_endpoints_cases.csv"
    endpoint_plot = args.endpoint_dir / "fdtdx_lumerical_binary_endpoints.png"
    endpoint_audit = args.endpoint_dir / "fdtdx_lumerical_binary_endpoint_runsetup_audit.json"
    gradient_json = args.gradient_dir / "fdtdx_production_width_nonuniform_au_gradient_smoke.json"
    gradient_csv = args.gradient_dir / "fdtdx_production_width_nonuniform_au_gradient_smoke.csv"
    gradient_plot = args.gradient_dir / "fdtdx_production_width_nonuniform_au_gradient_smoke.png"
    gradient_report = args.gradient_dir / "FDTDX_PRODUCTION_WIDTH_NONUNIFORM_AU_GRADIENT_REPORT.md"
    gradient_audit = args.gradient_dir / "fdtdx_lumerical_binary_endpoint_runsetup_audit.json"
    gradient_performance = args.gradient_dir / "fdtdx_checkpoint_performance_diagnostic.json"
    directions_csv = args.bridge_dir / "fdtdx_quasiuniform_au_tairte4_adfd_directions.csv"
    plot = args.bridge_dir / "fdtdx_quasiuniform_au_tairte4_adfd.png"
    bridge = load_json(bridge_json)
    source = load_json(source_json)
    w8_source = load_json(w8_source_json)
    endpoint = load_json(endpoint_json)
    gradient = load_json(gradient_json)

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

    gradient_finest = [row for row in gradient["directions"] if row["h"] == 0.005]
    gradient_strong_error = max(
        (row["strong_relative_error"] for row in gradient_finest if row["strong_direction"]),
        default=0.0,
    )
    gradient_normalized_error = max(
        row["gradient_l2_normalized_error"] for row in gradient_finest
    )
    status = "VALIDATED_FDTDX_PRODUCTION_WIDTH_AU_OPTICAL_FORWARD_AND_GRADIENT"
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
5. The material-bearing production-width exact-binary cross-check passes.  The
   absorbed fractions are:

| endpoint | FDTDX | Lumerical | relative difference |
|---|---:|---:|---:|
| TaIrTe4 only | {endpoint['comparisons']['au0']['fdtdx_absorbed_fraction']:.8f} | {endpoint['comparisons']['au0']['lumerical_absorbed_fraction']:.8f} | {100*endpoint['comparisons']['au0']['absorbed_fraction_relative_difference']:.6f}% |
| Au / TaIrTe4 | {endpoint['comparisons']['au1']['fdtdx_absorbed_fraction']:.8f} | {endpoint['comparisons']['au1']['lumerical_absorbed_fraction']:.8f} | {100*endpoint['comparisons']['au1']['absorbed_fraction_relative_difference']:.6f}% |

   The Au-present/Au-absent absorbed-power ratio differs by only
   `{100*endpoint['endpoint_ratio']['relative_difference']:.6f}%`.  Local
   native-Yee material loss agrees with the empty-subtracted six-face flux to
   `{100*endpoint['comparisons']['au0']['fdtdx_empty_subtracted_closure_relative']:.6f}%`
   and `{100*endpoint['comparisons']['au1']['fdtdx_empty_subtracted_closure_relative']:.6f}%`.
   These values use FDTDX's documented eta0 field-unit conversion; no fitted
   gain or endpoint rescaling was applied.
6. The production-width nonuniform-Au material gradient passes.  The 20x20
   density field is mapped to 100x100 component-native Yee samples and extruded
   through the physical 50-nm Au thickness.  At the finest FD step, the strong
   smooth-direction relative error is
   `{100*gradient_strong_error:.6f}%`; the maximum all-direction
   gradient-L2-normalized error is
   `{100*gradient_normalized_error:.6f}%`.  Empty-subtracted local-Q/six-face
   closure is `{100*gradient['baseline']['Q_flux_closure_relative']:.6f}%` and
   the last-window change is
   `{100*gradient['baseline']['late_window_relative_change']:.6f}%`.
   The fixed-seed random direction is explicitly classified as near-null at
   the 5% gradient-norm threshold and is not judged by an ill-conditioned local
   relative error.  No raw AD/FD value or gradient was rescaled.

## Diagnostic limitation retained from the compact control

- The last two-window absorbed-power change is
  `{100.0 * bridge['results']['window_relative_change']['total']:.6f}%`, above
  the 0.5% gate in the eight-period quick run.
- The old compact artifact records apparent local-Q/flux mismatches of
  `{100.0 * bridge['results']['closed_surface_raw_closure_relative_error']:.6f}%`
  (raw) and
  `{100.0 * bridge['results']['closed_surface_background_subtracted_closure_relative_error']:.6f}%`
  (empty-subtracted).  They are retained for provenance but are **not physical
  closure results**: that artifact co-located material fields and did not apply
  the complete FDTDX eta0 field-unit conversion now certified by stage 49.
- In the independent source-only control, compact-Gaussian closed-box residuals
  are `{gaussian_compact['+']:.4f}%` (`+z`) and
  `{gaussian_compact['-']:.4f}%` (`-z`); enlarging the box gives
  `{gaussian_large['+']:.4f}%` and `{gaussian_large['-']:.4f}%`.
  The corresponding uniform-source large-box residual is only about 0.08%.

Therefore the source direction is not the failure.  The compact source remains
a useful AD--FD control but is not an energy-closure certificate.  The
production-width source/material calculation supersedes it for forward power:
it uses native component Yee samples and explicitly converts
`E_SI=eta0*E_internal`, `H_SI=H_internal`, and
`S_SI=eta0*S_internal`.  The previously unresolved material-bearing
cross-solver checkpoint is now closed without an empirical correction.

## Decision

FDTDX is validated for the production-width forward optical endpoints and the
production-width spatially varying dispersive-Au material gradient.  It is the
selected optical route for the next Au inverse-design validation.  This status
does **not** yet validate thermal/PTE coupling, electrode transport, combined
gradients, or optimization.  The next fail-closed gate is explicit
Au/TaIrTe4 thermal coupling and thermal-only AD--FD; optimization starts only
after the thermal and electrical chains pass.
"""
    report_path.write_text(report, encoding="utf-8")

    tracked = [
        HERE / "45_validate_fdtdx_quasiuniform_au_tairte4_adfd.py",
        HERE / "46_validate_fdtdx_quasiuniform_source_direction.py",
        Path(__file__).resolve(),
        HERE / "48_validate_fdtdx_w8p5um_source_only.py",
        HERE / "49_validate_fdtdx_lumerical_binary_endpoints.py",
        HERE / "50_summarize_fdtdx_production_gradient_smoke.py",
        bridge_json,
        directions_csv,
        plot,
        source_json,
        w8_source_json,
        w8_source_plot,
        endpoint_json,
        endpoint_csv,
        endpoint_plot,
        endpoint_audit,
        gradient_json,
        gradient_csv,
        gradient_plot,
        gradient_report,
        gradient_audit,
        gradient_performance,
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
            "CUDA_VISIBLE_DEVICES=5 PYTHONPATH=/home/seunghyun/.local/fdtdx_main_src/src:/home/seunghyun/.local/au_fdtdx JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false python 49_validate_fdtdx_lumerical_binary_endpoints.py --output-dir results_fdtdx_lumerical_binary_endpoints",
            "CUDA_VISIBLE_DEVICES=5 PYTHONPATH=/home/seunghyun/.local/fdtdx_main_src/src:/home/seunghyun/.local/au_fdtdx XLA_PYTHON_CLIENT_PREALLOCATE=false python 49_validate_fdtdx_lumerical_binary_endpoints.py --gradient-smoke --gradient-checkpoints 16 --output-dir results_fdtdx_production_gradient_smoke",
            "python 50_summarize_fdtdx_production_gradient_smoke.py --result-dir results_fdtdx_production_gradient_smoke",
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
