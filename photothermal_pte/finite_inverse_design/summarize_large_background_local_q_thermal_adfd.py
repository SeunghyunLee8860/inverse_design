#!/usr/bin/env python3
"""Publish audited local-Q explicit-thermal AD--FD scenario results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .explicit_thermal import build_explicit_geometry
from .run_large_background_local_q_thermal_adfd import (
    STATUS_PASS,
    interface_diagnostics,
)


REPORT_NAME = "LARGE_BACKGROUND_LOCAL_Q_EXPLICIT_THERMAL_ADFD_REPORT.md"
SUMMARY_NAME = "large_background_local_q_explicit_thermal_adfd_summary.json"
CASES_NAME = "large_background_local_q_explicit_thermal_adfd_cases.csv"
MANIFEST_NAME = (
    "LARGE_BACKGROUND_LOCAL_Q_EXPLICIT_THERMAL_ADFD_RAW_ARTIFACT_MANIFEST.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--mapping-summary", action="append", required=True)
    parser.add_argument("--cell-size-nm", type=float, default=100.0)
    parser.add_argument("--lateral-domain-um", type=float, default=32.0)
    parser.add_argument("--si-depth-um", type=float, default=20.0)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(role: str, path: Path) -> dict[str, object]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {
        "role": role,
        "path": str(resolved),
        "byte_size": resolved.stat().st_size,
        "sha256": sha256(resolved),
    }


def relative_change(reference: float, value: float) -> float:
    return (value - reference) / reference


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    report_dir = Path(args.report_dir).expanduser().resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    raw_summary_path = (
        run_dir / "local_q_explicit_thermal_adfd_summary.json"
    )
    raw_cases_path = run_dir / "local_q_explicit_thermal_adfd_cases.csv"
    raw_npz_path = run_dir / "local_q_explicit_thermal_adfd_raw.npz"
    raw_summary = json.loads(raw_summary_path.read_text())
    if (
        raw_summary.get("status") != STATUS_PASS
        or not raw_summary.get("passed")
    ):
        raise RuntimeError("thermal AD-FD run is not validated")
    if raw_summary["raw_artifact"]["sha256"] != sha256(raw_npz_path):
        raise RuntimeError("thermal raw NPZ SHA-256 mismatch")

    mapping_summaries = []
    mapping_by_span = {}
    for raw_path in args.mapping_summary:
        path = Path(raw_path).expanduser().resolve()
        value = json.loads(path.read_text())
        if not value.get("passed"):
            raise RuntimeError(f"mapping did not pass: {path}")
        support = value.get("material_support_projection", {})
        if (
            support.get("outside_TaIrTe4_power_W") != 0.0
            or support.get("outside_TaIrTe4_nonzero_cell_count") != 0
        ):
            raise RuntimeError(f"mapping is outside TaIrTe4: {path}")
        span_um = float(value["thermal_target"]["flake_span_m"]) * 1.0e6
        mapping_summaries.append((path, value))
        mapping_by_span[span_um] = value

    with np.load(raw_npz_path, allow_pickle=False) as stored:
        scenarios = raw_summary["scenarios"]
        density_controls = []
        for scenario in scenarios:
            span_um = float(scenario["flake_span_um"])
            key = f"flake_{span_um:g}um".replace(".", "p")
            rho = np.asarray(stored[f"{key}_rho"], float)
            density_controls.append(rho)
            theta = np.asarray(stored[f"{key}_theta_K"], float)
            source = np.asarray(stored[f"{key}_source_W_m3"], float)
            geometry = build_explicit_geometry(
                rho,
                lateral_domain_m=args.lateral_domain_um * 1.0e-6,
                si_depth_m=args.si_depth_um * 1.0e-6,
                flake_span_m=span_um * 1.0e-6,
                cell_size_m=args.cell_size_nm * 1.0e-9,
            )
            if not np.array_equal(
                geometry.material_id,
                stored[f"{key}_material_id"],
            ):
                raise RuntimeError(f"{span_um:g} um material IDs changed")
            volume = (
                np.diff(geometry.x_edges_m)[:, None, None]
                * np.diff(geometry.y_edges_m)[None, :, None]
                * np.diff(geometry.z_edges_m)[None, None, :]
            )
            outside_power = float(
                np.sum(volume[~geometry.flake_mask] * source[
                    ~geometry.flake_mask
                ])
            )
            outside_count = int(
                np.count_nonzero(source[~geometry.flake_mask])
            )
            if outside_power != 0.0 or outside_count != 0:
                raise RuntimeError(f"{span_um:g} um source support changed")
            forward = SimpleNamespace(
                geometry=geometry,
                solved=SimpleNamespace(temperature_K=theta),
            )
            scenario["interface_diagnostics"] = interface_diagnostics(
                forward
            )
            scenario["thermal_geometry"] = {
                "lateral_domain_um": args.lateral_domain_um,
                "Si_depth_um": args.si_depth_um,
                "core_cell_size_nm": args.cell_size_nm,
                "flake_span_um": span_um,
                "flake_thickness_nm": 100.0,
                "design_span_um": 2.0,
                "design_height_nm": 600.0,
            }
            scenario["source_support"] = {
                "exact_TaIrTe4_only": True,
                "outside_TaIrTe4_power_W": outside_power,
                "outside_TaIrTe4_nonzero_cell_count": outside_count,
            }
            mapping = mapping_by_span[span_um]
            if (
                scenario["mapping_artifact"]["sha256"]
                != mapping["raw_artifact"]["sha256"]
            ):
                raise RuntimeError(f"{span_um:g} um mapping SHA changed")
        if not all(
            np.array_equal(density_controls[0], value)
            for value in density_controls[1:]
        ):
            raise RuntimeError("thermal density control differs by scenario")

    scenarios = sorted(
        raw_summary["scenarios"], key=lambda value: value["flake_span_um"]
    )
    if [value["flake_span_um"] for value in scenarios] != [4.0, 6.0]:
        raise RuntimeError("expected exactly the 4 um and 6 um scenarios")
    four, six = scenarios
    compared_fields = (
        "objective_central_flake_average_DeltaT_K",
        "Tmax_DeltaT_K",
        "TaIrTe4_Tmax_DeltaT_K",
        "TaIrTe4_volume_average_DeltaT_K",
    )
    comparison = {
        field: {
            "flake_4um": four[field],
            "flake_6um": six[field],
            "relative_change_6um_vs_4um": relative_change(
                four[field], six[field]
            ),
        }
        for field in compared_fields
    }
    raw_summary.update(
        {
            "published_status": STATUS_PASS,
            "thermal_density_control": {
                "formula": "rho=0.5+0.04*cos(pi*xhat)*cos(pi*yhat)",
                "shape": list(density_controls[0].shape),
                "minimum": float(np.min(density_controls[0])),
                "maximum": float(np.max(density_controls[0])),
                "optical_source_density": (
                    "uniform rho=0.5 from the immutable optical FSP"
                ),
                "Q_fixed_during_thermal_FD": True,
                "self_consistent_combined_optical_thermal": False,
                "purpose": (
                    "thermal-material/interface-only directional gradient "
                    "certificate"
                ),
            },
            "published_summary_audit": {
                "material_support_rechecked_from_raw": True,
                "hotspot_material_is_TaIrTe4_in_every_scenario": all(
                    value["hotspot_m"]["material_id"] == 3
                    for value in scenarios
                ),
                "interface_power_uses_only_selected_faces": True,
                "pre_support_mapping_promoted": False,
                "pte_or_optimization_promoted": False,
            },
            "scenario_comparison": comparison,
            "interpretation_limits": {
                "four_and_six_um_flakes": (
                    "named numerical footprint scenarios, not a confidence "
                    "interval or fabrication claim"
                ),
                "source": (
                    "local Omega_Q certificate only; not the complete "
                    "large-flake ideal-plane-wave absorption footprint"
                ),
                "objective": (
                    "central 2x2 um TaIrTe4 volume-average DeltaT, not "
                    "terminal PTE current"
                ),
                "boundary_flux": (
                    "far-x/y and bottom powers are numerical truncation "
                    "reservoir fluxes, not physical heat-path fractions"
                ),
                "spatial_discretization": (
                    "100 nm core-grid scenario; this run verifies the "
                    "discrete gradient, conservation, and residual, not a "
                    "new thermal mesh-convergence bound"
                ),
            },
            "mapping_summaries": [
                {
                    "path": str(path),
                    "sha256": sha256(path),
                    "raw_artifact": value["raw_artifact"],
                    "mapping_relative_power_error": value[
                        "mapping_relative_power_error"
                    ],
                    "transpose_relative_error": value[
                        "transpose_dot_test"
                    ]["relative_error"],
                    "material_support_projection": value[
                        "material_support_projection"
                    ],
                }
                for path, value in mapping_summaries
            ],
        }
    )

    summary_path = report_dir / SUMMARY_NAME
    summary_path.write_text(json.dumps(raw_summary, indent=2) + "\n")
    cases_path = report_dir / CASES_NAME
    with raw_cases_path.open(newline="") as input_stream:
        rows = list(csv.DictReader(input_stream))
    with cases_path.open("w", newline="") as output_stream:
        writer = csv.DictWriter(
            output_stream,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    artifacts = [
        artifact("thermal_run_summary", raw_summary_path),
        artifact("thermal_run_cases", raw_cases_path),
        artifact("thermal_raw_npz", raw_npz_path),
    ]
    for index, (path, value) in enumerate(mapping_summaries, start=1):
        artifacts.append(artifact(f"mapping_{index}_summary", path))
        artifacts.append(
            artifact(
                f"mapping_{index}_raw_npz",
                Path(value["raw_artifact"]["path"]),
            )
        )
        artifacts.append(
            artifact(
                f"mapping_{index}_immutable_input_npz",
                Path(value["optical_input"]["path"]),
            )
        )
    manifest = {
        "schema_version": 1,
        "raw_artifacts_committed_to_git": False,
        "note": (
            "Raw NPZ/FSP files remain outside Git. The rejected pre-support "
            "thermal run is not promoted or listed as an input."
        ),
        "generation_contract": {
            "mapping": (
                "validate_large_background_local_q_mapping with immutable "
                "native arrays, expected SHA-256, and exact TaIrTe4 support"
            ),
            "thermal": (
                "run_large_background_local_q_thermal_adfd with 4/6 um "
                "scenarios, 32 um lateral domain, 20 um Si depth, 100 nm "
                "core cells, and FD steps 0.01/0.005"
            ),
        },
        "artifacts": artifacts,
    }
    manifest_path = report_dir / MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    def pct(value: float) -> str:
        return f"{100.0 * value:.6g}%"

    report = f"""# Large-background local-Q explicit thermal AD–FD

Status: `{STATUS_PASS}`

This certificate validates the discrete thermal-material/interface gradient
for two named large-TaIrTe4 footprint scenarios. It does **not** promote a
complete ideal-plane-wave thermal source, a PTE current, an experimental
prediction, or an optimization result.

## Fail-closed source-support correction

The first geometric embedding conserved total optical power but placed
staggered Yee control-volume boundary pieces in adjacent oxide/design/air
thermal cells. That precheck was rejected and is not promoted. For the
dominant `Qx` component, the affected power fraction was
`{mapping_summaries[0][1]["components"]["x"]["relocated_power_fraction"]:.12g}`.

The validated map preserves every native source-cell energy and relocates only
the z-directed boundary pieces to the nearest exact TaIrTe4 thermal cell in
the same x-y column. It is a fixed linear operator with an exact transpose:
no clipping, deletion, smoothing, empirical gain, global rescaling, or tiling.

- native and mapped `P_Q`:
  `{four["source_power_W"]:.16e} W`;
- relative power error:
  `{mapping_summaries[0][1]["mapping_relative_power_error"]:.8e}`;
- 4 um / 6 um transpose errors:
  `{mapping_summaries[0][1]["transpose_dot_test"]["relative_error"]:.8e}` /
  `{mapping_summaries[1][1]["transpose_dot_test"]["relative_error"]:.8e}`;
- power and nonzero cells outside TaIrTe4: `0 W / 0` in both scenarios.

## Explicit thermal contract

- thermal domain: `32 × 32 um`, Si depth `20 um`;
- named TaIrTe4 footprints: `4 × 4 um` and `6 × 6 um`;
- protected design: `2 × 2 × 0.6 um`;
- core grid: `100 nm`; TaIrTe4 z cells: `25 nm`;
- TaIrTe4 kappa: `diag(14.4, 3.8, 1.0) W/(m K)`;
- bulk kappa: SiO2 `1.38`, Si `145`, air `0.026 W/(m K)`;
- gray design: `k=0.026+rho*(1.38-0.026)`;
- TaIrTe4/air sidewalls: `G=1 W/(m2 K)`, not adiabatic;
- TaIrTe4/bottom-SiO2: `G=7.37e6 W/(m2 K)`;
- gray top design contact:
  `G=1+rho*(7.37e4-1) W/(m2 K)`;
- SiO2/Si: named candidate `G=1.1e9 W/(m2 K)`;
- exposed top surface: Robin `h=10 W/(m2 K)`;
- far x/y and bottom Si: fixed `DeltaT=0` numerical truncation reservoirs.

The objective is the volume-average `DeltaT` in the central `2 × 2 um`
TaIrTe4 region. `Q` is fixed during these thermal-material FD checks.
The optical source came from a uniform `rho=0.5` optical forward, whereas
the thermal control uses
`rho=0.5+0.04*cos(pi*xhat)*cos(pi*yhat)` (range
`{np.min(density_controls[0]):.12g}` to
`{np.max(density_controls[0]):.12g}`). This deliberate mismatch excites the
thermal material/interface derivatives; it is not a self-consistent combined
optical-thermal design state.

## Scenario results

| quantity | 4 um flake | 6 um flake | 6 vs 4 |
|---|---:|---:|---:|
| central 2 um average `DeltaT` | `{four["objective_central_flake_average_DeltaT_K"]:.12e} K` | `{six["objective_central_flake_average_DeltaT_K"]:.12e} K` | `{pct(comparison["objective_central_flake_average_DeltaT_K"]["relative_change_6um_vs_4um"])}` |
| TaIrTe4 `Tmax` | `{four["TaIrTe4_Tmax_DeltaT_K"]:.12e} K` | `{six["TaIrTe4_Tmax_DeltaT_K"]:.12e} K` | `{pct(comparison["TaIrTe4_Tmax_DeltaT_K"]["relative_change_6um_vs_4um"])}` |
| whole-flake average `DeltaT` | `{four["TaIrTe4_volume_average_DeltaT_K"]:.12e} K` | `{six["TaIrTe4_volume_average_DeltaT_K"]:.12e} K` | `{pct(comparison["TaIrTe4_volume_average_DeltaT_K"]["relative_change_6um_vs_4um"])}` |
| worst AD–FD relative error | `{four["maximum_AD_FD_relative_error"]:.8e}` | `{six["maximum_AD_FD_relative_error"]:.8e}` | — |
| energy-balance error | `{four["energy_balance_relative_error"]:.8e}` | `{six["energy_balance_relative_error"]:.8e}` | — |
| forward residual | `{four["forward_linear_residual_relative"]:.8e}` | `{six["forward_linear_residual_relative"]:.8e}` | — |
| adjoint residual | `{four["adjoint_linear_residual_relative"]:.8e}` | `{six["adjoint_linear_residual_relative"]:.8e}` | — |

Both global hotspots are inside TaIrTe4 at approximately
`(x,y,z)=(0.05,0.05,-0.0125) um`. The much lower whole-flake average for the
6 um case mostly reflects averaging the same local source over a larger
unilluminated flake volume; the central 2 um objective is the more comparable
quantity.

## Interfaces and external boundaries

For 4 um / 6 um, the mean **contact-only** jumps are:

- TaIrTe4/bottom-SiO2:
  `{four["interface_diagnostics"]["TaIrTe4_bottom_SiO2"]["mean_contact_temperature_jump_K"]:.8e}` /
  `{six["interface_diagnostics"]["TaIrTe4_bottom_SiO2"]["mean_contact_temperature_jump_K"]:.8e} K`;
- SiO2/Si:
  `{four["interface_diagnostics"]["SiO2_Si"]["mean_contact_temperature_jump_K"]:.8e}` /
  `{six["interface_diagnostics"]["SiO2_Si"]["mean_contact_temperature_jump_K"]:.8e} K`;
- gray top design contact:
  `{four["interface_diagnostics"]["TaIrTe4_top_design_or_air"]["subinterfaces"]["gray_design_contact"]["mean_contact_temperature_jump_K"]:.8e}` /
  `{six["interface_diagnostics"]["TaIrTe4_top_design_or_air"]["subinterfaces"]["gray_design_contact"]["mean_contact_temperature_jump_K"]:.8e} K`.

The reported adjacent-cell jump is kept separately because it also contains
the two half-cell conduction drops and is not equal to `q''/G`.

Approximately
`{pct(sum(four["boundary_power_fraction_of_source"][name] for name in ("x_min", "x_max", "y_min", "y_max")))}` /
`{pct(sum(six["boundary_power_fraction_of_source"][name] for name in ("x_min", "x_max", "y_min", "y_max")))}` leaves through the four far lateral
reservoirs, while
`{pct(four["boundary_power_fraction_of_source"]["z_min"])}` /
`{pct(six["boundary_power_fraction_of_source"]["z_min"])}` leaves through the
bottom reservoir. These are numerical boundary flux partitions, not physical
heat-path fractions.

## Numerical error versus physical-model variation

The worst discrete AD–FD error is
`{raw_summary["gates"]["worst_AD_FD_relative_error"]:.8e}`;
the worst energy error is
`{raw_summary["gates"]["worst_energy_balance_relative_error"]:.8e}`;
and the worst linear residual is
`{raw_summary["gates"]["worst_linear_residual_relative"]:.8e}`.
These certify the assembled discrete equations and gradient.

The 4-to-6 um differences are a named footprint-scenario variation, not a
confidence interval. This run does not add a new spatial mesh-convergence
bound. More importantly, the source is only the validated local `Omega_Q`
certificate; the absorption outside that local volume under a truly extended
ideal plane wave is not included. Therefore these temperatures are not a
final plane-wave or experimental prediction.

No terminal PTE, transient, adjoint optimization, gradient-based optimization,
or geometry update was run.
"""
    (report_dir / REPORT_NAME).write_text(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
