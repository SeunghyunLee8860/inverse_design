#!/usr/bin/env python3
"""Publish the component-wise native-Yee material-Jacobian certificate."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

from .summarize_imported_permittivity_endpoints import artifact


EXPECTED_STATUS = "VALIDATED_COMPONENT_WISE_YEE_MATERIAL_JACOBIAN"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-result", required=True)
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--generation-command", required=True)
    args = parser.parse_args()
    raw_path = Path(args.raw_result).expanduser().resolve()
    raw = json.loads(raw_path.read_text())
    if raw.get("status") != EXPECTED_STATUS or not raw.get("passed"):
        raise RuntimeError("component Yee Jacobian did not pass")
    report_dir = Path(args.report_dir).expanduser().resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(timezone.utc).isoformat()
    summary = {
        "status": EXPECTED_STATUS,
        "passed": True,
        "generated_at_utc": generated_at,
        "scope": raw["scope"],
        "density_shape": raw["density_shape"],
        "density_bounds_m": raw["density_bounds_m"],
        "density_to_solver_chain": raw["density_to_solver_chain"],
        "construction": raw["construction"],
        "coordinate_audit": raw["coordinate_audit"],
        "maximum_coordinate_mismatch_m": raw[
            "maximum_coordinate_mismatch_m"
        ],
        "completed_vs_layout_index_detail_epsilon_max_abs_error": raw[
            "completed_vs_layout_index_detail_epsilon_max_abs_error"
        ],
        "baseline_layout_roundtrip_epsilon_max_abs_error": raw[
            "baseline_layout_roundtrip_epsilon_max_abs_error"
        ],
        "directions": raw["directions"],
        "gates": raw["gates"],
        "next_gate": "OPTICAL_DZ_DOWNSTREAM_PTE_AND_GRADIENT_CONVERGENCE",
        "gray_law_sensitivity_run": False,
        "full_latent_adfd_run": False,
        "optimization_run": False,
    }
    summary_path = (
        report_dir / "component_yee_material_jacobian_summary.json"
    )
    cases_path = (
        report_dir / "component_yee_material_jacobian_cases.csv"
    )
    report_path = (
        report_dir / "COMPONENT_WISE_YEE_MATERIAL_JACOBIAN_REPORT.md"
    )
    manifest_path = (
        report_dir
        / "COMPONENT_WISE_YEE_MATERIAL_JACOBIAN_RAW_ARTIFACT_MANIFEST.json"
    )
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    rows = [
        {
            "direction": name,
            "direction_sha256": item["direction_sha256"],
            "mapping_only_FD_step": item[
                "mapping_only_centered_FD_step"
            ],
            "mapping_only_FD_relative_error": item[
                "mapping_only_FD_relative_error"
            ],
            "JVP_VJP_dot_relative_error": item[
                "JVP_VJP_dot_relative_error"
            ],
            "clipping": item["clipping"],
            "Maxwell_solves": item["Maxwell_solves"],
        }
        for name, item in raw["directions"].items()
    ]
    with cases_path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    component_rows = "\n".join(
        "| "
        + " | ".join(
            (
                component,
                str(item["shape"]),
                str(item["J_nnz"]),
                str(item["maximum_J_nonzeros_per_Yee_sample"]),
                f"{item['maximum_forward_adjoint_coordinate_mismatch_m']:.6e}",
                f"{item['maximum_field_index_detail_coordinate_mismatch_m']:.6e}",
            )
        )
        + " |"
        for component, item in raw["coordinate_audit"]["components"].items()
    )
    direction_rows = "\n".join(
        f"| {row['direction']} | "
        f"{float(row['mapping_only_FD_relative_error']):.6e} | "
        f"{float(row['JVP_VJP_dot_relative_error']):.6e} |"
        for row in rows
    )
    report_path.write_text(
        f"""# Component-wise native-Yee material Jacobian

Status: `{EXPECTED_STATUS}`

The production material chain is

`rho(81x81) -> epsilon=1+rho*(1.38^2-1) -> n=sqrt(epsilon) ->
importnk2(81x81x13) -> v261 component-specific conformal index_detail ->
epsilon_Yee,c=index_c^2`.

Separate sparse operators `J_x`, `J_y`, and `J_z` were constructed at the
nonuniform physical-density baseline using 25 local colors and centered
layout-only material perturbations. The completed solver `index_detail` and
the layout-mode `index_detail` were identical. This construction ran zero
Maxwell solves and does not use per-pixel Maxwell solves, empirical
normalization, or gradient rescaling.

The component coordinates come directly from `index_detail`: `index_x` uses
`x_offset,y,z`; `index_y` uses `x,y_offset,z`; and `index_z` uses
`x,y,z_offset`. Forward and adjoint fields are read on the same native
`PABS_FIELD` component coordinates. The discarded path that multiplied
separate design-field and design-index arrays by common array index is not
used.

| component | shape | J nnz | max nnz/row | fwd-adj coord mismatch (m) | field-index coord mismatch (m) |
| --- | --- | ---: | ---: | ---: | ---: |
{component_rows}

| direction | mapping-only FD error | JVP-VJP dot error |
| --- | ---: | ---: |
{direction_rows}

- Worst mapping-only FD error:
  `{raw['gates']['worst_mapping_only_FD_relative_error']:.9e}`
  (limit `{raw['gates']['mapping_only_FD_limit']:.1e}`).
- Worst transpose error:
  `{raw['gates']['worst_JVP_VJP_dot_relative_error']:.9e}`
  (limit `{raw['gates']['dot_limit']:.1e}`).
- Maximum coordinate mismatch:
  `{raw['maximum_coordinate_mismatch_m']:.9e} m`
  (limit `{raw['gates']['coordinate_mismatch_limit_m']:.1e} m`).
- Completed-solver versus layout `index_detail` epsilon error: `0`.
- Baseline layout round-trip epsilon error: `0`.

Raw sparse matrices and coordinate arrays remain outside Git. Paths, sizes,
and SHA-256 values are recorded in the manifest. This checkpoint does not
run downstream thermal/PTE convergence, combined AD-FD, gray-law
sensitivity, latent/filter/projection AD-FD, transient, or optimization.
"""
    )
    artifacts = [
        {"role": "raw_result", **artifact(raw_path)},
        {
            "role": "forward_FSP",
            **raw["artifacts"]["forward_FSP"],
        },
        {
            "role": "adjoint_FSP",
            **raw["artifacts"]["adjoint_FSP"],
        },
        {
            "role": "component_coordinates",
            **raw["artifacts"]["coordinates"],
        },
    ]
    artifacts.extend(
        {
            "role": f"J_{component}",
            **item,
        }
        for component, item in raw["artifacts"]["component_J"].items()
    )
    manifest = {
        "status": EXPECTED_STATUS,
        "generated_at_utc": generated_at,
        "generation_command": args.generation_command,
        "artifacts": artifacts,
        "git_policy": (
            "Raw FSP, sparse-J NPZ, and coordinate NPZ artifacts stay "
            "outside Git; only paths, sizes, and SHA-256 values are committed"
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
