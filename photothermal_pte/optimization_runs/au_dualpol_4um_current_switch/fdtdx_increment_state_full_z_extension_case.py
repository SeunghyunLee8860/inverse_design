"""Canonical increment-state z16/z32 cases without changing the base ladder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_exact_binary_convergence import (
    MeshSpec,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_case_contract import (
    ANCHOR_CASE,
    FreshCaseSpec,
    TimeSpec,
    case_contract,
    case_for_axis,
    file_sha256,
)


EXTENSION_Z_FACTORS = {"z16": 16, "z32": 32}
TOTAL_PERIODS = 24
WINDOW_PERIODS = 4
COURANT_FACTOR = 0.5


def expected_extension_case(level: str) -> FreshCaseSpec:
    if level not in EXTENSION_Z_FACTORS:
        raise ValueError(f"extension level must be one of {tuple(EXTENSION_Z_FACTORS)}")
    return FreshCaseSpec(
        mesh=MeshSpec(z_factor=EXTENSION_Z_FACTORS[level]),
        time=TimeSpec(
            total_periods=TOTAL_PERIODS,
            window_periods=WINDOW_PERIODS,
            courant_factor=COURANT_FACTOR,
        ),
        pml_alpha_scale=ANCHOR_CASE.pml_alpha_scale,
        pml_target_reflection=ANCHOR_CASE.pml_target_reflection,
    )


def resolve_increment_state_case(
    mesh_axis: str,
    mesh_level: int,
    total_periods: int,
    window_periods: int,
    full_z_extension: str | None,
) -> FreshCaseSpec:
    if full_z_extension is not None:
        if mesh_axis != "anchor" or mesh_level != 0:
            raise ValueError(
                "--full-z-extension cannot be combined with a base mesh axis/level"
            )
        if total_periods != TOTAL_PERIODS or window_periods != WINDOW_PERIODS:
            raise ValueError("full-z extensions require the canonical 24/4 timing")
        return expected_extension_case(full_z_extension)
    return case_for_axis(
        mesh_axis,
        mesh_level,
        time=TimeSpec(
            total_periods=total_periods,
            window_periods=window_periods,
            courant_factor=ANCHOR_CASE.time.courant_factor,
        ),
        pml_alpha_scale=ANCHOR_CASE.pml_alpha_scale,
        pml_target_reflection=ANCHOR_CASE.pml_target_reflection,
    )


def write_extension_case(level: str, output: Path) -> dict:
    supplied = output.expanduser()
    resolved = supplied.resolve()
    if not supplied.is_absolute():
        raise RuntimeError("output path must be absolute")
    if not resolved.parent.is_dir() or resolved.exists():
        raise RuntimeError("output parent must exist and output must not exist")
    payload = case_contract(expected_extension_case(level))
    resolved.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {
        "level": level,
        "z_factor": EXTENSION_Z_FACTORS[level],
        "path": str(resolved),
        "file_sha256": file_sha256(resolved),
        "case_contract_sha256": payload["case_contract_sha256"],
        "grid_shape_xyz": payload["resolved_mesh"]["grid_shape_xyz"],
        "yee_cell_count": payload["resolved_mesh"]["yee_cell_count"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", choices=tuple(EXTENSION_Z_FACTORS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(write_extension_case(args.level, args.output), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
