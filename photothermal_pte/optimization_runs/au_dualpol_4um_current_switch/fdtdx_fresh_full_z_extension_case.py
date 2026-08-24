"""Write canonical finer full-domain-z case contracts without changing base ladders."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_exact_binary_convergence import (
    MeshSpec,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_case_contract import (
    FreshCaseSpec,
    TimeSpec,
    case_contract,
    file_sha256,
)


EXTENSION_FACTORS = {"z16": 16, "z32": 32}
TOTAL_PERIODS = 24
WINDOW_PERIODS = 4
COURANT_FACTOR = 0.25


def expected_extension_case(level: str) -> FreshCaseSpec:
    if level not in EXTENSION_FACTORS:
        raise ValueError(
            f"extension level must be one of {tuple(EXTENSION_FACTORS)}"
        )
    return FreshCaseSpec(
        mesh=MeshSpec(z_factor=EXTENSION_FACTORS[level]),
        time=TimeSpec(
            total_periods=TOTAL_PERIODS,
            window_periods=WINDOW_PERIODS,
            courant_factor=COURANT_FACTOR,
        ),
    )


def write_extension_case(level: str, output: Path) -> dict:
    supplied = output.expanduser()
    if not supplied.is_absolute():
        raise RuntimeError("output path must be absolute")
    resolved = supplied.resolve()
    if not resolved.parent.is_dir() or resolved.exists():
        raise RuntimeError("output parent must exist and output must not exist")
    payload = case_contract(expected_extension_case(level))
    resolved.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {
        "level": level,
        "z_factor": EXTENSION_FACTORS[level],
        "path": str(resolved),
        "file_sha256": file_sha256(resolved),
        "case_contract_sha256": payload["case_contract_sha256"],
        "grid_shape_xyz": payload["resolved_mesh"]["grid_shape_xyz"],
        "yee_cell_count": payload["resolved_mesh"]["yee_cell_count"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", choices=tuple(EXTENSION_FACTORS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = write_extension_case(args.level, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
