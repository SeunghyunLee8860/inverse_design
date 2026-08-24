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
SUPPORTED_TOTAL_PERIODS = (24, 32)
TOTAL_PERIODS = 24
WINDOW_PERIODS = 4
COURANT_FACTOR = 0.25


def expected_extension_case(
    level: str, total_periods: int = TOTAL_PERIODS
) -> FreshCaseSpec:
    if level not in EXTENSION_FACTORS:
        raise ValueError(
            f"extension level must be one of {tuple(EXTENSION_FACTORS)}"
        )
    if total_periods not in SUPPORTED_TOTAL_PERIODS:
        raise ValueError(
            f"total_periods must be one of {SUPPORTED_TOTAL_PERIODS}"
        )
    return FreshCaseSpec(
        mesh=MeshSpec(z_factor=EXTENSION_FACTORS[level]),
        time=TimeSpec(
            total_periods=total_periods,
            window_periods=WINDOW_PERIODS,
            courant_factor=COURANT_FACTOR,
        ),
    )


def write_extension_case(
    level: str, output: Path, total_periods: int = TOTAL_PERIODS
) -> dict:
    supplied = output.expanduser()
    if not supplied.is_absolute():
        raise RuntimeError("output path must be absolute")
    resolved = supplied.resolve()
    if not resolved.parent.is_dir() or resolved.exists():
        raise RuntimeError("output parent must exist and output must not exist")
    payload = case_contract(expected_extension_case(level, total_periods))
    resolved.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {
        "level": level,
        "z_factor": EXTENSION_FACTORS[level],
        "total_periods": total_periods,
        "path": str(resolved),
        "file_sha256": file_sha256(resolved),
        "case_contract_sha256": payload["case_contract_sha256"],
        "grid_shape_xyz": payload["resolved_mesh"]["grid_shape_xyz"],
        "yee_cell_count": payload["resolved_mesh"]["yee_cell_count"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", choices=tuple(EXTENSION_FACTORS), required=True)
    parser.add_argument(
        "--total-periods",
        type=int,
        choices=SUPPORTED_TOTAL_PERIODS,
        default=TOTAL_PERIODS,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = write_extension_case(args.level, args.output, args.total_periods)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
