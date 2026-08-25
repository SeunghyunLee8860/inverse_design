#!/usr/bin/env python3
"""Build a source-pair certificate for the full-domain z refinement."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_case_contract import (
    TimeSpec,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_increment_state_exact_binary_control import (
    _json_default,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_user_balanced_source_only import (
    STATUS_READY as CASE_STATUS,
    VERSION as CASE_VERSION,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_user_balanced_source_pair import (
    CASE_SCOPE,
    build_pair,
    sha256,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_user_balanced_z_refinement import (
    SUPPORTED_FACTORS,
    case_contract,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ea-report", type=Path, required=True)
    parser.add_argument("--ea-report-sha256", required=True)
    parser.add_argument("--eb-report", type=Path, required=True)
    parser.add_argument("--eb-report-sha256", required=True)
    parser.add_argument("--factor", type=int, choices=SUPPORTED_FACTORS, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.expanduser()
    if not output.is_absolute() or not output.parent.is_dir() or output.exists():
        parser.error("--output must be a new absolute file under an existing directory")
    time = TimeSpec(total_periods=24, window_periods=4, courant_factor=0.5)
    payload = build_pair(
        args.ea_report,
        args.ea_report_sha256,
        args.eb_report,
        args.eb_report_sha256,
        expected_case_contract=case_contract(time, args.factor),
        case_version=CASE_VERSION,
        case_status=CASE_STATUS,
        case_scope=CASE_SCOPE,
    )
    payload["full_domain_z_factor"] = args.factor
    temporary = output.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
            default=_json_default,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    print(
        json.dumps(
            {
                "output": str(output),
                "sha256": sha256(output),
                "ready": payload["ready"],
                "failed_gates": payload["failed_gates"],
                "relative_power_mismatch": payload["comparison"][
                    "relative_power_mismatch"
                ],
            },
            default=_json_default,
        )
    )
    return 0 if payload["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
