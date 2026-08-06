#!/usr/bin/env python3
"""Solver-free setup audit and fail-closed preflight for Run 002."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from photothermal_pte.optimization_runs import validate_run_directory  # noqa: E402
from photothermal_pte.optimization_runs.gaussian10_contract import (  # noqa: E402
    build_contract_audit,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--setup-audit", action="store_true")
    args = parser.parse_args()
    if args.setup_audit:
        print(json.dumps(build_contract_audit(), indent=2))
        print("SETUP_AUDIT_ONLY: no solver or optimizer was launched")
        return 0
    run_dir = Path(__file__).resolve().parent
    result = validate_run_directory(
        run_dir,
        repository_root=REPOSITORY_ROOT,
        require_external=False,
    )
    config = json.loads((run_dir / "run_config.json").read_text())
    if config["execution"]["enabled"]:
        raise RuntimeError("execution enabled before the reviewed production driver")
    print(json.dumps(result.as_dict(), indent=2))
    print(
        "PREFLIGHT_ONLY: source-only, production forward, component-Yee "
        "mapping, material-Q attribution, 3D thermal-grid deposition, and "
        "production CUDA thermal/PTE, fixed-Q thermal-material AD-FD, and "
        "exact thermal-to-native-Yee pullback controls passed; one "
        "production combined physical-rho Maxwell/CUDA-thermal AD-FD smoke "
        "also passed. Broader directions, gray-law and full-latent AD-FD, "
        "and design-window selection still block execution"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
