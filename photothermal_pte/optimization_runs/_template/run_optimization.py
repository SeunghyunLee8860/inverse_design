#!/usr/bin/env python3
"""Run-local entry point; currently provides a fail-closed preflight only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from photothermal_pte.optimization_runs import validate_run_directory  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if not args.preflight:
        parser.error(
            "optimization execution is not implemented; use --preflight"
        )
    run_dir = Path(__file__).resolve().parent
    result = validate_run_directory(
        run_dir,
        repository_root=REPOSITORY_ROOT,
        require_external=True,
    )
    config = json.loads((run_dir / "run_config.json").read_text())
    if config["execution"]["enabled"]:
        raise RuntimeError(
            "execution was enabled without a reviewed production driver"
        )
    print(json.dumps(result.as_dict(), indent=2))
    print("PREFLIGHT_ONLY: no Maxwell, thermal, adjoint, or optimizer solve run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
