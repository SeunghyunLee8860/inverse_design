#!/usr/bin/env python3
"""Validate one optimization run without launching a solver."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .run_contract import ValidationError, validate_run_directory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_directory", type=Path)
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument("--require-external", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = validate_run_directory(
            args.run_directory,
            repository_root=args.repository_root,
            require_external=args.require_external,
        )
    except ValidationError as exc:
        print(json.dumps({"valid": False, "error": str(exc)}, indent=2))
        return 1
    print(json.dumps(result.as_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
