#!/usr/bin/env python3
"""Audit Lumerical Maxwell/B200 plus custom GPU-PDE execution contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY = Path(__file__).resolve().parents[3]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.lumerical_maxwell_contract import (
    audit_environment,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-index", type=int)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--require-ready", action="store_true")
    args = parser.parse_args()
    payload = audit_environment(requested_gpu_index=args.gpu_index)
    rendered = json.dumps(payload, indent=2) + "\n"
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if (
        args.require_ready
        and payload["status"] != "READY_FOR_LUMERICAL_B200_MAXWELL_DEVELOPMENT"
    ):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
