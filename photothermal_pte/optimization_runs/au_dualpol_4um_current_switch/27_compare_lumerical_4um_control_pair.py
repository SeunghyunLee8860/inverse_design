#!/usr/bin/env python3
"""Compare two hash-verified Lumerical exact-control result bundles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.lumerical_4um_control_comparison import (
    compare_control_pair,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coarse-json", required=True, type=Path)
    parser.add_argument("--fine-json", required=True, type=Path)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    result = compare_control_pair(args.coarse_json, args.fine_json)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if result["all_gates_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
