#!/usr/bin/env python3
"""Run the four new Device-A Maxwell Q cases through both thermal bounds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
RUNNER = HERE / "run_device_a_explicit_thermal_pte.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry-contract", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    for label in ("sminus1", "splus1"):
        for polarization in ("a", "b"):
            parser.add_argument(f"--{label}-{polarization}", type=Path, required=True)
    return parser.parse_args()


def completed_summary(path: Path) -> dict[str, object]:
    summary = json.loads((path / "summary.json").read_text())
    if not str(summary["status"]).startswith("COMPLETED"):
        raise RuntimeError(f"thermal case failed: {path}: {summary['status']}")
    if float(summary["mapping"]["mapping_relative_power_error"]) >= 0.005:
        raise RuntimeError(f"mapping gate failed: {path}")
    if float(summary["thermal"]["linear_residual_relative"]) >= 1e-8:
        raise RuntimeError(f"residual gate failed: {path}")
    if float(summary["thermal"]["energy_balance_relative_error"]) >= 0.01:
        raise RuntimeError(f"energy gate failed: {path}")
    return summary


def main() -> int:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    records = []
    for label in ("sminus1", "splus1"):
        for polarization in ("a", "b"):
            optical = getattr(args, f"{label}_{polarization}")
            for short, scenario in (
                ("isolated", "isolated-lower-bound"),
                ("perfect", "perfect-to-flake-upper-bound"),
            ):
                output = args.output_root / (
                    f"thermal_{label}_{polarization}_{short}_60um_100nm_20260801"
                )
                if output.exists():
                    raise FileExistsError(
                        f"refusing to overwrite immutable output directory: {output}"
                    )
                command = [
                    sys.executable,
                    str(RUNNER),
                    "--optical-case-dir",
                    str(optical),
                    "--output-dir",
                    str(output),
                    "--thermal-domain-um",
                    "60",
                    "--si-depth-um",
                    "20",
                    "--core-step-nm",
                    "100",
                    "--flake-dz-nm",
                    "10",
                    "--geometry-contract-json",
                    str(args.geometry_contract),
                    "--geometry",
                    "device-a-polygon",
                    "--thermal-model",
                    "expanded",
                    "--metal-thermalization",
                    scenario,
                ]
                print(
                    f"[thermal-batch] start {label} E||{polarization} {scenario}",
                    flush=True,
                )
                subprocess.run(command, check=True)
                summary = completed_summary(output)
                record = {
                    "position_label": label,
                    "polarization": polarization,
                    "metal_thermalization": scenario,
                    "output_dir": str(output.resolve()),
                    "summary_path": str((output / "summary.json").resolve()),
                    "fields_path": str((output / "thermal_pte_fields.npz").resolve()),
                    "PTE_current_A": summary["PTE_current_A_at_285uW_incident"],
                    "mapping_relative_power_error": summary["mapping"][
                        "mapping_relative_power_error"
                    ],
                    "linear_residual_relative": summary["thermal"][
                        "linear_residual_relative"
                    ],
                    "energy_balance_relative_error": summary["thermal"][
                        "energy_balance_relative_error"
                    ],
                }
                records.append(record)
                (args.output_root / "device_a_position_maxwell_thermal_batch_index.json").write_text(
                    json.dumps(
                        {
                            "status": "IN_PROGRESS",
                            "thermal_contract": {
                                "lateral_domain_um": 60.0,
                                "Si_depth_um": 20.0,
                                "core_xy_cell_size_nm": 100.0,
                                "flake_dz_nm": 10.0,
                            },
                            "completed_cases": records,
                        },
                        indent=2,
                    )
                    + "\n"
                )
    payload = {
        "status": "COMPLETED_DEVICE_A_POSITION_MAXWELL_THERMAL_BATCH",
        "thermal_contract": {
            "lateral_domain_um": 60.0,
            "Si_depth_um": 20.0,
            "core_xy_cell_size_nm": 100.0,
            "flake_dz_nm": 10.0,
            "same_as_immutable_s0": True,
        },
        "completed_cases": records,
        "no_Q_clipping_smoothing_gain_rescaling": True,
    }
    (args.output_root / "device_a_position_maxwell_thermal_batch_index.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
