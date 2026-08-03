#!/usr/bin/env python3
"""Run the frozen 9-position Device-A optical matrix on GPU only.

Every position receives its own polarization-matched empty-stack reference.
The Device-A polygons, FDTD/PML bounds, monitors, and local-mesh rectangles
are invariant; only the Gaussian source is translated in the fixed Lumerical
``x=b, y=a`` frame.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
RUNNER = HERE / "run_lumerical_device_a_ir_q.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--position-contract", type=Path, required=True)
    parser.add_argument("--geometry-contract", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpu-device", default="GPU 0")
    parser.add_argument("--threads", default="3")
    parser.add_argument("--only-label", action="append", default=[])
    parser.add_argument("--only-polarization", choices=("a", "b"), default=None)
    return parser.parse_args()


def common_command(args: argparse.Namespace, case: dict[str, Any], polarization: str) -> list[str]:
    offset = case["beam_offset_from_fixed_baseline_um"]
    return [
        sys.executable,
        str(RUNNER),
        "--polarization", polarization,
        "--geometry", "device-a-polygon",
        "--device-a-geometry-json", str(args.geometry_contract),
        "--beam-offset-x-um", str(offset[0]),
        "--beam-offset-y-um", str(offset[1]),
        "--fixed-mesh-center-x-um", "-11.28125",
        "--fixed-mesh-center-y-um", "0",
        "--domain-um", "64",
        "--source-span-um", "50",
        "--waist-um", "8.75",
        "--source-object-waist-um", "8.610602974768",
        "--execution-contract", "paper-measured-reproduction",
        "--substrate-optical-model", "paper-kitamura-palik-nk-11um",
        "--source-pulse-contract", "frequency-centered-11um",
        "--matched-lossy-control-volume",
        "--pml-layers", "24",
        "--flake-dz-nm", "10",
        "--mesh-accuracy", "3",
        "--outer-local-xy-mesh-nm", "200",
        "--intermediate-xy-mesh-nm", "100",
        "--intermediate-half-span-um", "15.5",
        "--local-xy-mesh-nm", "50",
        "--refinement-half-span-um", "15",
        "--refinement-y-half-span-um", "15",
        "--simulation-time-ps", "4",
        "--auto-shutoff-min", "1e-5",
        "--epsilon-c-model", "paper-b-closure",
        "--gpu-device", args.gpu_device,
        "--threads", args.threads,
    ]


def validate_result(path: Path, case: dict[str, Any], polarization: str) -> dict[str, Any]:
    result_path = path / "case_result.json"
    if not result_path.is_file():
        raise RuntimeError(f"missing optical result: {result_path}")
    result = json.loads(result_path.read_text())
    acceptance = result.get("run_result", {}).get("acceptance", {})
    if result.get("status") != "COMPLETED" or not acceptance or not all(acceptance.values()):
        failed = [key for key, value in acceptance.items() if not value]
        raise RuntimeError(f"optical gates failed for {path}: {failed}")
    source = result["pre_run_contract"]["geometry"]["source"]
    realized = np.asarray(source["beam_center_m"], float) * 1.0e6
    expected = np.asarray(case["beam_center_lumerical_um"], float)
    if not np.allclose(realized, expected, rtol=0.0, atol=1.0e-9):
        raise RuntimeError(
            f"source-center readback mismatch for {path}: {realized} != {expected}"
        )
    if result["domain_um"] != 64.0 or result["pml_layers"] != 24:
        raise RuntimeError(f"domain/PML contract mismatch for {path}")
    if result["Q_clipped"] or result["Q_rescaled"] or result["periodic_Q_used"]:
        raise RuntimeError(f"forbidden Q processing recorded for {path}")
    run = result["run_result"]
    return {
        "label": case["label"],
        "polarization": polarization,
        "case": result["case"],
        "beam_center_lumerical_um": realized.tolist(),
        "output_dir": str(path.resolve()),
        "case_result": str(result_path.resolve()),
        "acceptance": acceptance,
        "P_Q_W_at_1_W_m2": run.get("P_Q_W"),
        "P_six_W_at_1_W_m2": run.get("P_six_face_W"),
        "six_face_closure": run.get("six_face_relative_closure"),
        "auto_shutoff_final": run["auto_shutoff"]["final_value"],
    }


def write_index(path: Path, contract: dict[str, Any], records: list[dict[str, Any]], status: str) -> None:
    payload = {
        "status": status,
        "position_contract": contract,
        "optical_contract": {
            "wavelength_um": 11.0,
            "scalar_Gaussian_waist_um": 8.75,
            "source_span_um": 50.0,
            "FDTD_lateral_span_um": 64.0,
            "six_PML_layers": 24,
            "mesh_accuracy": 3,
            "TaIrTe4_dz_nm": 10.0,
            "fixed_fine_mesh_nm": 50.0,
            "fixed_mesh_center_lumerical_um": [-11.28125, 0.0],
            "fixed_fine_mesh_half_spans_um": [15.0, 15.0],
            "fixed_intermediate_mesh_nm": 100.0,
            "fixed_intermediate_half_span_um": 15.5,
            "outer_mesh_nm": 200.0,
            "GPU_only": True,
            "CPU_FDTD_fallback": False,
            "Q_clipping_smoothing_gain_rescaling_tiling": False,
        },
        "completed_cases": records,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def main() -> int:
    args = parse_args()
    contract = json.loads(args.position_contract.read_text())
    if contract.get("status") != "FROZEN_DEVICE_A_NINE_POSITION_CONTRACT":
        raise RuntimeError("position contract is not frozen")
    selected = [
        case for case in contract["cases"]
        if not args.only_label or case["label"] in set(args.only_label)
    ]
    if not selected:
        raise RuntimeError("no selected position cases")
    polarizations = (args.only_polarization,) if args.only_polarization else ("a", "b")
    args.output_root.mkdir(parents=True, exist_ok=True)
    index_path = args.output_root / "device_a_nine_position_optical_batch_index.json"
    records: list[dict[str, Any]] = []
    for case in selected:
        for polarization in polarizations:
            case_root = args.output_root / case["label"] / polarization
            empty = case_root / "empty"
            finite = case_root / "finite"
            if not (empty / "case_result.json").is_file():
                command = common_command(args, case, polarization) + [
                    "--case", "empty-stack",
                    "--output-dir", str(empty),
                ]
                print(f"[nine-optical] empty {case['label']} E||{polarization}", flush=True)
                subprocess.run(command, check=True)
            empty_record = validate_result(empty, case, polarization)
            records.append(empty_record)
            write_index(index_path, contract, records, "IN_PROGRESS_EMPTY_COMPLETE")
            if not (finite / "case_result.json").is_file():
                command = common_command(args, case, polarization) + [
                    "--case", "finite-flake",
                    "--include-electrodes",
                    "--incident-reference", str(empty / "case_result.json"),
                    "--output-dir", str(finite),
                ]
                print(f"[nine-optical] finite {case['label']} E||{polarization}", flush=True)
                subprocess.run(command, check=True)
            finite_record = validate_result(finite, case, polarization)
            records.append(finite_record)
            write_index(index_path, contract, records, "IN_PROGRESS")
    write_index(index_path, contract, records, "COMPLETED_DEVICE_A_NINE_POSITION_OPTICAL_BATCH")
    print(index_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
