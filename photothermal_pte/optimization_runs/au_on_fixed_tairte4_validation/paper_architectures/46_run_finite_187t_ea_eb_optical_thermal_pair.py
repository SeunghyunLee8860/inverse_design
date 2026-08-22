#!/usr/bin/env python3
"""Complete the finite-187T Ea/Eb optical -> thermal/PTE pair."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
RAW = Path("/home/seunghyun/tairte4/raw_artifacts")
EA_Q_DIR = RAW / "paper_tairte4_finite_187T_w12_Q_11p825um_Ea"
EB_Q_DIR = RAW / "paper_tairte4_finite_187T_w12_Q_11p825um_Eb"
PAIR_DIR = RAW / "paper_tairte4_finite_187T_w12_ea_eb_thermal_pair"


def load_module(filename: str, name: str):
    path = HERE / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def invoke(module, argv: list[str]) -> int:
    old = sys.argv[:]
    try:
        sys.argv = argv
        return int(module.main())
    finally:
        sys.argv = old


def validated_optical(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text())
    if payload.get("status") != "VALIDATED_FINITE_187T_W12_VOLUMETRIC_Q":
        raise RuntimeError(f"optical pair member is not validated: {path}")
    return payload


def main() -> int:
    optical_runner = load_module("33_run_v261_finite_multi_t_gaussian_q.py", "finite_187t_pair_optical")
    thermal_runner = load_module("39_run_finite_187t_large_sheet_thermal_pte.py", "finite_187t_pair_thermal")
    if not (EA_Q_DIR / "FINITE_187T_W12_Q.json").is_file():
        code = invoke(
            optical_runner,
            [
                str(HERE / "33_run_v261_finite_multi_t_gaussian_q.py"),
                "--polarization", "Ea",
                "--output-dir", str(EA_Q_DIR),
            ],
        )
        if code != 0:
            raise RuntimeError("Ea optical solve failed")
    ea_optical_path = EA_Q_DIR / "FINITE_187T_W12_Q.json"
    eb_optical_path = EB_Q_DIR / "FINITE_187T_W12_Q_FINAL.json"
    optical = {
        "Ea": validated_optical(ea_optical_path),
        "Eb": validated_optical(eb_optical_path),
    }

    cuda_device = int(os.environ.get("CUDA_VISIBLE_PHYSICAL_DEVICE", "5"))
    thermal: dict[str, dict[str, object]] = {}
    for polarization, optical_path, q_dir in (
        ("Ea", ea_optical_path, EA_Q_DIR),
        ("Eb", eb_optical_path, EB_Q_DIR),
    ):
        case_output = PAIR_DIR / polarization
        result_path = case_output / f"FINITE_187T_LARGE_SHEET_THERMAL_PTE_{polarization}.json"
        if not result_path.is_file():
            code = invoke(
                thermal_runner,
                [
                    str(HERE / "39_run_finite_187t_large_sheet_thermal_pte.py"),
                    "--q-npz", str(q_dir / "finite_187T_w12_Q.npz"),
                    "--q-json", str(optical_path),
                    "--output-dir", str(case_output),
                    "--polarization", polarization,
                    "--cuda-device", str(cuda_device),
                ],
            )
            if code != 0:
                raise RuntimeError(f"{polarization} thermal solve failed")
        thermal[polarization] = json.loads(result_path.read_text())

    gates = {
        "both_optical_validated": all(
            item["status"] == "VALIDATED_FINITE_187T_W12_VOLUMETRIC_Q"
            for item in optical.values()
        ),
        "both_thermal_validated": all(
            str(item["status"]).startswith("VALIDATED") for item in thermal.values()
        ),
        "same_geometry": (
            optical["Ea"].get("contract", {}).get("array", optical["Ea"].get("array"))
            == optical["Eb"].get("contract", {}).get("array", optical["Eb"].get("array"))
        ),
        "same_report_incident_power": all(
            item["illumination"]["reported_incident_power_W"] == 285.0e-6
            for item in thermal.values()
        ),
    }
    summary = {
        "status": (
            "VALIDATED_FINITE_187T_EA_EB_OPTICAL_THERMAL_PAIR"
            if all(gates.values())
            else "FAILED_FINITE_187T_EA_EB_PAIR_GATE"
        ),
        "classification": (
            "paired finite nonperiodic Maxwell Q and identical large-sheet thermal/PTE diagnostic; "
            "not experimental finite-contact prediction"
        ),
        "axis_mapping": "Lumerical x=b, y=a, z=c",
        "cases": {
            pol: {
                "P_Q_native_W": optical[pol]["P_Q_native_W"],
                "closure_relative": optical[pol]["six_face_closure_relative"],
                "Tmax_K_at_285uW": thermal[pol]["thermal"]["Tmax_K"],
                "max_gradient_K_m": thermal[pol]["thermal"]["max_gradient_K_m"],
                "short_circuit_current_A": thermal[pol]["electrical"]["short_circuit_current_A"],
                "thermal_json": str(PAIR_DIR / pol / f"FINITE_187T_LARGE_SHEET_THERMAL_PTE_{pol}.json"),
            }
            for pol in ("Ea", "Eb")
        },
        "gates": gates,
    }
    PAIR_DIR.mkdir(parents=True, exist_ok=True)
    (PAIR_DIR / "FINITE_187T_EA_EB_OPTICAL_THERMAL_PAIR.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))
    return 0 if all(gates.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
