#!/usr/bin/env python3
"""Run the corrected Z-M2 linear-polarization pair as one fail-closed job."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
RAW_ROOT = Path("/home/seunghyun/tairte4/raw_artifacts")
PAIR_DIR = RAW_ROOT / "paper_z2022_m2_figure_digitized_ea_eb_pair_v2_matched_cv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_runner():
    path = HERE / "41_run_v261_z2022_m2_selected_q.py"
    spec = importlib.util.spec_from_file_location("z2022_selected_q_pair_runner", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run_case(runner, polarization: str, output: Path, gpu: str) -> dict[str, object]:
    if output.exists() and (output / "Z2022_M2_selected_Q.json").is_file():
        payload = json.loads((output / "Z2022_M2_selected_Q.json").read_text())
        if str(payload.get("status", "")).startswith("COMPLETED"):
            return payload
        raise RuntimeError(f"refusing to overwrite incomplete case: {output}")
    old_argv = sys.argv[:]
    try:
        sys.argv = [
            str(HERE / "41_run_v261_z2022_m2_selected_q.py"),
            "--output-dir", str(output),
            "--gpu-device", gpu,
            "--handedness", "LH",
            "--polarization", polarization,
            "--geometry-variant", "figure_axis_corrected_v2",
            "--wavelength-um", "5.3",
            "--duration-ps", "6.0",
        ]
        code = int(runner.main())
    finally:
        sys.argv = old_argv
    payload = json.loads((output / "Z2022_M2_selected_Q.json").read_text())
    if code != 0 or not str(payload.get("status", "")).startswith("COMPLETED"):
        raise RuntimeError(f"{polarization} failed: {payload.get('status')}")
    return payload


def main() -> int:
    PAIR_DIR.mkdir(parents=True, exist_ok=True)
    gpu = os.environ.get("LUMERICAL_SESSION_GPU_DEVICE", "GPU 0")
    runner = load_runner()
    cases = {
        "Ea": run_case(runner, "y_a", RAW_ROOT / "paper_z2022_m2_figure_digitized_Ea_5p3um_v2_matched_cv", gpu),
        "Eb": run_case(runner, "x_b", RAW_ROOT / "paper_z2022_m2_figure_digitized_Eb_5p3um_v2_matched_cv", gpu),
    }
    pair_gates = {
        "both_completed": all(str(item["status"]).startswith("COMPLETED") for item in cases.values()),
        "same_geometry": cases["Ea"]["geometry"] == cases["Eb"]["geometry"],
        "same_wavelength": cases["Ea"]["wavelength_um"] == cases["Eb"]["wavelength_um"],
        "both_closure_lt_0p5pct": all(float(item["closure_relative"]) < 0.005 for item in cases.values()),
        "both_auto_shutoff_lt_1e_5": all(
            float(item["log_audit"]["final_auto_shutoff"]) < 1.0e-5
            for item in cases.values()
        ),
    }
    payload = {
        "status": (
            "VALIDATED_Z2022_M2_FIGURE_DIGITIZED_EA_EB_OPTICAL_PAIR"
            if all(pair_gates.values())
            else "FAILED_Z2022_M2_FIGURE_DIGITIZED_EA_EB_PAIR_GATE"
        ),
        "classification": (
            "paper-dimension, figure-axis-corrected, corner-joined digitized closure; "
            "not exact author CAD"
        ),
        "axis_mapping": "Lumerical x=b, y=a, z=c",
        "cases": {
            key: {
                "status": value["status"],
                "polarization": value["polarization"],
                "P_Q_W": value["P_Q_pabs_periodic_W"],
                "P_flux_W": value["P_flux_absorbed_W"],
                "closure_relative": value["closure_relative"],
                "Q_component_power_native_W": value["Q_component_power_native_W"],
                "raw_artifacts": value["raw_artifacts"],
            }
            for key, value in cases.items()
        },
        "gates": pair_gates,
        "scope_exclusions": ["thermal", "weighting potential", "PTE", "adjoint", "optimization"],
    }
    json_path = PAIR_DIR / "Z2022_M2_FIGURE_DIGITIZED_EA_EB_PAIR.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    payload["summary_sha256"] = sha256(json_path)
    print(json.dumps(payload, indent=2))
    return 0 if all(pair_gates.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
