#!/usr/bin/env python3
"""Extract finite-187T Q from the already completed, saved GPU FSP."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[3]
RAW = Path(
    "/home/seunghyun/tairte4/raw_artifacts/"
    "paper_tairte4_finite_187T_w12_Q_11p825um_Eb"
)
FSP = RAW / "finite_187T_w12_Q.fsp"
NPZ = RAW / "finite_187T_w12_Q.npz"
RESULT = RAW / "FINITE_187T_W12_Q_FINAL.json"
WAVELENGTH_M = 11.825e-6
FREQUENCY_HZ = 299_792_458.0 / WAVELENGTH_M

if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from photothermal_pte.finite_inverse_design.native_yee_q import extract_native_yee_q  # noqa: E402
from photothermal_pte.finite_inverse_design.probe_v261_cpu_tfsf_device import (  # noqa: E402
    PABS_FIELD,
    PABS_GROUP,
    PABS_INDEX,
)
from photothermal_pte.validation.paper_ir_sanity import validate_paper_ir_source_only_gpu as audit  # noqa: E402


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    driver = load_module(HERE / "33_run_v261_finite_multi_t_gaussian_q.py", "finite_187T_driver_helpers")
    os.environ["VC_LUMERICAL_ROOT"] = str(audit.APPROVED_ROOT)
    os.environ["LUMERICAL_ROOT"] = str(audit.APPROVED_ROOT)
    os.environ["LUMERICAL_PYTHONPATH"] = str(audit.APPROVED_API)
    os.environ["PATH"] = f"{audit.APPROVED_ROOT / 'bin'}:{os.environ.get('PATH', '')}"
    sys.path.insert(0, str(audit.APPROVED_API))
    import lumapi

    fdtd = lumapi.FDTD(hide=True, serverArgs={"platform": "offscreen"})
    try:
        fdtd.load(str(FSP))
        source_power = driver.scalar(
            fdtd.sourcepower(FREQUENCY_HZ, 2, "finite_T_scalar_Gaussian"),
            "sourcepower",
        )
        six_face = driver.face_fluxes(fdtd, driver.existing_flux_box(), source_power)
        fdtd.runanalysis(PABS_GROUP)
        q = extract_native_yee_q(
            fdtd,
            field_monitor=PABS_FIELD,
            index_monitor=PABS_INDEX,
            wavelength_m=WAVELENGTH_M,
        )
        stage1 = REPOSITORY / "photothermal_pte/validation/photothermal_stage1"
        if str(stage1) not in sys.path:
            sys.path.insert(0, str(stage1))
        common_module = load_module(
            stage1 / "27_validate_finite_2um_optical_q.py",
            "finite_187T_common_q_extract",
        )
        common_module.PABS_FIELD = PABS_FIELD
        common_module.PABS_INDEX = PABS_INDEX
        common = common_module.common_grid_component_q(fdtd, FREQUENCY_HZ)
        p_native = float(q["P_Q_W"])
        p_pabs = driver.scalar(
            fdtd.getresult(PABS_GROUP, "Pabs_total")["Pabs_total"],
            "Pabs_total",
        ) * source_power
        p_six = float(six_face["net_inward_power_W"])
        closure = abs(p_native - p_six) / max(abs(p_six), np.finfo(float).tiny)
        pabs_delta = abs(p_native - p_pabs) / max(abs(p_pabs), np.finfo(float).tiny)
        negative = {
            component: int(np.count_nonzero(np.asarray(q["Q_components"][component]) < 0.0))
            for component in "xyz"
        }
        finite = all(
            np.all(np.isfinite(np.asarray(q["Q_components"][component])))
            for component in "xyz"
        )
        q_total_common = np.asarray(common["Q_native_W_m3"], float)
        hotspot_index = np.unravel_index(int(np.argmax(q_total_common)), q_total_common.shape)
        hotspot = {
            "x_m": float(common["x_m"][hotspot_index[0]]),
            "y_m": float(common["y_m"][hotspot_index[1]]),
            "z_m": float(common["z_m"][hotspot_index[2]]),
            "Q_W_m3": float(q_total_common[hotspot_index]),
            "classification": "linearly collocated common-grid diagnostic; native Yee power is authoritative",
        }
        arrays: dict[str, np.ndarray] = {
            "common_x_m": np.asarray(common["x_m"]),
            "common_y_m": np.asarray(common["y_m"]),
            "common_z_m": np.asarray(common["z_m"]),
            "Q_common_W_m3": q_total_common,
        }
        for component in "xyz":
            arrays[f"Q{component}_W_m3"] = np.asarray(q["Q_components"][component])
            for axis in "xyz":
                arrays[f"Q{component}_{axis}_m"] = np.asarray(q["native_coordinates"][component][axis])
        np.savez_compressed(NPZ, **arrays)
        log = audit.log_audit(RAW)
        gates = {
            "GPU_completed": bool(log["simulation_completed_successfully"]),
            "auto_shutoff_lt_1e_5": log["final_auto_shutoff"] is not None and log["final_auto_shutoff"] < 1e-5,
            "six_face_closure_lt_0p5pct": closure < 0.005,
            "native_vs_pabs_lt_0p5pct": pabs_delta < 0.005,
            "all_Q_arrays_finite": bool(finite),
            "no_negative_Q": sum(negative.values()) == 0,
        }
        payload = {
            "status": "VALIDATED_FINITE_187T_W12_VOLUMETRIC_Q" if all(gates.values()) else "FAILED_FINITE_187T_W12_VOLUMETRIC_Q_GATE",
            "classification": "read-only extraction from completed finite nonperiodic GPU FSP",
            "solver_version": str(fdtd.version()),
            "source": {
                "wavelength_um": 11.825,
                "polarization": "E||b; Lumerical x=b",
                "target_realized_w0_um": 12.0,
                "Lumerical_source_object_w0_um": 11.85757138436561,
                "span_um": 50.0,
            },
            "array": {"nx": 11, "ny": 17, "count": 187, "span_um": [16.5, 17.0]},
            "domain": {"x_um": 60.0, "y_um": 60.0, "boundaries": "six PML, 24 layers"},
            "control_volume_bounds_m": driver.CONTROL_BOUNDS_M,
            "source_power_W": source_power,
            "P_Q_native_W": p_native,
            "P_Q_pabs_W": p_pabs,
            "P_six_face_W": p_six,
            "six_face_closure_relative": closure,
            "native_vs_pabs_relative": pabs_delta,
            "Q_component_power_native_W": q["component_power_W"],
            "common_grid_component_power_W": common["common_component_power_W"],
            "common_grid_interpolation_relative_error": common["component_interpolation_relative_error"],
            "hotspot": hotspot,
            "negative_Q_cell_count": negative,
            "all_Q_arrays_finite": bool(finite),
            "six_face": six_face,
            "log_audit": log,
            "gates": gates,
            "scope_exclusions": ["thermal", "weighting potential", "PTE", "adjoint", "optimization"],
            "raw_artifacts": [
                {"path": str(path), "size_bytes": path.stat().st_size, "sha256": driver.sha256(path)}
                for path in (FSP, NPZ)
            ],
        }
        RESULT.write_text(json.dumps(payload, indent=2, default=str) + "\n")
        print(json.dumps(payload, indent=2, default=str))
        return 0 if payload["status"].startswith("VALIDATED") else 1
    finally:
        fdtd.close()


if __name__ == "__main__":
    raise SystemExit(main())
