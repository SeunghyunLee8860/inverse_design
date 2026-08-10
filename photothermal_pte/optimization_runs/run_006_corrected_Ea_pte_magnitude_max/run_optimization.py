#!/usr/bin/env python3
"""Run 006: corrected-axis E||a PTE-magnitude optimization."""

from pathlib import Path
import os
import runpy
import sys

HERE = Path(__file__).resolve().parent
ENGINE = HERE.parent / "run_005_lowbeta_topology_exploration_pilot"
os.environ.update({
    "PTE_OPTIMIZATION_RUN_DIR": str(HERE),
    "PTE_OPTIMIZATION_RAW_ROOT": "/data/seunghyun/tairte4/raw_artifacts/run006_corrected_Ea_optimization_20260809",
    "PTE_OPTIMIZATION_TAG_PREFIX": "run006_Ea",
    "PTE_OPTIMIZATION_RUNNING_STATUS": "RUNNING_CORRECTED_EA_PTE_MAGNITUDE_OPTIMIZATION",
    "PTE_OPTIMIZATION_AXIS_CONTRACT": "lumerical_x_b_y_a",
    "PTE_OPTIMIZATION_POLARIZATION_LABEL": "E_parallel_a",
    "PTE_OPTIMIZATION_POLARIZATION_ANGLE_DEG": "90",
    "PTE_OPTIMIZATION_OBJECTIVE_SIGN": "-1",
    "PTE_OPTIMIZATION_INITIAL_RESULT": "/data/seunghyun/tairte4/raw_artifacts/run006_corrected_Ea_magnitude_initial_20260809/selected_full_latent_adjoint_preparation_result.json",
    "PTE_OPTIMIZATION_INITIAL_RESULT_SHA": "317ea871743b804b213eaa71b76fa33c08c9302fd144b41ccc4b8cad9d57dbd3",
    "PTE_OPTIMIZATION_INITIAL_RAW": "/data/seunghyun/tairte4/raw_artifacts/run006_corrected_Ea_magnitude_initial_20260809/selected_full_latent_adjoint_preparation.npz",
    "PTE_OPTIMIZATION_INITIAL_RAW_SHA": "a398687c45cb379fff7ed183b5769450220307ea410dfbe426a12f2938bf4ab6",
})
sys.path.insert(0, str(ENGINE))
runpy.run_path(str(ENGINE / "run_optimization.py"), run_name="__main__")
