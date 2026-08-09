#!/usr/bin/env python3
"""Run 007: corrected-axis E||b PTE-magnitude optimization."""

from pathlib import Path
import os
import runpy
import sys

HERE = Path(__file__).resolve().parent
ENGINE = HERE.parent / "run_005_lowbeta_topology_exploration_pilot"
os.environ.update({
    "PTE_OPTIMIZATION_RUN_DIR": str(HERE),
    "PTE_OPTIMIZATION_RAW_ROOT": "/data/seunghyun/tairte4/raw_artifacts/run007_corrected_Eb_optimization_20260809",
    "PTE_OPTIMIZATION_TAG_PREFIX": "run007_Eb",
    "PTE_OPTIMIZATION_RUNNING_STATUS": "RUNNING_CORRECTED_EB_PTE_MAGNITUDE_OPTIMIZATION",
    "PTE_OPTIMIZATION_AXIS_CONTRACT": "lumerical_x_b_y_a",
    "PTE_OPTIMIZATION_POLARIZATION_LABEL": "E_parallel_b",
    "PTE_OPTIMIZATION_POLARIZATION_ANGLE_DEG": "0",
    "PTE_OPTIMIZATION_OBJECTIVE_SIGN": "-1",
    "PTE_OPTIMIZATION_INITIAL_RESULT": "/data/seunghyun/tairte4/raw_artifacts/run007_corrected_Eb_magnitude_initial_20260809/selected_full_latent_adjoint_preparation_result.json",
    "PTE_OPTIMIZATION_INITIAL_RESULT_SHA": "146e5b7989726d894f47d9f5c01d1cd18ab1e860a22ed09a8a08d7208434c4d2",
    "PTE_OPTIMIZATION_INITIAL_RAW": "/data/seunghyun/tairte4/raw_artifacts/run007_corrected_Eb_magnitude_initial_20260809/selected_full_latent_adjoint_preparation.npz",
    "PTE_OPTIMIZATION_INITIAL_RAW_SHA": "868f7aacb735e028f53f3c0df30ca498f05b6a12cc4cfdaa5e0fd9237a4d4dc4",
})
sys.path.insert(0, str(ENGINE))
runpy.run_path(str(ENGINE / "run_optimization.py"), run_name="__main__")
