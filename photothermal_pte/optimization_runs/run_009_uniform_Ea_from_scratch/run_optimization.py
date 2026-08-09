#!/usr/bin/env python3
"""Run009: corrected E||a optimization from exact uniform rho=0.5."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import runpy
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
ENGINE = HERE.parent / "run_005_lowbeta_topology_exploration_pilot"
INITIAL_DIR = Path(
    "/data/seunghyun/tairte4/raw_artifacts/run009_uniform_Ea_initial_20260809"
)
INITIAL_SOLVER_DIR = INITIAL_DIR / "solver_baseline"
INITIAL_RESULT = INITIAL_SOLVER_DIR / "selected_full_latent_adjoint_preparation_result.json"
INITIAL_RAW = INITIAL_SOLVER_DIR / "selected_full_latent_adjoint_preparation.npz"
INITIAL_AUDIT = INITIAL_DIR / "uniform_initial_audit.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_uniform_initial() -> None:
    for path in (INITIAL_RESULT, INITIAL_RAW, INITIAL_AUDIT):
        if not path.is_file():
            raise RuntimeError(f"missing Run009 uniform initial artifact: {path}")
    audit = json.loads(INITIAL_AUDIT.read_text())
    if audit.get("status") != "VALIDATED_EXACT_UNIFORM_INITIAL_DENSITY":
        raise RuntimeError("Run009 initial-density audit did not pass")
    result = json.loads(INITIAL_RESULT.read_text())
    if not result.get("passed") or float(result.get("beta", -1)) != 2.0:
        raise RuntimeError("Run009 uniform solver baseline did not pass")
    latent = np.asarray(np.load(INITIAL_RAW)["latent"], dtype=np.float64)
    expected = np.full((373, 373), 0.5, dtype=np.float64)
    if not np.array_equal(latent, expected):
        raise RuntimeError("Run009 must start from exact uniform latent=0.5")
    for field in ("filtered", "rho"):
        values = np.asarray(np.load(INITIAL_RAW)[field], dtype=np.float64)
        if not np.array_equal(values, expected):
            raise RuntimeError(f"Run009 initial {field} is not exact uniform 0.5")


verify_uniform_initial()
RAW_ROOT = "/data/seunghyun/tairte4/raw_artifacts/run009_uniform_Ea_optimization_20260809"
os.environ.update({
    "PTE_OPTIMIZATION_RUN_DIR": str(HERE),
    "PTE_OPTIMIZATION_RAW_ROOT": RAW_ROOT,
    "RUN005_STAGE_CAPS_FILE": f"{RAW_ROOT}/stage_caps.json",
    "PTE_OPTIMIZATION_TAG_PREFIX": "run009_Ea_uniform",
    "PTE_OPTIMIZATION_RUNNING_STATUS": "RUNNING_EXACT_UNIFORM_EA_OBJECTIVE_FIRST_OPTIMIZATION",
    "PTE_OPTIMIZATION_AXIS_CONTRACT": "lumerical_x_b_y_a",
    "PTE_OPTIMIZATION_POLARIZATION_LABEL": "E_parallel_a",
    "PTE_OPTIMIZATION_POLARIZATION_ANGLE_DEG": "90",
    "PTE_OPTIMIZATION_OBJECTIVE_SIGN": "-1",
    "PTE_OPTIMIZATION_INITIAL_RESULT": str(INITIAL_RESULT),
    "PTE_OPTIMIZATION_INITIAL_RESULT_SHA": sha256(INITIAL_RESULT),
    "PTE_OPTIMIZATION_INITIAL_RAW": str(INITIAL_RAW),
    "PTE_OPTIMIZATION_INITIAL_RAW_SHA": sha256(INITIAL_RAW),
    # Wide low-beta morphology envelopes keep the first stages objective-led.
    # Exact 500 nm DRC remains diagnostic until the documented high-beta phase.
    "PTE_OPTIMIZATION_BETA2_CAPS_JSON": "[0.002, 0.002]",
    "PTE_OPTIMIZATION_STAGE_MAX_UPDATES_JSON": "{\"2\":40,\"4\":32,\"8\":24,\"16\":20,\"32\":10,\"64\":8,\"128\":12}",
    "PTE_OPTIMIZATION_STAGE_BUDGET_POLICY": "fail_closed",
    "PTE_OPTIMIZATION_ALLOW_MINIMUM_MOVE_TRANSITION": "0",
    "PTE_OPTIMIZATION_ALLOW_LOW_BETA_MORPHOLOGY_TRANSITION": "0",
    "PTE_OPTIMIZATION_SMOOTH_FEASIBILITY_GATE_START_BETA": "8",
    "PTE_OPTIMIZATION_PHASEWISE_NONINCREASE_START_BETA": "8",
    "PTE_OPTIMIZATION_STAGE_CAP_TARGET_OCCUPANCY_JSON": "{\"4\":0.80,\"8\":0.88,\"16\":0.92,\"32\":1.05,\"64\":1.10,\"128\":1.15}",
})

sys.path.insert(0, str(ENGINE))
runpy.run_path(str(ENGINE / "run_optimization.py"), run_name="__main__")
