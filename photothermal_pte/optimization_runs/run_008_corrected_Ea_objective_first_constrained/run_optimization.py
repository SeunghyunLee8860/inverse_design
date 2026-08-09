#!/usr/bin/env python3
"""Run 008: corrected E||a, objective-first then constrained continuation."""

from pathlib import Path
import os
import runpy
import sys

HERE = Path(__file__).resolve().parent
ENGINE = HERE.parent / "run_005_lowbeta_topology_exploration_pilot"
RAW_ROOT = "/data/seunghyun/tairte4/raw_artifacts/run008_corrected_Ea_objective_first_20260809"

os.environ.update({
    "PTE_OPTIMIZATION_RUN_DIR": str(HERE),
    "PTE_OPTIMIZATION_RAW_ROOT": RAW_ROOT,
    "RUN005_STAGE_CAPS_FILE": f"{RAW_ROOT}/stage_caps.json",
    "PTE_OPTIMIZATION_TAG_PREFIX": "run008_Ea",
    "PTE_OPTIMIZATION_RUNNING_STATUS": "RUNNING_CORRECTED_EA_OBJECTIVE_FIRST_CONSTRAINED_OPTIMIZATION",
    "PTE_OPTIMIZATION_AXIS_CONTRACT": "lumerical_x_b_y_a",
    "PTE_OPTIMIZATION_POLARIZATION_LABEL": "E_parallel_a",
    "PTE_OPTIMIZATION_POLARIZATION_ANGLE_DEG": "90",
    "PTE_OPTIMIZATION_OBJECTIVE_SIGN": "-1",
    "PTE_OPTIMIZATION_INITIAL_RESULT": "/data/seunghyun/tairte4/raw_artifacts/run006_corrected_Ea_magnitude_initial_20260809/selected_full_latent_adjoint_preparation_result.json",
    "PTE_OPTIMIZATION_INITIAL_RESULT_SHA": "317ea871743b804b213eaa71b76fa33c08c9302fd144b41ccc4b8cad9d57dbd3",
    "PTE_OPTIMIZATION_INITIAL_RAW": "/data/seunghyun/tairte4/raw_artifacts/run006_corrected_Ea_magnitude_initial_20260809/selected_full_latent_adjoint_preparation.npz",
    "PTE_OPTIMIZATION_INITIAL_RAW_SHA": "a398687c45cb379fff7ed183b5769450220307ea410dfbe426a12f2938bf4ab6",
    # The initial design occupies 80% of both beta=2 caps, so topology search
    # starts feasible and the morphology multipliers do not dictate step one.
    "PTE_OPTIMIZATION_BETA2_CAPS_JSON": "[0.0014902195929630476, 0.000032042875522131275]",
    # Low-beta stages must satisfy the measured plateau gates. A work budget
    # is only a fail-closed watchdog and never authorizes an early beta jump.
    "PTE_OPTIMIZATION_STAGE_MAX_UPDATES_JSON": "{\"2\":40,\"4\":32,\"8\":24,\"16\":20,\"32\":10,\"64\":8,\"128\":12}",
    "PTE_OPTIMIZATION_STAGE_BUDGET_POLICY": "fail_closed",
    # Do not let the legacy move-size or thresholded-pixel heuristics advance
    # low beta while the solver-backed FOM is still improving.
    "PTE_OPTIMIZATION_ALLOW_MINIMUM_MOVE_TRANSITION": "0",
    "PTE_OPTIMIZATION_ALLOW_LOW_BETA_MORPHOLOGY_TRANSITION": "0",
    # At beta 2/4, constraints remain bounded candidate guards but cannot hold
    # the stage after the objective and density have genuinely plateaued.
    "PTE_OPTIMIZATION_SMOOTH_FEASIBILITY_GATE_START_BETA": "8",
    # Preserve topology freedom at beta 2/4. Begin phase-wise nonincrease at 8,
    # then retain the existing exact nonincrease gate from beta 32.
    "PTE_OPTIMIZATION_PHASEWISE_NONINCREASE_START_BETA": "8",
    "PTE_OPTIMIZATION_STAGE_CAP_TARGET_OCCUPANCY_JSON": "{\"4\":0.80,\"8\":0.88,\"16\":0.92,\"32\":1.05,\"64\":1.10,\"128\":1.15}",
})

sys.path.insert(0, str(ENGINE))
runpy.run_path(str(ENGINE / "run_optimization.py"), run_name="__main__")
