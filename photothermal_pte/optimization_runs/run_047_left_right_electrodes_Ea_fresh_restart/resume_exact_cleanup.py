#!/usr/bin/env python3
"""Resume Run047 at its immutable beta=32 checkpoint after cleanup bug fix."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from photothermal_pte.optimization_runs.tairte4_flake_topology.optimization_support import (
    exact_binary_audit,
)
from photothermal_pte.optimization_runs.tairte4_flake_topology.run_ansys_dfm_ld_mma_optimization import (
    REFERENCE_INCIDENT_POWER_W,
    SCHEMA,
    evaluate_exact_cleanup_candidates,
    final_geometry_gate,
)


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]
PYTHON = Path("/home/eidl/miniconda3/envs/EIDL-Lumapi/bin/python")
CONDA_LIBRARY_DIR = PYTHON.parents[1] / "lib"
ROOT = Path("/data/seunghyun/tairte4/artifacts/tairte4_left_right_contact_anchored")
RAW = ROOT / "run047_Ea_fresh_current_max"
CHECKPOINT = RAW / "stage_0006_beta32.npz"
CHECKPOINT_SHA256 = "7e199b2bb798b3815e07f5ca42e043079893853f34e01913044e8a04fde7dd9c"
BASE_FSP = ROOT / "uniform_rho0p5_Ea_forward_queued/attempt_0002/tairte4_flake_forward_Ea.fsp"
BASE_SHA256 = "6274627f8e84cc61a8b5925472fc131041e7662b06d77141f3b52353d3578aa6"
JACOBIAN = ROOT / "component_yee_jacobian_v1"
RESULTS = HERE / "results"
STATE = HERE / "PIPELINE_STATE.json"
REFERENCE_EVALUATION = RESULTS / "evaluation_0069.json"
GPU = int(os.environ.get("RUN047_GPU", "5"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_state(status: str, **extra: object) -> None:
    payload = {
        "stage": "exact_cleanup_recovery",
        "status": status,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
        "gpu": GPU,
        "checkpoint": str(CHECKPOINT),
        "restart_beta": 32.0,
        **extra,
    }
    temporary = STATE.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(STATE)


def main() -> int:
    if not CHECKPOINT.is_file() or sha256(CHECKPOINT) != CHECKPOINT_SHA256:
        raise RuntimeError("Run047 beta=32 recovery checkpoint is missing or changed")
    if not BASE_FSP.is_file() or sha256(BASE_FSP) != BASE_SHA256:
        raise RuntimeError("Run047 base FSP is missing or changed")
    final = RESULTS / "FINAL_RESULT.json"
    if final.exists():
        raise RuntimeError("Run047 already has a final result")
    cleanup = RAW / "forced_exact_500nm_cleanup"
    if not cleanup.is_dir() or any(cleanup.iterdir()):
        raise RuntimeError("expected the failed Run047 cleanup directory to be empty")

    env = dict(os.environ)
    inherited = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = ":".join(
        part for part in (str(CONDA_LIBRARY_DIR), inherited) if part
    )
    env.update(
        {
            "PYTHONPATH": str(REPOSITORY),
            "TAIRTE4_TOPOLOGY_GEOMETRY": "left_right_contact_anchored",
            "CUDA_VISIBLE_DEVICES": str(GPU),
            "LUMERICAL_LICENSE_RETRY_SECONDS": "30",
            "LUMERICAL_GPU_ENGINE_LOCK": "/tmp/seunghyun_lumerical_fdtd_gpu_engine.lock",
            "XDG_CONFIG_HOME": "/tmp/seunghyun_lumerical_run047_cleanup",
            "MPLCONFIGDIR": "/tmp/seunghyun_matplotlib_run047_cleanup",
        }
    )
    os.environ.clear()
    os.environ.update(env)
    if not REFERENCE_EVALUATION.is_file():
        raise RuntimeError("Run047 evaluation 69 publication is missing")
    reference = json.loads(REFERENCE_EVALUATION.read_text())
    with np.load(CHECKPOINT) as loaded:
        rho = np.asarray(loaded["rho"], dtype=np.float64)

    write_state("running", action="direct_exact_cleanup_from_beta32_checkpoint")
    forced = evaluate_exact_cleanup_candidates(
        rho,
        raw_root=RAW,
        base_fsp=BASE_FSP,
        base_sha256=BASE_SHA256,
        polarization="Ea",
        gpu=GPU,
        reference_objective_A=float(reference["objective_A"]),
    )
    selected = str(forced["selected"])
    selected_row = forced["candidates"][selected]
    with np.load(selected_row["density"]["path"]) as loaded:
        binary = np.asarray(loaded["rho"], dtype=np.float64)
    exact, _ = exact_binary_audit(binary)
    if not exact["passed"]:
        raise RuntimeError("selected Run047 cleanup candidate failed exact 500 nm audit")
    binary_path = RAW / "final_exact_binary_density.npz"
    np.savez_compressed(binary_path, rho=binary)
    binary_result = selected_row["result"]
    result = {
        "schema": SCHEMA,
        "passed": bool(binary_result.get("passed")),
        "status": (
            "VALIDATED_ANSYS_STYLE_DFM_LD_MMA_EXACT_BINARY_PTE_OPTIMIZATION"
            if binary_result.get("passed")
            else "COMPLETED_EXACT_BINARY_WITH_OBJECTIVE_PRESERVATION_GATE_FAILED"
        ),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "polarization": "Ea",
        "algorithm": "NLopt LD_MMA",
        "objective": "signed full-flake terminal PTE current",
        "reference_incident_power_W": REFERENCE_INCIDENT_POWER_W,
        "final_beta": 32.0,
        "full_physics_evaluations": 69,
        "completed_stages": 6,
        "final_geometry_gate": final_geometry_gate(binary),
        "binary_result": binary_result,
        "manual_move_limit": None,
        "connectivity_constraint": False,
        "symmetry_constraint": False,
        "volume_constraint": False,
        "posthoc_morphology_repair": True,
        "forced_exact_cleanup": forced,
        "recovery": {
            "reason": "solid_first cleanup cycle previously aborted before void_first",
            "checkpoint": str(CHECKPOINT),
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "reference_evaluation": 69,
            "additional_gradient_evaluations": 0,
        },
    }
    manifest_path = RESULTS / "RAW_ARTIFACT_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["forced_exact_cleanup"] = forced
    manifest["cleanup_recovery"] = result["recovery"]
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    (RESULTS / "FORCED_EXACT_CLEANUP.json").write_text(
        json.dumps(forced, indent=2) + "\n"
    )
    final.write_text(json.dumps(result, indent=2) + "\n")
    write_state("complete", final_status=result.get("status"), passed=result.get("passed"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
