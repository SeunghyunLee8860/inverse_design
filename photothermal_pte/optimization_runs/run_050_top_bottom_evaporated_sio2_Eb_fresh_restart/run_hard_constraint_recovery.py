#!/usr/bin/env python3
"""Recover Run050 performance from an exact-feasible inverse-filter seed."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]
PYTHON = Path("/home/eidl/miniconda3/envs/EIDL-Lumapi/bin/python")
CONDA_LIBRARY_DIR = PYTHON.parents[1] / "lib"
ARTIFACT_ROOT = Path("/data/seunghyun/tairte4/artifacts/tairte4_contact_anchored")
SOURCE_RUN = ARTIFACT_ROOT / "run050_Eb_evaporated_fresh_current_max"
RECOVERY_ROOT = ARTIFACT_ROOT / "run050_Eb_evaporated_hard_constraint_recovery_v1"
RAW_ROOT = RECOVERY_ROOT / "optimization_v2"
SEED = RECOVERY_ROOT / "exact_feasible_latent_seed_beta8_v2.npz"
SEED_REPORT = RECOVERY_ROOT / "EXACT_FEASIBLE_LATENT_SEED_REPORT_v2.json"
PUBLISHED = HERE / "hard_constraint_recovery_v2" / "results"
BASE_FSP = ARTIFACT_ROOT / (
    "production_input_uniform_rho0p5_Ea_forward_v1/"
    "tairte4_flake_forward_Ea.fsp"
)
BASE_SHA256 = "454fa83bc918b4db0e25d28f7debf23de38977038bd716c8d7dc539d6b3e3d83"
JACOBIAN_ROOT = ARTIFACT_ROOT / "production_input_component_yee_jacobian_v1"
GPU = int(os.environ.get("RUN050_GPU", "0"))


def environment() -> dict[str, str]:
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = ":".join(
        part
        for part in (str(CONDA_LIBRARY_DIR), env.get("LD_LIBRARY_PATH", ""))
        if part
    )
    env.update(
        {
            "PYTHONPATH": str(REPOSITORY),
            "TAIRTE4_TOPOLOGY_GEOMETRY": "contact_anchored",
            "TAIRTE4_SIO2_INTERFACE_SCENARIO": "evaporated",
            "CUDA_VISIBLE_DEVICES": str(GPU),
            "LUMERICAL_LICENSE_RETRY_SECONDS": "30",
            "LUMERICAL_GPU_ENGINE_LOCK": "/tmp/seunghyun_lumerical_fdtd_gpu_engine.lock",
            "XDG_CONFIG_HOME": "/tmp/seunghyun_lumerical_run050_hard_recovery_v2",
            "MPLCONFIGDIR": "/tmp/seunghyun_matplotlib_run050_hard_recovery_v2",
        }
    )
    return env


def main() -> int:
    if not SEED.is_file() or not SEED_REPORT.is_file():
        raise RuntimeError("exact-feasible inverse-filter seed is missing")
    seed_report = json.loads(SEED_REPORT.read_text())
    if not seed_report.get("passed"):
        raise RuntimeError("exact-feasible inverse-filter seed did not pass")
    if int(seed_report["exact_500nm_audit"]["total_bad_cell_count"]) != 0:
        raise RuntimeError("seed report contains an exact 500-nm violation")
    history_path = PUBLISHED / "optimization_history.json"
    manifest_path = PUBLISHED / "RAW_ARTIFACT_MANIFEST.json"
    recovery_append = history_path.is_file() and manifest_path.is_file()
    initial_latent = SEED
    output_slug = "run050_exact_feasible_hard_constraint_recovery"
    if recovery_append:
        history = json.loads(history_path.read_text())
        manifest = json.loads(manifest_path.read_text())
        if not history:
            raise RuntimeError("recovery history exists but is empty")
        evaluation_id = int(history[-1]["evaluation_id"])
        entry = manifest["evaluations"][f"{evaluation_id:04d}"]
        initial_latent = Path(entry["latent_design"]["path"])
        if not initial_latent.is_file():
            raise RuntimeError("latest fully evaluated latent checkpoint is missing")
        output_slug = f"run050_hard_constraint_resume_after_{evaluation_id:04d}"
    command = [
        str(PYTHON),
        "-m",
        "photothermal_pte.optimization_runs.tairte4_flake_topology."
        "run_ansys_dfm_ld_mma_optimization",
        "--polarization",
        "Eb",
        "--raw-root",
        str(RAW_ROOT),
        "--published-dir",
        str(PUBLISHED),
        "--gpu",
        str(GPU),
        "--base-fsp",
        str(BASE_FSP),
        "--base-sha256",
        BASE_SHA256,
        "--jacobian-dir",
        str(JACOBIAN_ROOT),
        "--constraint-device",
        "cuda:0",
        "--initial-latent-npz",
        str(initial_latent),
        "--start-beta",
        "8",
        "--hard-morphology-constraints",
        "--hard-cap-relative-slack",
        "0.01",
        "--hard-rho-init",
        "0.01",
        "--output-slug",
        output_slug,
    ]
    if recovery_append:
        command.append("--recovery-append")
    print(
        json.dumps(
            {
                "purpose": "recover PTE current while retaining explicit solid/void LD_MMA inequalities",
                "source_run": str(SOURCE_RUN),
                "seed": str(initial_latent),
                "seed_exact_bad_nodes": 0,
                "recovery_append": recovery_append,
                "start_beta": 8,
                "hard_constraints": ["500nm_solid_opening", "500nm_void_opening"],
                "scalar_dfm_penalty_weight": 0.0,
                "hard_constraint_ccsa_parameters": {"rho_init": 0.01},
                "gpu": GPU,
                "command": command,
            },
            indent=2,
        ),
        flush=True,
    )
    return subprocess.run(command, cwd=REPOSITORY, env=environment()).returncode


if __name__ == "__main__":
    raise SystemExit(main())
