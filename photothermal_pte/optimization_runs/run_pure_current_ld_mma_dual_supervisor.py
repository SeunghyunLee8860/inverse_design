#!/usr/bin/env python3
"""Run pure-terminal-current LD_MMA for Ea and then Eb on one GPU.

No terminal-conductance/connectivity inequality is passed to either run.  The
top/bottom electrode weighting-potential boundary conditions remain part of
the electrical physics solved in every objective/gradient evaluation.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


REPOSITORY = Path(__file__).resolve().parents[2]
PARENT = Path("/data/seunghyun/tairte4/artifacts/tairte4_contact_anchored")
BASE_FSP = PARENT / "run012_uniform_rho0p5_Ea_forward_retry_20260810/tairte4_flake_forward_Ea.fsp"
BASE_SHA256 = "454fa83bc918b4db0e25d28f7debf23de38977038bd716c8d7dc539d6b3e3d83"
JACOBIAN = PARENT / "run012_component_yee_jacobian_retry_20260810"
PREFLIGHT = REPOSITORY / "photothermal_pte/optimization_runs/true_mma_preflight/TRUE_MMA_PREFLIGHT.json"
# Preserve the GPU-0 placement used by Run036 unless the caller deliberately
# overrides it.  A recovery must not silently move to a different device.
GPU = int(os.environ.get("TAIRTE4_PURE_CURRENT_LD_MMA_GPU", "0"))
STATUS = REPOSITORY / "photothermal_pte/optimization_runs/PURE_CURRENT_LD_MMA_DUAL_RUN_STATUS.json"
RUNS = (
    (
        "Run036",
        "Ea",
        PARENT / "run036_pure_current_ld_mma_morphology_from_beta1_Ea_20260810",
        REPOSITORY / "photothermal_pte/optimization_runs/run_036_pure_current_ld_mma_morphology_from_beta1_Ea_current_max",
        PARENT / "run036_pure_current_ld_mma_morphology_from_beta1_Ea_20260810/evaluation_0012_beta1_pure_current_ld_mma_latent.npz",
        True,
        PARENT / "run036_pure_current_ld_mma_morphology_from_beta1_Ea_20260810/evaluation_0012_retry1_beta1_pure_current_ld_mma/objective_gradient_result.json",
    ),
    (
        "Run037",
        "Eb",
        PARENT / "run037_pure_current_ld_mma_morphology_from_beta1_Eb_20260811",
        REPOSITORY / "photothermal_pte/optimization_runs/run_037_pure_current_ld_mma_morphology_from_beta1_Eb_current_max",
        None,
        False,
        None,
    ),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_status(status: str, **values: object) -> None:
    payload = {
        "schema": "pure-terminal-current-nlopt-ld-mma-dual-supervisor-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "gpu": GPU,
        "algorithm": "NLopt LD_MMA",
        "objective": "signed full-flake terminal PTE current",
        "top_bottom_weighting_boundaries": {"top": 1.0, "bottom": 0.0},
        "terminal_conductance_constraint": False,
        "terminal_conductance_role": "diagnostic_only",
        "manual_move_limit": None,
        **values,
    }
    temporary = STATUS.with_suffix(STATUS.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(STATUS)
    print(json.dumps(payload), flush=True)


def passed(path: Path) -> bool:
    return path.is_file() and bool(json.loads(path.read_text()).get("passed"))


def main() -> int:
    if not BASE_FSP.is_file() or sha256(BASE_FSP) != BASE_SHA256:
        raise RuntimeError("immutable base FSP is missing or SHA-mismatched")
    if not passed(PREFLIGHT):
        raise RuntimeError("existing optical/thermal/electrical/AD-FD preflight is not passed")
    certificate = JACOBIAN / "component_yee_jacobian_result.json"
    if not passed(certificate):
        raise RuntimeError("component-Yee Jacobian certificate is not passed")
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = str(GPU)
    environment["TAIRTE4_TOPOLOGY_GEOMETRY"] = "contact_anchored"
    conda_library = "/home/eidl/miniconda3/envs/EIDL-Lumapi/lib"
    current_library_path = environment.get("LD_LIBRARY_PATH", "")
    environment["LD_LIBRARY_PATH"] = (
        conda_library if not current_library_path
        else f"{conda_library}:{current_library_path}"
    )
    for (
        label,
        polarization,
        raw_root,
        published,
        initial_latent,
        recovery_append,
        recovery_validation,
    ) in RUNS:
        final = published / "FINAL_RESULT.json"
        if passed(final):
            continue
        if raw_root.exists() and any(raw_root.iterdir()) and not recovery_append:
            raise RuntimeError(
                f"{label} raw root already contains data; refusing ambiguous overwrite: {raw_root}"
            )
        if recovery_append and not raw_root.is_dir():
            raise RuntimeError(f"{label} recovery raw root is missing: {raw_root}")
        raw_root.mkdir(parents=True, exist_ok=True)
        published.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            "-m",
            "photothermal_pte.optimization_runs.tairte4_flake_topology.run_pure_current_ld_mma_optimization",
            "--polarization", polarization,
            "--raw-root", str(raw_root),
            "--published-dir", str(published),
            "--gpu", str(GPU),
            "--base-fsp", str(BASE_FSP),
            "--base-sha256", BASE_SHA256,
            "--jacobian-dir", str(JACOBIAN),
            "--constraint-device", "cuda:0",
        ]
        if initial_latent is not None:
            if not initial_latent.is_file():
                raise RuntimeError(f"{label} warm-start latent is missing: {initial_latent}")
            command.extend(["--initial-latent-npz", str(initial_latent)])
        if recovery_append:
            command.append("--recovery-append")
            if recovery_validation is not None:
                command.extend(["--recovery-validation-result", str(recovery_validation)])
        write_status(
            "RUNNING_PURE_CURRENT_NLOPT_LD_MMA",
            run=label,
            polarization=polarization,
            native_ld_mma_warm_restart=initial_latent is not None,
            recovery_append=recovery_append,
            command=command,
        )
        completed = subprocess.run(command, cwd=REPOSITORY, env=environment)
        if completed.returncode or not passed(final):
            write_status(
                "BLOCKED_PURE_CURRENT_NLOPT_LD_MMA_RUN",
                run=label,
                polarization=polarization,
                returncode=completed.returncode,
            )
            raise RuntimeError(f"{label} pure-current NLopt LD_MMA run failed closed")
    write_status("VALIDATED_PURE_CURRENT_NLOPT_LD_MMA_EA_EB_OPTIMIZATIONS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
