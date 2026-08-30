#!/usr/bin/env python3
"""Fresh top/bottom-electrode Ea optimization with evaporated-SiO2 G."""

from __future__ import annotations

from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]
PYTHON = Path("/home/eidl/miniconda3/envs/EIDL-Lumapi/bin/python")
CONDA_LIBRARY_DIR = PYTHON.parents[1] / "lib"
ARTIFACT_ROOT = Path(
    "/data/seunghyun/tairte4/artifacts/tairte4_contact_anchored"
)
BASE_FSP = ARTIFACT_ROOT / (
    "production_input_uniform_rho0p5_Ea_forward_v1/"
    "tairte4_flake_forward_Ea.fsp"
)
BASE_SHA256 = "454fa83bc918b4db0e25d28f7debf23de38977038bd716c8d7dc539d6b3e3d83"
JACOBIAN_ROOT = ARTIFACT_ROOT / "production_input_component_yee_jacobian_v1"
ADFD_ROOT = ARTIFACT_ROOT / "combined_adfd_Ea_evaporated_run049_precheck"
ADFD_RESULT = ADFD_ROOT / "tairte4_flake_combined_adfd.json"
OPTIMIZATION_ROOT = ARTIFACT_ROOT / "run049_Ea_evaporated_fresh_current_max"
RESULTS = HERE / "results"
STATE = HERE / "PIPELINE_STATE.json"
LOCK = Path("/tmp/seunghyun_run049_top_bottom_evaporated_Ea.lock")
GPU = int(os.environ.get("RUN049_GPU", "5"))
INTERFACE_SCENARIO = "evaporated"
INTERFACE_G_W_M2K = 7.37e4


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_state(stage: str, **extra: object) -> None:
    payload = {
        "stage": stage,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
        "gpu": GPU,
        "polarization": "Ea",
        "polarization_angle_deg": 90,
        "axis_contract": "Lumerical x=b, y=a, z=c",
        "electrical_terminals": {"bottom": 0.0, "top": 1.0},
        "geometry_mode": "contact_anchored",
        "thermal_interface_scenario": INTERFACE_SCENARIO,
        "G_TaIrTe4_SiO2_W_m2K": INTERFACE_G_W_M2K,
        "fresh_restart": True,
        "initial_density": 0.5,
        "warm_restart": False,
        "mma_internal_state_reused": False,
        **extra,
    }
    STATE.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATE.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(STATE)


def environment() -> dict[str, str]:
    env = dict(os.environ)
    inherited_library_path = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = ":".join(
        part for part in (str(CONDA_LIBRARY_DIR), inherited_library_path) if part
    )
    env.update(
        {
            "PYTHONPATH": str(REPOSITORY),
            "TAIRTE4_TOPOLOGY_GEOMETRY": "contact_anchored",
            "TAIRTE4_SIO2_INTERFACE_SCENARIO": INTERFACE_SCENARIO,
            "CUDA_VISIBLE_DEVICES": str(GPU),
            "LUMERICAL_LICENSE_RETRY_SECONDS": "30",
            "LUMERICAL_GPU_ENGINE_LOCK": "/tmp/seunghyun_lumerical_fdtd_gpu_engine.lock",
            "XDG_CONFIG_HOME": "/tmp/seunghyun_lumerical_run049",
            "MPLCONFIGDIR": "/tmp/seunghyun_matplotlib_run049",
        }
    )
    return env


def run_checked(command: list[str], *, stage: str) -> None:
    write_state(stage, status="running", command=command)
    completed = subprocess.run(command, cwd=REPOSITORY, env=environment())
    if completed.returncode:
        write_state(stage, status="failed", returncode=completed.returncode)
        raise RuntimeError(f"{stage} failed with return code {completed.returncode}")


def main() -> int:
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    lock_stream = LOCK.open("w")
    try:
        fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("Run049 pipeline is already active") from error

    if not BASE_FSP.is_file() or sha256(BASE_FSP) != BASE_SHA256:
        raise RuntimeError("immutable top/bottom uniform-rho0.5 Ea FSP changed")
    certificate = JACOBIAN_ROOT / "component_yee_jacobian_result.json"
    if not certificate.is_file() or not json.loads(certificate.read_text()).get("passed"):
        raise RuntimeError("top/bottom component-Yee Jacobian certificate unavailable")
    if OPTIMIZATION_ROOT.exists() and any(OPTIMIZATION_ROOT.iterdir()):
        raise RuntimeError("Run049 raw output is non-empty; refusing non-fresh start")
    if RESULTS.exists() and any(RESULTS.iterdir()):
        raise RuntimeError("Run049 published output is non-empty; refusing non-fresh start")

    if ADFD_ROOT.exists():
        if not ADFD_RESULT.is_file() or not json.loads(ADFD_RESULT.read_text()).get("passed"):
            raise RuntimeError("existing evaporated-interface AD-FD output is incomplete")
        contract = json.loads(ADFD_RESULT.read_text()).get("thermal_interface_contract", {})
        if contract.get("TaIrTe4_SiO2_scenario") != INTERFACE_SCENARIO:
            raise RuntimeError("existing AD-FD interface scenario does not match Run049")
    else:
        run_checked(
            [
                str(PYTHON), "-m",
                "photothermal_pte.optimization_runs.tairte4_flake_topology.validate_combined_adfd",
                "--base-fsp", str(BASE_FSP),
                "--base-sha256", BASE_SHA256,
                "--jacobian-dir", str(JACOBIAN_ROOT),
                "--output-dir", str(ADFD_ROOT),
                "--gpu-device", f"GPU {GPU}",
                "--cuda-device", "0",
                "--step", "0.005",
                "--polarization", "Ea",
            ],
            stage="combined_adfd_Ea_evaporated",
        )
    precheck = json.loads(ADFD_RESULT.read_text())
    if not precheck.get("passed"):
        raise RuntimeError("Run049 evaporated-interface combined AD-FD failed")
    contract = precheck.get("thermal_interface_contract", {})
    if (
        contract.get("TaIrTe4_SiO2_scenario") != INTERFACE_SCENARIO
        or float(contract.get("G_TaIrTe4_SiO2_W_m2K", 0.0)) != INTERFACE_G_W_M2K
    ):
        raise RuntimeError("Run049 AD-FD did not use the requested evaporated G")

    RESULTS.mkdir(parents=True, exist_ok=True)
    OPTIMIZATION_ROOT.mkdir(parents=True, exist_ok=True)
    run_checked(
        [
            str(PYTHON), "-m",
            "photothermal_pte.optimization_runs.tairte4_flake_topology.run_ansys_dfm_ld_mma_optimization",
            "--polarization", "Ea",
            "--raw-root", str(OPTIMIZATION_ROOT),
            "--published-dir", str(RESULTS),
            "--gpu", str(GPU),
            "--base-fsp", str(BASE_FSP),
            "--base-sha256", BASE_SHA256,
            "--jacobian-dir", str(JACOBIAN_ROOT),
            "--constraint-device", "cuda:0",
        ],
        stage="optimization_Ea_evaporated",
    )
    final_result = RESULTS / "FINAL_RESULT.json"
    if not final_result.is_file():
        raise RuntimeError("Run049 returned without FINAL_RESULT.json")
    write_state("complete", status="complete", final_result=str(final_result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
