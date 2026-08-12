#!/usr/bin/env python3
"""License-aware fail-closed pipeline for Run 046.

Only transient FlexNet exhaustion is retried.  Every other failure stops the
pipeline.  Failed, regenerable FSP files are removed while their JSON/log
diagnostics are retained.
"""

from __future__ import annotations

from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]
PYTHON = Path("/home/eidl/miniconda3/envs/EIDL-Lumapi/bin/python")
CONDA_LIBRARY_DIR = PYTHON.parents[1] / "lib"
ARTIFACT_ROOT = Path(
    "/data/seunghyun/tairte4/artifacts/tairte4_left_right_contact_anchored"
)
RUNSETUP = ARTIFACT_ROOT / "runsetup_100nm_v1/tairte4_flake_optical_runsetup.fsp"
RUNSETUP_SHA256 = "3fee2632e906a181edf1f83f49ba2e4eb8a24918755100ea85eb4763195350d2"
FORWARD_ROOT = ARTIFACT_ROOT / "uniform_rho0p5_Ea_forward_queued"
JACOBIAN_ROOT = ARTIFACT_ROOT / "component_yee_jacobian_v1"
ADFD_ROOT = ARTIFACT_ROOT / "combined_adfd_Ea_reserved_v3"
OPTIMIZATION_ROOT = ARTIFACT_ROOT / "run046_Ea_current_max"
RESULTS = HERE / "results"
STATE = HERE / "PIPELINE_STATE.json"
LOCK = Path("/tmp/seunghyun_run046_left_right_Ea.lock")
GPU = int(os.environ.get("RUN046_GPU", "1"))
RETRY_SECONDS = float(os.environ.get("RUN046_LICENSE_RETRY_SECONDS", "60"))
LICENSE_MARKERS = (
    "insufficient flexnet publisher",
    "licensed number of users already reached",
    "unable to checkout the requested hpc license",
    "flexnet licensing error:-4,132",
)


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
        **extra,
    }
    temporary = STATE.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(STATE)


def artifact_text(root: Path) -> str:
    chunks: list[str] = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".json", ".log", ".txt"}:
            try:
                chunks.append(path.read_text(errors="replace").lower())
            except OSError:
                pass
    return "\n".join(chunks)


def transient_license_failure(root: Path) -> bool:
    text = artifact_text(root)
    return any(marker in text for marker in LICENSE_MARKERS)


def remove_regenerable_fsp(root: Path) -> None:
    for path in root.rglob("*.fsp"):
        if path.is_file():
            path.unlink()


def environment() -> dict[str, str]:
    env = dict(os.environ)
    inherited_library_path = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = ":".join(
        part
        for part in (str(CONDA_LIBRARY_DIR), inherited_library_path)
        if part
    )
    env.update(
        {
            "PYTHONPATH": str(REPOSITORY),
            "TAIRTE4_TOPOLOGY_GEOMETRY": "left_right_contact_anchored",
            "CUDA_VISIBLE_DEVICES": str(GPU),
            "LUMERICAL_LICENSE_RETRY_SECONDS": str(RETRY_SECONDS),
            "LUMERICAL_GPU_ENGINE_LOCK": "/tmp/seunghyun_lumerical_fdtd_gpu_engine.lock",
            "XDG_CONFIG_HOME": "/tmp/seunghyun_lumerical_run046",
            "MPLCONFIGDIR": "/tmp/seunghyun_matplotlib_run046",
        }
    )
    return env


def run_retryable(stage: str, command_factory, success_factory) -> Path:
    attempt = 1
    while True:
        output = command_factory(attempt)
        write_state(stage, attempt=attempt, output=str(output), status="running")
        completed = subprocess.run(
            command_factory(attempt, command_only=True),
            cwd=REPOSITORY,
            env=environment(),
        )
        success = success_factory(output)
        if completed.returncode == 0 and success.is_file():
            write_state(stage, attempt=attempt, output=str(output), status="passed")
            return success
        if not transient_license_failure(output):
            write_state(stage, attempt=attempt, output=str(output), status="failed_nonlicense")
            raise RuntimeError(f"{stage} failed for a non-license reason: {output}")
        remove_regenerable_fsp(output)
        write_state(stage, attempt=attempt, output=str(output), status="waiting_for_license")
        time.sleep(RETRY_SECONDS)
        attempt += 1


class ForwardCommand:
    def __call__(self, attempt: int, command_only: bool = False):
        output = FORWARD_ROOT / f"attempt_{attempt:04d}"
        if not command_only:
            return output
        return [
            str(PYTHON), "-m",
            "photothermal_pte.optimization_runs.tairte4_flake_topology.run_forward_gpu",
            "--input-project", str(RUNSETUP),
            "--input-sha256", RUNSETUP_SHA256,
            "--output-dir", str(output),
            "--gpu-device", f"GPU {GPU}",
            "--polarization", "a",
        ]


def forward_success(output: Path) -> Path:
    result = output / "tairte4_flake_forward_Ea.json"
    if not result.is_file() or not json.loads(result.read_text()).get("passed"):
        return output / "MISSING_FORWARD_PASS"
    return result


def existing_passed_forward() -> Path | None:
    if not FORWARD_ROOT.is_dir():
        return None
    for result in sorted(
        FORWARD_ROOT.glob("attempt_*/tairte4_flake_forward_Ea.json"), reverse=True
    ):
        if json.loads(result.read_text()).get("passed"):
            return result
    return None


def optimization_recovery_arguments() -> list[str]:
    """Resume from the last fully evaluated design, never a failed candidate."""
    history_path = RESULTS / "optimization_history.json"
    manifest_path = RESULTS / "RAW_ARTIFACT_MANIFEST.json"
    if not history_path.is_file() or not manifest_path.is_file():
        return []
    history = json.loads(history_path.read_text())
    if not isinstance(history, list) or not history:
        raise RuntimeError("Run046 recovery history is malformed or empty")
    last = history[-1]
    evaluation_id = int(last["evaluation_id"])
    beta = float(last["beta"])
    matches = sorted(
        OPTIMIZATION_ROOT.glob(
            f"evaluation_{evaluation_id:04d}_beta*_latent.npz"
        )
    )
    if len(matches) != 1:
        raise RuntimeError(
            "expected exactly one last-successful latent checkpoint for "
            f"evaluation {evaluation_id}, found {len(matches)}"
        )
    return [
        "--initial-latent-npz", str(matches[0]),
        "--recovery-append",
        "--start-beta", f"{beta:.12g}",
        "--output-slug", "ansys_dfm_ld_mma_gpu_resource_recovery1",
    ]


def main() -> int:
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    lock_stream = LOCK.open("w")
    try:
        fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("Run046 pipeline is already active") from error
    if sha256(RUNSETUP) != RUNSETUP_SHA256:
        raise RuntimeError("runsetup SHA mismatch")
    for path in (FORWARD_ROOT, Path(environment()["XDG_CONFIG_HOME"]), Path(environment()["MPLCONFIGDIR"])):
        path.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    forward_json = existing_passed_forward()
    if forward_json is None:
        forward_json = run_retryable("uniform_Ea_forward", ForwardCommand(), forward_success)
    else:
        write_state(
            "uniform_Ea_forward",
            status="reused_passed_checkpoint",
            output=str(forward_json.parent),
        )
    forward = json.loads(forward_json.read_text())
    base_fsp = Path(forward["output_project"]["path"])
    base_sha = str(forward["output_project"]["sha256"])

    jacobian_result = JACOBIAN_ROOT / "component_yee_jacobian_result.json"
    if jacobian_result.is_file() and json.loads(jacobian_result.read_text()).get("passed"):
        write_state("component_yee_jacobian", status="reused_passed_checkpoint")
    else:
        write_state("component_yee_jacobian", status="running")
        command = [
            str(PYTHON), "-m",
            "photothermal_pte.optimization_runs.legacy_v261_optical_support.build_nonuniform_complex_yee_jacobian",
            "--base-project", str(base_fsp),
            "--base-sha256", base_sha,
            "--output-dir", str(JACOBIAN_ROOT),
            "--geometry", "tairte4_flake",
        ]
        completed = subprocess.run(command, cwd=REPOSITORY, env=environment())
        if completed.returncode or not jacobian_result.is_file() or not json.loads(jacobian_result.read_text()).get("passed"):
            raise RuntimeError("component Yee Jacobian gate failed")

    adfd_result = ADFD_ROOT / "tairte4_flake_combined_adfd.json"
    if adfd_result.is_file() and json.loads(adfd_result.read_text()).get("passed"):
        write_state(
            "combined_adfd_Ea",
            status="reused_passed_checkpoint",
            output=str(adfd_result),
        )
    else:
        write_state("combined_adfd_Ea", status="running")
        command = [
            str(PYTHON), "-m",
            "photothermal_pte.optimization_runs.tairte4_flake_topology.validate_combined_adfd",
            "--base-fsp", str(base_fsp),
            "--base-sha256", base_sha,
            "--jacobian-dir", str(JACOBIAN_ROOT),
            "--output-dir", str(ADFD_ROOT),
            "--gpu-device", f"GPU {GPU}",
            "--cuda-device", "0",
            "--step", "0.005",
            "--polarization", "Ea",
        ]
        completed = subprocess.run(command, cwd=REPOSITORY, env=environment())
        if completed.returncode or not adfd_result.is_file() or not json.loads(adfd_result.read_text()).get("passed"):
            raise RuntimeError("combined Ea AD-FD gate failed")

    write_state("optimization_Ea", status="running")
    command = [
        str(PYTHON), "-m",
        "photothermal_pte.optimization_runs.tairte4_flake_topology.run_ansys_dfm_ld_mma_optimization",
        "--polarization", "Ea",
        "--raw-root", str(OPTIMIZATION_ROOT),
        "--published-dir", str(RESULTS),
        "--gpu", str(GPU),
        "--base-fsp", str(base_fsp),
        "--base-sha256", base_sha,
        "--jacobian-dir", str(JACOBIAN_ROOT),
        "--constraint-device", "cuda:0",
    ]
    recovery_arguments = optimization_recovery_arguments()
    if recovery_arguments:
        command.extend(recovery_arguments)
        write_state(
            "optimization_Ea",
            status="warm_restart_from_last_successful_evaluation",
            recovery_arguments=recovery_arguments,
        )
    completed = subprocess.run(command, cwd=REPOSITORY, env=environment())
    final_result = RESULTS / "FINAL_RESULT.json"
    if completed.returncode or not final_result.is_file():
        write_state(
            "optimization_Ea",
            status="failed",
            returncode=completed.returncode,
        )
        raise RuntimeError("Run046 optimization did not produce a final result")
    write_state("complete", status="complete", final_result=str(final_result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
