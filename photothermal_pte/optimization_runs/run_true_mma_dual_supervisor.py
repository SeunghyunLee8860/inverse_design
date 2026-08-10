#!/usr/bin/env python3
"""Fail-closed preflight, then fresh Run016 Ea followed by Run017 Eb."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


REPOSITORY = Path(__file__).resolve().parents[2]
PARENT = Path("/data/seunghyun/tairte4/artifacts/tairte4_contact_anchored")
BASE_ROOT = PARENT / "run012_uniform_rho0p5_Ea_forward_retry_20260810"
BASE_FSP = BASE_ROOT / "tairte4_flake_forward_Ea.fsp"
BASE_JSON = BASE_ROOT / "tairte4_flake_forward_Ea.json"
BASE_NPZ = BASE_ROOT / "tairte4_flake_native_Q_Ea.npz"
BASE_SHA256 = "454fa83bc918b4db0e25d28f7debf23de38977038bd716c8d7dc539d6b3e3d83"
JACOBIAN = PARENT / "run012_component_yee_jacobian_retry_20260810"
PREFLIGHT = PARENT / "run016_017_true_mma_preflight_20260810"
GPU = int(os.environ.get("TAIRTE4_TRUE_MMA_GPU", "5"))
STATUS = REPOSITORY / "photothermal_pte/optimization_runs/TRUE_MMA_DUAL_RUN_STATUS.json"
AUDIT_OUTPUT = REPOSITORY / "photothermal_pte/optimization_runs/true_mma_preflight"
RUNS = (
    (
        "Run016",
        "Ea",
        PARENT / "run016_true_mma_Ea_20260810",
        REPOSITORY / "photothermal_pte/optimization_runs/run_016_true_mma_contact_anchored_Ea_current_max",
    ),
    (
        "Run017",
        "Eb",
        PARENT / "run017_true_mma_Eb_20260810",
        REPOSITORY / "photothermal_pte/optimization_runs/run_017_true_mma_contact_anchored_Eb_current_max",
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
        "schema": "true-mma-dual-supervisor-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "gpu": GPU,
        **values,
    }
    temporary = STATUS.with_suffix(STATUS.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(STATUS)
    print(json.dumps(payload), flush=True)


def run(command: list[str], label: str, environment: dict[str, str]) -> None:
    write_status("RUNNING_COMMAND", label=label, command=command)
    completed = subprocess.run(command, cwd=REPOSITORY, env=environment)
    if completed.returncode:
        write_status("BLOCKED_COMMAND_FAILED", label=label, returncode=completed.returncode)
        raise RuntimeError(f"{label} failed with return code {completed.returncode}")


def run_restartable(
    command: list[str],
    label: str,
    environment: dict[str, str],
    final_result: Path,
) -> None:
    for attempt in range(1, 9):
        if passed(final_result):
            return
        write_status(
            "RUNNING_RESTARTABLE_OPTIMIZATION",
            label=label,
            attempt=attempt,
            command=command,
        )
        completed = subprocess.run(command, cwd=REPOSITORY, env=environment)
        if completed.returncode == 0 and passed(final_result):
            return
        write_status(
            "RETRYING_FROM_IMMUTABLE_CHECKPOINT",
            label=label,
            attempt=attempt,
            returncode=completed.returncode,
        )
        time.sleep(30.0)
    raise RuntimeError(f"{label} exceeded eight restart attempts")


def passed(path: Path) -> bool:
    return path.is_file() and bool(json.loads(path.read_text()).get("passed"))


def ensure_runsetup(
    output: Path,
    *,
    domain_um: float,
    interface_nm: float,
    environment: dict[str, str],
) -> Path:
    result = output / "tairte4_flake_optical_runsetup_audit.json"
    if not passed(result):
        if output.exists() and any(output.iterdir()):
            raise RuntimeError(f"incomplete preflight output requires manual preservation: {output}")
        run(
            [
                sys.executable,
                "-m",
                "photothermal_pte.optimization_runs.tairte4_flake_topology.audit_optical_runsetup",
                "--output-dir", str(output),
                "--domain-um", str(domain_um),
                "--interface-xy-nm", str(interface_nm),
            ],
            f"runsetup_domain{domain_um:g}_mesh{interface_nm:g}",
            environment,
        )
    return output / "tairte4_flake_optical_runsetup.fsp"


def ensure_forward(
    runsetup: Path,
    output: Path,
    *,
    environment: dict[str, str],
) -> tuple[Path, Path]:
    result = output / "tairte4_flake_forward_Ea.json"
    if not passed(result):
        if output.exists() and any(output.iterdir()):
            raise RuntimeError(f"incomplete preflight output requires manual preservation: {output}")
        run(
            [
                sys.executable,
                "-m",
                "photothermal_pte.optimization_runs.tairte4_flake_topology.run_forward_gpu",
                "--input-project", str(runsetup),
                "--input-sha256", sha256(runsetup),
                "--output-dir", str(output),
                "--gpu-device", f"GPU {GPU}",
                "--polarization", "a",
            ],
            f"forward_{output.name}",
            environment,
        )
    return result, output / "tairte4_flake_native_Q_Ea.npz"


def ensure_comparison(
    output: Path,
    *,
    fine_json: Path,
    fine_npz: Path,
    comparison: str,
    environment: dict[str, str],
) -> None:
    stem = "tairte4_flake_100nm_50nm" if comparison == "mesh" else "tairte4_flake_40um_48um"
    result = output / f"{stem}_{comparison}_comparison.json"
    if not passed(result):
        if output.exists() and any(output.iterdir()):
            raise RuntimeError(f"incomplete comparison output requires preservation: {output}")
        run(
            [
                sys.executable,
                "-m",
                "photothermal_pte.optimization_runs.tairte4_flake_topology.compare_forward_meshes",
                "--coarse-json", str(BASE_JSON),
                "--coarse-npz", str(BASE_NPZ),
                "--fine-json", str(fine_json),
                "--fine-npz", str(fine_npz),
                "--output-dir", str(output),
                "--comparison", comparison,
            ],
            f"compare_{comparison}",
            environment,
        )


def main() -> int:
    if not BASE_FSP.is_file() or sha256(BASE_FSP) != BASE_SHA256:
        raise RuntimeError("immutable contact-anchored base FSP is missing/SHA-mismatched")
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = str(GPU)
    environment["TAIRTE4_TOPOLOGY_GEOMETRY"] = "contact_anchored"
    PREFLIGHT.mkdir(parents=True, exist_ok=True)

    domain_setup = ensure_runsetup(
        PREFLIGHT / "domain48_runsetup",
        domain_um=48.0,
        interface_nm=100.0,
        environment=environment,
    )
    domain_json, domain_npz = ensure_forward(
        domain_setup,
        PREFLIGHT / "domain48_forward",
        environment=environment,
    )
    ensure_comparison(
        PREFLIGHT / "domain_comparison",
        fine_json=domain_json,
        fine_npz=domain_npz,
        comparison="domain",
        environment=environment,
    )

    mesh_setup = ensure_runsetup(
        PREFLIGHT / "mesh50_runsetup",
        domain_um=40.0,
        interface_nm=50.0,
        environment=environment,
    )
    mesh_json, mesh_npz = ensure_forward(
        mesh_setup,
        PREFLIGHT / "mesh50_forward",
        environment=environment,
    )
    ensure_comparison(
        PREFLIGHT / "mesh_comparison",
        fine_json=mesh_json,
        fine_npz=mesh_npz,
        comparison="mesh",
        environment=environment,
    )

    eb_result = PREFLIGHT / "combined_Eb/tairte4_flake_combined_adfd.json"
    if not passed(eb_result):
        if eb_result.parent.exists() and any(eb_result.parent.iterdir()):
            raise RuntimeError("incomplete Eb combined AD-FD output requires preservation")
        run(
            [
                sys.executable,
                "-m",
                "photothermal_pte.optimization_runs.tairte4_flake_topology.validate_combined_adfd",
                "--base-fsp", str(BASE_FSP),
                "--base-sha256", BASE_SHA256,
                "--jacobian-dir", str(JACOBIAN),
                "--output-dir", str(eb_result.parent),
                "--gpu-device", f"GPU {GPU}",
                "--cuda-device", "0",
                "--polarization", "Eb",
            ],
            "combined_Eb_ADFD",
            environment,
        )

    run(
        [
            sys.executable,
            "-m",
            "photothermal_pte.optimization_runs.audit_true_mma_preflight",
            "--output-dir", str(AUDIT_OUTPUT),
            "--preflight-root", str(PREFLIGHT),
        ],
        "close_true_mma_preflight",
        environment,
    )

    for label, polarization, raw, published in RUNS:
        final = published / "FINAL_RESULT.json"
        if passed(final):
            continue
        raw.mkdir(parents=True, exist_ok=True)
        published.mkdir(parents=True, exist_ok=True)
        run_restartable(
            [
                sys.executable,
                "-m",
                "photothermal_pte.optimization_runs.tairte4_flake_topology.run_true_mma_optimization",
                "--polarization", polarization,
                "--raw-root", str(raw),
                "--published-dir", str(published),
                "--gpu", str(GPU),
                "--base-fsp", str(BASE_FSP),
                "--base-sha256", BASE_SHA256,
                "--jacobian-dir", str(JACOBIAN),
                "--connectivity-fraction", "0.10",
                "--constraint-device", "cuda:0",
            ],
            f"{label}_{polarization}_true_MMA",
            environment,
            final,
        )
    write_status("VALIDATED_TRUE_MMA_EA_EB_COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
