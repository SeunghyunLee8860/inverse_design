#!/usr/bin/env python3
"""Re-evaluate a saved source-only solve without running FDTD again."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from photothermal_pte.validation.paper_ir_sanity import (
    validate_paper_ir_source_only_gpu as runner,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def artifact_record(path: Path, role: str) -> dict[str, Any]:
    return {
        "role": role,
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": runner.sha256(path),
    }


def saved_plane_fields(
    raw: np.lib.npyio.NpzFile,
    name: str,
) -> dict[str, Any]:
    return {
        "coordinates": {
            "x": raw[f"{name}_x_m"],
            "y": raw[f"{name}_y_m"],
            "z": np.asarray([runner.MONITORS[name]], float),
        },
        "electric": {
            axis: raw[f"{name}_E{axis}"] for axis in "xyz"
        },
        "magnetic": {
            axis: raw[f"{name}_H{axis}"] for axis in "xyz"
        },
    }


def load_saved_mesh(project: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    os.environ["VC_LUMERICAL_ROOT"] = str(runner.APPROVED_ROOT)
    os.environ["LUMERICAL_ROOT"] = str(runner.APPROVED_ROOT)
    os.environ["LUMERICAL_PYTHONPATH"] = str(runner.APPROVED_API)
    os.environ["PATH"] = (
        f"{runner.APPROVED_ROOT / 'bin'}:{os.environ.get('PATH', '')}"
    )
    for path in (runner.STAGE1, runner.REPOSITORY / "photothermal_pte"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    helper = runner.load_module(
        runner.API_HELPER,
        "paper_ir_source_readonly_lumerical_api",
    )
    installation = SimpleNamespace(
        version_key="v261",
        root=runner.APPROVED_ROOT,
        lumapi_path=runner.APPROVED_API / "lumapi.py",
        device_executable=runner.APPROVED_ROOT / "bin" / "device",
    )
    lumapi = helper.load_lumapi(installation)
    fdtd = None
    try:
        fdtd = lumapi.FDTD(
            filename=str(project),
            hide=True,
            serverArgs={"platform": "offscreen"},
        )
        version = str(fdtd.version())
        provenance = runner.v261_session_provenance(
            solver_version=version,
            loaded_lumapi_path=Path(lumapi.__file__),
            installation_version_key=installation.version_key,
            installation_root=installation.root,
        )
        mesh = runner.mesh_readback(fdtd)
        if not provenance["all"]:
            raise RuntimeError(f"invalid saved-session provenance: {provenance}")
        if not mesh["available"]:
            raise RuntimeError(f"saved native mesh unavailable: {mesh}")
        return mesh, provenance
    finally:
        if fdtd is not None:
            try:
                fdtd.close()
            except Exception:
                pass


def main() -> int:
    args = parse_args()
    artifact_dir = Path(args.artifact_dir).expanduser().resolve()
    original_path = artifact_dir / "source_only_case_result.json"
    field_path = artifact_dir / "paper_ir_source_only_fields.npz"
    project_path = artifact_dir / "paper_ir_source_only.fsp"
    log_paths = sorted(artifact_dir.glob("*.log"))
    for required in (original_path, field_path, project_path):
        if not required.is_file():
            raise FileNotFoundError(required)

    original = read_json(original_path)
    recorded_field = original["field_artifact"]
    if (
        field_path.stat().st_size != recorded_field["size_bytes"]
        or runner.sha256(field_path) != recorded_field["sha256"]
    ):
        raise RuntimeError("saved field artifact SHA/size mismatch")

    source_power = float(original["source_power_readback_W"])
    with np.load(field_path, allow_pickle=False) as raw:
        profile, _ = runner.source_profile_from_arrays(
            raw["source_profile_x_m"],
            raw["source_profile_y_m"],
            raw["source_profile_E"],
        )
        planes: dict[str, dict[str, Any]] = {}
        for name in runner.MONITORS:
            metrics, _ = runner.plane_metrics(
                saved_plane_fields(raw, name),
                source_power,
            )
            planes[name] = metrics

    mesh, session = load_saved_mesh(project_path)
    log = runner.log_audit(artifact_dir)
    acceptance = runner.source_acceptance(
        focus=planes["flake_target_plane"],
        profile_metrics=profile,
        post_run_mesh=mesh,
        log=log,
        planes=planes,
        auto_shutoff_min=float(original["pre_run"]["built_contract"][
            "domain"
        ].get("auto_shutoff_min", 1.0e-5)),
    )
    passed = all(acceptance.values())
    command = shlex.join([sys.executable, *sys.argv])
    result = {
        "status": (
            "VALIDATED_PAPER_LIKE_SCALAR_GAUSSIAN_SOURCE_ONLY"
            if passed
            else "FAILED_PAPER_LIKE_SCALAR_GAUSSIAN_SOURCE_ONLY_GATE"
        ),
        "read_only_recovery": True,
        "new_FDTD_solve_executed": False,
        "CPU_FDTD_fallback": False,
        "generation_command": command,
        "generation_commit": runner.git_commit(),
        "source_result_path": str(original_path),
        "source_result_sha256": runner.sha256(original_path),
        "source_generation_commit": original["generation_commit"],
        "session_provenance": session,
        "source_power_readback_W": source_power,
        "source_object_profile": profile,
        "planes": planes,
        "post_run_mesh": {
            key: value
            for key, value in mesh.items()
            if key != "coordinate_arrays"
        },
        "log_audit": log,
        "acceptance": acceptance,
        "source_only_gate_passed": passed,
        "successor_sequence_authorized": passed,
        "failed_gates": [
            key for key, value in acceptance.items() if not value
        ],
        "thin_lens_status": "OPTIONAL_FUTURE_DIAGNOSTIC",
    }
    output = artifact_dir / "source_only_readonly_recovery.json"
    runner.write_json(output, result)
    artifacts = [
        artifact_record(original_path, "immutable_original_source_result"),
        artifact_record(field_path, "immutable_source_field_npz"),
        artifact_record(project_path, "immutable_source_project_fsp"),
        *[
            artifact_record(path, "immutable_solver_log")
            for path in log_paths
        ],
        artifact_record(output, "read_only_recovered_result"),
    ]
    manifest = {
        "raw_artifacts_committed_to_git": False,
        "new_FDTD_solve_executed": False,
        "generation_command": command,
        "generation_commit": runner.git_commit(),
        "artifacts": artifacts,
    }
    runner.write_json(
        artifact_dir / "READONLY_RECOVERY_MANIFEST.json",
        manifest,
    )
    print(json.dumps({
        "status": result["status"],
        "failed_gates": result["failed_gates"],
        "result": str(output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
