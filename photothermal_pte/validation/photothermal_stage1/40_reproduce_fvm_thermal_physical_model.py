#!/usr/bin/env python3
"""Fail-closed clean-checkout entry point for FVM physical scenarios."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
from pathlib import Path

import config_stage1 as config
from lumerical_api import utc_timestamp, write_json


PR3_COMMIT = "053260da6fd0caec28ce155221bd18f683a0e5e7"
PR4_CHECKPOINT = "437ec0644b15a4b9a6919a0151e4aa531fb1e0ab"
EXPECTED_SHA256 = (
    "7a63f82842751e7623e895701bac4ce92558679ed71bf13a3d404695a150e794"
)
IMPORT_SCRIPT = Path(__file__).with_name(
    "35_validate_finite_q_fvm_import.py"
)
SCENARIO_SCRIPT = Path(__file__).with_name(
    "39_validate_fvm_thermal_physical_model.py"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pr3-q-artifact", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--phase", choices=("material", "boundary", "all"), default="all"
    )
    parser.add_argument("--report-dir")
    parser.add_argument("--precheck-only", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(command: list[str]) -> None:
    completed = subprocess.run(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed ({completed.returncode}): "
            f"{shlex.join(command)}\n{completed.stdout}"
        )


def ancestry_contains(commit: str) -> bool:
    completed = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=config.REPOSITORY_ROOT.parent,
        check=False,
    )
    if completed.returncode not in (0, 1):
        raise RuntimeError(f"cannot evaluate ancestry for {commit}")
    return completed.returncode == 0


def main() -> int:
    args = parse_args()
    artifact = Path(args.pr3_q_artifact).expanduser().resolve()
    if not artifact.is_file():
        raise FileNotFoundError(
            f"required external PR #3 Q artifact is missing: {artifact}"
        )
    actual_sha = sha256_file(artifact)
    if actual_sha != EXPECTED_SHA256:
        raise RuntimeError(
            f"PR #3 Q SHA mismatch: expected {EXPECTED_SHA256}, "
            f"got {actual_sha}; no import or thermal solve was started"
        )

    output_root = Path(args.output_root).expanduser().resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(
            f"refusing non-empty reproduction output root: {output_root}"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    dependency = {
        "PR3_commit": PR3_COMMIT,
        "PR4_immutable_checkpoint": PR4_CHECKPOINT,
        "PR3_in_current_ancestry": ancestry_contains(PR3_COMMIT),
        "PR4_checkpoint_in_current_ancestry": ancestry_contains(
            PR4_CHECKPOINT
        ),
        "external_PR3_artifact_required": True,
        "artifact_path": str(artifact),
        "expected_SHA256": EXPECTED_SHA256,
        "actual_SHA256": actual_sha,
        "SHA_verified_before_output_or_solver": True,
    }
    write_json(output_root / "dependency_precheck.json", dependency)
    if args.precheck_only:
        print(json.dumps(dependency, indent=2))
        return 0

    import_output = output_root / "finite_q_import"
    import_reports = output_root / "finite_q_import_reports"
    import_command = [
        sys.executable,
        str(IMPORT_SCRIPT),
        "--q-artifact",
        str(artifact),
        "--output-dir",
        str(import_output),
        "--report-dir",
        str(import_reports),
        "--allow-missing-pr3-git-object",
    ]
    run(import_command)
    exact_source = import_output / "finite_q_exact_flake_source.npz"
    if not exact_source.is_file():
        raise RuntimeError("conservative exact-flake source was not produced")

    scenario_command = [
        sys.executable,
        str(SCENARIO_SCRIPT),
        "--source-artifact",
        str(exact_source),
        "--output-dir",
        str(output_root / "physical_model_scenarios"),
        "--phase",
        args.phase,
    ]
    if args.report_dir:
        scenario_command.extend(["--report-dir", args.report_dir])
    run(scenario_command)
    result = {
        "schema_version": 1,
        "generated_at_utc": utc_timestamp(),
        "status": "REPRODUCED_FVM_THERMAL_PHYSICAL_MODEL_SCENARIOS",
        "dependency_precheck": dependency,
        "import_command": shlex.join(import_command),
        "scenario_command": shlex.join(scenario_command),
        "raw_NPZ_committed_to_git": False,
        "transient_PTE_adjoint_gradient_optimization_executed": False,
    }
    write_json(output_root / "reproduction_summary.json", result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
