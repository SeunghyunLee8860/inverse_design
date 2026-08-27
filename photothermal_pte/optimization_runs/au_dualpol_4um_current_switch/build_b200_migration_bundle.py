#!/usr/bin/env python3
"""Seal one stopped continuation checkpoint into a portable Git bundle."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np


CHECKPOINT_SCHEMA = "au-lumerical-continuation-checkpoint-v2"
BUNDLE_SCHEMA = "au-lumerical-b200-migration-bundle-v1"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--confirm-no-live-process",
        action="store_true",
        help="Required acknowledgement after checking tmux/runres/FDTD processes.",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact(path: Path, *, portable_name: str | None = None) -> dict[str, Any]:
    return {
        "path": portable_name if portable_name is not None else str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _load_checkpoint(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as arrays:
        schema = str(np.asarray(arrays["schema"]).item())
        if schema != CHECKPOINT_SCHEMA:
            raise RuntimeError(f"unexpected checkpoint schema: {schema}")
        latent = np.asarray(arrays["latent"], dtype=np.float64)
        if latent.shape != (81, 81):
            raise RuntimeError(f"unexpected latent shape: {latent.shape}")
        if (
            not np.all(np.isfinite(latent))
            or float(np.min(latent)) < 0.0
            or float(np.max(latent)) > 1.0
        ):
            raise RuntimeError("checkpoint latent is outside finite [0,1]")
        return {
            "latent": latent,
            "beta_index": int(np.asarray(arrays["beta_index"]).item()),
            "attempt": int(np.asarray(arrays["attempt"]).item()),
            "dfm_caps": np.asarray(arrays["dfm_caps"], dtype=np.float64),
            "grayness_cap": float(np.asarray(arrays["grayness_cap"]).item()),
        }


def main() -> int:
    args = _parse_args()
    if not args.confirm_no_live_process:
        raise RuntimeError("refusing to package before no-live-process confirmation")
    source = args.source_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing nonempty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    source_checkpoint = source / "continuation_checkpoint.npz"
    source_manifest_path = source / "production_manifest.json"
    if not source_checkpoint.is_file() or not source_manifest_path.is_file():
        raise FileNotFoundError("source checkpoint or production manifest is absent")
    state = _load_checkpoint(source_checkpoint)
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    stages = source_manifest.get("stages")
    if not isinstance(stages, list) or not stages:
        raise RuntimeError("source manifest has no completed stage")
    latest = source_manifest.get("latest")
    if not isinstance(latest, dict):
        raise RuntimeError("source manifest has no latest completed-stage record")
    terminal_record = stages[-1].get("state_artifact")
    if not isinstance(terminal_record, dict):
        raise RuntimeError("source manifest has no terminal state artifact")
    terminal_source = Path(str(terminal_record.get("path", ""))).resolve()
    if not terminal_source.is_file():
        raise FileNotFoundError(terminal_source)
    terminal_actual = _artifact(terminal_source)
    if (
        terminal_actual["size_bytes"] != int(terminal_record.get("size_bytes", -1))
        or terminal_actual["sha256"] != terminal_record.get("sha256")
    ):
        raise RuntimeError("source terminal stage artifact changed")
    with np.load(terminal_source, allow_pickle=False) as arrays:
        terminal_latent = np.asarray(arrays["latent_final"], dtype=np.float64)
    if not np.array_equal(terminal_latent, state["latent"]):
        raise RuntimeError("checkpoint latent differs from completed terminal state")
    if int(state["beta_index"]) != 0 or float(latest.get("beta", np.nan)) != 1.0:
        raise RuntimeError("this migration tool expects the audited beta-1 checkpoint")

    checkpoint_out = output / "continuation_checkpoint.npz"
    terminal_out = output / "terminal_stage_state.npz"
    shutil.copyfile(source_checkpoint, checkpoint_out)
    shutil.copyfile(terminal_source, terminal_out)
    portable_terminal = _artifact(
        terminal_out, portable_name=terminal_out.name
    )
    restart_manifest = {
        "schema": source_manifest.get("schema"),
        "status": "STOPPED_FOR_B200_MIGRATION_AFTER_RUNTIME_JACOBIAN_SELF_AUDIT",
        "passed": False,
        "git_commit": source_manifest.get("git_commit"),
        "blocking_attempts": int(state["attempt"]),
        "latest": latest,
        "stages": [
            {
                "beta": stages[-1].get("beta"),
                "attempt": stages[-1].get("attempt"),
                "status": stages[-1].get("status"),
                "state_artifact": portable_terminal,
            }
        ],
        "migration_audit": {
            "schema": BUNDLE_SCHEMA,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "original_manifest": _artifact(source_manifest_path),
            "original_status": source_manifest.get("status"),
            "original_error": source_manifest.get("error"),
            "checkpoint_terminal_latent_exact": True,
            "no_partial_eval_7_state_promoted": True,
            "restart_semantics": (
                "resume beta 1 at logical attempt 4 from the last completed "
                "attempt-3 feasible point"
            ),
        },
    }
    restart_manifest_path = output / "restart_manifest.json"
    _write_json(restart_manifest_path, restart_manifest)

    bundle_manifest = {
        "schema": BUNDLE_SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_root_recorded_for_audit_only": str(source),
        "source_git_commit": source_manifest.get("git_commit"),
        "checkpoint": {
            "schema": CHECKPOINT_SCHEMA,
            "beta_index": int(state["beta_index"]),
            "logical_attempt": int(state["attempt"]),
            "latent_shape": list(state["latent"].shape),
            "latent_range": [
                float(np.min(state["latent"])),
                float(np.max(state["latent"])),
            ],
            "terminal_latent_exact": True,
        },
        "last_completed_physics": latest,
        "files": {
            "checkpoint": _artifact(
                checkpoint_out, portable_name=checkpoint_out.name
            ),
            "terminal_stage_state": portable_terminal,
            "restart_manifest": _artifact(
                restart_manifest_path, portable_name=restart_manifest_path.name
            ),
        },
        "excluded": [
            "partial attempt-4 eval-7 forward/Jacobian artifacts",
            "GPU-UUID-bound RTX source calibrations",
            "FSP, H5, raw-Q, and CUDA pullback transients",
            "credentials, SSH keys, license files, and Codex state",
        ],
    }
    _write_json(output / "bundle_manifest.json", bundle_manifest)
    print(json.dumps(bundle_manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
