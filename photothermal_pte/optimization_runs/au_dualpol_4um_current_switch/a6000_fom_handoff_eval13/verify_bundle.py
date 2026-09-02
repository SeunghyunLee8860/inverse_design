#!/usr/bin/env python3
"""Fail-closed verification for the A6000 FOM-change handoff bundle."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    bundle = json.loads((ROOT / "bundle_manifest.json").read_text())
    if bundle.get("schema") != "au-lumerical-a6000-fom-handoff-bundle-v1":
        raise RuntimeError("unexpected bundle schema")
    for key in (
        "checkpoint",
        "stage_final_state",
        "latest_successful_state",
        "restart_manifest",
        "objective_history",
        "source_production_manifest",
    ):
        record = bundle[key]
        path = ROOT / record["path"]
        if path.stat().st_size != record["size_bytes"]:
            raise RuntimeError(f"{key} size mismatch")
        if sha256(path) != record["sha256"]:
            raise RuntimeError(f"{key} hash mismatch")
    with np.load(ROOT / "continuation_checkpoint.npz", allow_pickle=False) as checkpoint:
        checkpoint_latent = np.asarray(checkpoint["latent"])
        if int(checkpoint["beta_index"]) != 1 or int(checkpoint["attempt"]) != 6:
            raise RuntimeError("checkpoint beta/attempt mismatch")
    with np.load(ROOT / "stage_final_state.npz", allow_pickle=False) as state:
        terminal_latent = np.asarray(state["latent_final"])
    with np.load(ROOT / "latest_successful_state.npz", allow_pickle=False) as live:
        live_latent = np.asarray(live["latent"])
    if not np.array_equal(checkpoint_latent, terminal_latent):
        raise RuntimeError("checkpoint and terminal latent differ")
    if not np.array_equal(checkpoint_latent, live_latent):
        raise RuntimeError("checkpoint and live latent differ")
    restart = json.loads((ROOT / "restart_manifest.json").read_text())
    if restart.get("status") != "STOPPED_FOR_A6000_FOM_CHANGE_HANDOFF":
        raise RuntimeError("restart manifest is not explicitly stopped")
    if restart.get("restart_checkpoint_attempt") != 6:
        raise RuntimeError("restart attempt mismatch")
    if not all(bundle.get("gates", {}).values()):
        raise RuntimeError("bundle construction gate failed")
    print(json.dumps({"status": "PASSED_A6000_FOM_HANDOFF_BUNDLE", "snapshot": bundle["snapshot_latest"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
