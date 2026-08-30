#!/usr/bin/env python3
"""Resume Run052 v4 at beta=8 from the last committed beta=4 evaluation."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]
RAW = Path(
    "/data/seunghyun/tairte4/artifacts/tairte4_contact_anchored/"
    "run052_constraint_aware_Eb_evaporated_v4"
)
CHECKPOINT = RAW / "evaluation_0375_beta4_ansys_dfm_ld_mma_latent.npz"
CHECKPOINT_SHA256 = "4949347ad7f33e799b757ce89e5e42bc6fd3f43e21c7e13f3f57fd105da87e02"
PUBLISHED = HERE / "run_052_Eb_results_v4"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    if not CHECKPOINT.is_file() or sha256(CHECKPOINT) != CHECKPOINT_SHA256:
        raise RuntimeError("Run052 evaluation 375 latent checkpoint is missing or changed")
    history = json.loads((PUBLISHED / "optimization_history.json").read_text())
    if int(history[-1]["evaluation_id"]) != 375:
        raise RuntimeError(
            "published Run052 history must end at committed evaluation 375 before resume"
        )
    environment = dict(os.environ)
    environment.update(
        {
            "CONSTRAINT_AWARE_POLARIZATION": "Eb",
            "CONSTRAINT_AWARE_GPU": "1",
            "CONSTRAINT_AWARE_GENERATION": "v4",
            "CONSTRAINT_AWARE_FAST_CONTINUATION": "1",
            "CONSTRAINT_AWARE_RECOVERY_INITIAL_LATENT": str(CHECKPOINT),
            "CONSTRAINT_AWARE_RECOVERY_START_BETA": "8",
            "CONSTRAINT_AWARE_RECOVERY_OUTPUT_SLUG": (
                "ansys_dfm_ld_mma_v7_beta8_resume_from_eval375"
            ),
        }
    )
    return subprocess.run(
        [sys.executable, "-u", str(HERE / "run_one.py")],
        cwd=REPOSITORY,
        env=environment,
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
