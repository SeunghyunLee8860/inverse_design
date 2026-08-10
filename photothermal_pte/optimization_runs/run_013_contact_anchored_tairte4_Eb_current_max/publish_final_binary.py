#!/usr/bin/env python3
"""Publish Run013 E||b by reusing the common exact-binary certificate renderer."""

from pathlib import Path

from photothermal_pte.optimization_runs.run_012_contact_anchored_tairte4_Ea_current_max import (
    publish_final_binary as shared,
)


shared.HERE = Path(__file__).resolve().parent
shared.RUN_LABEL = "Run013 E∥b"
shared.POLARIZATION_LABEL = "E∥b"
shared.FINALIZATION = Path(
    "/data/seunghyun/tairte4/artifacts/tairte4_contact_anchored/"
    "run013_Eb_exact_binary_finalization_20260810/binary_finalization_result.json"
)
shared.OBJECTIVE = Path(
    "/data/seunghyun/tairte4/artifacts/tairte4_contact_anchored/"
    "run013_Eb_exact_binary_objective_20260810/binary_objective_result.json"
)


if __name__ == "__main__":
    raise SystemExit(shared.main())
