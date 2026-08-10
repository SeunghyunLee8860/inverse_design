#!/usr/bin/env python3
"""Apply an exact +/-1 objective convention to a passed signed-current gate."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, object]:
    return {"path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": sha256(path)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-result", type=Path, required=True)
    parser.add_argument("--source-raw", type=Path, required=True)
    parser.add_argument("--objective-sign", type=float, choices=(-1.0, 1.0), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    source_result = args.source_result.resolve()
    source_raw = args.source_raw.resolve()
    result = json.loads(source_result.read_text())
    if not result.get("passed"):
        raise RuntimeError("source signed-current preparation did not pass")
    recorded = result["raw_artifact"]["sha256"]
    if sha256(source_raw) != recorded:
        raise RuntimeError("source raw SHA mismatch")
    old_sign = float(result.get("objective_sign", 1.0))
    if old_sign != 1.0:
        raise RuntimeError("source gate is not the unmodified signed-current objective")
    data = np.load(source_raw)
    raw_out = output / "selected_full_latent_adjoint_preparation.npz"
    arrays = {name: np.asarray(data[name]) for name in data.files}
    for name in (
        "gradient_physical_A",
        "gradient_latent_A",
        "gradient_optical_A",
        "gradient_thermal_A",
    ):
        arrays[name] = args.objective_sign * np.asarray(arrays[name])
    np.savez_compressed(raw_out, **arrays)
    signed_current = float(result["objective_A"])
    result.update(
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "objective_A": args.objective_sign * signed_current,
            "signed_current_A": signed_current,
            "objective_sign": args.objective_sign,
            "objective_definition": "objective_sign * signed_current_A / incident power",
            "exact_fixed_sign_transform": True,
            "source_signed_gate": {
                "result": artifact(source_result),
                "raw": artifact(source_raw),
            },
            "raw_artifact": artifact(raw_out),
            "Maxwell_forward_solves_for_transform": 0,
            "Maxwell_adjoint_solves_for_transform": 0,
            "thermal_solves_for_transform": 0,
            "empirical_normalization": False,
            "gradient_rescaling": False,
        }
    )
    result_out = output / "selected_full_latent_adjoint_preparation_result.json"
    result_out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"result": artifact(result_out), "raw": artifact(raw_out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
