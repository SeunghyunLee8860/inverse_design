#!/usr/bin/env python3
"""Force Run051 evaluation 223 to an exact 500-nm binary design."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("TAIRTE4_TOPOLOGY_GEOMETRY", "contact_anchored")
os.environ.setdefault("TAIRTE4_SIO2_INTERFACE_SCENARIO", "evaporated")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from photothermal_pte.optimization_runs.tairte4_flake_topology.contract import CONTRACT
from photothermal_pte.optimization_runs.tairte4_flake_topology.optimization_support import (
    exact_binary_audit,
)
from photothermal_pte.optimization_runs.tairte4_flake_topology.run_ansys_dfm_ld_mma_optimization import (
    SCHEMA,
    evaluate_exact_cleanup_candidates,
    final_geometry_gate,
)


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]
RAW = Path(
    "/data/seunghyun/tairte4/artifacts/tairte4_contact_anchored/"
    "run051_constraint_aware_Ea_evaporated_v3"
)
PUBLISHED = HERE / "run_051_Ea_results_v3"
REFERENCE_EVALUATION = PUBLISHED / "evaluation_0223.json"
REFERENCE_RHO = RAW / (
    "evaluation_0223_beta8_ansys_dfm_ld_mma_adaptive_v6_beta8_gpu1_rho.npz"
)
REFERENCE_RHO_SHA256 = "61a24c01f10362c97417aafbcd59de79eab6b4d11a60f788f06463b7d2190445"
REFERENCE_JSON_SHA256 = "c9bc0bc097e6188341a70bbd0d3ea9af1196150fb39773e66fc1e34d6d84a192"
BASE_FSP = RAW.parent / (
    "production_input_uniform_rho0p5_Ea_forward_v1/tairte4_flake_forward_Ea.fsp"
)
BASE_SHA256 = "454fa83bc918b4db0e25d28f7debf23de38977038bd716c8d7dc539d6b3e3d83"
GPU = int(os.environ.get("RUN051_CLEANUP_GPU", "1"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def plot_binary(binary: np.ndarray, output: Path) -> None:
    bounds = CONTRACT.design_bounds_m
    extent = [
        bounds["x"][0] * 1.0e6,
        bounds["x"][1] * 1.0e6,
        bounds["y"][0] * 1.0e6,
        bounds["y"][1] * 1.0e6,
    ]
    figure, axis = plt.subplots(figsize=(8.5, 7.2), constrained_layout=True)
    image = axis.imshow(
        binary.T,
        origin="lower",
        extent=extent,
        cmap="gray_r",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
        aspect="equal",
    )
    axis.set_title("Run051 final exact binary density: black=TaIrTe4, white=void")
    axis.set_xlabel("Lumerical x = b (um)")
    axis.set_ylabel("Lumerical y = a (um)")
    figure.colorbar(image, ax=axis, label="physical density")
    figure.savefig(output, dpi=220)
    plt.close(figure)


def main() -> int:
    if sha256(REFERENCE_RHO) != REFERENCE_RHO_SHA256:
        raise RuntimeError("Run051 evaluation 223 rho checkpoint changed")
    if sha256(REFERENCE_EVALUATION) != REFERENCE_JSON_SHA256:
        raise RuntimeError("Run051 evaluation 223 publication changed")
    if sha256(BASE_FSP) != BASE_SHA256:
        raise RuntimeError("Run051 base FSP changed")
    with np.load(REFERENCE_RHO) as loaded:
        rho = np.asarray(loaded["rho"], dtype=np.float64)
    reference = json.loads(REFERENCE_EVALUATION.read_text())
    forced = evaluate_exact_cleanup_candidates(
        rho,
        raw_root=RAW,
        base_fsp=BASE_FSP,
        base_sha256=BASE_SHA256,
        polarization="Ea",
        gpu=GPU,
        reference_objective_A=float(reference["objective_A"]),
        attempt_label="user_forced_eval0223",
    )
    if forced.get("selected") is None:
        raise RuntimeError("no exact-binary cleanup candidate passed the numerical gates")
    selected = str(forced["selected"])
    selected_row = forced["candidates"][selected]
    with np.load(selected_row["density"]["path"]) as loaded:
        binary = np.asarray(loaded["rho"], dtype=np.float64)
    exact, _ = exact_binary_audit(binary)
    if not exact["passed"] or not np.all((binary == 0.0) | (binary == 1.0)):
        raise RuntimeError("selected cleanup is not exact binary and 500-nm feasible")
    final_density = RAW / "final_user_forced_exact_binary_density.npz"
    np.savez_compressed(final_density, rho=binary)
    plot_path = PUBLISHED / "final_user_forced_exact_binary_density.png"
    plot_binary(binary, plot_path)
    binary_result = selected_row["result"]
    final = {
        "schema": SCHEMA,
        "status": "COMPLETED_USER_FORCED_EXACT_BINARY_500NM",
        "passed": True,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "polarization": "Ea",
        "substrate_interface": "evaporated_SiO2",
        "axis_contract": "Lumerical x=b, y=a",
        "reference_evaluation": 223,
        "reference_continuous_objective_A": float(reference["objective_A"]),
        "reference_equivalent_285uW_A": float(reference["objective_at_reference_power_A"]),
        "selected_order": selected,
        "selected_binary_objective_A": float(binary_result["objective_A"]),
        "selected_binary_equivalent_285uW_A": float(
            binary_result["equivalent_objective_at_285uW_A"]
        ),
        "relative_objective_change_from_continuous": float(
            binary_result["relative_objective_change_from_continuous"]
        ),
        "objective_preserved_within_one_percent": bool(
            binary_result["binary_objective_preserved_within_one_percent"]
        ),
        "selection_policy": (
            "user-requested exact cleanup; select the higher fresh unrescaled "
            "terminal current among solid-first and void-first candidates"
        ),
        "final_geometry_gate": final_geometry_gate(binary),
        "forced_exact_cleanup": forced,
        "final_density": {
            "path": str(final_density),
            "size_bytes": final_density.stat().st_size,
            "sha256": sha256(final_density),
        },
        "plot": str(plot_path),
    }
    write_json(PUBLISHED / "FORCED_EXACT_CLEANUP_USER_REQUESTED.json", forced)
    write_json(PUBLISHED / "FINAL_USER_FORCED_EXACT_BINARY.json", final)
    manifest_path = PUBLISHED / "RAW_ARTIFACT_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["user_requested_forced_exact_cleanup"] = forced
    manifest["user_requested_final_exact_binary"] = final
    write_json(manifest_path, manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
