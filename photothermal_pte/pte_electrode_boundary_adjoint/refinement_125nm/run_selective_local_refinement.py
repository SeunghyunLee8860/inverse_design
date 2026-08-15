from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tairte4_boundary_adjoint.baseline import ElectricalModel, load_config  # noqa: E402
from tairte4_boundary_adjoint.optimization import run_signed_slsqp  # noqa: E402
from tairte4_boundary_adjoint.robin import DifferentiableContactModel  # noqa: E402
from tairte4_boundary_adjoint.scaled import ScaledDesign, SignedBranchObjective  # noqa: E402


STARTS_PER_BRANCH = 4
TRANSITION_M = 0.50e-6
MESH_CHANGE_THRESHOLD = 0.01


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def canonical_key(x: np.ndarray) -> tuple[float, ...]:
    value = np.asarray(x, dtype=float).copy()
    value[[0, 2]] %= 1.0
    return tuple(np.round(value, 13))


def local_starts(strongest: np.ndarray, perimeter, minimum_gap_m: float) -> list[np.ndarray]:
    delta = 0.125e-6 / perimeter.perimeter_m
    candidates = [
        strongest,
        strongest[[2, 3, 0, 1]],
        strongest + delta * np.asarray([+1.0, 0.0, -1.0, 0.0]),
        strongest + delta * np.asarray([-1.0, 0.0, +1.0, 0.0]),
        strongest + delta * np.asarray([+1.0, 0.0, +1.0, 0.0]),
        strongest + delta * np.asarray([-1.0, 0.0, -1.0, 0.0]),
    ]
    starts = []
    seen = set()
    for candidate in candidates:
        constraints, _ = perimeter.separation_constraints_scaled(
            candidate, minimum_gap_m / perimeter.perimeter_m
        )
        key = canonical_key(candidate)
        if np.min(constraints) >= -1e-12 and key not in seen:
            starts.append(np.asarray(candidate, dtype=float).copy())
            seen.add(key)
        if len(starts) == STARTS_PER_BRANCH:
            break
    if len(starts) != STARTS_PER_BRANCH:
        raise RuntimeError("could not construct four feasible local starts")
    return starts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--beam-index", type=int, required=True)
    args = parser.parse_args()
    beam_index = args.beam_index

    config_path = PROJECT_ROOT / "configs" / "per_beam_125nm.json"
    fields_path = HERE / "per_beam_125nm_fields.npz"
    thermal_path = HERE / "per_beam_125nm_thermal.json"
    relaxation_path = HERE / "relaxation_125nm.json"
    final_250_path = PROJECT_ROOT / "refinement" / "transition_width_final_250nm.json"
    config = load_config(config_path)
    thermal = json.loads(thermal_path.read_text(encoding="utf-8"))
    relaxation = json.loads(relaxation_path.read_text(encoding="utf-8"))
    final_250 = json.loads(final_250_path.read_text(encoding="utf-8"))
    if thermal["status"] != "PASS" or relaxation["status"] != "PASS":
        raise RuntimeError("0.125 um thermal and relaxation gates must pass")
    thermal_row = thermal["beams"][beam_index]
    if abs(thermal_row["relative_current_change_250_to_125"]) <= MESH_CHANGE_THRESHOLD:
        raise ValueError(f"beam {beam_index} does not require selective refinement")
    with np.load(fields_path) as fields:
        centers = np.asarray(fields["beam_centers_m"])
        temperature = np.asarray(fields["temperature_nodes_K"][beam_index])
    electrical = ElectricalModel(config)
    selected_g = float(relaxation["selected_g_S_m2"])
    model = DifferentiableContactModel(
        electrical,
        temperature,
        contact_conductance_S_m2=selected_g,
        transition_m=TRANSITION_M,
        contact_discretization="nodal_lumped",
    )
    objective = SignedBranchObjective(model)
    econf = config["electrodes"]
    minimum_length_m = econf["min_contact_length_um"] * 1e-6
    maximum_length_m = econf["max_contact_fraction"] * (
        config["geometry"]["flake_width_um"] - 2 * econf["edge_clearance_um"]
    ) * 1e-6
    minimum_gap_m = econf["same_side_min_gap_um"] * 1e-6
    strongest = np.asarray(
        final_250["beams"][beam_index]["final_best"]["canonical_scaled"],
        dtype=float,
    )
    starts = local_starts(strongest, model.perimeter, minimum_gap_m)

    checkpoint_path = HERE / f"local_refinement_beam{beam_index:02d}_checkpoint.json"
    output_path = HERE / f"local_refinement_beam{beam_index:02d}_125nm.json"
    identity = {
        "beam_index": beam_index,
        "config_sha256": digest(config_path),
        "fields_sha256": digest(fields_path),
        "thermal_sha256": digest(thermal_path),
        "relaxation_sha256": digest(relaxation_path),
        "source_250nm_sha256": digest(final_250_path),
        "starts_per_branch": STARTS_PER_BRANCH,
        "transition_m": TRANSITION_M,
        "selected_g_S_m2": selected_g,
    }
    if checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint.get("identity") != identity:
            raise RuntimeError("selective refinement checkpoint identity mismatch")
    else:
        checkpoint = {"identity": identity, "runs": []}
    completed = {
        (int(row["branch_sign"]), int(row["start_index"])): row
        for row in checkpoint["runs"]
    }

    candidates = []
    for start_index, start in enumerate(starts):
        p = ScaledDesign.from_array(start).canonical().to_physical(
            model.perimeter.perimeter_m
        )
        hard = model.hard_evaluate(p)
        candidates.append(
            {
                "source": "initial_hard",
                "start_index": start_index,
                "hard_current_A": hard.current_A,
                "hard_abs_current_A": abs(hard.current_A),
                "canonical_scaled": ScaledDesign.from_array(start).canonical().as_array().tolist(),
            }
        )
    for branch_sign in (+1, -1):
        for start_index, start in enumerate(starts):
            run_key = (branch_sign, start_index)
            if run_key in completed:
                row = completed[run_key]
                print(
                    f"beam={beam_index:02d} branch={branch_sign:+d} "
                    f"start={start_index} resumed",
                    flush=True,
                )
            else:
                run = run_signed_slsqp(
                    objective,
                    start,
                    branch_sign=branch_sign,
                    minimum_length_m=minimum_length_m,
                    maximum_length_m=maximum_length_m,
                    minimum_gap_m=minimum_gap_m,
                    max_iterations=250,
                    function_tolerance=1e-11,
                )
                row = {
                    "branch_sign": branch_sign,
                    "start_index": start_index,
                    "success": bool(run.scipy_result.success),
                    "status_code": int(run.scipy_result.status),
                    "message": str(run.scipy_result.message),
                    "iterations": int(run.scipy_result.nit),
                    "endpoint_smooth_current_A": run.smooth.current_A,
                    "endpoint_hard_current_A": run.hard.current_A,
                    "endpoint_hard_abs_current_A": abs(run.hard.current_A),
                    "endpoint_canonical_scaled": run.smooth.canonical_design.as_array().tolist(),
                    "state_residual_relative": run.smooth.forward.state_residual_relative,
                    "adjoint_residual_relative": run.smooth.forward.adjoint_residual_relative,
                }
                checkpoint["runs"].append(row)
                atomic_json(checkpoint_path, checkpoint)
                completed[run_key] = row
                print(
                    f"beam={beam_index:02d} branch={branch_sign:+d} "
                    f"start={start_index} success={row['success']} "
                    f"nit={row['iterations']} hard={row['endpoint_hard_abs_current_A']:.6e}",
                    flush=True,
                )
            candidates.append(
                {
                    "source": "slsqp_endpoint_hard",
                    "branch_sign": branch_sign,
                    "start_index": start_index,
                    "hard_current_A": row["endpoint_hard_current_A"],
                    "hard_abs_current_A": row["endpoint_hard_abs_current_A"],
                    "canonical_scaled": row["endpoint_canonical_scaled"],
                }
            )

    best = max(candidates, key=lambda row: row["hard_abs_current_A"])
    transferred = abs(float(thermal_row["same_geometry_hard_current_125nm_A"]))
    result = {
        "status": "PASS" if all(row["success"] for row in checkpoint["runs"]) else "FAIL",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "beam_index": beam_index,
        "beam_center_um": (centers[beam_index] * 1e6).tolist(),
        "mesh_step_um": 0.125,
        "selected_g_S_m2": selected_g,
        "transition_width_um": TRANSITION_M * 1e6,
        "starts_per_signed_branch": STARTS_PER_BRANCH,
        "transferred_hard_abs_current_A": transferred,
        "best": best,
        "relative_improvement_over_transferred": (
            best["hard_abs_current_A"] - transferred
        ) / transferred,
        "all_slsqp_runs_successful": all(row["success"] for row in checkpoint["runs"]),
        "starts": [np.asarray(start).tolist() for start in starts],
        "runs": checkpoint["runs"],
        "identity": identity,
    }
    atomic_json(output_path, result)
    print(json.dumps({
        "status": result["status"],
        "beam_index": beam_index,
        "best_hard_abs_current_A": best["hard_abs_current_A"],
        "relative_improvement_over_transferred": result["relative_improvement_over_transferred"],
        "output": str(output_path),
    }, indent=2))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
