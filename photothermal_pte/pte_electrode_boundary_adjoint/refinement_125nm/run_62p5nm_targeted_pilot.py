from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tairte4_boundary_adjoint.baseline import load_config  # noqa: E402
from tairte4_boundary_adjoint.robin import DifferentiableContactModel  # noqa: E402
from tairte4_boundary_adjoint.scaled import ScaledDesign  # noqa: E402
from tairte4_pte.model import PTEModel  # noqa: E402


TARGET_INDICES = (0, 1)
# At 6.05 million cells the post-solve unpreconditioned matvec residual reaches
# a roughly 3e-10 floating-point cancellation floor despite normal CG exit and
# 1e-12 energy balance.  Keep this gate separate from the 1% mesh-current gate.
RESIDUAL_GATE = 5e-10
ENERGY_GATE = 1e-8
MESH_TOLERANCE = 0.01


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    config_path = PROJECT_ROOT / "configs" / "per_beam_62p5nm.json"
    final_125_path = HERE / "final_125nm.json"
    config = load_config(config_path)
    centers_um = np.asarray(config["beam"]["centers_um"], dtype=float)
    final_125 = json.loads(final_125_path.read_text(encoding="utf-8"))
    identity = {
        "config_sha256": digest(config_path),
        "source_125nm_sha256": digest(final_125_path),
        "target_indices": list(TARGET_INDICES),
    }
    checkpoint_path = HERE / "targeted_62p5nm_checkpoint.json"
    if checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint.get("identity") != identity:
            raise RuntimeError("62.5 nm pilot checkpoint identity mismatch")
    else:
        checkpoint = {"identity": identity, "beams": []}
    completed = {int(row["beam_index"]): row for row in checkpoint["beams"]}

    config["beam"]["centers_um"] = [centers_um[index].tolist() for index in TARGET_INDICES]
    assembly_start = perf_counter()
    print("assembling explicit 62.5 nm targeted thermal model", flush=True)
    model = PTEModel(config)
    assembly_seconds = perf_counter() - assembly_start
    print(
        f"thermal shape={model.thermal.system.shape}, "
        f"electrical shape={model.electrical.mesh.shape}, "
        f"assembly={assembly_seconds:.2f}s",
        flush=True,
    )
    field_dir = HERE / "field_checkpoints_62p5nm"
    field_dir.mkdir(exist_ok=True)
    for beam_index in TARGET_INDICES:
        field_path = field_dir / f"beam_{beam_index:02d}.npz"
        if beam_index in completed and field_path.exists():
            if digest(field_path) != completed[beam_index]["field_sha256"]:
                raise RuntimeError(f"62.5 nm field hash mismatch for beam {beam_index}")
            print(f"beam={beam_index:02d} resumed", flush=True)
            continue
        solve_start = perf_counter()
        thermal = model.thermal.solve_beam(tuple(centers_um[beam_index] * 1e-6))
        solve_seconds = perf_counter() - solve_start
        temperature = np.asarray(thermal.temperature_nodes_K)
        np.savez_compressed(
            field_path,
            beam_index=np.asarray(beam_index),
            beam_center_m=centers_um[beam_index] * 1e-6,
            temperature_nodes_K=temperature,
        )
        contact_model = DifferentiableContactModel(
            model.electrical,
            temperature,
            contact_conductance_S_m2=1e14,
            transition_m=0.50e-6,
            contact_discretization="nodal_lumped",
        )
        scaled = np.asarray(
            final_125["beams"][beam_index]["best"]["canonical_scaled"],
            dtype=float,
        )
        parameters = ScaledDesign.from_array(scaled).canonical().to_physical(
            contact_model.perimeter.perimeter_m
        )
        hard = contact_model.hard_evaluate(parameters)
        current_125 = float(final_125["beams"][beam_index]["best_hard_abs_current_125nm_A"])
        current_62p5 = abs(hard.current_A)
        row = {
            "beam_index": beam_index,
            "beam_center_um": centers_um[beam_index].tolist(),
            "solve_seconds": solve_seconds,
            "iterations": thermal.iterations,
            "solver": thermal.solver,
            "linear_residual_relative": thermal.linear_residual_relative,
            "energy_balance_relative_error": thermal.energy_balance_relative_error,
            "source_power_W": thermal.source_power_W,
            "temperature_min_K": float(np.min(temperature)),
            "temperature_max_K": float(np.max(temperature)),
            "hard_current_125nm_A": current_125,
            "same_geometry_hard_current_62p5nm_A": current_62p5,
            "relative_current_change_125_to_62p5": (current_62p5 - current_125)
            / current_125,
            "hard_residual_relative": hard.residual_relative,
            "canonical_scaled": scaled.tolist(),
            "field_path": str(field_path),
            "field_sha256": digest(field_path),
        }
        checkpoint["beams"] = [
            prior for prior in checkpoint["beams"]
            if int(prior["beam_index"]) != beam_index
        ] + [row]
        checkpoint["beams"].sort(key=lambda item: int(item["beam_index"]))
        atomic_json(checkpoint_path, checkpoint)
        completed[beam_index] = row
        print(
            f"beam={beam_index:02d} iterations={thermal.iterations} "
            f"time={solve_seconds:.2f}s "
            f"delta_I={100*row['relative_current_change_125_to_62p5']:+.3f}%",
            flush=True,
        )

    rows = [completed[index] for index in TARGET_INDICES]
    max_change = max(abs(row["relative_current_change_125_to_62p5"]) for row in rows)
    thermal_pass = all(
        row["linear_residual_relative"] <= RESIDUAL_GATE
        and abs(row["energy_balance_relative_error"]) <= ENERGY_GATE
        for row in rows
    )
    report = {
        "status": "PASS" if thermal_pass else "FAIL",
        "mesh_convergence_status": (
            "TARGETED_PASS_1PCT" if max_change <= MESH_TOLERANCE
            else "TARGETED_REFINEMENT_REQUIRED"
        ),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mesh_step_um": 0.0625,
        "target_indices": list(TARGET_INDICES),
        "mesh_relative_tolerance": MESH_TOLERANCE,
        "thermal_cell_shape": list(model.thermal.system.shape),
        "electrical_node_shape": list(model.electrical.mesh.shape),
        "assembly_seconds": assembly_seconds,
        "maximum_absolute_relative_current_change": max_change,
        "residual_gate": RESIDUAL_GATE,
        "energy_gate": ENERGY_GATE,
        "config_path": str(config_path),
        "config_sha256": digest(config_path),
        "source_125nm_sha256": digest(final_125_path),
        "reference_solver_provenance": model.thermal.reference_solver_provenance,
        "beams": rows,
    }
    output = HERE / "targeted_62p5nm_pilot.json"
    atomic_json(output, report)
    print(json.dumps({
        "status": report["status"],
        "mesh_convergence_status": report["mesh_convergence_status"],
        "maximum_absolute_relative_current_change": max_change,
        "output": str(output),
    }, indent=2))
    return 0 if thermal_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
