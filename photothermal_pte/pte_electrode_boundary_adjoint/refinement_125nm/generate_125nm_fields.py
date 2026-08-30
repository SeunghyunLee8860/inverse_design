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


RESIDUAL_GATE = 7e-11
ENERGY_GATE = 1e-8


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    config_path = PROJECT_ROOT / "configs" / "per_beam_125nm.json"
    final_250_path = PROJECT_ROOT / "refinement" / "transition_width_final_250nm.json"
    config = load_config(config_path)
    final_250 = json.loads(final_250_path.read_text(encoding="utf-8"))
    centers_um = np.asarray(config["beam"]["centers_um"], dtype=float)
    identity = {
        "config_sha256": digest(config_path),
        "source_250nm_sha256": digest(final_250_path),
        "residual_gate": RESIDUAL_GATE,
        "energy_gate": ENERGY_GATE,
    }
    checkpoint_path = HERE / "fields_125nm_checkpoint.json"
    if checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint.get("identity") != identity:
            raise RuntimeError("0.125 um field checkpoint identity mismatch")
    else:
        checkpoint = {"identity": identity, "beams": []}
    completed = {int(row["beam_index"]): row for row in checkpoint["beams"]}

    print("assembling explicit 0.125 um air/TaIrTe4/SiO2/Si FVM", flush=True)
    assembly_start = perf_counter()
    model = PTEModel(config)
    assembly_seconds = perf_counter() - assembly_start
    field_dir = HERE / "field_checkpoints"
    field_dir.mkdir(exist_ok=True)
    for beam_index, center_um in enumerate(centers_um):
        field_path = field_dir / f"beam_{beam_index:02d}.npz"
        if beam_index in completed and field_path.exists():
            if digest(field_path) != completed[beam_index]["field_sha256"]:
                raise RuntimeError(f"field checkpoint hash mismatch for beam {beam_index}")
            print(f"beam={beam_index:02d}/08 resumed", flush=True)
            continue
        solve_start = perf_counter()
        thermal = model.thermal.solve_beam(tuple(center_um * 1e-6))
        solve_seconds = perf_counter() - solve_start
        temperature = np.asarray(thermal.temperature_nodes_K)
        np.savez_compressed(
            field_path,
            beam_index=np.asarray(beam_index),
            beam_center_m=center_um * 1e-6,
            temperature_nodes_K=temperature,
        )
        contact_model = DifferentiableContactModel(
            model.electrical,
            temperature,
            contact_conductance_S_m2=1e14,
            transition_m=0.75e-6,
            contact_discretization="nodal_lumped",
        )
        scaled = np.asarray(
            final_250["beams"][beam_index]["final_best"]["canonical_scaled"],
            dtype=float,
        )
        parameters = ScaledDesign.from_array(scaled).canonical().to_physical(
            contact_model.perimeter.perimeter_m
        )
        hard = contact_model.hard_evaluate(parameters)
        current_250 = float(
            final_250["beams"][beam_index]["final_best"]["hard_abs_current_A"]
        )
        current_125 = abs(hard.current_A)
        row = {
            "beam_index": beam_index,
            "beam_center_um": center_um.tolist(),
            "solve_seconds": solve_seconds,
            "iterations": thermal.iterations,
            "solver": thermal.solver,
            "linear_residual_relative": thermal.linear_residual_relative,
            "energy_balance_relative_error": thermal.energy_balance_relative_error,
            "source_power_W": thermal.source_power_W,
            "in_flake_power_fraction": thermal.in_flake_power_fraction,
            "temperature_min_K": float(np.min(temperature)),
            "temperature_max_K": float(np.max(temperature)),
            "hard_current_250nm_A": current_250,
            "same_geometry_hard_current_125nm_A": current_125,
            "relative_current_change_250_to_125": (current_125 - current_250)
            / current_250,
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
            f"beam={beam_index:02d}/08 iterations={thermal.iterations} "
            f"time={solve_seconds:.2f}s "
            f"delta_I={100*row['relative_current_change_250_to_125']:+.3f}%",
            flush=True,
        )

    rows = [completed[index] for index in range(len(centers_um))]
    fields = []
    centers_m = []
    for row in rows:
        with np.load(row["field_path"]) as field:
            fields.append(np.asarray(field["temperature_nodes_K"]))
            centers_m.append(np.asarray(field["beam_center_m"]))
    fields_path = HERE / "per_beam_125nm_fields.npz"
    np.savez_compressed(
        fields_path,
        beam_centers_m=np.asarray(centers_m),
        temperature_nodes_K=np.asarray(fields),
    )
    max_change = max(abs(row["relative_current_change_250_to_125"]) for row in rows)
    thermal_pass = all(
        row["linear_residual_relative"] <= RESIDUAL_GATE
        and abs(row["energy_balance_relative_error"]) <= ENERGY_GATE
        for row in rows
    )
    report = {
        "status": "PASS" if thermal_pass else "FAIL",
        "mesh_convergence_status": (
            "PASS_1PCT" if max_change <= 0.01 else "REFINEMENT_REQUIRED"
        ),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mesh_step_um": 0.125,
        "thermal_cell_shape": list(model.thermal.system.shape),
        "electrical_node_shape": list(model.electrical.mesh.shape),
        "assembly_seconds": assembly_seconds,
        "maximum_absolute_relative_current_change": max_change,
        "residual_gate": RESIDUAL_GATE,
        "energy_gate": ENERGY_GATE,
        "config_path": str(config_path),
        "config_sha256": digest(config_path),
        "source_250nm_sha256": digest(final_250_path),
        "temperature_fields_path": str(fields_path),
        "temperature_fields_sha256": digest(fields_path),
        "reference_solver_provenance": model.thermal.reference_solver_provenance,
        "beams": rows,
    }
    output = HERE / "per_beam_125nm_thermal.json"
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
