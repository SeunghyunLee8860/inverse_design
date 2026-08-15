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


REPRESENTATIVE_INDICES = (0, 1, 3, 4)


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
    all_centers = np.asarray(config["beam"]["centers_um"], dtype=float)
    config["beam"]["centers_um"] = [
        all_centers[index].tolist() for index in REPRESENTATIVE_INDICES
    ]
    final_250 = json.loads(final_250_path.read_text(encoding="utf-8"))
    if final_250["status"] != "PASS":
        raise RuntimeError("0.25 um final transition audit is not PASS")

    start = perf_counter()
    print("assembling explicit 0.125 um thermal/electrical model", flush=True)
    model = PTEModel(config)
    assembly_seconds = perf_counter() - start
    print(
        f"thermal shape={model.thermal.system.shape}, "
        f"electrical shape={model.electrical.mesh.shape}, "
        f"assembly={assembly_seconds:.2f}s",
        flush=True,
    )

    fields = []
    beam_rows = []
    total_start = perf_counter()
    for local_index, beam_index in enumerate(REPRESENTATIVE_INDICES):
        center_m = tuple(all_centers[beam_index] * 1e-6)
        solve_start = perf_counter()
        thermal = model.thermal.solve_beam(center_m)
        solve_seconds = perf_counter() - solve_start
        temperature = np.asarray(thermal.temperature_nodes_K)
        fields.append(temperature)
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
            "beam_center_um": all_centers[beam_index].tolist(),
            "solve_seconds": solve_seconds,
            "iterations": thermal.iterations,
            "solver": thermal.solver,
            "linear_residual_relative": thermal.linear_residual_relative,
            "energy_balance_relative_error": thermal.energy_balance_relative_error,
            "source_power_W": thermal.source_power_W,
            "temperature_min_K": float(np.min(temperature)),
            "temperature_max_K": float(np.max(temperature)),
            "hard_current_250nm_A": current_250,
            "same_geometry_hard_current_125nm_A": current_125,
            "relative_current_change_250_to_125": (current_125 - current_250)
            / current_250,
            "hard_residual_relative": hard.residual_relative,
            "canonical_scaled": scaled.tolist(),
        }
        beam_rows.append(row)
        print(
            f"beam={beam_index} center={row['beam_center_um']} "
            f"iterations={thermal.iterations} time={solve_seconds:.2f}s "
            f"delta_I={100*row['relative_current_change_250_to_125']:+.3f}%",
            flush=True,
        )

    fields_path = HERE / "representative_fields_125nm.npz"
    np.savez_compressed(
        fields_path,
        beam_indices=np.asarray(REPRESENTATIVE_INDICES, dtype=int),
        beam_centers_m=all_centers[list(REPRESENTATIVE_INDICES)] * 1e-6,
        temperature_nodes_K=np.asarray(fields),
    )
    max_change = max(abs(row["relative_current_change_250_to_125"]) for row in beam_rows)
    thermal_pass = all(
        row["linear_residual_relative"] <= 1e-10
        and abs(row["energy_balance_relative_error"]) <= 1e-8
        for row in beam_rows
    )
    report = {
        "status": "PASS" if thermal_pass else "FAIL",
        "mesh_convergence_status": (
            "REPRESENTATIVE_PASS_1PCT" if max_change <= 0.01
            else "REPRESENTATIVE_REFINEMENT_REQUIRED"
        ),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mesh_step_um": 0.125,
        "representative_indices": list(REPRESENTATIVE_INDICES),
        "thermal_cell_shape": list(model.thermal.system.shape),
        "electrical_node_shape": list(model.electrical.mesh.shape),
        "assembly_seconds": assembly_seconds,
        "total_solve_and_evaluation_seconds": perf_counter() - total_start,
        "maximum_absolute_relative_current_change": max_change,
        "config_path": str(config_path),
        "config_sha256": digest(config_path),
        "source_250nm_sha256": digest(final_250_path),
        "fields_path": str(fields_path),
        "fields_sha256": digest(fields_path),
        "reference_solver_provenance": model.thermal.reference_solver_provenance,
        "beams": beam_rows,
    }
    output = HERE / "representative_pilot_125nm.json"
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
