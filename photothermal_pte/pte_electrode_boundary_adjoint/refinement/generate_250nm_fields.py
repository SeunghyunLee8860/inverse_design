from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tairte4_boundary_adjoint.baseline import load_config  # noqa: E402
from tairte4_pte.model import PTEModel  # noqa: E402


def file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def main() -> int:
    config_path = PROJECT_ROOT / "configs" / "per_beam_250nm.json"
    fields_path = HERE / "per_beam_250nm_fields.npz"
    report_path = HERE / "per_beam_250nm_thermal.json"
    config = load_config(config_path)
    print("assembling explicit 0.25 um air/TaIrTe4/SiO2/Si FVM", flush=True)
    model = PTEModel(config)
    print(f"thermal cell shape={model.thermal.system.shape}", flush=True)
    beams = model.solve_beams()
    temperature = np.asarray([beam.temperature_nodes_K for beam in beams])
    centers = np.asarray([beam.center_m for beam in beams])
    np.savez_compressed(
        fields_path,
        beam_centers_m=centers,
        temperature_nodes_K=temperature,
    )
    report = {
        "status": "PASS" if all(
            beam.linear_residual_relative <= 1e-10
            and beam.energy_balance_relative_error <= 1e-8
            for beam in beams
        ) else "FAIL",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "config_sha256": file_sha256(config_path),
        "reference_solver_provenance": model.thermal.reference_solver_provenance,
        "thermal_cell_shape": list(model.thermal.system.shape),
        "electrical_node_shape": list(model.electrical.mesh.shape),
        "beam_count": len(beams),
        "temperature_fields_path": str(fields_path),
        "temperature_fields_sha256": file_sha256(fields_path),
        "beams": [
            {
                "beam_index": index,
                "beam_center_um": (np.asarray(beam.center_m) * 1e6).tolist(),
                "temperature_min_K": float(np.min(beam.temperature_nodes_K)),
                "temperature_max_K": float(np.max(beam.temperature_nodes_K)),
                "in_flake_power_fraction": beam.in_flake_power_fraction,
                "source_power_W": beam.source_power_W,
                "linear_residual_relative": beam.linear_residual_relative,
                "energy_balance_relative_error": beam.energy_balance_relative_error,
                "solver": beam.solver,
                "iterations": beam.iterations,
            }
            for index, beam in enumerate(beams)
        ],
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
