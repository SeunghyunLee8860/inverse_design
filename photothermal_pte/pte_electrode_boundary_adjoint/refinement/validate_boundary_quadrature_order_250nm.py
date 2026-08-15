from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tairte4_boundary_adjoint.baseline import ElectricalModel, load_config  # noqa: E402
from tairte4_boundary_adjoint.robin import DifferentiableContactModel  # noqa: E402
from tairte4_boundary_adjoint.scaled import ScaledDesign  # noqa: E402


ORDERS = (3, 5, 7, 9)
WIDTHS_UM = (0.25, 0.50, 0.75, 1.00)
CONTACT_G_S_M2 = 1e14
TOLERANCE = 1e-12


def main() -> int:
    final = json.loads(
        (HERE / "transition_width_final_250nm.json").read_text(encoding="utf-8")
    )
    if final["status"] != "PASS":
        raise RuntimeError("transition-width gate must pass first")
    config = load_config(PROJECT_ROOT / "configs" / "per_beam_250nm.json")
    electrical = ElectricalModel(config)
    with np.load(HERE / "per_beam_250nm_fields.npz") as fields:
        temperatures = np.asarray(fields["temperature_nodes_K"])

    cases = []
    maxima = {
        "current_relative_difference": 0.0,
        "gradient_scaled_max_difference": 0.0,
        "psi_relative_l2_difference": 0.0,
        "contact_integral_relative_difference": 0.0,
    }
    for beam in final["beams"]:
        beam_index = int(beam["beam_index"])
        scaled = np.asarray(beam["final_best"]["canonical_scaled"], dtype=float)
        for width_um in WIDTHS_UM:
            evaluations = {}
            for order in ORDERS:
                model = DifferentiableContactModel(
                    electrical,
                    temperatures[beam_index],
                    contact_conductance_S_m2=CONTACT_G_S_M2,
                    transition_m=width_um * 1e-6,
                    quadrature_order=order,
                    contact_discretization="nodal_lumped",
                )
                parameters = ScaledDesign.from_array(scaled).canonical().to_physical(
                    model.perimeter.perimeter_m
                )
                evaluation = model.evaluate(parameters)
                evaluations[order] = evaluation
            reference = evaluations[ORDERS[-1]]
            order_rows = []
            for order in ORDERS:
                evaluation = evaluations[order]
                current_difference = abs(evaluation.current_A - reference.current_A) / max(
                    abs(reference.current_A), np.finfo(float).tiny
                )
                gradient_difference = np.max(
                    np.abs(
                        evaluation.current_gradient_A_per_m
                        - reference.current_gradient_A_per_m
                    )
                ) / max(
                    np.max(np.abs(reference.current_gradient_A_per_m)),
                    np.finfo(float).tiny,
                )
                psi_difference = np.linalg.norm(
                    evaluation.weighting_potential - reference.weighting_potential
                ) / max(
                    np.linalg.norm(reference.weighting_potential),
                    np.finfo(float).tiny,
                )
                integral_difference = np.max(
                    np.abs(
                        np.asarray(evaluation.contact_integrals_m)
                        - np.asarray(reference.contact_integrals_m)
                    )
                ) / max(
                    np.max(np.abs(reference.contact_integrals_m)),
                    np.finfo(float).tiny,
                )
                values = {
                    "quadrature_order": order,
                    "current_A": evaluation.current_A,
                    "gradient_A_per_m": evaluation.current_gradient_A_per_m.tolist(),
                    "current_relative_difference_from_order_9": current_difference,
                    "gradient_scaled_max_difference_from_order_9": gradient_difference,
                    "psi_relative_l2_difference_from_order_9": psi_difference,
                    "contact_integral_relative_difference_from_order_9": integral_difference,
                }
                order_rows.append(values)
                maxima["current_relative_difference"] = max(
                    maxima["current_relative_difference"], current_difference
                )
                maxima["gradient_scaled_max_difference"] = max(
                    maxima["gradient_scaled_max_difference"], gradient_difference
                )
                maxima["psi_relative_l2_difference"] = max(
                    maxima["psi_relative_l2_difference"], psi_difference
                )
                maxima["contact_integral_relative_difference"] = max(
                    maxima["contact_integral_relative_difference"], integral_difference
                )
            cases.append(
                {
                    "beam_index": beam_index,
                    "beam_center_um": beam["beam_center_um"],
                    "transition_width_um": width_um,
                    "orders": order_rows,
                }
            )

    status = "PASS" if max(maxima.values()) <= TOLERANCE else "FAIL"
    result = {
        "status": status,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mesh_step_um": 0.25,
        "contact_discretization": "nodal_lumped",
        "contact_g_S_m2": CONTACT_G_S_M2,
        "quadrature_orders": list(ORDERS),
        "transition_widths_um": list(WIDTHS_UM),
        "tolerance": TOLERANCE,
        "explanation": (
            "The production nodal-lumped contact uses boundary-node trapezoidal "
            "weights; Gaussian edge quadrature arrays are intentionally unused. "
            "The order sweep is therefore an invariance gate, not an asymptotic "
            "consistent-edge convergence claim."
        ),
        "maxima": maxima,
        "cases": cases,
    }
    output = HERE / "boundary_quadrature_order_250nm.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "maxima": maxima, "output": str(output)}, indent=2))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
