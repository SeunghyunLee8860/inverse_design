from __future__ import annotations

import csv
from hashlib import sha256
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tairte4_boundary_adjoint.baseline import (  # noqa: E402
    BASELINE_ROOT,
    ElectricalModel,
    load_config,
)
from tairte4_boundary_adjoint.robin import DifferentiableContactModel  # noqa: E402
from tairte4_boundary_adjoint.scaled import ScaledDesign  # noqa: E402


VARIABLES = ("u0=c0/P", "l0=L0/P", "u1=c1/P", "l1=L1/P")
DESIGN_PHYSICAL_UM = np.asarray([6.1, 7.3, 58.4, 10.7], dtype=float)
FD_STEPS_SCALED = np.asarray(
    [1e-2, 5e-3, 2e-3, 1e-3, 5e-4, 2e-4, 1e-4, 5e-5, 2e-5, 1e-5, 5e-6],
    dtype=float,
)
CONTACT_CONDUCTANCE_S_M2 = 1.0e12
TRANSITION_WIDTH_M = 0.75e-6
BOUNDARY_QUADRATURE_ORDER = 5
PASS_MAX_COMPONENT_RELATIVE_ERROR = 1.0e-4
PASS_SECOND_ORDER_RANGE = (1.8, 2.2)


def file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    config_path = BASELINE_ROOT / "configs" / "per_beam_500nm.json"
    fields_path = (
        BASELINE_ROOT / "results" / "per_beam_500nm_final" / "per_beam_fields.npz"
    )
    config = load_config(config_path)
    with np.load(fields_path) as fields:
        centers_m = np.asarray(fields["beam_centers_m"], dtype=float)
        beam_index = int(np.argmin(np.linalg.norm(centers_m, axis=1)))
        temperature_nodes_K = np.asarray(
            fields["temperature_nodes_K"][beam_index], dtype=float
        )

    electrical = ElectricalModel(config)
    model = DifferentiableContactModel(
        electrical,
        temperature_nodes_K,
        contact_conductance_S_m2=CONTACT_CONDUCTANCE_S_M2,
        transition_m=TRANSITION_WIDTH_M,
        quadrature_order=BOUNDARY_QUADRATURE_ORDER,
    )
    perimeter_m = model.perimeter.perimeter_m
    x0 = DESIGN_PHYSICAL_UM * 1.0e-6 / perimeter_m
    parameters = ScaledDesign.from_array(x0).to_physical(perimeter_m)
    constraints, _ = model.perimeter.separation_constraints_scaled(
        x0, gap_fraction=0.5e-6 / perimeter_m
    )
    if np.any(constraints <= 0.0):
        raise RuntimeError(f"gradient-check design is not strictly feasible: {constraints}")

    base = model.evaluate(parameters)
    adjoint_scaled = perimeter_m * base.current_gradient_A_per_m
    gradient_floor = max(np.linalg.norm(adjoint_scaled), np.finfo(float).tiny) * 1e-12
    rows: list[dict[str, float]] = []
    arrays: list[dict[str, np.ndarray | float]] = []
    for step in FD_STEPS_SCALED:
        fd = np.zeros(4, dtype=float)
        plus_current = np.zeros(4, dtype=float)
        minus_current = np.zeros(4, dtype=float)
        for index in range(4):
            plus_x = x0.copy()
            minus_x = x0.copy()
            plus_x[index] += step
            minus_x[index] -= step
            plus = model.evaluate(
                ScaledDesign.from_array(plus_x).to_physical(perimeter_m)
            )
            minus = model.evaluate(
                ScaledDesign.from_array(minus_x).to_physical(perimeter_m)
            )
            plus_current[index] = plus.current_A
            minus_current[index] = minus.current_A
            fd[index] = (plus.current_A - minus.current_A) / (2.0 * step)
        absolute_error = np.abs(fd - adjoint_scaled)
        denominator = np.maximum.reduce(
            (np.abs(fd), np.abs(adjoint_scaled), np.full(4, gradient_floor))
        )
        relative_error = absolute_error / denominator
        vector_relative_error = float(
            np.linalg.norm(fd - adjoint_scaled)
            / max(np.linalg.norm(fd), np.linalg.norm(adjoint_scaled), gradient_floor)
        )
        row: dict[str, float] = {
            "h_scaled": float(step),
            "h_physical_um": float(step * perimeter_m * 1e6),
            "vector_relative_error": vector_relative_error,
            "max_component_relative_error": float(np.max(relative_error)),
        }
        for index, name in enumerate(("u0", "l0", "u1", "l1")):
            row[f"adjoint_{name}_A"] = float(adjoint_scaled[index])
            row[f"central_fd_{name}_A"] = float(fd[index])
            row[f"absolute_error_{name}_A"] = float(absolute_error[index])
            row[f"relative_error_{name}"] = float(relative_error[index])
            row[f"plus_current_{name}_A"] = float(plus_current[index])
            row[f"minus_current_{name}_A"] = float(minus_current[index])
        rows.append(row)
        arrays.append(
            {
                "step": float(step),
                "fd": fd,
                "absolute_error": absolute_error,
                "relative_error": relative_error,
            }
        )

    best_index = int(np.argmin([row["max_component_relative_error"] for row in rows]))
    best = rows[best_index]
    best_arrays = arrays[best_index]
    component_pass = np.asarray(best_arrays["relative_error"]) <= (
        PASS_MAX_COMPONENT_RELATIVE_ERROR
    )
    residual_pass = (
        base.state_residual_relative <= 1e-10
        and base.adjoint_residual_relative <= 1e-10
    )
    tail_max_errors = np.asarray(
        [entry["max_component_relative_error"] for entry in rows[-3:]], dtype=float
    )
    tail_steps = FD_STEPS_SCALED[-3:]
    tail_observed_orders = np.log(tail_max_errors[:-1] / tail_max_errors[1:]) / np.log(
        tail_steps[:-1] / tail_steps[1:]
    )
    second_order_pass = bool(
        np.all(tail_observed_orders >= PASS_SECOND_ORDER_RANGE[0])
        and np.all(tail_observed_orders <= PASS_SECOND_ORDER_RANGE[1])
    )
    passed = bool(np.all(component_pass) and residual_pass and second_order_pass)

    summary = {
        "status": "PASS" if passed else "FAIL",
        "scope": "raw current gradient only; no optimizer was called",
        "baseline_root": str(BASELINE_ROOT),
        "config_path": str(config_path),
        "config_sha256": file_sha256(config_path),
        "temperature_fields_path": str(fields_path),
        "temperature_fields_sha256": file_sha256(fields_path),
        "beam_index": beam_index,
        "beam_center_um": (centers_m[beam_index] * 1e6).tolist(),
        "temperature_shape": list(temperature_nodes_K.shape),
        "temperature_min_K": float(np.min(temperature_nodes_K)),
        "temperature_max_K": float(np.max(temperature_nodes_K)),
        "mesh_step_um": float(electrical.step_m * 1e6),
        "perimeter_um": float(perimeter_m * 1e6),
        "scaled_variable_order": list(VARIABLES),
        "design_physical_um": DESIGN_PHYSICAL_UM.tolist(),
        "design_scaled": x0.tolist(),
        "separation_constraints": constraints.tolist(),
        "contact_conductance_S_m2": CONTACT_CONDUCTANCE_S_M2,
        "transition_width_um": TRANSITION_WIDTH_M * 1e6,
        "boundary_quadrature_order": BOUNDARY_QUADRATURE_ORDER,
        "contact_discretization": base.contact_discretization,
        "current_A": base.current_A,
        "adjoint_gradient_A_per_scaled_variable": adjoint_scaled.tolist(),
        "state_residual_relative": base.state_residual_relative,
        "adjoint_residual_relative": base.adjoint_residual_relative,
        "matrix_relative_asymmetry": base.matrix_relative_asymmetry,
        "pass_tolerance_max_component_relative_error": (
            PASS_MAX_COMPONENT_RELATIVE_ERROR
        ),
        "best_h_scaled": best["h_scaled"],
        "best_h_physical_um": best["h_physical_um"],
        "best_central_fd_gradient_A_per_scaled_variable": np.asarray(
            best_arrays["fd"]
        ).tolist(),
        "best_component_absolute_errors_A": np.asarray(
            best_arrays["absolute_error"]
        ).tolist(),
        "best_component_relative_errors": np.asarray(
            best_arrays["relative_error"]
        ).tolist(),
        "best_vector_relative_error": best["vector_relative_error"],
        "best_max_component_relative_error": best[
            "max_component_relative_error"
        ],
        "component_pass": component_pass.tolist(),
        "residual_pass": residual_pass,
        "tail_steps_scaled_for_order_check": tail_steps.tolist(),
        "tail_max_component_relative_errors": tail_max_errors.tolist(),
        "tail_observed_central_fd_orders": tail_observed_orders.tolist(),
        "pass_second_order_range": list(PASS_SECOND_ORDER_RANGE),
        "second_order_pass": second_order_pass,
        "evaluations": rows,
    }
    output_json = HERE / "phase3_gradient_check.json"
    output_csv = HERE / "phase3_gradient_check.csv"
    output_plot = HERE / "phase3_gradient_check.png"
    output_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_csv(output_csv, rows)

    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    steps = np.asarray([entry["step"] for entry in arrays], dtype=float)
    for index, variable in enumerate(("u0", "l0", "u1", "l1")):
        errors = np.asarray(
            [np.asarray(entry["relative_error"])[index] for entry in arrays]
        )
        ax.loglog(steps, errors, "o-", label=variable)
    ax.axhline(
        PASS_MAX_COMPONENT_RELATIVE_ERROR,
        color="black",
        linestyle="--",
        linewidth=1.0,
        label="pass tolerance",
    )
    ax.set_xlabel("central-FD step h in scaled variable")
    ax.set_ylabel("componentwise relative error")
    ax.set_title("Actual baseline T: adjoint vs central FD for raw current")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.savefig(output_plot, dpi=180)
    plt.close(fig)

    print(json.dumps(summary, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
