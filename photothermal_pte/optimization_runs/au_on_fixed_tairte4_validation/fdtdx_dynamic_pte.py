"""In-memory explicit-PTE evaluation and native-Yee source pullback.

The functions here are iteration-local: they consume the current native-Yee
Maxwell heat source, solve the current thermal/electrical systems, and return
both direct density gradients and the current ``dI/dp_Yee`` weights.  No raw-Q
file round trip, frozen source weight, clipping, or rescaling is used.
"""

from __future__ import annotations

from typing import Any

import numpy as np


MATERIALS = ("au", "tairte4", "sio2")
COMPONENTS = ("x", "y", "z")


def component_coordinates(realized_grid, grid_slice, component: int):
    coordinates = []
    widths = []
    for axis, part in enumerate(grid_slice):
        edges = np.asarray(realized_grid.edges(axis), dtype=np.float64)
        centers = 0.5 * (edges[:-1] + edges[1:])
        primal_width = np.diff(edges)
        edge_dual_width = 0.5 * (
            np.concatenate((primal_width[:1], primal_width[:-1])) + primal_width
        )
        sample = centers if axis == component else edges[:-1]
        metric = primal_width if axis == component else edge_dual_width
        coordinates.append(sample[int(part.start) : int(part.stop)])
        widths.append(metric[int(part.start) : int(part.stop)])
    return tuple(coordinates), tuple(widths)


def evaluate_and_pullback(
    *,
    rho: np.ndarray,
    q_fields_W_m3: dict[str, np.ndarray],
    dual_volumes_m3: dict[str, np.ndarray],
    material_slices: dict[str, tuple[slice, slice, slice]],
    realized_grid,
    scenario: str,
    cuda_device: int,
    overlap,
    forward,
    stage67,
    electrical,
    coupled,
    topology,
    fvm,
) -> dict[str, Any]:
    """Return current objective, direct gradients, and native Yee weights."""

    state = forward._thermal_state(
        rho, forward.G_TA_SIO2_SCENARIOS[scenario], topology, fvm
    )
    source_power = np.zeros(state["system"].shape, dtype=np.float64)
    operators: dict[str, dict[str, Any]] = {}
    mapping: dict[str, dict[str, float]] = {}
    native_power: dict[str, list[np.ndarray]] = {}

    for material in MATERIALS:
        material_mask = state["masks"][material]
        indices = (
            np.flatnonzero(np.any(material_mask, axis=(1, 2))),
            np.flatnonzero(np.any(material_mask, axis=(0, 2))),
            np.flatnonzero(np.any(material_mask, axis=(0, 1))),
        )
        explicit_edges = tuple(
            state["edges"][axis][index[0] : index[-1] + 2]
            for axis, index in enumerate(indices)
        )

        component_geometry = {
            component: component_coordinates(
                realized_grid, material_slices[material], component_index
            )
            for component_index, component in enumerate(COMPONENTS)
        }
        primal_edges = tuple(
            overlap._primal_edges(
                component_geometry[COMPONENTS[axis]][0][axis],
                component_geometry[COMPONENTS[axis]][1][axis],
            )
            for axis in range(3)
        )
        primal_centers = tuple(0.5 * (edge[:-1] + edge[1:]) for edge in primal_edges)
        primal_widths = tuple(np.diff(edge) for edge in primal_edges)
        first_operators = {}
        native_power[material] = []
        primal_total = np.zeros(
            tuple(len(edge) - 1 for edge in primal_edges), dtype=np.float64
        )
        for component_index, component in enumerate(COMPONENTS):
            coordinates, widths = component_geometry[component]
            component_operators = tuple(
                overlap._overlap_operator(
                    coordinates[axis], widths[axis], primal_edges[axis]
                )[0]
                for axis in range(3)
            )
            first_operators[component] = component_operators
            power = (
                np.asarray(q_fields_W_m3[material][component_index], dtype=np.float64)
                * np.asarray(
                    dual_volumes_m3[material][component_index], dtype=np.float64
                )
            )
            native_power[material].append(power)
            primal_total += overlap._forward(power, component_operators)

        second_operators = tuple(
            overlap._overlap_operator(
                primal_centers[axis], primal_widths[axis], explicit_edges[axis]
            )[0]
            for axis in range(3)
        )
        explicit_power = overlap._forward(primal_total, second_operators)
        source_power[np.ix_(*indices)] += explicit_power
        source_value = float(
            sum(np.sum(value) for value in native_power[material])
        )
        target_value = float(np.sum(explicit_power))
        mapping[material] = {
            "source_power_W": source_value,
            "mapped_power_W": target_value,
            "relative_error": abs(source_value - target_value)
            / max(abs(source_value), np.finfo(float).tiny),
        }
        operators[material] = {
            "indices": indices,
            "first": first_operators,
            "second": second_operators,
        }

    evaluated = stage67._evaluate(
        rho,
        source_power,
        scenario,
        cuda_device,
        need_gradient=True,
        forward=forward,
        electrical=electrical,
        coupled=coupled,
        topology=topology,
        fvm=fvm,
    )
    thermal_adjoint = np.asarray(evaluated["thermal_adjoint"], dtype=np.float64).reshape(
        state["system"].shape
    )
    weights: dict[str, np.ndarray] = {}
    native_contraction = 0.0
    explicit_contraction = float(np.vdot(thermal_adjoint, source_power))
    for material in MATERIALS:
        item = operators[material]
        explicit_lambda = thermal_adjoint[np.ix_(*item["indices"])]
        primal_lambda = overlap._transpose(explicit_lambda, item["second"])
        components = []
        for component_index, component in enumerate(COMPONENTS):
            native_weight = overlap._transpose(
                primal_lambda, item["first"][component]
            )
            components.append(native_weight)
            native_contraction += float(
                np.vdot(native_weight, native_power[material][component_index])
            )
        weights[material] = np.stack(components)

    return {
        "objective_A": float(evaluated["objective_A"]),
        "temperature": evaluated["temperature"],
        "weighting": evaluated["weighting"],
        "thermal_adjoint": evaluated["thermal_adjoint"],
        "gradient_thermal_A": np.asarray(evaluated["gradient_thermal"]),
        "gradient_electrical_A": np.asarray(evaluated["gradient_electrical"]),
        "native_weights_A_W": weights,
        "source_power_W": source_power,
        "mapping": mapping,
        "native_weighted_contraction_A": native_contraction,
        "explicit_weighted_contraction_A": explicit_contraction,
        "weighted_contraction_relative_error": abs(
            native_contraction - explicit_contraction
        ) / max(abs(explicit_contraction), np.finfo(float).tiny),
        "thermal_residual": evaluated["thermal_residual"],
        "thermal_adjoint_residual": evaluated["thermal_adjoint_residual"],
        "thermal_energy_balance": evaluated["thermal_energy_balance"],
        "electrical_residual": evaluated["electrical_residual"],
        "electrical_adjoint_residual": evaluated["electrical_adjoint_residual"],
        "electrical_terminal_balance": evaluated["electrical_balance"],
    }
