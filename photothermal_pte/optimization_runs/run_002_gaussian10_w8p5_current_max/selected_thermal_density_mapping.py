"""Exact 50 nm nodal-density to 100 nm thermal-cell mapping for Run 002.

The selected optical design has 373 nodes on [-9.3, 9.3] um.  The thermal
core has 186 cells over the same interval.  Physical density is interpreted
as bilinear between optical nodes, so each thermal-cell average is the exact
integral over two fine intervals per axis.
"""

from __future__ import annotations

import numpy as np


SELECTED_NODAL_SHAPE = (373, 373)
SELECTED_THERMAL_CELL_SHAPE = (186, 186)
SELECTED_BOUNDS_M = (-9.3e-6, 9.3e-6)
NODAL_STEP_M = 50.0e-9
THERMAL_CELL_STEP_M = 100.0e-9


def selected_nodal_to_thermal_cell(density: np.ndarray) -> np.ndarray:
    """Return exact bilinear-field averages on the 186x186 thermal cells."""

    values = np.asarray(density, float)
    if values.shape != SELECTED_NODAL_SHAPE:
        raise ValueError(
            f"selected nodal density shape {values.shape} != {SELECTED_NODAL_SHAPE}"
        )
    weights = np.asarray([1.0, 2.0, 1.0])
    result = np.zeros(SELECTED_THERMAL_CELL_SHAPE, float)
    for di, wi in enumerate(weights):
        for dj, wj in enumerate(weights):
            result += wi * wj * values[di : di + 372 : 2, dj : dj + 372 : 2]
    return result / 16.0


def selected_nodal_to_thermal_cell_transpose(cell_values: np.ndarray) -> np.ndarray:
    """Apply the exact Euclidean transpose of selected_nodal_to_thermal_cell."""

    values = np.asarray(cell_values, float)
    if values.shape != SELECTED_THERMAL_CELL_SHAPE:
        raise ValueError(
            f"selected thermal-cell shape {values.shape} != {SELECTED_THERMAL_CELL_SHAPE}"
        )
    result = np.zeros(SELECTED_NODAL_SHAPE, float)
    weights = np.asarray([1.0, 2.0, 1.0])
    for di, wi in enumerate(weights):
        for dj, wj in enumerate(weights):
            result[di : di + 372 : 2, dj : dj + 372 : 2] += wi * wj * values / 16.0
    return result


def bilinear_integral(density: np.ndarray) -> float:
    """Integrate a bilinear nodal field over the selected square."""

    values = np.asarray(density, float)
    if values.shape != SELECTED_NODAL_SHAPE:
        raise ValueError("unexpected selected nodal shape")
    one_d = np.ones(373, float)
    one_d[[0, -1]] = 0.5
    return float(np.einsum("ij,i,j->", values, one_d, one_d) * NODAL_STEP_M**2)
