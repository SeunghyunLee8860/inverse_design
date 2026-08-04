"""Exact nonperiodic 81x81 nodal-density coupling operators."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse


def _strict(values: np.ndarray, name: str) -> np.ndarray:
    out = np.asarray(values, float).reshape(-1)
    if out.size < 2 or not np.all(np.isfinite(out)):
        raise ValueError(f"{name} must contain at least two finite values")
    if not np.all(np.diff(out) > 0.0):
        raise ValueError(f"{name} must be strictly increasing")
    return out


def build_piecewise_linear_cell_average(
    *,
    source_nodes_m: np.ndarray,
    target_edges_m: np.ndarray,
    tolerance_m: float = 2.0e-18,
) -> sparse.csr_matrix:
    """Map nodal samples to exact cell averages of their linear interpolant."""

    nodes = _strict(source_nodes_m, "source nodes")
    edges = _strict(target_edges_m, "target edges")
    if (
        edges[0] < nodes[0] - tolerance_m
        or edges[-1] > nodes[-1] + tolerance_m
    ):
        raise ValueError("target cells extend outside the nodal support")
    if not np.allclose(
        edges[[0, -1]],
        nodes[[0, -1]],
        rtol=0.0,
        atol=tolerance_m,
    ):
        raise ValueError(
            "target and nodal bounds must match; no crop, pad, or tiling"
        )
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    for target in range(edges.size - 1):
        lower = edges[target]
        upper = edges[target + 1]
        width = upper - lower
        for segment in range(nodes.size - 1):
            left = max(lower, nodes[segment])
            right = min(upper, nodes[segment + 1])
            if right <= left:
                continue
            segment_width = nodes[segment + 1] - nodes[segment]
            left_basis_integral = (
                nodes[segment + 1] * (right - left)
                - 0.5 * (right**2 - left**2)
            ) / segment_width
            right_basis_integral = (
                0.5 * (right**2 - left**2)
                - nodes[segment] * (right - left)
            ) / segment_width
            rows.extend((target, target))
            columns.extend((segment, segment + 1))
            values.extend(
                (
                    left_basis_integral / width,
                    right_basis_integral / width,
                )
            )
    operator = sparse.coo_matrix(
        (values, (rows, columns)),
        shape=(edges.size - 1, nodes.size),
    ).tocsr()
    operator.sum_duplicates()
    if np.min(operator.data) < -2.0e-14:
        raise RuntimeError("cell-average operator contains negative weights")
    constant_error = float(
        np.max(np.abs(operator @ np.ones(nodes.size) - 1.0))
    )
    if constant_error > 2.0e-13:
        raise RuntimeError(
            f"cell-average operator does not preserve constants: "
            f"{constant_error}"
        )
    return operator


@dataclass(frozen=True)
class NodalPhysicalCoupling:
    """Couple one 2D nodal physical density to optical and thermal grids."""

    x_nodes_m: np.ndarray
    y_nodes_m: np.ndarray
    optical_z_nodes_m: np.ndarray
    thermal_x_edges_m: np.ndarray
    thermal_y_edges_m: np.ndarray

    def __post_init__(self) -> None:
        x = _strict(self.x_nodes_m, "x nodes")
        y = _strict(self.y_nodes_m, "y nodes")
        z = _strict(self.optical_z_nodes_m, "optical z nodes")
        tx = _strict(self.thermal_x_edges_m, "thermal x edges")
        ty = _strict(self.thermal_y_edges_m, "thermal y edges")
        object.__setattr__(self, "x_nodes_m", x)
        object.__setattr__(self, "y_nodes_m", y)
        object.__setattr__(self, "optical_z_nodes_m", z)
        object.__setattr__(self, "thermal_x_edges_m", tx)
        object.__setattr__(self, "thermal_y_edges_m", ty)
        object.__setattr__(
            self,
            "thermal_x_operator",
            build_piecewise_linear_cell_average(
                source_nodes_m=x, target_edges_m=tx
            ),
        )
        object.__setattr__(
            self,
            "thermal_y_operator",
            build_piecewise_linear_cell_average(
                source_nodes_m=y, target_edges_m=ty
            ),
        )

    @property
    def physical_shape(self) -> tuple[int, int]:
        return self.x_nodes_m.size, self.y_nodes_m.size

    @property
    def optical_shape(self) -> tuple[int, int, int]:
        return (*self.physical_shape, self.optical_z_nodes_m.size)

    @property
    def thermal_shape(self) -> tuple[int, int]:
        return (
            self.thermal_x_edges_m.size - 1,
            self.thermal_y_edges_m.size - 1,
        )

    def _physical(self, rho: np.ndarray) -> np.ndarray:
        value = np.asarray(rho, float)
        if value.shape != self.physical_shape:
            raise ValueError(
                f"physical rho shape {value.shape} != {self.physical_shape}"
            )
        if not np.all(np.isfinite(value)):
            raise ValueError("physical rho contains NaN or Inf")
        return value

    def optical(self, rho: np.ndarray) -> np.ndarray:
        value = self._physical(rho)
        return np.repeat(
            value[:, :, None], self.optical_z_nodes_m.size, axis=2
        )

    def optical_jvp(self, direction: np.ndarray) -> np.ndarray:
        return self.optical(direction)

    def optical_vjp(self, sensitivity: np.ndarray) -> np.ndarray:
        value = np.asarray(sensitivity, float)
        if value.shape != self.optical_shape:
            raise ValueError(
                f"optical sensitivity shape {value.shape} != "
                f"{self.optical_shape}"
            )
        return np.sum(value, axis=2)

    def thermal(self, rho: np.ndarray) -> np.ndarray:
        value = self._physical(rho)
        return np.asarray(
            self.thermal_x_operator
            @ value
            @ self.thermal_y_operator.T
        )

    def thermal_jvp(self, direction: np.ndarray) -> np.ndarray:
        return self.thermal(direction)

    def thermal_vjp(self, sensitivity: np.ndarray) -> np.ndarray:
        value = np.asarray(sensitivity, float)
        if value.shape != self.thermal_shape:
            raise ValueError(
                f"thermal sensitivity shape {value.shape} != "
                f"{self.thermal_shape}"
            )
        return np.asarray(
            self.thermal_x_operator.T
            @ value
            @ self.thermal_y_operator
        )
