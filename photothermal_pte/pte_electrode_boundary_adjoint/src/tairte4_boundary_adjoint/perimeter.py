from __future__ import annotations

from dataclasses import dataclass
from math import pi

import numpy as np


SIDES = ("bottom", "right", "top", "left")


@dataclass(frozen=True)
class PerimeterParameters:
    center_0_m: float
    length_0_m: float
    center_1_m: float
    length_1_m: float

    def as_array(self) -> np.ndarray:
        return np.asarray(
            [self.center_0_m, self.length_0_m, self.center_1_m, self.length_1_m],
            dtype=float,
        )

    @classmethod
    def from_array(cls, values: np.ndarray) -> "PerimeterParameters":
        p = np.asarray(values, dtype=float)
        if p.shape != (4,):
            raise ValueError("perimeter parameters must have shape (4,)")
        return cls(float(p[0]), float(p[1]), float(p[2]), float(p[3]))


@dataclass(frozen=True)
class PerimeterDiscretization:
    width_m: float
    height_m: float
    perimeter_m: float
    quadrature_node_ids: np.ndarray
    quadrature_shape: np.ndarray
    quadrature_weight_m: np.ndarray
    quadrature_s_m: np.ndarray
    boundary_node_ids: np.ndarray
    boundary_node_s_m: np.ndarray

    @classmethod
    def from_mesh(cls, mesh, order: int = 5) -> "PerimeterDiscretization":
        if order < 2:
            raise ValueError("boundary quadrature order must be at least two")
        x = np.asarray(mesh.x_m, float)
        y = np.asarray(mesh.y_m, float)
        width = float(x[-1] - x[0])
        height = float(y[-1] - y[0])
        perimeter = 2.0 * (width + height)
        ids = mesh.ids
        edge_nodes: list[tuple[int, int]] = []
        edge_start: list[float] = []
        edge_length: list[float] = []

        for i in range(x.size - 1):
            edge_nodes.append((int(ids[i, 0]), int(ids[i + 1, 0])))
            edge_start.append(float(x[i] - x[0]))
            edge_length.append(float(x[i + 1] - x[i]))
        for j in range(y.size - 1):
            edge_nodes.append((int(ids[-1, j]), int(ids[-1, j + 1])))
            edge_start.append(width + float(y[j] - y[0]))
            edge_length.append(float(y[j + 1] - y[j]))
        for i in range(x.size - 1, 0, -1):
            edge_nodes.append((int(ids[i, -1]), int(ids[i - 1, -1])))
            edge_start.append(width + height + float(x[-1] - x[i]))
            edge_length.append(float(x[i] - x[i - 1]))
        for j in range(y.size - 1, 0, -1):
            edge_nodes.append((int(ids[0, j]), int(ids[0, j - 1])))
            edge_start.append(2.0 * width + height + float(y[-1] - y[j]))
            edge_length.append(float(y[j] - y[j - 1]))

        points, weights = np.polynomial.legendre.leggauss(order)
        t = 0.5 * (points + 1.0)
        w = 0.5 * weights
        nodes_q = []
        shape_q = []
        weight_q = []
        s_q = []
        for nodes, start, length in zip(edge_nodes, edge_start, edge_length):
            for local_t, local_w in zip(t, w):
                nodes_q.append(nodes)
                shape_q.append((1.0 - local_t, local_t))
                weight_q.append(local_w * length)
                s_q.append((start + local_t * length) % perimeter)

        boundary_ids: list[int] = []
        boundary_s: list[float] = []
        for i in range(x.size):
            boundary_ids.append(int(ids[i, 0]))
            boundary_s.append(float(x[i] - x[0]))
        for j in range(1, y.size):
            boundary_ids.append(int(ids[-1, j]))
            boundary_s.append(width + float(y[j] - y[0]))
        for i in range(x.size - 2, -1, -1):
            boundary_ids.append(int(ids[i, -1]))
            boundary_s.append(width + height + float(x[-1] - x[i]))
        for j in range(y.size - 2, 0, -1):
            boundary_ids.append(int(ids[0, j]))
            boundary_s.append(2.0 * width + height + float(y[-1] - y[j]))

        if len(set(boundary_ids)) != len(boundary_ids):
            raise RuntimeError("boundary node list is not unique")
        return cls(
            width_m=width,
            height_m=height,
            perimeter_m=perimeter,
            quadrature_node_ids=np.asarray(nodes_q, dtype=np.int64),
            quadrature_shape=np.asarray(shape_q, dtype=float),
            quadrature_weight_m=np.asarray(weight_q, dtype=float),
            quadrature_s_m=np.asarray(s_q, dtype=float),
            boundary_node_ids=np.asarray(boundary_ids, dtype=np.int64),
            boundary_node_s_m=np.asarray(boundary_s, dtype=float),
        )

    def wrap_center(self, center_m: float) -> float:
        return float(center_m % self.perimeter_m)

    def periodic_distance(self, s_m: np.ndarray, center_m: float) -> np.ndarray:
        delta = np.mod(np.asarray(s_m) - center_m + 0.5 * self.perimeter_m,
                       self.perimeter_m) - 0.5 * self.perimeter_m
        return np.abs(delta)

    def hard_contact_nodes(self, center_m: float, length_m: float) -> np.ndarray:
        if not 0.0 < length_m < self.perimeter_m:
            raise ValueError("hard contact length must lie in (0,P)")
        selected = self.periodic_distance(
            self.boundary_node_s_m, center_m
        ) <= 0.5 * length_m + 1e-15
        nodes = self.boundary_node_ids[selected]
        if nodes.size < 2:
            raise ValueError("hard contact must contain at least two boundary nodes")
        return nodes

    def mask_and_derivatives(
        self,
        center_m: float,
        length_m: float,
        transition_m: float,
        delta_floor: float = 1e-8,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not 0.0 < length_m < self.perimeter_m:
            raise ValueError("contact length must lie in (0,P)")
        if transition_m <= 0.0:
            raise ValueError("transition width must be positive")
        theta = 2.0 * pi * (self.quadrature_s_m - center_m) / self.perimeter_m
        ell = pi * length_m / self.perimeter_m
        scale = 2.0 * pi * transition_m / self.perimeter_m
        z = np.cos(theta) - np.cos(ell)
        delta = delta_floor + scale * np.sin(ell)
        u = z / delta
        mask = np.zeros_like(u)
        dm_du = np.zeros_like(u)
        transition = (u > 0.0) & (u < 1.0)
        core = u >= 1.0
        ut = u[transition]
        mask[transition] = 6.0 * ut**5 - 15.0 * ut**4 + 10.0 * ut**3
        dm_du[transition] = 30.0 * ut**2 * (ut - 1.0)**2
        mask[core] = 1.0
        dz_dc = (2.0 * pi / self.perimeter_m) * np.sin(theta)
        dz_dL = (pi / self.perimeter_m) * np.sin(ell)
        ddelta_dL = scale * (pi / self.perimeter_m) * np.cos(ell)
        du_dc = dz_dc / delta
        du_dL = (dz_dL * delta - z * ddelta_dL) / delta**2
        return mask, dm_du * du_dc, dm_du * du_dL

    def separation_constraints(
        self, values: np.ndarray, gap_m: float
    ) -> tuple[np.ndarray, np.ndarray]:
        p = np.asarray(values, dtype=float)
        c0, l0, c1, l1 = p
        required = pi * (l0 + l1 + 2.0 * gap_m) / self.perimeter_m
        delta = 2.0 * pi * (c1 - c0) / self.perimeter_m
        constraints = np.asarray([
            np.cos(required) - np.cos(delta),
            self.perimeter_m - l0 - l1 - 2.0 * gap_m,
        ])
        jacobian = np.zeros((2, 4), dtype=float)
        jacobian[0] = (
            -(2.0 * pi / self.perimeter_m) * np.sin(delta),
            -(pi / self.perimeter_m) * np.sin(required),
            +(2.0 * pi / self.perimeter_m) * np.sin(delta),
            -(pi / self.perimeter_m) * np.sin(required),
        )
        jacobian[1] = (0.0, -1.0, 0.0, -1.0)
        return constraints, jacobian

    def separation_constraints_scaled(
        self, values: np.ndarray, gap_fraction: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Constraints for x=(u0,l0,u1,l1), scaled by perimeter P.

        The lifted centers u0/u1 are intentionally not wrapped or bounded.
        Cosine makes both the constraint and its Jacobian exactly 1-periodic.
        """
        x = np.asarray(values, dtype=float)
        if x.shape != (4,):
            raise ValueError("scaled perimeter parameters must have shape (4,)")
        if gap_fraction < 0.0:
            raise ValueError("gap fraction cannot be negative")
        u0, ell0, u1, ell1 = x
        required = pi * (ell0 + ell1 + 2.0 * gap_fraction)
        delta = 2.0 * pi * (u1 - u0)
        constraints = np.asarray([
            np.cos(required) - np.cos(delta),
            1.0 - ell0 - ell1 - 2.0 * gap_fraction,
        ])
        jacobian = np.zeros((2, 4), dtype=float)
        jacobian[0] = (
            -2.0 * pi * np.sin(delta),
            -pi * np.sin(required),
            +2.0 * pi * np.sin(delta),
            -pi * np.sin(required),
        )
        jacobian[1] = (0.0, -1.0, 0.0, -1.0)
        return constraints, jacobian

    def side_coordinate_to_s(self, side: str, tangent_m: float) -> float:
        """Map the legacy side/tangent convention to the oriented perimeter."""
        x_min = -0.5 * self.width_m
        x_max = +0.5 * self.width_m
        y_min = -0.5 * self.height_m
        y_max = +0.5 * self.height_m
        if side == "bottom":
            s = tangent_m - x_min
        elif side == "right":
            s = self.width_m + tangent_m - y_min
        elif side == "top":
            s = self.width_m + self.height_m + x_max - tangent_m
        elif side == "left":
            s = 2.0 * self.width_m + self.height_m + y_max - tangent_m
        else:
            raise ValueError(f"unknown side {side!r}; choose from {SIDES}")
        return self.wrap_center(s)

    def s_to_side_coordinate(self, s_m: float) -> tuple[str, float]:
        """Return the side and legacy tangent coordinate containing perimeter s."""
        s = self.wrap_center(s_m)
        if s < self.width_m:
            return "bottom", s - 0.5 * self.width_m
        if s < self.width_m + self.height_m:
            return "right", s - self.width_m - 0.5 * self.height_m
        if s < 2.0 * self.width_m + self.height_m:
            return "top", 0.5 * self.width_m - (s - self.width_m - self.height_m)
        return "left", 0.5 * self.height_m - (
            s - 2.0 * self.width_m - self.height_m
        )
