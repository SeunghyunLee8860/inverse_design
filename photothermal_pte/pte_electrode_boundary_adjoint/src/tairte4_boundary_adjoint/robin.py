from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla

from .perimeter import PerimeterDiscretization, PerimeterParameters


@dataclass(frozen=True)
class RobinEvaluation:
    parameters: PerimeterParameters
    weighting_potential: np.ndarray
    current_A: float
    current_gradient_A_per_m: np.ndarray
    squared_current_A2_diagnostic: float
    squared_current_gradient_A2_per_m_diagnostic: np.ndarray
    terminal_conductance_S: float
    state_residual_relative: float
    adjoint_residual_relative: float
    matrix_relative_asymmetry: float
    contact_integrals_m: tuple[float, float]


@dataclass(frozen=True)
class HardEvaluation:
    parameters: PerimeterParameters
    weighting_potential: np.ndarray
    current_A: float
    squared_current_A2_diagnostic: float
    terminal_conductance_S: float
    residual_relative: float
    contact_0_nodes: np.ndarray
    contact_1_nodes: np.ndarray


def build_current_vector(electrical, temperature_nodes_K: np.ndarray) -> np.ndarray:
    mesh = electrical.mesh
    temperature = np.asarray(temperature_nodes_K, dtype=float)
    if temperature.shape != mesh.shape:
        raise ValueError("temperature does not match electrical mesh")
    tri = mesh.triangles
    grad_temperature = np.einsum(
        "eai,ei->ea", mesh.gradients_m_inv, temperature.reshape(-1)[tri]
    )
    local = -electrical.thickness_m * mesh.area_m2[:, None] * np.einsum(
        "eai,eab,eb->ei",
        mesh.gradients_m_inv,
        electrical.alpha,
        grad_temperature,
    )
    q = np.zeros(mesh.nodes_m.shape[0], dtype=float)
    np.add.at(q, tri.ravel(), local.ravel())
    return q


class DifferentiableContactModel:
    def __init__(
        self,
        electrical,
        temperature_nodes_K: np.ndarray,
        *,
        contact_conductance_S_m2: float,
        transition_m: float,
        quadrature_order: int = 5,
    ):
        if contact_conductance_S_m2 <= 0.0:
            raise ValueError("contact conductance must be positive")
        self.electrical = electrical
        self.temperature = np.asarray(temperature_nodes_K, float)
        self.g_S_m2 = float(contact_conductance_S_m2)
        self.transition_m = float(transition_m)
        self.perimeter = PerimeterDiscretization.from_mesh(
            electrical.mesh, order=quadrature_order
        )
        self.q_A = build_current_vector(electrical, self.temperature)
        self.bulk = electrical.matrix.tocsr()
        self.node_count = electrical.mesh.nodes_m.shape[0]

    def _assemble_contact(
        self,
        center_m: float,
        length_m: float,
        derivative: str | None = None,
    ) -> tuple[sparse.csr_matrix, np.ndarray, float]:
        mask, dm_dc, dm_dL = self.perimeter.mask_and_derivatives(
            center_m, length_m, self.transition_m
        )
        if derivative is None:
            field = mask
        elif derivative == "center":
            field = dm_dc
        elif derivative == "length":
            field = dm_dL
        else:
            raise ValueError("derivative must be None, center, or length")
        nodes = self.perimeter.quadrature_node_ids
        shape = self.perimeter.quadrature_shape
        coefficient = (
            self.electrical.thickness_m
            * self.g_S_m2
            * self.perimeter.quadrature_weight_m
            * field
        )
        local = coefficient[:, None, None] * shape[:, :, None] * shape[:, None, :]
        rows = np.repeat(nodes, 2, axis=1).ravel()
        columns = np.tile(nodes, (1, 2)).ravel()
        matrix = sparse.coo_matrix(
            (local.ravel(), (rows, columns)),
            shape=(self.node_count, self.node_count),
        ).tocsr()
        vector = np.zeros(self.node_count, dtype=float)
        np.add.at(vector, nodes.ravel(), (coefficient[:, None] * shape).ravel())
        integral = float(np.sum(self.perimeter.quadrature_weight_m * field))
        return matrix, vector, integral

    @staticmethod
    def _relative_residual(matrix, solution, rhs) -> float:
        return float(
            np.linalg.norm(matrix @ solution - rhs)
            / max(np.linalg.norm(rhs), np.finfo(float).tiny)
        )

    @staticmethod
    def _relative_asymmetry(matrix: sparse.csr_matrix) -> float:
        delta = (matrix - matrix.T).tocoo()
        numerator = np.linalg.norm(delta.data) if delta.nnz else 0.0
        return float(numerator / np.linalg.norm(matrix.data))

    def evaluate(self, parameters: PerimeterParameters) -> RobinEvaluation:
        p = parameters
        b0, v0, int0 = self._assemble_contact(p.center_0_m, p.length_0_m)
        b1, v1, int1 = self._assemble_contact(p.center_1_m, p.length_1_m)
        matrix = (self.bulk + b0 + b1).tocsc()
        rhs = v1  # V0=0 and V1=1.
        psi = np.asarray(spla.spsolve(matrix, rhs), dtype=float)
        state_residual = self._relative_residual(matrix, psi, rhs)
        current = float(self.q_A @ psi)
        objective = current**2
        # Differentiate signed current first.  I^2 is retained only as a
        # diagnostic; production optimization uses the two signed I branches.
        adjoint_rhs = self.q_A
        adjoint = np.asarray(spla.spsolve(matrix.T, adjoint_rhs), dtype=float)
        adjoint_residual = self._relative_residual(matrix.T, adjoint, adjoint_rhs)
        derivatives = (
            (0, "center", 0.0),
            (0, "length", 0.0),
            (1, "center", 1.0),
            (1, "length", 1.0),
        )
        current_gradient = np.zeros(4, dtype=float)
        for index, (contact, variable, voltage) in enumerate(derivatives):
            if contact == 0:
                dmatrix, dvector, _ = self._assemble_contact(
                    p.center_0_m, p.length_0_m, variable
                )
            else:
                dmatrix, dvector, _ = self._assemble_contact(
                    p.center_1_m, p.length_1_m, variable
                )
            current_gradient[index] = float(
                adjoint @ (voltage * dvector - dmatrix @ psi)
            )
        squared_current_gradient = 2.0 * current * current_gradient
        terminal_current = float(
            self.electrical.thickness_m * self.g_S_m2 * np.sum(
                self.perimeter.quadrature_weight_m
                * self.perimeter.mask_and_derivatives(
                    p.center_1_m, p.length_1_m, self.transition_m
                )[0]
                * (1.0 - np.sum(
                    self.perimeter.quadrature_shape
                    * psi[self.perimeter.quadrature_node_ids], axis=1
                ))
            )
        )
        return RobinEvaluation(
            parameters=p,
            weighting_potential=psi.reshape(self.electrical.mesh.shape),
            current_A=current,
            current_gradient_A_per_m=current_gradient,
            squared_current_A2_diagnostic=objective,
            squared_current_gradient_A2_per_m_diagnostic=squared_current_gradient,
            terminal_conductance_S=terminal_current,
            state_residual_relative=state_residual,
            adjoint_residual_relative=adjoint_residual,
            matrix_relative_asymmetry=self._relative_asymmetry(matrix.tocsr()),
            contact_integrals_m=(int0, int1),
        )

    def hard_evaluate(self, parameters: PerimeterParameters) -> HardEvaluation:
        p = parameters
        nodes_0 = self.perimeter.hard_contact_nodes(p.center_0_m, p.length_0_m)
        nodes_1 = self.perimeter.hard_contact_nodes(p.center_1_m, p.length_1_m)
        if np.intersect1d(nodes_0, nodes_1).size:
            raise ValueError("hard perimeter contacts overlap on boundary nodes")
        fixed = np.unique(np.concatenate((nodes_0, nodes_1)))
        fixed_values = np.isin(fixed, nodes_1).astype(float)
        free_mask = np.ones(self.node_count, dtype=bool)
        free_mask[fixed] = False
        free = np.flatnonzero(free_mask)
        reduced = self.bulk[free][:, free].tocsc()
        rhs = -np.asarray(self.bulk[free][:, fixed] @ fixed_values).reshape(-1)
        psi = np.zeros(self.node_count, dtype=float)
        psi[fixed] = fixed_values
        psi[free] = spla.spsolve(reduced, rhs)
        residual = self._relative_residual(reduced, psi[free], rhs)
        current = float(self.q_A @ psi)
        conductance = float(psi @ (self.bulk @ psi))
        return HardEvaluation(
            parameters=p,
            weighting_potential=psi.reshape(self.electrical.mesh.shape),
            current_A=current,
            squared_current_A2_diagnostic=current**2,
            terminal_conductance_S=conductance,
            residual_relative=residual,
            contact_0_nodes=nodes_0,
            contact_1_nodes=nodes_1,
        )
