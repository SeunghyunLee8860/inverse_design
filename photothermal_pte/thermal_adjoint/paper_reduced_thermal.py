"""Paper-aligned reduced TaIrTe4 thermal boundary model.

This module implements the Supplementary Note S5-style model used by the
AD--FD certificates:

* only the TaIrTe4 flake is a bulk thermal domain;
* in-plane and cross-plane conduction use diag(14.4, 3.8, 1.0) W/(m K);
* the substrate face has a fixed TaIrTe4/thermally-grown-SiO2 Robin law;
* the design face interpolates TaIrTe4/air and TaIrTe4/SiO2 conductance by
  projected contact-area fraction.

The optical endpoint remains a separate n=4 proxy.  No bulk air, SiO2, or Si
thermal material is introduced here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
from scipy.sparse import linalg as sparse_linalg


HERE = Path(__file__).resolve().parent
FVM = HERE.parent / "validation" / "photothermal_stage1"
if str(FVM) not in sys.path:
    sys.path.insert(0, str(FVM))

from anisotropic_heat_fvm import (  # noqa: E402
    AssembledThermalSystem,
    SteadyHeatResult,
    assemble_steady_diagonal_kappa,
    solve_assembled_thermal_system,
)
from local_pte_functional import (  # noqa: E402
    LocalPTEFunctional,
    build_local_pte_functional,
)


T_BATH_K = 300.0
KAPPA_TAIRTE4_W_MK = (14.4, 3.8, 1.0)
G_AIR_W_M2K = 1.0
G_THERMAL_SIO2_W_M2K = 7.37e6
G_EVAPORATED_SIO2_W_M2K = 7.37e4


@dataclass(frozen=True)
class DesignSurfaceMap:
    """Linear optical-physical-density to thermal-face-density map.

    The production optical density has a duplicated x/y periodic fencepost
    and is exactly extruded in z.  The unique 240 x 240 torus is averaged in
    10 x 10 blocks onto the 24 x 24 thermal surface.  The z mean makes this a
    defined linear map even when an independent physical-density probe is not
    perfectly extruded.
    """

    physical_shape: tuple[int, int, int]
    face_shape: tuple[int, int]

    def __post_init__(self) -> None:
        nx, ny, nz = self.physical_shape
        fx, fy = self.face_shape
        if nx < 2 or ny < 2 or nz < 1:
            raise ValueError("invalid optical physical shape")
        if (nx - 1) % fx or (ny - 1) % fy:
            raise ValueError(
                "unique optical grid must divide exactly into thermal faces"
            )

    @property
    def block_shape(self) -> tuple[int, int]:
        return (
            (self.physical_shape[0] - 1) // self.face_shape[0],
            (self.physical_shape[1] - 1) // self.face_shape[1],
        )

    def apply(self, physical_density: np.ndarray) -> np.ndarray:
        physical = np.asarray(physical_density, float)
        if physical.shape != self.physical_shape:
            raise ValueError(
                f"physical density {physical.shape} != {self.physical_shape}"
            )
        if not np.all(np.isfinite(physical)):
            raise ValueError("physical density contains NaN or Inf")
        bx, by = self.block_shape
        unique = np.mean(physical[:-1, :-1, :], axis=2)
        return np.mean(
            unique.reshape(
                self.face_shape[0],
                bx,
                self.face_shape[1],
                by,
            ),
            axis=(1, 3),
        )

    def transpose(self, face_sensitivity: np.ndarray) -> np.ndarray:
        sensitivity = np.asarray(face_sensitivity, float)
        if sensitivity.shape != self.face_shape:
            raise ValueError(
                f"face sensitivity {sensitivity.shape} != {self.face_shape}"
            )
        bx, by = self.block_shape
        nz = self.physical_shape[2]
        unique = np.repeat(
            np.repeat(sensitivity / (bx * by), bx, axis=0),
            by,
            axis=1,
        )
        physical = np.zeros(self.physical_shape, float)
        physical[:-1, :-1, :] = unique[:, :, None] / nz
        return physical


@dataclass(frozen=True)
class ReducedThermalEvaluation:
    system: AssembledThermalSystem
    solved: SteadyHeatResult
    functional: LocalPTEFunctional
    temperature_active_K: np.ndarray
    adjoint_active: np.ndarray
    objective_A_m: float
    gradient_Q_active_A_m4_W: np.ndarray
    gradient_rho_face_A_m: np.ndarray
    rho_face: np.ndarray
    G_design_W_m2K: np.ndarray


def reduced_flake_grid(
    *,
    lateral_period_m: float = 6.0e-6,
    flake_thickness_m: float = 100.0e-9,
    nx: int = 24,
    ny: int = 24,
    nz: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if min(nx, ny, nz) < 1:
        raise ValueError("grid counts must be positive")
    return (
        np.linspace(
            -0.5 * lateral_period_m,
            0.5 * lateral_period_m,
            nx + 1,
        ),
        np.linspace(
            -0.5 * lateral_period_m,
            0.5 * lateral_period_m,
            ny + 1,
        ),
        np.linspace(-flake_thickness_m, 0.0, nz + 1),
    )


def design_conductance_W_m2K(
    rho_face: np.ndarray,
    *,
    G_sio2_W_m2K: float = G_THERMAL_SIO2_W_M2K,
    G_air_W_m2K: float = G_AIR_W_M2K,
) -> np.ndarray:
    rho = np.asarray(rho_face, float)
    if not np.all(np.isfinite(rho)):
        raise ValueError("surface density contains NaN or Inf")
    if np.any(rho < 0.0) or np.any(rho > 1.0):
        raise ValueError("surface density must lie in [0, 1]")
    if not (G_sio2_W_m2K > 0.0 and G_air_W_m2K > 0.0):
        raise ValueError("surface conductances must be positive")
    return G_air_W_m2K + rho * (G_sio2_W_m2K - G_air_W_m2K)


def assemble_reduced_paper_system(
    rho_face: np.ndarray,
    *,
    G_sio2_W_m2K: float = G_THERMAL_SIO2_W_M2K,
    G_substrate_W_m2K: float = G_THERMAL_SIO2_W_M2K,
    G_air_W_m2K: float = G_AIR_W_M2K,
    bath_temperature_K: float = T_BATH_K,
    grid: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> tuple[AssembledThermalSystem, np.ndarray]:
    x_edges, y_edges, z_edges = (
        reduced_flake_grid() if grid is None else grid
    )
    shape = (
        x_edges.size - 1,
        y_edges.size - 1,
        z_edges.size - 1,
    )
    rho = np.asarray(rho_face, float)
    if rho.shape != shape[:2]:
        raise ValueError(f"surface density {rho.shape} != {shape[:2]}")
    kappa = np.empty((*shape, 3), float)
    kappa[...] = KAPPA_TAIRTE4_W_MK
    design_g = design_conductance_W_m2K(
        rho,
        G_sio2_W_m2K=G_sio2_W_m2K,
        G_air_W_m2K=G_air_W_m2K,
    )
    if not np.isfinite(bath_temperature_K):
        raise ValueError("bath temperature must be finite")
    # Solve the exactly equivalent temperature-rise equation
    # theta = T - T_bath.  Its Robin load is zero, avoiding cancellation
    # between ~300 K diagonal/load terms when the optical rise is tiny.
    system = assemble_steady_diagonal_kappa(
        x_edges_m=x_edges,
        y_edges_m=y_edges,
        z_edges_m=z_edges,
        kappa_W_mK=kappa,
        dirichlet_temperature_K={},
        periodic_axes=("x", "y"),
        surface_robin_heat_transfer_W_m2K={
            "z_min": float(G_substrate_W_m2K),
            "z_max": design_g,
        },
        surface_robin_temperature_K={
            "z_min": 0.0,
            "z_max": 0.0,
        },
    )
    return system, design_g


def default_local_pte_mask(
    x_edges_m: np.ndarray,
    y_edges_m: np.ndarray,
    z_edges_m: np.ndarray,
) -> np.ndarray:
    x = 0.5 * (x_edges_m[:-1] + x_edges_m[1:])
    y = 0.5 * (y_edges_m[:-1] + y_edges_m[1:])
    z = 0.5 * (z_edges_m[:-1] + z_edges_m[1:])
    return (
        (np.abs(x[:, None, None]) <= 2.0e-6)
        & (np.abs(y[None, :, None]) <= 1.75e-6)
        & np.ones((1, 1, z.size), bool)
    )


def evaluate_reduced_paper_thermal(
    *,
    rho_face: np.ndarray,
    source_W_m3: np.ndarray,
    G_sio2_W_m2K: float = G_THERMAL_SIO2_W_M2K,
    G_substrate_W_m2K: float = G_THERMAL_SIO2_W_M2K,
    G_air_W_m2K: float = G_AIR_W_M2K,
    bath_temperature_K: float = T_BATH_K,
    grid: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    fom_mask: np.ndarray | None = None,
) -> ReducedThermalEvaluation:
    x_edges, y_edges, z_edges = (
        reduced_flake_grid() if grid is None else grid
    )
    system, design_g = assemble_reduced_paper_system(
        rho_face,
        G_sio2_W_m2K=G_sio2_W_m2K,
        G_substrate_W_m2K=G_substrate_W_m2K,
        G_air_W_m2K=G_air_W_m2K,
        bath_temperature_K=bath_temperature_K,
        grid=(x_edges, y_edges, z_edges),
    )
    source = np.asarray(source_W_m3, float)
    if source.shape != system.shape:
        raise ValueError(f"source {source.shape} != {system.shape}")
    solved = solve_assembled_thermal_system(
        system, source_W_m3=source
    )
    if fom_mask is None:
        fom_mask = default_local_pte_mask(x_edges, y_edges, z_edges)
    functional = build_local_pte_functional(
        x_edges_m=x_edges,
        y_edges_m=y_edges,
        z_edges_m=z_edges,
        active_mask=system.active_mask,
        active_ids=system.active_ids,
        fom_mask=fom_mask,
        periodic_axes=(),
    )
    temperature = solved.temperature_K[system.active_mask]
    objective = functional.evaluate_active(temperature)
    adjoint = sparse_linalg.spsolve(
        system.matrix_W_K.T.tocsc(),
        functional.temperature_source_A_m_K,
    )
    gradient_q = np.asarray(
        system.source_volume_operator_m3.T @ adjoint
    ).reshape(-1)
    top_ids, _, _ = system.boundary_terms["surface_robin_z_max"]
    dx = np.diff(x_edges)
    dy = np.diff(y_edges)
    area = dx[:, None] * dy[None, :]
    delta_g = float(G_sio2_W_m2K - G_air_W_m2K)
    top_temperature = temperature[top_ids].reshape(area.shape)
    top_adjoint = np.asarray(adjoint)[top_ids].reshape(area.shape)
    # In absolute temperature this is lambda*A*dG*(T_bath - T).  Since the
    # solve uses theta=T-T_bath, the stable equivalent is -lambda*A*dG*theta.
    gradient_rho = -top_adjoint * area * delta_g * top_temperature
    return ReducedThermalEvaluation(
        system=system,
        solved=solved,
        functional=functional,
        temperature_active_K=temperature,
        adjoint_active=np.asarray(adjoint),
        objective_A_m=objective,
        gradient_Q_active_A_m4_W=gradient_q,
        gradient_rho_face_A_m=gradient_rho,
        rho_face=np.asarray(rho_face, float),
        G_design_W_m2K=design_g,
    )


def boundary_diagnostics(
    evaluation: ReducedThermalEvaluation,
    *,
    endpoint_tolerance: float = 1.0e-12,
) -> dict:
    rho = evaluation.rho_face
    top_ids, top_conductance, _ = evaluation.system.boundary_terms[
        "surface_robin_z_max"
    ]
    bottom_ids, bottom_conductance, _ = evaluation.system.boundary_terms[
        "surface_robin_z_min"
    ]
    return {
        "top_face_count": int(top_ids.size),
        "bottom_face_count": int(bottom_ids.size),
        "air_endpoint_face_count": int(
            np.count_nonzero(rho <= endpoint_tolerance)
        ),
        "sio2_endpoint_face_count": int(
            np.count_nonzero(rho >= 1.0 - endpoint_tolerance)
        ),
        "gray_face_count": int(
            np.count_nonzero(
                (rho > endpoint_tolerance)
                & (rho < 1.0 - endpoint_tolerance)
            )
        ),
        "air_equivalent_area_fraction": float(np.mean(1.0 - rho)),
        "sio2_equivalent_area_fraction": float(np.mean(rho)),
        "G_design_min_W_m2K": float(
            np.min(evaluation.G_design_W_m2K)
        ),
        "G_design_max_W_m2K": float(
            np.max(evaluation.G_design_W_m2K)
        ),
        "G_design_area_weighted_mean_W_m2K": float(
            np.mean(evaluation.G_design_W_m2K)
        ),
        "top_active_area_m2": float(
            np.sum(
                top_conductance
                / evaluation.G_design_W_m2K.reshape(-1)
            )
        ),
        "bottom_active_area_m2": float(
            np.sum(
                bottom_conductance
                / G_THERMAL_SIO2_W_M2K
            )
        ),
        "boundary_power_out_W": dict(
            evaluation.solved.boundary_power_out_W
        ),
    }
