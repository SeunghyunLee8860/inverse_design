"""Fail-closed crystal-to-solver coordinate contracts for PTE optimization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


KAPPA_A_B_C_W_MK = (14.4, 3.8, 1.0)
SIGMA_A_B_S_M = (4.91e5, 1.10e5)
SEEBECK_A_B_V_K = (-6.0e-6, 27.0e-6)


@dataclass(frozen=True)
class AxisContract:
    name: str
    crystal_axis_by_solver_axis: tuple[str, str, str]
    epsilon_axis_by_solver_axis: tuple[str, str, str]
    kappa_xyz_W_mK: tuple[float, float, float]
    sigma_xy_S_m: tuple[float, float]
    seebeck_xy_V_K: tuple[float, float]
    polarization_angle_deg: dict[str, float]

    def validate_polarization(self, label: str, angle_deg: float) -> None:
        expected = self.polarization_angle_deg[label]
        if not np.isclose(float(angle_deg), expected, rtol=0.0, atol=1.0e-12):
            raise RuntimeError(
                f"polarization/axis-contract mismatch: {label} requires "
                f"{expected:g} deg, read back {angle_deg:g} deg"
            )


X_B_Y_A = AxisContract(
    name="lumerical_x_b_y_a",
    crystal_axis_by_solver_axis=("b", "a", "c"),
    epsilon_axis_by_solver_axis=("b", "a", "b_closure_for_c"),
    kappa_xyz_W_mK=(3.8, 14.4, 1.0),
    sigma_xy_S_m=(1.10e5, 4.91e5),
    seebeck_xy_V_K=(27.0e-6, -6.0e-6),
    polarization_angle_deg={"E_parallel_b": 0.0, "E_parallel_a": 90.0},
)


LEGACY_X_A_Y_B = AxisContract(
    name="legacy_lumerical_x_a_y_b",
    crystal_axis_by_solver_axis=("a", "b", "c"),
    epsilon_axis_by_solver_axis=("a", "b", "legacy_c"),
    kappa_xyz_W_mK=(14.4, 3.8, 1.0),
    sigma_xy_S_m=(4.91e5, 1.10e5),
    seebeck_xy_V_K=(-6.0e-6, 27.0e-6),
    polarization_angle_deg={"E_parallel_a": 0.0, "E_parallel_b": 90.0},
)


CONTRACTS = {contract.name: contract for contract in (X_B_Y_A, LEGACY_X_A_Y_B)}


def get_axis_contract(name: str) -> AxisContract:
    try:
        return CONTRACTS[name]
    except KeyError as exc:
        raise ValueError(f"unknown axis contract: {name}") from exc
