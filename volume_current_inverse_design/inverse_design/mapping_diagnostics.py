"""Solver-free checks for filtering, periodicity, z extrusion and binarity."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class MappingDiagnostics:
    shape: tuple[int, int, int]
    unique_shape: tuple[int, int]
    rho_min: float
    rho_max: float
    # rail statistics are INFORMATIONAL only.  The solver-safe affine layer in
    # VolumeCurrentEvaluator keeps the Jacobian probe inside [0,1] regardless of
    # how close the geometry density is to a rail, so a small margin is normal
    # and is NOT a mapping failure (that gate caused the old beta deadlock).
    centered_probe_margin: float
    fraction_exact_zero: float
    fraction_exact_one: float
    fraction_between_rails: float
    periodic_x_max_abs_error: float
    periodic_y_max_abs_error: float
    z_extrusion_max_abs_error: float
    grayness: float
    binarization: float
    is_exact_binary: bool

    def to_dict(self) -> dict:
        value = asdict(self)
        value["shape"] = list(self.shape)
        value["unique_shape"] = list(self.unique_shape)
        return value


def mapping_diagnostics(physical: np.ndarray, rho_step: float = 0.0) -> MappingDiagnostics:
    rho = np.asarray(physical, dtype=float)
    if rho.ndim != 3:
        raise ValueError(f"physical density must be 3D, got {rho.shape}")
    if not np.all(np.isfinite(rho)):
        raise FloatingPointError("physical density contains NaN or Inf")
    rho_min = float(np.min(rho))
    rho_max = float(np.max(rho))
    margin = float(min(rho_min, 1.0 - rho_max))
    periodic_x = float(np.max(np.abs(rho[-1, :, :] - rho[0, :, :])))
    periodic_y = float(np.max(np.abs(rho[:, -1, :] - rho[:, 0, :])))
    reference = rho[:, :, :1]
    z_error = float(np.max(np.abs(rho - reference)))
    unique = rho[:-1, :-1, 0]
    grayness = float(np.mean(4.0 * unique * (1.0 - unique)))
    exact_zero = float(np.mean(unique <= 0.0))
    exact_one = float(np.mean(unique >= 1.0))
    between = float(np.mean((unique > 0.0) & (unique < 1.0)))
    is_binary = bool(np.all(np.isin(np.unique(unique), (0.0, 1.0))))
    return MappingDiagnostics(
        shape=tuple(int(v) for v in rho.shape),
        unique_shape=(int(unique.shape[0]), int(unique.shape[1])),
        rho_min=rho_min,
        rho_max=rho_max,
        centered_probe_margin=margin,
        fraction_exact_zero=exact_zero,
        fraction_exact_one=exact_one,
        fraction_between_rails=between,
        periodic_x_max_abs_error=periodic_x,
        periodic_y_max_abs_error=periodic_y,
        z_extrusion_max_abs_error=z_error,
        grayness=grayness,
        binarization=float(1.0 - grayness),
        is_exact_binary=is_binary,
    )


def assert_mapping_contract(
    diagnostics: MappingDiagnostics,
    expected_shape: tuple[int, int, int],
    rho_step: float = 0.0,
    tolerance: float = 1e-12,
) -> None:
    """Structural contract only -- shape, [0,1] range, periodic fencepost, z
    extrusion.  Rail margin is NOT a failure condition (see class docstring)."""
    if diagnostics.shape != tuple(expected_shape):
        raise RuntimeError(
            f"mapped shape {diagnostics.shape} != expected {tuple(expected_shape)}"
        )
    if diagnostics.rho_min < -tolerance or diagnostics.rho_max > 1.0 + tolerance:
        raise RuntimeError("mapped density is outside [0,1]")
    if diagnostics.periodic_x_max_abs_error > tolerance:
        raise RuntimeError("x periodic fencepost is not exactly wrapped")
    if diagnostics.periodic_y_max_abs_error > tolerance:
        raise RuntimeError("y periodic fencepost is not exactly wrapped")
    if diagnostics.z_extrusion_max_abs_error > tolerance:
        raise RuntimeError("z layers are not an exact vertical extrusion")

