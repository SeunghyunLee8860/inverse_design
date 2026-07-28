"""Finite, non-periodic density filter/projection and exact transpose."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from .contract import (
    CERTIFICATE_BETA,
    DESIGN_DX_M,
    DESIGN_DY_M,
    DESIGN_NX,
    DESIGN_NY,
    DESIGN_NZ,
    FILTER_RADIUS_M,
    PROJECTION_ETA,
)


def _finite_conic_filter(
    nx: int,
    ny: int,
    dx_m: float,
    dy_m: float,
    radius_m: float,
) -> sparse.csr_matrix:
    """Row-normalized finite conic filter with no periodic wrap."""
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    rx = int(np.ceil(radius_m / dx_m))
    ry = int(np.ceil(radius_m / dy_m))
    for i in range(nx):
        for j in range(ny):
            row = i * ny + j
            local_columns = []
            local_weights = []
            for ii in range(max(0, i - rx), min(nx, i + rx + 1)):
                for jj in range(max(0, j - ry), min(ny, j + ry + 1)):
                    distance = np.hypot(
                        (ii - i) * dx_m, (jj - j) * dy_m
                    )
                    weight = max(0.0, radius_m - distance)
                    if weight > 0.0:
                        local_columns.append(ii * ny + jj)
                        local_weights.append(weight)
            normalization = float(np.sum(local_weights))
            if normalization <= 0.0:
                raise RuntimeError("finite filter row has no support")
            rows.extend([row] * len(local_columns))
            columns.extend(local_columns)
            values.extend(
                [weight / normalization for weight in local_weights]
            )
    matrix = sparse.coo_matrix(
        (values, (rows, columns)),
        shape=(nx * ny, nx * ny),
    ).tocsr()
    constant_error = float(
        np.max(np.abs(matrix @ np.ones(nx * ny) - 1.0))
    )
    # Sparse row assembly order can move this last-bit error slightly across
    # SciPy builds.  The matrix itself is unchanged; accept only a small
    # multiple of binary64 roundoff.
    if constant_error > 1.0e-14:
        raise RuntimeError(
            f"finite filter does not preserve constants: {constant_error}"
        )
    return matrix


def _projection(value: np.ndarray, beta: float, eta: float) -> np.ndarray:
    denominator = np.tanh(beta * eta) + np.tanh(beta * (1.0 - eta))
    return (
        np.tanh(beta * eta) + np.tanh(beta * (value - eta))
    ) / denominator


def _projection_derivative(
    value: np.ndarray, beta: float, eta: float
) -> np.ndarray:
    denominator = np.tanh(beta * eta) + np.tanh(beta * (1.0 - eta))
    return beta * (1.0 - np.tanh(beta * (value - eta)) ** 2) / denominator


@dataclass(frozen=True)
class FiniteDensityMapping:
    nx: int = DESIGN_NX
    ny: int = DESIGN_NY
    nz: int = DESIGN_NZ
    dx_m: float = DESIGN_DX_M
    dy_m: float = DESIGN_DY_M
    filter_radius_m: float = FILTER_RADIUS_M
    eta: float = PROJECTION_ETA

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "filter_matrix",
            _finite_conic_filter(
                self.nx,
                self.ny,
                self.dx_m,
                self.dy_m,
                self.filter_radius_m,
            ),
        )

    @property
    def latent_shape(self) -> tuple[int, int]:
        return self.nx, self.ny

    @property
    def physical_shape(self) -> tuple[int, int, int]:
        return self.nx, self.ny, self.nz

    def filtered(self, latent: np.ndarray) -> np.ndarray:
        value = np.asarray(latent, float).reshape(-1)
        if value.size != self.nx * self.ny:
            raise ValueError("latent density has the wrong size")
        if not np.all(np.isfinite(value)):
            raise ValueError("latent density contains NaN or Inf")
        return np.asarray(self.filter_matrix @ value).reshape(self.nx, self.ny)

    def physical(
        self, latent: np.ndarray, beta: float = CERTIFICATE_BETA
    ) -> np.ndarray:
        projected = _projection(self.filtered(latent), beta, self.eta)
        return np.repeat(projected[:, :, None], self.nz, axis=2)

    def jvp(
        self,
        latent: np.ndarray,
        direction: np.ndarray,
        beta: float = CERTIFICATE_BETA,
    ) -> np.ndarray:
        filtered = self.filtered(latent)
        filtered_direction = np.asarray(
            self.filter_matrix @ np.asarray(direction, float).reshape(-1)
        ).reshape(self.nx, self.ny)
        projected_direction = (
            _projection_derivative(filtered, beta, self.eta)
            * filtered_direction
        )
        return np.repeat(projected_direction[:, :, None], self.nz, axis=2)

    def vjp(
        self,
        latent: np.ndarray,
        physical_sensitivity: np.ndarray,
        beta: float = CERTIFICATE_BETA,
    ) -> np.ndarray:
        sensitivity = np.asarray(physical_sensitivity, float)
        if sensitivity.shape != self.physical_shape:
            raise ValueError("physical sensitivity has the wrong shape")
        filtered = self.filtered(latent)
        projected_sensitivity = np.sum(sensitivity, axis=2)
        filtered_sensitivity = (
            projected_sensitivity
            * _projection_derivative(filtered, beta, self.eta)
        )
        return np.asarray(
            self.filter_matrix.T @ filtered_sensitivity.reshape(-1)
        ).reshape(self.latent_shape)
