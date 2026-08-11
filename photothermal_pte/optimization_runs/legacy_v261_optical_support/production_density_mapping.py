"""Efficient finite nonperiodic conic filter and tanh projection for Run 002."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage


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
class ProductionDensityMapping:
    """Row-normalized finite conic mapping with its exact discrete transpose.

    The forward filter is ``H = D^-1 C``, where ``C`` is zero-padded finite
    convolution and ``D`` contains the truncated kernel sums. Because the
    conic kernel is symmetric, the transpose is ``H.T = C D^-1``. This order
    matters at the finite design boundary.
    """

    shape: tuple[int, int] = (373, 373)
    spacing_m: float = 50.0e-9
    radius_m: float = 500.0e-9
    eta: float = 0.5

    def __post_init__(self) -> None:
        if len(self.shape) != 2 or min(self.shape) < 2:
            raise ValueError("mapping shape must be a two-dimensional grid")
        if self.spacing_m <= 0.0 or self.radius_m <= 0.0:
            raise ValueError("spacing and radius must be positive")
        extent = int(np.ceil(self.radius_m / self.spacing_m))
        offsets = np.arange(-extent, extent + 1, dtype=float) * self.spacing_m
        xx, yy = np.meshgrid(offsets, offsets, indexing="ij")
        kernel = np.maximum(0.0, self.radius_m - np.hypot(xx, yy))
        kernel[np.hypot(xx, yy) >= self.radius_m] = 0.0
        normalization = ndimage.convolve(
            np.ones(self.shape, dtype=float),
            kernel,
            mode="constant",
            cval=0.0,
        )
        if not np.all(normalization > 0.0):
            raise RuntimeError("finite filter has an empty row")
        object.__setattr__(self, "kernel", kernel)
        object.__setattr__(self, "normalization", normalization)

    def _array(self, value: np.ndarray, label: str) -> np.ndarray:
        result = np.asarray(value, dtype=float)
        if result.shape != self.shape:
            raise ValueError(f"{label} must have shape {self.shape}")
        if not np.all(np.isfinite(result)):
            raise ValueError(f"{label} contains NaN or Inf")
        return result

    def _convolve(self, value: np.ndarray) -> np.ndarray:
        return ndimage.convolve(value, self.kernel, mode="constant", cval=0.0)

    def filtered(self, latent: np.ndarray) -> np.ndarray:
        value = self._array(latent, "latent")
        return self._convolve(value) / self.normalization

    def filter_transpose(self, cotangent: np.ndarray) -> np.ndarray:
        value = self._array(cotangent, "filter cotangent")
        return self._convolve(value / self.normalization)

    def physical(self, latent: np.ndarray, beta: float) -> np.ndarray:
        return _projection(self.filtered(latent), float(beta), self.eta)

    def jvp(
        self,
        latent: np.ndarray,
        direction: np.ndarray,
        beta: float,
    ) -> np.ndarray:
        filtered = self.filtered(latent)
        filtered_direction = self.filtered(direction)
        return _projection_derivative(filtered, float(beta), self.eta) * filtered_direction

    def vjp(
        self,
        latent: np.ndarray,
        physical_cotangent: np.ndarray,
        beta: float,
    ) -> np.ndarray:
        filtered = self.filtered(latent)
        cotangent = self._array(physical_cotangent, "physical cotangent")
        projected = cotangent * _projection_derivative(
            filtered, float(beta), self.eta
        )
        return self.filter_transpose(projected)

    def audit(self) -> dict[str, object]:
        constant_error = float(
            np.max(np.abs(self.filtered(np.ones(self.shape, dtype=float)) - 1.0))
        )
        return {
            "shape": list(self.shape),
            "spacing_nm": self.spacing_m * 1.0e9,
            "radius_nm": self.radius_m * 1.0e9,
            "eta": self.eta,
            "kernel_shape": list(self.kernel.shape),
            "kernel_nonzero_count": int(np.count_nonzero(self.kernel)),
            "normalization_min": float(np.min(self.normalization)),
            "normalization_max": float(np.max(self.normalization)),
            "constant_preservation_max_abs": constant_error,
            "periodic_wrap": False,
            "forward_operator": "D^-1 C with zero padding",
            "transpose_operator": "C D^-1 with symmetric conic kernel",
        }
