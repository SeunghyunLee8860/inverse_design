"""Fresh 81x81 latent-to-80x80 cell mapping for FDTDX parity.

The optimizer carrier is bounded 81x81 nodal latent density.  A finite,
nonperiodic 500-nm conic filter with truncated-row normalization is followed
by the tanh projection and the exact four-node cell average.  This module has
no dependency on the historical FDTDX path or a Lumerical API.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any

import numpy as np
from scipy import ndimage


NODE_SHAPE = (81, 81)
CELL_SHAPE = (80, 80)
PITCH_M = 100.0e-9
FILTER_RADIUS_M = 500.0e-9
PROJECTION_ETA = 0.5
FIRST_CERTIFICATE_BETA = 4.0


def _projection(value: Any, *, beta: float, eta: float, xp: Any) -> Any:
    denominator = xp.tanh(beta * eta) + xp.tanh(beta * (1.0 - eta))
    return (
        xp.tanh(beta * eta) + xp.tanh(beta * (value - eta))
    ) / denominator


def _projection_derivative(value: np.ndarray, *, beta: float, eta: float) -> np.ndarray:
    denominator = np.tanh(beta * eta) + np.tanh(beta * (1.0 - eta))
    return beta * (1.0 - np.tanh(beta * (value - eta)) ** 2) / denominator


def _validated_beta(beta: float) -> float:
    value = float(beta)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("projection beta must be finite and positive")
    return value


def nodal_to_cell_average(projected_nodes: Any) -> np.ndarray:
    value = np.asarray(projected_nodes, dtype=np.float64)
    if value.shape != NODE_SHAPE or not np.all(np.isfinite(value)):
        raise ValueError(f"projected nodes must be finite with shape {NODE_SHAPE}")
    return np.ascontiguousarray(
        0.25
        * (
            value[:-1, :-1]
            + value[1:, :-1]
            + value[:-1, 1:]
            + value[1:, 1:]
        )
    )


def nodal_to_cell_jvp(direction: Any) -> np.ndarray:
    value = np.asarray(direction, dtype=np.float64)
    if value.shape != NODE_SHAPE or not np.all(np.isfinite(value)):
        raise ValueError(f"nodal direction must be finite with shape {NODE_SHAPE}")
    return np.ascontiguousarray(
        0.25
        * (
            value[:-1, :-1]
            + value[1:, :-1]
            + value[:-1, 1:]
            + value[1:, 1:]
        )
    )


def nodal_to_cell_vjp(cell_cotangent: Any) -> np.ndarray:
    value = np.asarray(cell_cotangent, dtype=np.float64)
    if value.shape != CELL_SHAPE or not np.all(np.isfinite(value)):
        raise ValueError(f"cell cotangent must be finite with shape {CELL_SHAPE}")
    result = np.zeros(NODE_SHAPE, dtype=np.float64)
    quarter = 0.25 * value
    result[:-1, :-1] += quarter
    result[1:, :-1] += quarter
    result[:-1, 1:] += quarter
    result[1:, 1:] += quarter
    return result


@dataclass(frozen=True)
class ParityNodalDesignMapping:
    shape: tuple[int, int] = NODE_SHAPE
    spacing_m: float = PITCH_M
    radius_m: float = FILTER_RADIUS_M
    eta: float = PROJECTION_ETA

    def __post_init__(self) -> None:
        if self.shape != NODE_SHAPE:
            raise ValueError("FDTDX parity mapping must use 81x81 nodes")
        if self.spacing_m <= 0.0 or self.radius_m <= 0.0:
            raise ValueError("spacing and filter radius must be positive")
        if not 0.0 < self.eta < 1.0:
            raise ValueError("projection eta must lie strictly inside (0,1)")
        extent = int(np.ceil(self.radius_m / self.spacing_m))
        offsets = np.arange(-extent, extent + 1, dtype=np.float64) * self.spacing_m
        xx, yy = np.meshgrid(offsets, offsets, indexing="ij")
        distance = np.hypot(xx, yy)
        kernel = np.maximum(0.0, self.radius_m - distance)
        kernel[distance >= self.radius_m] = 0.0
        normalization = ndimage.convolve(
            np.ones(self.shape, dtype=np.float64),
            kernel,
            mode="constant",
            cval=0.0,
        )
        if not np.all(normalization > 0.0):
            raise RuntimeError("finite conic filter has an empty row")
        object.__setattr__(self, "extent", extent)
        object.__setattr__(self, "kernel", np.ascontiguousarray(kernel))
        object.__setattr__(
            self, "normalization", np.ascontiguousarray(normalization)
        )

    def _array(self, value: Any, *, label: str, bounded: bool = False) -> np.ndarray:
        result = np.asarray(value, dtype=np.float64)
        if result.shape != self.shape or not np.all(np.isfinite(result)):
            raise ValueError(f"{label} must be finite with shape {self.shape}")
        if bounded and (np.min(result) < 0.0 or np.max(result) > 1.0):
            raise ValueError(f"{label} must lie in [0,1]")
        return result

    def _convolve(self, value: np.ndarray) -> np.ndarray:
        return ndimage.convolve(value, self.kernel, mode="constant", cval=0.0)

    def filtered(self, latent: Any) -> np.ndarray:
        value = self._array(latent, label="latent density", bounded=True)
        return self._convolve(value) / self.normalization

    def filter_direction(self, direction: Any) -> np.ndarray:
        value = self._array(direction, label="latent direction")
        return self._convolve(value) / self.normalization

    def filter_transpose(self, cotangent: Any) -> np.ndarray:
        value = self._array(cotangent, label="filter cotangent")
        return self._convolve(value / self.normalization)

    def physical_nodes(self, latent: Any, beta: float) -> np.ndarray:
        beta_value = _validated_beta(beta)
        result = np.asarray(
            _projection(
                self.filtered(latent), beta=beta_value, eta=self.eta, xp=np
            ),
            dtype=np.float64,
        )
        if np.min(result) < -1.0e-14 or np.max(result) > 1.0 + 1.0e-14:
            raise RuntimeError("projected density left [0,1]")
        return np.ascontiguousarray(result)

    def jvp(self, latent: Any, direction: Any, beta: float) -> np.ndarray:
        beta_value = _validated_beta(beta)
        filtered = self.filtered(latent)
        return np.ascontiguousarray(
            _projection_derivative(filtered, beta=beta_value, eta=self.eta)
            * self.filter_direction(direction)
        )

    def vjp(self, latent: Any, projected_cotangent: Any, beta: float) -> np.ndarray:
        beta_value = _validated_beta(beta)
        filtered = self.filtered(latent)
        cotangent = self._array(
            projected_cotangent, label="projected-density cotangent"
        )
        return self.filter_transpose(
            cotangent
            * _projection_derivative(filtered, beta=beta_value, eta=self.eta)
        )

    def cell_density(self, latent: Any, beta: float) -> np.ndarray:
        return nodal_to_cell_average(self.physical_nodes(latent, beta))

    def cell_jvp(self, latent: Any, direction: Any, beta: float) -> np.ndarray:
        return nodal_to_cell_jvp(self.jvp(latent, direction, beta))

    def cell_vjp(self, latent: Any, cell_cotangent: Any, beta: float) -> np.ndarray:
        return self.vjp(latent, nodal_to_cell_vjp(cell_cotangent), beta)

    def jax_filtered(self, latent: Any) -> Any:
        """Differentiable float32/float64 realization of the same finite filter."""

        import jax.numpy as jnp
        from jax import lax

        value = jnp.asarray(latent)
        if value.shape != self.shape:
            raise ValueError(f"JAX latent density must have shape {self.shape}")
        kernel = jnp.asarray(self.kernel, dtype=value.dtype)
        lhs = value[None, :, :, None]
        rhs = kernel[:, :, None, None]
        convolved = lax.conv_general_dilated(
            lhs,
            rhs,
            window_strides=(1, 1),
            padding=((self.extent, self.extent), (self.extent, self.extent)),
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
        )[0, :, :, 0]
        return convolved / jnp.asarray(self.normalization, dtype=value.dtype)

    def jax_physical_nodes(self, latent: Any, beta: float) -> Any:
        import jax.numpy as jnp

        beta_value = _validated_beta(beta)
        return _projection(
            self.jax_filtered(latent), beta=beta_value, eta=self.eta, xp=jnp
        )

    def jax_cell_density(self, latent: Any, beta: float) -> Any:
        nodes = self.jax_physical_nodes(latent, beta)
        return 0.25 * (
            nodes[:-1, :-1]
            + nodes[1:, :-1]
            + nodes[:-1, 1:]
            + nodes[1:, 1:]
        )

    def coefficient_sha256(self) -> str:
        digest = hashlib.sha256()
        digest.update(b"fdtdx-parity-nodal-design-map-v1")
        digest.update(np.asarray(self.shape, dtype="<i8").tobytes())
        digest.update(
            np.asarray(
                [self.spacing_m, self.radius_m, self.eta], dtype="<f8"
            ).tobytes()
        )
        digest.update(np.asarray(self.kernel, dtype="<f8").tobytes())
        digest.update(np.asarray(self.normalization, dtype="<f8").tobytes())
        return digest.hexdigest()

    def audit(self) -> dict[str, Any]:
        constant_errors = {
            str(value): float(
                np.max(
                    np.abs(
                        self.physical_nodes(
                            np.full(self.shape, value, dtype=np.float64),
                            FIRST_CERTIFICATE_BETA,
                        )
                        - value
                    )
                )
            )
            for value in (0.0, 0.5, 1.0)
        }
        return {
            "schema": "fdtdx_4um_parity_nodal_filter_projection_v1",
            "status": (
                "PASS" if max(constant_errors.values()) < 1.0e-14 else "FAIL"
            ),
            "latent_shape": list(self.shape),
            "projected_nodal_shape": list(self.shape),
            "physical_cell_shape": list(CELL_SHAPE),
            "spacing_m": self.spacing_m,
            "filter_radius_m": self.radius_m,
            "projection_eta": self.eta,
            "first_certificate_beta": FIRST_CERTIFICATE_BETA,
            "kernel_shape": list(self.kernel.shape),
            "kernel_nonzero_count": int(np.count_nonzero(self.kernel)),
            "finite_nonperiodic_boundary": True,
            "forward_filter": "D^-1 C with zero padding and truncated-row normalization",
            "transpose_filter": "C D^-1 with the same symmetric conic kernel",
            "nodal_to_cell": "exact four-node average with committed transpose",
            "constant_projection_max_abs_errors": constant_errors,
            "coefficient_sha256": self.coefficient_sha256(),
            "optimizer_enabled": False,
        }


MAPPING = ParityNodalDesignMapping()


def deterministic_gray_latent() -> np.ndarray:
    """Smooth, bounded, non-symmetric control; never an optimizer candidate."""

    x = np.linspace(-1.0, 1.0, NODE_SHAPE[0], dtype=np.float64)[:, None]
    y = np.linspace(-1.0, 1.0, NODE_SHAPE[1], dtype=np.float64)[None, :]
    latent = (
        0.5
        + 0.16 * np.sin(0.8 * np.pi * x) * np.cos(0.6 * np.pi * y)
        + 0.07 * np.cos(1.1 * np.pi * x + 0.3) * np.sin(0.9 * np.pi * y)
    )
    if np.min(latent) <= 0.0 or np.max(latent) >= 1.0:
        raise RuntimeError("deterministic gray latent is not strictly interior")
    return np.ascontiguousarray(latent)


def control_density(case: str, *, beta: float = FIRST_CERTIFICATE_BETA) -> dict[str, Any]:
    if case == "empty":
        latent = np.zeros(NODE_SHAPE, dtype=np.float64)
    elif case == "full":
        latent = np.ones(NODE_SHAPE, dtype=np.float64)
    elif case == "nonuniform_gray":
        latent = deterministic_gray_latent()
    else:
        raise ValueError(f"unknown density control {case!r}")
    projected = MAPPING.physical_nodes(latent, beta)
    cells = nodal_to_cell_average(projected)
    return {
        "case": case,
        "beta": float(beta),
        "latent": latent,
        "projected_nodes": projected,
        "cells": cells,
        "ranges": {
            "latent": [float(np.min(latent)), float(np.max(latent))],
            "projected_nodes": [float(np.min(projected)), float(np.max(projected))],
            "cells": [float(np.min(cells)), float(np.max(cells))],
        },
    }


def main() -> int:
    import json

    payload = MAPPING.audit()
    payload["nonuniform_gray_beta4"] = control_density("nonuniform_gray")["ranges"]
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
