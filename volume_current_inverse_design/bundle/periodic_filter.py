"""Endpoint-free periodic conic filter on the unique 240x240 design grid.

The legacy ``msopt.Filters.conic_filter`` is entangled with the
``L*resolution + 1`` node convention: it rebuilds the grid from
``mesh_grid(radius, Lx, Ly, resolution)`` and reshapes the input to that many
nodes.  On a 240x240 *unique* periodic latent that convention is off-by-one --
the last row/column of a 241-node period leaks into the interior and is then
discarded on output.

This module implements a clean circular (torus) conic filter that operates on
exactly ``(Nux, Nuy)`` unique nodes.  Properties (all covered by
``inverse_design/tests/test_periodic_unique_mapping.py``):

  * shape in == shape out == (Nux, Nuy)
  * circular convolution using the minimal torus offset for every node
  * kernel ``max(0, 1 - r/R)`` normalised to unit sum
  * a constant field maps to exactly the same constant
  * roll (x or y) equivariance to machine precision
  * an impulse maps to the (symmetric) kernel
  * autograd-differentiable exact VJP

The kernel FFT is precomputed once with plain numpy (it is a constant and
carries no gradient); only the design array flows through the autograd FFT, so
the VJP is exact and cheap.
"""

from __future__ import annotations

from functools import lru_cache

import autograd.numpy as anp
import numpy as np


@lru_cache(maxsize=32)
def _kernel_fft(nux: int, nuy: int, radius_um: float, dx_um: float, dy_um: float):
    """Return fft2 of the unit-sum conic kernel centred at the torus origin."""
    # np.fft.fftfreq(N)*N == [0, 1, ..., N/2-1, -N/2, ..., -1]: the signed
    # minimal torus offset of each node from the origin.  Using it as the
    # convolution origin makes the kernel exactly symmetric on the torus.
    ox = np.fft.fftfreq(nux) * nux
    oy = np.fft.fftfreq(nuy) * nuy
    xx = ox[:, None] * float(dx_um)
    yy = oy[None, :] * float(dy_um)
    r = np.sqrt(xx * xx + yy * yy)
    h = np.maximum(0.0, 1.0 - r / float(radius_um))
    total = h.sum()
    if total <= 0:
        raise ValueError(f"conic kernel radius {radius_um} um too small for grid")
    h = h / total
    return np.fft.fft2(h)


def periodic_conic_filter_unique(x_unique, radius_um, dx_um, dy_um):
    """Circular conic filter of a 2-D unique periodic field.

    Args:
        x_unique: (Nux, Nuy) design weights on the unique periodic grid.
        radius_um: conic filter radius in micrometres.
        dx_um, dy_um: node spacing in micrometres.

    Returns:
        (Nux, Nuy) filtered field, autograd-differentiable.
    """
    x2d = anp.reshape(x_unique, (x_unique.shape[0], x_unique.shape[1]))
    nux, nuy = int(x2d.shape[0]), int(x2d.shape[1])
    hk = _kernel_fft(nux, nuy, float(radius_um), float(dx_um), float(dy_um))
    # Circular convolution via FFT.  hk is a plain-numpy constant, so only x2d
    # carries gradient; anp.fft provides the exact VJP through fft2/ifft2.
    y = anp.real(anp.fft.ifft2(anp.fft.fft2(x2d) * hk))
    return y


def measured_kernel(nux, nuy, radius_um, dx_um, dy_um):
    """Real-space conic kernel (for tests / inspection), centred at origin."""
    ox = np.fft.fftfreq(nux) * nux
    oy = np.fft.fftfreq(nuy) * nuy
    xx = ox[:, None] * float(dx_um)
    yy = oy[None, :] * float(dy_um)
    r = np.sqrt(xx * xx + yy * yy)
    h = np.maximum(0.0, 1.0 - r / float(radius_um))
    return h / h.sum()
