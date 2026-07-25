"""Independent DRC fixtures: 19 fail / 21 pass for solid, void, diagonal, seam."""

import numpy as np
import pytest

from geometry_drc import geometry_drc

N = 240
H = 0.025


def solid_bar(w):
    m = np.zeros((N, N), np.uint8)
    m[:, N // 2 : N // 2 + w] = 1
    return m


def void_channel(w):
    m = np.ones((N, N), np.uint8)
    m[:, N // 2 : N // 2 + w] = 0
    return m


def diagonal_band(w):
    ii, jj = np.indices((N, N))
    band = (ii - jj) % N
    m = np.zeros((N, N), np.uint8)
    m[band < w] = 1
    return m


def seam_bar(w):
    """Solid bar straddling the x=0 seam (wraps)."""
    m = np.zeros((N, N), np.uint8)
    m[:, : w // 2] = 1
    m[:, N - (w - w // 2) :] = 1
    return m


@pytest.mark.parametrize("w,expect", [(18, False), (19, False), (21, True), (22, True)])
def test_solid_bar(w, expect):
    r = geometry_drc(solid_bar(w), H, 0.5, 0.5)
    assert r["pass"] is expect


@pytest.mark.parametrize("w,expect", [(18, False), (19, False), (21, True), (22, True)])
def test_void_channel(w, expect):
    r = geometry_drc(void_channel(w), H, 0.5, 0.5)
    assert r["pass"] is expect


def test_diagonal_fail_then_pass():
    # a band |i-j|<w has TRUE perpendicular width w/sqrt(2); the DRC measures the
    # real (diagonal-aware) thickness, so w=19 -> ~13 cells (fail) and w=32 ->
    # ~22.6 cells (pass).  This is exactly why a run-length DRC is insufficient.
    assert geometry_drc(diagonal_band(19), H, 0.5, 0.5)["pass"] is False
    assert geometry_drc(diagonal_band(32), H, 0.5, 0.5)["pass"] is True


def test_seam_crossing_measured_periodically():
    # a 21-cell bar split across the seam must still be seen as one 21-cell bar
    r = geometry_drc(seam_bar(21), H, 0.5, 0.5)
    assert r["minimum_solid_width_um"] is not None


def test_non_binary_rejected():
    m = 0.4 * np.ones((N, N))
    with pytest.raises(ValueError):
        geometry_drc(m, H, 0.5, 0.5)


def test_two_islands_gap_rule():
    # two 40-cell solid squares separated by ~19-cell void gap -> gap rule fails
    m = np.zeros((N, N), np.uint8)
    m[100:140, 40:80] = 1
    m[100:140, 99:139] = 1  # 19-cell gap between the two squares
    r = geometry_drc(m, H, 0.5, 0.5, min_gap_um=0.5)
    assert r["minimum_gap_um"] is not None
    assert r["pass"] is False
