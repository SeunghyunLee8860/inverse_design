"""Independent DRC adversarial fixtures (opposing-boundary linewidth).

Covers the failure modes flagged in review: trivial phases must FAIL, large
sharp/rounded features must PASS (no curvature false-fail), thin bars/fingers/
frames must FAIL, seam-crossing measured periodically, disconnected-island gap.
"""

import numpy as np
import pytest

from geometry_drc import geometry_drc

N = 240
H = 0.025


def vbar(w, at=120):
    m = np.zeros((N, N), np.uint8)
    m[:, at : at + w] = 1
    return m


def void_channel(w, at=120):
    m = np.ones((N, N), np.uint8)
    m[:, at : at + w] = 0
    return m


# --- trivial phases must FAIL (regression: they used to PASS) ---
def test_all_void_fails():
    r = geometry_drc(np.zeros((N, N), np.uint8), H, 0.5, 0.5)
    assert r["pass"] is False and r["trivial_topology"] is True


def test_all_solid_fails():
    r = geometry_drc(np.ones((N, N), np.uint8), H, 0.5, 0.5)
    assert r["pass"] is False and r["trivial_topology"] is True


# --- large features must PASS (regression: 2um square false-failed at 150nm) ---
def test_large_square_passes():
    sq = np.zeros((N, N), np.uint8)
    sq[80:160, 80:160] = 1
    assert geometry_drc(sq, H, 0.5, 0.5)["pass"] is True


def test_500nm_frame_passes():
    fr = np.zeros((N, N), np.uint8)
    fr[60:180, 60:180] = 1
    fr[80:160, 80:160] = 0                 # 20-cell (500 nm) wide ring
    assert geometry_drc(fr, H, 0.5, 0.5)["pass"] is True


def test_two_large_islands_pass():
    m = np.zeros((N, N), np.uint8)
    m[40:100, 40:100] = 1
    m[40:100, 140:200] = 1
    assert geometry_drc(m, H, 0.5, 0.5, min_gap_um=0.5)["pass"] is True


# --- thin features must FAIL ---
@pytest.mark.parametrize("w,expect", [(8, False), (19, False), (21, True), (22, True)])
def test_solid_bar(w, expect):
    assert geometry_drc(vbar(w), H, 0.5, 0.5)["pass"] is expect


@pytest.mark.parametrize("w,expect", [(8, False), (19, False), (21, True), (22, True)])
def test_void_channel(w, expect):
    assert geometry_drc(void_channel(w), H, 0.5, 0.5)["pass"] is expect


def test_thin_finger_on_bulk_fails():
    m = np.zeros((N, N), np.uint8)
    m[:, :120] = 1
    m[112:120, 120:170] = 1               # 8-cell finger attached to bulk
    r = geometry_drc(m, H, 0.5, 0.5)
    assert r["pass"] is False
    assert r["minimum_solid_width_um"] < 0.5


def test_diagonal_true_width():
    ii, jj = np.indices((N, N))
    def band(w):
        m = np.zeros((N, N), np.uint8)
        m[((ii - jj) % N) < w] = 1
        return m
    # perpendicular width ~ w/sqrt(2): 19 -> ~13 (FAIL), 40 -> ~28 (PASS)
    assert geometry_drc(band(19), H, 0.5, 0.5)["pass"] is False
    assert geometry_drc(band(40), H, 0.5, 0.5)["pass"] is True


def test_non_binary_rejected():
    with pytest.raises(ValueError):
        geometry_drc(0.4 * np.ones((N, N)), H, 0.5, 0.5)


def test_disconnected_gap_rule():
    m = np.zeros((N, N), np.uint8)
    m[100:140, 40:80] = 1
    m[100:140, 99:139] = 1                 # 19-cell gap between two islands
    r = geometry_drc(m, H, 0.5, 0.5, min_gap_um=0.5)
    assert r["minimum_gap_um"] is not None and r["pass"] is False
