#!/usr/bin/env python3
"""DRC-driven deterministic binary-mask cleanup for the finalizer.

Why this exists (measured, official run r5 2026-07-28): the optimisation
length-scale constraints sit at their reachable floor (g_solid=-1.85e-5,
g_void=-1.0e-6, fully feasible) while the exact-binary projection still
carries sub-rule detail the independent DRC rejects.  The DRC only REPORTS
the single minimum-width pixel, but measuring the whole r5 mask found 333
solid + 712 void pixels below rule: tapering wedge tips, edge bumps and
12-px "ledges" on block sides.  The autograd constraint has NO gradient
against these (it is already feasible), so more FDTD iterations or higher
beta cannot remove them; only a deterministic mask-level cleanup can.

Algorithm (each stage measured on the real r5 failure):
  1. Periodic morphological open+close with a rule-diameter disk.  This
     removes the bulk of the debris (~1000 px -> ~200 px below rule) but
     CANNOT finish the job: opening guarantees every surviving pixel is
     covered by an inscribed rule-disk, yet the DRC's opposing-boundary ray
     through a pixel near the disk rim can still read far below the rule
     (measured: a 12-px ledge beside a large block survives opening because
     the disk nestles into the corner, but its vertical linewidth is 12 px).
  2. Site-wise MONOTONE local surgery on the residue: violating pixels are
     clustered (periodic 8-connectivity); each cluster is resolved inside a
     local window by flipping in ONE direction only -- thin solid is carved
     to void, thin void is filled with solid -- re-measuring with the SAME
     opposing-boundary width map as the DRC (imported from geometry_drc,
     deliberately not reimplemented) until the window is locally clean.
     One-directional flips make the solid monotone within a site, so the
     global carve<->fill oscillation of a naive "flip everything thin"
     loop (measured: 24,020 flips, limit cycle, no convergence) cannot occur.
  3. The final verdict is geometry_drc itself on the cleaned mask; on
     non-convergence the caller must fail closed (the finalizer keeps its
     geometry_infeasible abort).

This does NOT weaken any SUCCESS gate: the independent DRC still judges the
(cleaned) mask, the exact-binary FDTD measures the (cleaned) mask, and the
finalizer records pre/post masks, the per-stage log and the changed pixel
count in the manifest and SUCCESS.json.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_closing, binary_opening, label as ndlabel

from geometry_drc import _width_map_px, geometry_drc

CLEANUP_VERSION = "drc_cleanup/v2-morphology+monotone-sites"


def _disk(radius_px: float) -> np.ndarray:
    n = int(np.ceil(radius_px))
    y, x = np.mgrid[-n : n + 1, -n : n + 1]
    return (x * x + y * y) <= radius_px * radius_px


def _wrap_morph(mask: np.ndarray, se: np.ndarray, op) -> np.ndarray:
    """Periodic morphology: wrap-pad by the FULL SE size (a half-SE pad leaves
    the pad ring's own erosion wrong within one half-SE of the crop edge)."""
    pad = se.shape[0]
    return op(np.pad(mask, pad, mode="wrap"), structure=se)[pad:-pad, pad:-pad]


def _violations(mask: np.ndarray, rule_px: float, r_cap: int):
    """(solid_pts, void_pts) below rule, as (k,2) int arrays."""
    out = []
    for phase in (mask, ~mask):
        w, ij = _width_map_px(phase, r_cap)
        out.append(ij[w < rule_px] if w.size else np.empty((0, 2), int))
    return out[0], out[1]


def _periodic_clusters(pts: np.ndarray, shape: tuple) -> list:
    """Group violating pixels by periodic 8-connectivity."""
    grid = np.zeros(shape, bool)
    grid[pts[:, 0], pts[:, 1]] = True
    tile = np.tile(grid, (2, 2))          # enough to see wrap-adjacency
    lab, n = ndlabel(tile, structure=np.ones((3, 3), int))
    seen, clusters = set(), []
    for i, j in pts:
        root = lab[i, j]
        # unify with wrap copies
        for a, b in ((i + shape[0], j), (i, j + shape[1]), (i + shape[0], j + shape[1])):
            root = min(root, lab[a, b]) if lab[a, b] else root
        if root in seen:
            continue
        seen.add(root)
        members = pts[[lab[p, q] == root or lab[p + shape[0], q] == root
                       or lab[p, q + shape[1]] == root
                       or lab[p + shape[0], q + shape[1]] == root
                       for p, q in pts]]
        clusters.append(members)
    return clusters


def _window(shape: tuple, pts: np.ndarray, margin: int) -> np.ndarray:
    """Periodic boolean window: within `margin` (Chebyshev) of any pt."""
    win = np.zeros(shape, bool)
    win[pts[:, 0], pts[:, 1]] = True
    for ax in (0, 1):
        acc = win.copy()
        for s in range(1, margin + 1):
            acc |= np.roll(win, s, axis=ax) | np.roll(win, -s, axis=ax)
        win = acc
    return win


def _resolve_site(solid: np.ndarray, pts: np.ndarray, phase_is_solid: bool,
                  rule_px: float, r_cap: int, margin: int,
                  max_iters: int = 80, budget: int = 4000):
    """Monotone local surgery: carve thin solid / fill thin void, window-only.

    Mutates `solid` in place.  Returns (converged, pixels_flipped).
    """
    win = _window(solid.shape, pts, margin)
    flipped = 0
    for _ in range(max_iters):
        phase = solid if phase_is_solid else ~solid
        w, ij = _width_map_px(phase, r_cap)
        if w.size == 0:
            return False, flipped              # phase vanished: caller fails
        thin = (w < rule_px) & win[ij[:, 0], ij[:, 1]]
        if not thin.any():
            return True, flipped
        k = int(thin.sum())
        if flipped + k > budget:
            return False, flipped              # runaway cascade: give up site
        solid[ij[thin, 0], ij[thin, 1]] = not phase_is_solid
        flipped += k
    return False, flipped


def drc_cleanup(
    mask: np.ndarray,
    spacing_um: float = 0.025,
    min_solid_width_um: float = 0.5,
    min_void_width_um: float = 0.5,
    min_gap_um: float | None = None,
    conservative_cells: float = 0.5,
    morph_rounds: int = 2,
    site_rounds: int = 5,
) -> dict:
    """Return {'mask', 'pass', 'pixels_changed', 'stages', 'drc', 'version'}.

    `mask` must be exact binary on the unique grid.  `pass` is the verdict of
    geometry_drc on the returned mask -- False means cleanup could not reach a
    rule-clean design and the caller must treat the design as infeasible.
    """
    arr = np.asarray(mask)
    uniq = np.unique(arr)
    if not np.all(np.isin(uniq, (0, 1))):
        raise ValueError(f"mask is not binary; unique values = {uniq[:8]}")
    solid = arr.astype(bool).copy()

    guard = conservative_cells * spacing_um
    rule_um = max(min_solid_width_um, min_void_width_um) + guard
    r_cap = int(np.ceil(rule_um / spacing_um)) + 2   # same cap formula as the DRC
    # one strict rule for both phases: cleanup targets the tighter requirement
    rule_px = rule_um / spacing_um
    margin = int(np.ceil(2 * rule_px))

    def _drc(m):
        return geometry_drc(m.astype(np.uint8), spacing_um=spacing_um,
                            min_solid_width_um=min_solid_width_um,
                            min_void_width_um=min_void_width_um,
                            min_gap_um=min_gap_um,
                            conservative_cells=conservative_cells)

    stages = []
    drc = _drc(solid)

    # --- stage 1: periodic morphology (bulk debris) ---
    if not drc["pass"]:
        se = _disk(rule_px / 2.0)
        for n in range(1, morph_rounds + 1):
            solid = _wrap_morph(solid, se, binary_opening)
            solid = _wrap_morph(solid, se, binary_closing)
        drc = _drc(solid)
        sv, vv = _violations(solid, rule_px, r_cap)
        stages.append({"stage": f"morphology x{morph_rounds}",
                       "drc_pass": bool(drc["pass"]),
                       "solid_below_rule": int(len(sv)),
                       "void_below_rule": int(len(vv))})

    # --- stage 2: site-wise monotone surgery on the residue ---
    for rnd in range(1, site_rounds + 1):
        if drc["pass"]:
            break
        sv, vv = _violations(solid, rule_px, r_cap)
        if len(sv) == 0 and len(vv) == 0:
            break                                  # DRC fail for another reason
        round_flips, unresolved = 0, 0
        for phase_is_solid, pts in ((True, sv), (False, vv)):
            if len(pts) == 0:
                continue
            for cluster in _periodic_clusters(pts, solid.shape):
                ok, k = _resolve_site(solid, cluster, phase_is_solid,
                                      rule_px, r_cap, margin)
                round_flips += k
                unresolved += 0 if ok else 1
        drc = _drc(solid)
        stages.append({"stage": f"site-surgery round {rnd}",
                       "sites_solid": int(len(sv) > 0), "flips": int(round_flips),
                       "unresolved_sites": int(unresolved),
                       "drc_pass": bool(drc["pass"])})
        if round_flips == 0:
            break                                  # no progress possible

    net = int((solid != arr.astype(bool)).sum())
    return {
        "version": CLEANUP_VERSION,
        "mask": solid.astype(np.uint8),
        "pass": bool(drc["pass"]),
        "pixels_changed": net,
        "pixels_changed_fraction": net / float(arr.size),
        "stages": stages,
        "drc": drc,
    }
