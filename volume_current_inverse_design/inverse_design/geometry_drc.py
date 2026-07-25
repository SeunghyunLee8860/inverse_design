#!/usr/bin/env python3
"""Independent periodic binary-geometry design-rule check (DRC).

The *operational* 500 nm guarantee.  Deliberately a different algorithm than the
optimisation length-scale constraints (`geometric_constraints.py`).

Minimum feature/gap width = **opposing-boundary distance** (a.k.a. local
linewidth), NOT an inscribed-disk / rolling-ball radius:

  for each foreground pixel p, let v = nearest background pixel (periodic EDT
  feature transform); march from p in the +(-p->v) direction (INTO the phase)
  until the opposite boundary; width(p) = dist(p,v) + dist to that opposite
  boundary.  The minimum over the phase is the minimum linewidth.

Why this and not "largest inscribed disk covering p": the disk measure conflates
minimum linewidth with minimum radius of curvature, so it FALSELY fails sharp
convex corners of otherwise-large features (a 2 um square measured 150 nm).  The
opposing-boundary measure returns the true width: a square edge -> W, a square
corner -> ~W*sqrt2 (large, not flagged), a thin bar/finger/ring -> its width.

Trivial-phase gate: an all-solid or all-void array has no second phase and is
never a valid 500 nm two-phase design -> it FAILS (previously such arrays PASSED,
a critical bug).

If the `imageruler` package is importable its measurement is RECORDED alongside
(field ``imageruler_cross_check_um``) for comparison, but the opposing-boundary
measure is authoritative for pass/fail (the cross-check is informational, not
enforced).  Output is a plain JSON-serialisable dict; no autograd/FDTD.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt, label as ndlabel

DRC_VERSION = "geometry_drc/v2-opposing-boundary"


def _tile3(mask: np.ndarray) -> np.ndarray:
    return np.tile(mask, (3, 3))


def _center(tiled: np.ndarray, nx: int, ny: int) -> np.ndarray:
    return tiled[nx : 2 * nx, ny : 2 * ny]


def _min_width_px(mask: np.ndarray, r_cap: int) -> tuple:
    """Minimum opposing-boundary width (in pixels) of the True phase, periodic.

    Returns (min_width_px, (ix,iy) in unique coords) or (inf, None) if the phase
    is empty or full (no opposing boundary exists).
    """
    nx, ny = mask.shape
    if not mask.any() or mask.all():
        return float("inf"), None
    tile = _tile3(mask)
    edt, inds = distance_transform_edt(tile, return_indices=True)
    edt_c = _center(edt, nx, ny)
    cand = mask & (edt_c <= r_cap + 1.0)          # only near-boundary (thin) pixels
    ij = np.argwhere(cand)
    if ij.size == 0:
        return float("inf"), None
    P = ij + np.array([nx, ny])                    # tile coords of candidates
    d0 = edt[P[:, 0], P[:, 1]]                      # distance to nearest background
    V = np.stack([inds[0][P[:, 0], P[:, 1]], inds[1][P[:, 0], P[:, 1]]], axis=1)
    dvec = (P - V).astype(float)                    # from background INTO the phase
    norm = np.hypot(dvec[:, 0], dvec[:, 1])
    norm[norm == 0.0] = 1.0
    u = dvec / norm[:, None]
    maxstep = int(2 * r_cap) + 2
    d2 = np.full(len(P), np.inf)
    remaining = np.ones(len(P), bool)
    for s in range(1, maxstep + 1):
        rp = P[remaining] + s * u[remaining]
        pos = np.round(rp).astype(int)
        pos[:, 0] %= 3 * nx
        pos[:, 1] %= 3 * ny
        isbg = ~tile[pos[:, 0], pos[:, 1]]
        ridx = np.where(remaining)[0][isbg]
        d2[ridx] = s
        remaining[np.where(remaining)[0][isbg]] = False
        if not remaining.any():
            break
    width = d0 + d2
    k = int(np.argmin(width))
    return float(width[k]), (int(ij[k, 0]), int(ij[k, 1]))


def _periodic_components(mask: np.ndarray) -> tuple:
    nx, ny = mask.shape
    tile = _tile3(mask)
    lab, _ = ndlabel(tile, structure=np.ones((3, 3), int))
    parent: dict[int, int] = {}

    def find(a: int) -> int:
        parent.setdefault(a, a)
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    fg = np.argwhere(mask)
    for i, j in fg:
        base = int(lab[nx + i, ny + j])
        for a, b in ((i, ny + j), (2 * nx + i, ny + j), (nx + i, j), (nx + i, 2 * ny + j)):
            other = int(lab[a, b])
            if other > 0:
                union(base, other)
    out = np.zeros((nx, ny), int)
    ids: dict[int, int] = {}
    nextid = 0
    for i, j in fg:
        root = find(int(lab[nx + i, ny + j]))
        if root not in ids:
            nextid += 1
            ids[root] = nextid
        out[i, j] = ids[root]
    return out, nextid


def _min_disconnected_gap_um(mask: np.ndarray, spacing_um: float) -> tuple:
    labels, ncomp = _periodic_components(mask)
    if ncomp < 2:
        return float("inf"), None
    nx, ny = mask.shape
    best, best_at = float("inf"), None
    for cid in range(1, ncomp + 1):
        this = labels == cid
        others = mask & (labels != cid)
        if not others.any():
            continue
        dist_c = _center(distance_transform_edt(~_tile3(this)), nx, ny)
        cand = np.where(others, dist_c, np.inf)
        idx = int(np.argmin(cand))
        val = float(cand.ravel()[idx])
        if val < best:
            best = val
            best_at = tuple(int(v) for v in np.unravel_index(idx, mask.shape))
    if not np.isfinite(best):
        return float("inf"), None
    return float(best * spacing_um), best_at


def _imageruler_widths(mask, spacing_um):
    """Optional cross-check via the imageruler package (returns (solid,void) um)."""
    try:
        import imageruler
    except Exception:
        return None
    try:
        solid = imageruler.minimum_length_scale(mask.astype(bool), periodic=(True, True))
        void = imageruler.minimum_length_scale(~mask.astype(bool), periodic=(True, True))
        return float(solid) * spacing_um, float(void) * spacing_um
    except Exception:
        return None


def geometry_drc(
    mask: np.ndarray,
    spacing_um: float = 0.025,
    min_solid_width_um: float = 0.5,
    min_void_width_um: float = 0.5,
    min_gap_um: float | None = None,
    conservative_cells: float = 0.5,
) -> dict:
    arr = np.asarray(mask)
    uniq = np.unique(arr)
    if not np.all(np.isin(uniq, (0, 1))):
        raise ValueError(f"mask is not binary; unique values = {uniq[:8]}")
    solid = arr.astype(bool)
    void = ~solid
    frac = float(solid.mean())
    guard = conservative_cells * spacing_um
    rule_um = max(min_solid_width_um, min_void_width_um) + guard
    r_cap = int(np.ceil(rule_um / spacing_um)) + 2

    # trivial-phase gate: a valid two-phase 500 nm design needs BOTH phases.
    trivial = (not solid.any()) or (not void.any())

    solid_w, solid_at = _min_width_px(solid, r_cap)
    void_w, void_at = _min_width_px(void, r_cap)
    solid_w_um = solid_w * spacing_um
    void_w_um = void_w * spacing_um
    solid_ok = (not trivial) and solid_w_um >= min_solid_width_um + guard
    void_ok = (not trivial) and void_w_um >= min_void_width_um + guard

    result = {
        "pass": bool(solid_ok and void_ok),
        "trivial_topology": bool(trivial),
        "periodic": True,
        "spacing_um": float(spacing_um),
        "solid_fraction": frac,
        "minimum_solid_width_um": None if not np.isfinite(solid_w) else round(solid_w_um, 6),
        "minimum_void_width_um": None if not np.isfinite(void_w) else round(void_w_um, 6),
        "minimum_gap_um": None,
        "solid_violations": [] if solid_ok else [{"pixel": solid_at, "width_um": solid_w_um}],
        "void_violations": [] if void_ok else [{"pixel": void_at, "width_um": void_w_um}],
        "gap_violations": [],
        "rules": {
            "minimum_solid_width_um": min_solid_width_um,
            "minimum_void_width_um": min_void_width_um,
            "minimum_disconnected_boundary_gap_um": min_gap_um,
            "conservative_cells": conservative_cells,
        },
        "measurement_convention": (
            "opposing-boundary (local linewidth); pass requires width >= rule + "
            "conservative_cells*spacing AND both phases present"
        ),
        "code_version": DRC_VERSION,
    }
    if min_gap_um is not None:
        gap, gap_at = _min_disconnected_gap_um(solid, spacing_um)
        # P1-2: a single island tiled with a moat is ONE torus component, so the
        # disconnected-component search returns inf.  Its separation from its own
        # periodic image IS the moat = the void local width; use that so the gap
        # rule is actually measured (previously reported None but SUCCESS said
        # gap ok).
        if not np.isfinite(gap):
            gap, gap_at = void_w_um, void_at
        gap_ok = (not trivial) and np.isfinite(gap) and gap >= min_gap_um + guard
        result["minimum_gap_um"] = None if not np.isfinite(gap) else round(gap, 6)
        result["gap_violations"] = [] if gap_ok else [{"pixel": gap_at, "gap_um": gap}]
        result["pass"] = bool(result["pass"] and gap_ok)

    ir = _imageruler_widths(solid, spacing_um)
    if ir is not None:
        result["imageruler_cross_check_um"] = {"solid": ir[0], "void": ir[1]}
    return result


def _load_mask(path: Path) -> np.ndarray:
    data = np.load(path)
    for key in ("mask", "mask_unique", "physical", "binary"):
        if key in getattr(data, "files", []):
            arr = np.asarray(data[key])
            break
    else:
        raise KeyError(f"{path} has no mask/mask_unique/physical array")
    if arr.ndim == 3:
        arr = arr[:, :, arr.shape[2] // 2]
    if arr.shape[0] == arr.shape[1] and arr.shape[0] % 2 == 1:
        arr = arr[:-1, :-1]
    # P1-4: do NOT silently threshold a gray array.  An exact-binary DRC must be
    # fed an exact-binary mask; geometry_drc() raises on non-binary input.
    return arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mask_npz")
    ap.add_argument("--spacing-um", type=float, default=0.025)
    ap.add_argument("--min-solid-um", type=float, default=0.5)
    ap.add_argument("--min-void-um", type=float, default=0.5)
    ap.add_argument("--min-gap-um", type=float, default=None)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    mask = _load_mask(Path(args.mask_npz).resolve())
    report = geometry_drc(mask, args.spacing_um, args.min_solid_um,
                          args.min_void_um, args.min_gap_um)
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        Path(args.output).write_text(text + "\n")
    raise SystemExit(0 if report["pass"] else 2)


if __name__ == "__main__":
    main()
