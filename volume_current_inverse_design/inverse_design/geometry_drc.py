#!/usr/bin/env python3
"""Independent periodic binary-geometry design-rule check (DRC).

This is the *operational* 500 nm guarantee.  It is deliberately implemented
with a different algorithm than the optimisation length-scale constraints
(``geometric_constraints.py``, which use conic-filter indicators): here we use a
morphological **local-thickness** measure (largest inscribed disk covering each
pixel) computed on the torus.  Calling the same indicator twice would not be an
independent check.

Measurement convention
----------------------
* pixels are unit cells of side ``spacing`` (=25 nm); a solid feature spanning
  ``W`` cells has physical width ``W * spacing``.
* local thickness of a foreground pixel = diameter (in physical units) of the
  largest solid disk that covers it, evaluated on the periodic tiling.
* the minimum solid/void width is the minimum local thickness over the
  respective foreground.
* the disk-covering measure rounds a W-cell bar UP by at most one cell, so the
  pass rule is intentionally conservative: a feature passes only when its
  measured width is at least ``min_width_um`` AND the measure cannot be inflating
  a sub-target feature past the bar.  Empirically (see
  ``tests/test_geometry_drc.py``) this makes 19-cell fail and 21-cell pass; the
  20-cell (exactly 500 nm) case is treated as a *fail* -- conservative by design.

Output is a plain JSON-serialisable dict; no autograd, no FDTD.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt, label as ndlabel

DRC_VERSION = "geometry_drc/v1"


def _tile3(mask: np.ndarray) -> np.ndarray:
    return np.tile(mask, (3, 3))


def _center(tiled: np.ndarray, nx: int, ny: int) -> np.ndarray:
    return tiled[nx : 2 * nx, ny : 2 * ny]


def _periodic_local_thickness_px(mask: np.ndarray, r_cap: int) -> np.ndarray:
    """Local thickness (in pixels) of the foreground of `mask`, periodic.

    thickness(p) = diameter of the largest inscribed foreground disk covering p,
    evaluated on the 3x3 torus tiling.  Radii are searched from ``r_cap`` down to
    1; a feature thicker than ``2*r_cap`` is reported as exactly ``2*r_cap`` (it
    is thick enough for any rule at or below the cap, so the truncation is always
    on the safe side).  Dilation by a disk of radius ``r`` is computed as
    ``EDT(~centers) <= r`` -- two O(N) distance transforms per radius instead of
    an O(N*r^2) structuring-element dilation, which keeps the sweep fast even
    when the foreground fills most of the cell.
    """
    nx, ny = mask.shape
    if not mask.any():
        return np.zeros_like(mask, float)
    tile = _tile3(mask)
    dist = distance_transform_edt(tile)  # px to nearest background
    thick = np.zeros_like(tile, dtype=float)
    for r in range(int(r_cap), 0, -1):
        centers = dist >= r
        if not centers.any():
            continue
        covered = distance_transform_edt(~centers) <= r  # dilation by disk(r)
        update = covered & tile & (thick == 0.0)
        thick[update] = 2.0 * r
    thick[tile & (thick == 0.0)] = 1.0  # thinner than 1 px radius (isolated)
    return _center(thick, nx, ny)


def _min_width_um(mask: np.ndarray, spacing_um: float, r_cap: int) -> tuple:
    """Return (min_width_um, argmin (ix,iy)) for the foreground of mask."""
    if not mask.any():
        return float("inf"), None
    thick = _periodic_local_thickness_px(mask, r_cap)
    thick_fg = np.where(mask, thick, np.inf)
    idx = int(np.argmin(thick_fg))
    ix, iy = np.unravel_index(idx, mask.shape)
    return float(thick_fg[ix, iy] * spacing_um), (int(ix), int(iy))


def _periodic_components(mask: np.ndarray) -> tuple:
    """Label connected foreground components on the torus (8-connectivity).

    A wrapped feature is split into several labels by ``ndlabel`` on the 3x3
    tile; we union the tile labels of every center pixel with the tile labels of
    its four periodic-image copies, so a feature that crosses the seam becomes a
    single component.  Returns (labels_center (nx,ny), ncomp).
    """
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
    """Minimum periodic separation between DISTINCT solid components."""
    labels, ncomp = _periodic_components(mask)
    if ncomp < 2:
        return float("inf"), None
    nx, ny = mask.shape
    best = float("inf")
    best_at = None
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


def geometry_drc(
    mask: np.ndarray,
    spacing_um: float = 0.025,
    min_solid_width_um: float = 0.5,
    min_void_width_um: float = 0.5,
    min_gap_um: float | None = None,
    conservative_cells: float = 0.5,
) -> dict:
    """Independent periodic binary DRC.

    Args:
        mask: 2-D array; nonzero = solid.  Must be exactly binary.
        spacing_um: cell size (um).
        min_solid_width_um / min_void_width_um: rules.
        min_gap_um: optional disconnected-component separation rule
            (None -> not applicable).
        conservative_cells: measured width must clear the rule by this many
            cells to pass (guards the +/-1 cell disk-covering rounding).
    """
    arr = np.asarray(mask)
    uniq = np.unique(arr)
    if not np.all(np.isin(uniq, (0, 1))):
        raise ValueError(f"mask is not binary; unique values = {uniq[:8]}")
    solid = arr.astype(bool)
    void = ~solid
    guard = conservative_cells * spacing_um
    # only need thickness resolved up to a little past the rule; anything
    # thicker passes any rule at/below the cap.
    rule_um = max(min_solid_width_um, min_void_width_um) + guard
    r_cap = int(np.ceil(rule_um / spacing_um)) + 2

    solid_w, solid_at = _min_width_um(solid, spacing_um, r_cap)
    void_w, void_at = _min_width_um(void, spacing_um, r_cap)
    solid_ok = solid_w >= min_solid_width_um + guard
    void_ok = void_w >= min_void_width_um + guard

    result = {
        "pass": bool(solid_ok and void_ok),
        "periodic": True,
        "spacing_um": float(spacing_um),
        "solid_fraction": float(solid.mean()),
        "minimum_solid_width_um": None if not np.isfinite(solid_w) else round(solid_w, 6),
        "minimum_void_width_um": None if not np.isfinite(void_w) else round(void_w, 6),
        "minimum_gap_um": None,
        "solid_violations": [] if solid_ok else [{"pixel": solid_at, "width_um": solid_w}],
        "void_violations": [] if void_ok else [{"pixel": void_at, "width_um": void_w}],
        "gap_violations": [],
        "rules": {
            "minimum_solid_width_um": min_solid_width_um,
            "minimum_void_width_um": min_void_width_um,
            "minimum_disconnected_boundary_gap_um": min_gap_um,
            "conservative_cells": conservative_cells,
        },
        "measurement_convention": (
            "periodic disk-covering local thickness; pass requires measured "
            "width >= rule + conservative_cells*spacing"
        ),
        "code_version": DRC_VERSION,
    }
    if min_gap_um is not None:
        gap, gap_at = _min_disconnected_gap_um(solid, spacing_um)
        gap_ok = gap >= min_gap_um + guard
        result["minimum_gap_um"] = None if not np.isfinite(gap) else round(gap, 6)
        result["gap_violations"] = [] if gap_ok else [{"pixel": gap_at, "gap_um": gap}]
        result["pass"] = bool(result["pass"] and gap_ok)
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
        arr = arr[:-1, :-1]  # drop fencepost to unique grid
    return (arr >= 0.5).astype(np.uint8)


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
    report = geometry_drc(
        mask, args.spacing_um, args.min_solid_um, args.min_void_um, args.min_gap_um
    )
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        Path(args.output).write_text(text + "\n")
    raise SystemExit(0 if report["pass"] else 2)


if __name__ == "__main__":
    main()
