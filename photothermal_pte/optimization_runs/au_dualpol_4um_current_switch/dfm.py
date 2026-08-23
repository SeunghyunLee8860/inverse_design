"""Finite-window 500 nm solid/void audit for the 100 nm Au design grid."""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def physical_disk_footprint(radius_m: float, spacing_m: float) -> np.ndarray:
    if radius_m <= 0.0 or spacing_m <= 0.0:
        raise ValueError("radius and spacing must be positive")
    extent = int(np.ceil(radius_m / spacing_m))
    offset = np.arange(-extent, extent + 1, dtype=float) * spacing_m
    xx, yy = np.meshgrid(offset, offset, indexing="ij")
    result = np.hypot(xx, yy) <= radius_m + 1.0e-15 * spacing_m
    if not result[extent, extent]:
        raise RuntimeError("morphology footprint lost its origin")
    return result


def exact_500nm_audit(
    physical_density: np.ndarray,
    spacing_m: float = 100.0e-9,
    minimum_feature_m: float = 500.0e-9,
) -> dict[str, object]:
    """Audit thresholded Au and void with a physical radius-250 nm opening.

    Outside the finite design window is void.  The test is an exact discrete
    audit on the chosen 100 nm grid; it is not used as a differentiable
    optimizer constraint and no geometry is silently repaired here.
    """

    rho = np.asarray(physical_density, dtype=float)
    if rho.ndim != 2 or not np.all(np.isfinite(rho)):
        raise ValueError("physical density must be a finite 2-D array")
    binary = rho >= 0.5
    footprint = physical_disk_footprint(0.5 * minimum_feature_m, spacing_m)
    solid_open = ndimage.binary_opening(binary, structure=footprint, border_value=0)
    void_open = ndimage.binary_opening(~binary, structure=footprint, border_value=1)
    bad_solid = binary & ~solid_open
    bad_void = (~binary) & ~void_open
    return {
        "minimum_feature_nm": minimum_feature_m * 1.0e9,
        "opening_radius_nm": 0.5 * minimum_feature_m * 1.0e9,
        "spacing_nm": spacing_m * 1.0e9,
        "footprint_pixel_count": int(np.count_nonzero(footprint)),
        "solid_bad_cell_count": int(np.count_nonzero(bad_solid)),
        "void_bad_cell_count": int(np.count_nonzero(bad_void)),
        "solid_pass": bool(not np.any(bad_solid)),
        "void_pass": bool(not np.any(bad_void)),
        "binary": binary,
        "bad_solid": bad_solid,
        "bad_void": bad_void,
    }

