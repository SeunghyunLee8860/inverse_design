"""Historical shared-linear FDTDX material-fraction baseline.

``rho`` is the projected topology variable.  During continuous optimization it
is only a numerical relaxation; only an exact-binary density is promoted as a
physical Au/void geometry.  This module removed the historical O3/TE1 software
mismatch, but it is not the selected Lumerical optical constitutive law.  The
Lumerical route uses the same projected ``rho`` with the published nonlinear
``n-k`` interpolation in ``au_density_relaxation.py``.
"""

from __future__ import annotations


AU_MATERIAL_FRACTION_LAW = "shared_linear_projected_density"
AU_MATERIAL_FRACTION_EXPONENT = 1.0


def au_material_fraction(rho):
    """Return the common relaxed Au fraction for NumPy, JAX, or Torch arrays."""

    return rho**AU_MATERIAL_FRACTION_EXPONENT


def d_au_material_fraction_drho(rho):
    """Derivative of the common Au fraction with respect to projected rho."""

    return (
        AU_MATERIAL_FRACTION_EXPONENT
        * rho ** (AU_MATERIAL_FRACTION_EXPONENT - 1.0)
    )


def audit() -> dict[str, object]:
    return {
        "scope": "historical_fdtdx_consistency_baseline",
        "law": AU_MATERIAL_FRACTION_LAW,
        "exponent": AU_MATERIAL_FRACTION_EXPONENT,
        "optical_fraction": "au_material_fraction(rho)",
        "thermal_fraction": "au_material_fraction(rho)",
        "electrical_fraction": "au_material_fraction(rho)",
        "gray_density_is_physical_geometry": False,
        "promotion_requires_exact_binary_density": True,
        "selected_lumerical_optical_law": "christiansen_nk_then_square_v1",
        "rho_cubed_used_in_selected_lumerical_optical_law": False,
    }
