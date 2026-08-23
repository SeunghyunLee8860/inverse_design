"""Single Au material-fraction contract shared by all three physics.

``rho`` is the projected topology variable.  During continuous optimization it
is only a numerical relaxation; only an exact-binary density is promoted as a
physical Au/void geometry.  Maxwell, thermal, and electrical operators must
nevertheless see the same relaxed Au fraction so one subsystem cannot exploit
a different amount of gray Au from another subsystem.
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
        "law": AU_MATERIAL_FRACTION_LAW,
        "exponent": AU_MATERIAL_FRACTION_EXPONENT,
        "optical_fraction": "au_material_fraction(rho)",
        "thermal_fraction": "au_material_fraction(rho)",
        "electrical_fraction": "au_material_fraction(rho)",
        "gray_density_is_physical_geometry": False,
        "promotion_requires_exact_binary_density": True,
    }
