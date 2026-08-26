"""Relaxed Au occupancy shared by the Lumerical and custom-CUDA physics.

``rho`` is a filtered/projected topology variable, not carrier density and not
a claim that a gray cell is a physical alloy.  During continuation the same
occupancy is supplied to every physics block, while each block uses its own
documented constitutive relation.  Only the final exact 0/1 mask is promoted
as a physical Au/void design.
"""

from __future__ import annotations


AU_MATERIAL_FRACTION_LAW = "shared_linear_projected_density"
AU_MATERIAL_FRACTION_EXPONENT = 1.0


def au_material_fraction(rho):
    """Return the relaxed Au occupancy for NumPy-compatible arrays."""

    return rho**AU_MATERIAL_FRACTION_EXPONENT


def d_au_material_fraction_drho(rho):
    """Return the derivative of relaxed Au occupancy with respect to rho."""

    return (
        AU_MATERIAL_FRACTION_EXPONENT
        * rho ** (AU_MATERIAL_FRACTION_EXPONENT - 1.0)
    )


def audit() -> dict[str, object]:
    return {
        "scope": "lumerical_maxwell_custom_cuda_pde_topology_relaxation",
        "law": AU_MATERIAL_FRACTION_LAW,
        "exponent": AU_MATERIAL_FRACTION_EXPONENT,
        "optical_occupancy": "same projected rho; Lumerical n-k constitutive map",
        "thermal_occupancy": "au_material_fraction(rho)",
        "electrical_occupancy": "au_material_fraction(rho)",
        "gray_density_is_physical_geometry": False,
        "promotion_requires_exact_binary_density": True,
        "rho_cubed_used": False,
    }
