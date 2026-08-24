"""Published density relaxation for a 4-um Au/background design layer.

The topology variable is not an electron or hole density and it is not a
physical Au concentration.  It is the filtered/projected occupancy used to
make the topology problem differentiable.  Binarization belongs to the
filter/projection continuation, while this module supplies the constitutive
map used by the single-frequency Lumerical Maxwell gate.

For the optical relaxation we interpolate refractive index ``n`` and
extinction coefficient ``k`` and then form ``epsilon=(n+1j*k)**2``.  This is
the nonlinear metal/dielectric interpolation proposed by Christiansen et al.
(CMAME 343, 23-39, 2019; DOI 10.1016/j.cma.2018.08.034) and used in the
plasmonic FDTD topology framework of Zeng et al. (ACS Photonics 8, 2021;
DOI 10.1021/acsphotonics.1c00260).  It is not a SIMP rho**3 law.

The relaxation is only a candidate until its Lumerical B200 forward,
bandwidth, resonance-sweep, and same-step AD-FD gates pass.  A promoted final
mask must be re-evaluated with ordinary sampled-data dispersive Au.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ORDAL_AU_DATA = (
    HERE.parent
    / "au_on_fixed_tairte4_validation"
    / "data"
    / "au_ordal_1987_nk.csv"
)


@dataclass(frozen=True)
class AuDensityRelaxationContract:
    law: str = "christiansen_nk_then_square_v1"
    role: str = "filtered_projected_topology_occupancy_not_carrier_density"
    wavelength_m: float = 4.0e-6
    background_n: float = 1.0
    background_k: float = 0.0
    endpoint_source: str = "Ordal et al. 1987, DOI 10.1364/AO.26.000744"
    binarization_mechanism: str = "density_filter_plus_tanh_projection_beta_continuation"
    optical_rho_power: float | None = None
    exact_binary_required_during_relaxed_iterations: bool = False
    exact_binary_required_for_final_promotion: bool = True
    final_material: str = "ordinary_sampled_data_dispersive_Au"


CONTRACT = AuDensityRelaxationContract()


def _validate_projected_density(rho: np.ndarray) -> np.ndarray:
    value = np.asarray(rho, dtype=np.float64)
    if value.size == 0:
        raise ValueError("projected density must be non-empty")
    if not np.all(np.isfinite(value)):
        raise ValueError("projected density contains a non-finite value")
    tolerance = 1.0e-12
    if np.any(value < -tolerance) or np.any(value > 1.0 + tolerance):
        raise ValueError("projected density must remain in [0,1]")
    return np.clip(value, 0.0, 1.0)


def ordal_au_index(wavelength_m: float = CONTRACT.wavelength_m) -> complex:
    """Return the passive Ordal Au index without endpoint extrapolation."""

    table = np.genfromtxt(ORDAL_AU_DATA, delimiter=",", names=True)
    wavelength_um = np.asarray(table["wavelength_um"], dtype=np.float64)
    query_um = float(wavelength_m) * 1.0e6
    if not wavelength_um[0] <= query_um <= wavelength_um[-1]:
        raise ValueError(
            f"Au wavelength {query_um:g} um outside Ordal table "
            f"[{wavelength_um[0]:g},{wavelength_um[-1]:g}] um"
        )
    n = float(np.interp(query_um, wavelength_um, table["n"]))
    k = float(np.interp(query_um, wavelength_um, table["k"]))
    if n <= 0.0 or k < 0.0:
        raise RuntimeError("Ordal Au endpoint is not on the passive n+ik branch")
    return complex(n, k)


def nk_relaxation(
    projected_density: np.ndarray,
    *,
    au_index: complex | None = None,
    background_index: complex = complex(CONTRACT.background_n, CONTRACT.background_k),
) -> np.ndarray:
    """Map projected occupancy to the Christiansen ``n-k`` relaxation."""

    rho = _validate_projected_density(projected_density)
    au = ordal_au_index() if au_index is None else complex(au_index)
    background = complex(background_index)
    if au.real <= 0.0 or au.imag < 0.0:
        raise ValueError("Au index must use a passive n+ik convention")
    if background.real <= 0.0 or background.imag < 0.0:
        raise ValueError("background index must use a passive n+ik convention")
    return background + rho * (au - background)


def epsilon_relaxation(
    projected_density: np.ndarray,
    *,
    au_index: complex | None = None,
    background_index: complex = complex(CONTRACT.background_n, CONTRACT.background_k),
) -> np.ndarray:
    """Return ``epsilon=(n+ik)^2`` for the relaxed design state."""

    index = nk_relaxation(
        projected_density,
        au_index=au_index,
        background_index=background_index,
    )
    return index**2


def d_epsilon_d_projected_density(
    projected_density: np.ndarray,
    *,
    au_index: complex | None = None,
    background_index: complex = complex(CONTRACT.background_n, CONTRACT.background_k),
) -> np.ndarray:
    """Analytic complex derivative of the nonlinear optical relaxation."""

    index = nk_relaxation(
        projected_density,
        au_index=au_index,
        background_index=background_index,
    )
    au = ordal_au_index() if au_index is None else complex(au_index)
    return 2.0 * index * (au - complex(background_index))


def lumerical_import_index(
    projected_density_xy: np.ndarray,
    *,
    z_samples: int = 2,
) -> np.ndarray:
    """Build the complex ``importnk2`` array for a uniform-thickness layer.

    The returned array has shape ``(nx, ny, nz)``.  This function performs no
    Lumerical solve and does not claim that the imported material is already a
    certified Au topology carrier.
    """

    rho = _validate_projected_density(projected_density_xy)
    if rho.ndim != 2:
        raise ValueError("Au projected density must be a 2-D array")
    if int(z_samples) != z_samples or z_samples < 2:
        raise ValueError("z_samples must be an integer >= 2")
    index_xy = nk_relaxation(rho)
    return np.repeat(index_xy[:, :, None], int(z_samples), axis=2)


def audit(sample_count: int = 1001) -> dict[str, object]:
    """Return solver-free endpoint, passivity, and analytic-law evidence."""

    if sample_count < 3:
        raise ValueError("sample_count must be >= 3")
    rho = np.linspace(0.0, 1.0, int(sample_count))
    index = nk_relaxation(rho)
    epsilon = index**2
    au = ordal_au_index()
    source_hash = hashlib.sha256(ORDAL_AU_DATA.read_bytes()).hexdigest()
    return {
        **asdict(CONTRACT),
        "sample_count": int(sample_count),
        "au_index": [au.real, au.imag],
        "au_epsilon": [(au**2).real, (au**2).imag],
        "background_epsilon": [1.0, 0.0],
        "minimum_index_imaginary": float(np.min(index.imag)),
        "minimum_epsilon_imaginary": float(np.min(epsilon.imag)),
        "passive_on_uniform_density_sweep": bool(np.all(epsilon.imag >= 0.0)),
        "exact_background_endpoint": bool(epsilon[0] == 1.0 + 0.0j),
        "exact_au_endpoint": bool(epsilon[-1] == au**2),
        "rho_cubed_used": False,
        "ordal_data": str(ORDAL_AU_DATA),
        "ordal_data_sha256": source_hash,
        "remaining_gates": [
            "Lumerical B200 imported-index versus ordinary dispersive-Au endpoint parity",
            "source-band material-fit and passivity audit",
            "uniform-density field/Q resonance sweep",
            "nonuniform density-to-Yee Jacobian transpose and centered-FD tests",
            "complete optical/thermal/electrical latent-variable AD-FD",
            "independent exact-binary dispersive-Au final reevaluation",
        ],
    }
