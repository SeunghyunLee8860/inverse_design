from __future__ import annotations

import numpy as np
import pytest

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch import (
    lumerical_au_np_drude,
)


def test_4um_au_drude_carrier_has_exact_passive_endpoints() -> None:
    wavelength_m, target = lumerical_au_np_drude.load_frozen_au()
    carrier = lumerical_au_np_drude.fit_drude_carrier(target, wavelength_m)
    values = carrier.epsilon(np.asarray([0.0, 0.25, 0.5, 0.75, 1.0]))
    assert values[0] == pytest.approx(1.0 + 0.0j)
    assert values[-1] == pytest.approx(target, rel=1.0e-14)
    assert np.all(values.imag >= 0.0)
    assert np.allclose(values, 1.0 + np.linspace(0.0, 1.0, 5) * (target - 1.0))


def test_4um_au_drude_carrier_parameters_match_gold_scale() -> None:
    wavelength_m, target = lumerical_au_np_drude.load_frozen_au()
    carrier = lumerical_au_np_drude.fit_drude_carrier(target, wavelength_m)
    assert carrier.electron_density_cm3 == pytest.approx(5.9283726e22, rel=1.0e-7)
    assert carrier.electron_mobility_cm2_Vs == pytest.approx(24.418819, rel=1.0e-7)
    assert carrier.omega_p_rad_s == pytest.approx(1.3735968e16, rel=1.0e-7)
    assert carrier.gamma_rad_s == pytest.approx(7.2027236e13, rel=1.0e-7)


@pytest.mark.parametrize("bad", [-0.01, 1.01, np.nan])
def test_4um_au_drude_carrier_rejects_invalid_fraction(bad: float) -> None:
    wavelength_m, target = lumerical_au_np_drude.load_frozen_au()
    carrier = lumerical_au_np_drude.fit_drude_carrier(target, wavelength_m)
    with pytest.raises(ValueError):
        carrier.epsilon(bad)
