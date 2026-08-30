from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parents[1]


def load_module():
    path = HERE / "17_run_v261_t2024_periodic_broadband_rta.py"
    spec = importlib.util.spec_from_file_location("test_t2024_broadband_module", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_signed_flux_conversion() -> None:
    module = load_module()
    reflection, transmission, absorption = module.rta_from_signed_transmission(
        np.array([-0.2, -0.8]), np.array([-0.1, -0.05])
    )
    np.testing.assert_allclose(reflection, [0.8, 0.2])
    np.testing.assert_allclose(transmission, [0.1, 0.05])
    np.testing.assert_allclose(absorption, [0.1, 0.75])
    np.testing.assert_allclose(reflection + transmission + absorption, 1.0)


def test_broadband_contract_covers_published_t_and_z_mir_range() -> None:
    module = load_module()
    assert module.WAVELENGTH_MIN_M == 4.0e-6
    assert module.WAVELENGTH_MAX_M == 12.0e-6
    assert module.WAVELENGTH_POINTS >= 321
