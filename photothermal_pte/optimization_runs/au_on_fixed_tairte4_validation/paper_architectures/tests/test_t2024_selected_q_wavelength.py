from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


HERE = Path(__file__).resolve().parents[1]


def load_module():
    path = HERE / "07_run_v261_t2024_tairte4_optical_smoke.py"
    spec = importlib.util.spec_from_file_location("test_t2024_selected_q_module", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_historical_4750nm_source_window_is_immutable() -> None:
    module = load_module()
    module.configure_wavelength(4.75)
    assert module.source_window_m() == (4.5e-6, 5.0e-6)


def test_selected_q_wavelength_changes_frequency_and_window() -> None:
    module = load_module()
    module.configure_wavelength(8.0)
    assert module.WAVELENGTH_M == pytest.approx(8.0e-6)
    assert module.FREQUENCY_HZ == pytest.approx(module.C0 / 8.0e-6)
    start, stop = module.source_window_m()
    assert start == pytest.approx(7.6e-6)
    assert stop == pytest.approx(8.4e-6)


@pytest.mark.parametrize("wavelength_um", [3.999, 12.001])
def test_selected_q_wavelength_must_come_from_screen(wavelength_um: float) -> None:
    module = load_module()
    with pytest.raises(ValueError):
        module.configure_wavelength(wavelength_um)
