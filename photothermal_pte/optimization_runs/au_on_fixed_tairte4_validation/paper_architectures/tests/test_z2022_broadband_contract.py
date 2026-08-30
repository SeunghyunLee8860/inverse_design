from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


def load_module():
    path = HERE / "19_run_v261_z2022_m2_periodic_broadband_rta.py"
    spec = importlib.util.spec_from_file_location("test_z2022_broadband_module", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_z_broadband_contract_is_4_to_12_um() -> None:
    module = load_module()
    assert module.WAVELENGTH_MIN_M == 4e-6
    assert module.WAVELENGTH_MAX_M == 12e-6
    assert module.WAVELENGTH_POINTS == 321
