from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


REPOSITORY = Path(__file__).resolve().parents[3]
WRAPPER = (
    REPOSITORY
    / "photothermal_pte"
    / "optimization_runs"
    / "run_002_gaussian10_w8p5_current_max"
    / "audit_source_only_gpu.py"
)


def load_wrapper():
    spec = importlib.util.spec_from_file_location("run002_source_wrapper", WRAPPER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_run002_source_wrapper_sets_frequency_centered_contract():
    wrapper = load_wrapper()
    wrapper.configure_source_audit()
    audit = wrapper.source_audit
    assert audit.contract.WAVELENGTH_M == 10.0e-6
    assert audit.contract.SELECTED_W0_M == 8.5e-6
    assert audit.contract.SOURCE_SPAN_M == 40.0e-6
    assert audit.contract.LATERAL_DOMAIN_M == 48.0e-6
    f_center = 0.5 * audit.C0 * (
        1.0 / audit.SOURCE_START_M + 1.0 / audit.SOURCE_STOP_M
    )
    assert np.isclose(f_center, audit.C0 / 10.0e-6, rtol=0.0, atol=1.0)
    assert audit.PML_LAYERS == 24
    assert audit.MESH_ACCURACY == 3
    assert set(audit.MONITORS) == {
        "source_plane",
        "flake_target_plane",
        "downstream_plane",
    }
    assert audit.setup is not wrapper.BASE_SETUP
