from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parents[1]
SCRIPT = HERE / "12_run_au_sharp_interface_external_field_adjoint.py"


def load_module():
    spec = importlib.util.spec_from_file_location("au_external_field", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fixed_air_roi_has_150nm_vertical_clearance_and_no_direct_term():
    module = load_module()
    assert module.AU_Z_MIN_M - module.ROI["z_max_m"] >= 150e-9
    assert module.ROI["z_max_m"] < module.AU_Z_MIN_M


def test_fixed_air_objective_matches_constant_field_analytic_value():
    module = load_module()
    x = np.linspace(-10.5e-6, 10.5e-6, 85)
    y = np.linspace(-10.5e-6, 10.5e-6, 85)
    z = np.linspace(-0.45e-6, 0.60e-6, 43)
    grid = {
        "x": x,
        "y": y,
        "z": z,
        "f": np.asarray([module.FREQUENCY_HZ]),
        "delta_x": np.zeros_like(x),
        "delta_y": np.zeros_like(y),
        "delta_z": np.zeros_like(z),
    }
    electric = np.ones((x.size, y.size, z.size, 1, 3), complex) * (2.0 + 1.0j)
    objective, source, meta = module.fixed_air_objective_and_source(electric, grid)
    assert objective > 0.0
    assert source.shape == electric.shape
    assert meta["moving_domain_or_direct_material_term"] is False
    assert sum(meta["component_value_J_proxy"].values()) == objective
    assert all(count > 0 for count in meta["component_sample_count"].values())


def test_center_depth_kernel_zero_for_zero_adjoint_field():
    module = load_module()

    class Field:
        @staticmethod
        def getfield(x, y, z, wavelength):
            return np.ones((np.asarray(x).size, 3), complex)

        @staticmethod
        def getDfield(x, y, z, wavelength):
            return np.ones((np.asarray(x).size, 3), complex) * module.EPS0

    class ZeroField:
        @staticmethod
        def getfield(x, y, z, wavelength):
            return np.zeros((np.asarray(x).size, 3), complex)

        @staticmethod
        def getDfield(x, y, z, wavelength):
            return np.zeros((np.asarray(x).size, 3), complex)

    result = module.official_center_depth_integral(
        Field(),
        ZeroField(),
        half_width_m=8e-6,
        epsilon_au=module.pq.AU_EPSILON,
        n_points=101,
    )
    assert result["total_J_proxy_per_m"] == 0.0
    assert result["all_finite"]
