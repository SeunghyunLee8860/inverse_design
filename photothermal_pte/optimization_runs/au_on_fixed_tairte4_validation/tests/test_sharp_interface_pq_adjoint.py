from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "10_run_au_sharp_interface_pq_adjoint.py"


def load_module():
    spec = importlib.util.spec_from_file_location("au_shape_pq_adjoint", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ConstantFields:
    def __init__(self, electric: np.ndarray, displacement: np.ndarray):
        self.electric = np.asarray(electric, complex)
        self.displacement = np.asarray(displacement, complex)

    def getfield(self, x, *_):
        return np.broadcast_to(self.electric, np.asarray(x).shape + (3,))

    def getDfield(self, x, *_):
        return np.broadcast_to(self.displacement, np.asarray(x).shape + (3,))


def test_two_face_surface_integral_matches_constant_field_analytic_result():
    module = load_module()
    ef = np.asarray([0.0, 1.0, 0.0], complex)
    ea = np.asarray([0.0, 2.0, 0.0], complex)
    forward = ConstantFields(ef, module.EPS0 * module.AU_EPSILON * ef)
    adjoint = ConstantFields(ea, module.EPS0 * module.AU_EPSILON * ea)
    result = module.surface_integrals(
        forward,
        adjoint,
        half_width_m=8.0e-6,
        dy_m=1.0e-6,
        dz_m=5.0e-9,
    )
    area_two_faces = (
        2.0
        * (2.0 * module.AU_HALF_Y_M)
        * (module.AU_Z_MAX_M - module.AU_Z_MIN_M)
    )
    expected_indirect = float(
        np.real(
            2.0
            * module.EPS0
            * (module.AU_EPSILON - module.AIR_EPSILON)
            * 2.0
        )
    ) * area_two_faces
    expected_direct = (
        0.5
        * module.EPS0
        * (2.0 * np.pi * module.FREQUENCY_HZ)
        * float(np.imag(module.AU_EPSILON))
        * area_two_faces
    )
    assert np.isclose(result["indirect_W_per_m"], expected_indirect)
    assert np.isclose(result["direct_W_per_m"], expected_direct)
    assert np.isclose(
        result["total_W_per_m"], expected_indirect + expected_direct
    )
    assert result["all_finite"] is True
