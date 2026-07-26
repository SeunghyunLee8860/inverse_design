from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


PHOTOTHERMAL = Path(__file__).resolve().parents[2]
THERMAL = PHOTOTHERMAL / "thermal_adjoint"
VOLUME_CURRENT = PHOTOTHERMAL.parent / "volume_current_inverse_design"
for path in (THERMAL, VOLUME_CURRENT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from collocated_coherent_fom import fieldregion_periodic_source_right_inverse
from volume_current_yee_metric import component_periodic_yee_volumes
from yee_absorption_functional import weighted_yee_absorption_and_wirtinger


def _fixture():
    shape = (7, 6, 5, 1, 3)
    rng = np.random.default_rng(2026072605)
    field = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    delta_x = np.linspace(8e-9, 13e-9, shape[0])
    delta_y = np.linspace(9e-9, 12e-9, shape[1])
    delta_z = np.linspace(2e-9, 4e-9, shape[2])
    volumes = component_periodic_yee_volumes(delta_x, delta_y, delta_z)
    weight = rng.normal(size=(*shape[:3], 3))
    return rng, field, volumes, weight


def test_complex_directional_derivative_matches_wirtinger_source():
    rng, field, volumes, weight = _fixture()
    direction = rng.normal(size=field.shape) + 1j * rng.normal(size=field.shape)
    loss = np.asarray([50.848086107787424, 9.289194887622557, 0.0])
    kwargs = dict(
        frequency_Hz=299792458.0 / 4e-6,
        epsilon_imaginary=loss,
        component_volume_m3=volumes,
        density_weight=weight,
    )
    base = weighted_yee_absorption_and_wirtinger(
        electric_field_V_m=field, **kwargs
    )
    analytic = 2.0 * np.real(np.vdot(base.wirtinger_source, direction))
    step = 2e-6
    plus = weighted_yee_absorption_and_wirtinger(
        electric_field_V_m=field + step * direction, **kwargs
    ).weighted_value
    minus = weighted_yee_absorption_and_wirtinger(
        electric_field_V_m=field - step * direction, **kwargs
    ).weighted_value
    finite_difference = (plus - minus) / (2.0 * step)
    assert np.isclose(finite_difference, analytic, rtol=2e-10, atol=1e-30)


def test_component_powers_sum_and_lossless_z_has_zero_source():
    _, field, volumes, _ = _fixture()
    result = weighted_yee_absorption_and_wirtinger(
        electric_field_V_m=field,
        frequency_Hz=299792458.0 / 4e-6,
        epsilon_imaginary=[50.0, 10.0, 0.0],
        component_volume_m3=volumes,
    )
    assert np.isclose(result.power_total_W, np.sum(result.power_component_W))
    assert result.power_component_W[2] == 0.0
    assert np.all(result.wirtinger_source[..., 2] == 0.0)


def test_periodic_fieldregion_fold_preserves_pairing():
    rng, field, volumes, weight = _fixture()
    q = weighted_yee_absorption_and_wirtinger(
        electric_field_V_m=field,
        frequency_Hz=299792458.0 / 4e-6,
        epsilon_imaginary=[50.0, 10.0, 0.0],
        component_volume_m3=volumes,
        density_weight=weight,
    ).wirtinger_source
    folded = fieldregion_periodic_source_right_inverse(q)
    assert np.isclose(np.vdot(folded, field), np.vdot(q, field), rtol=1e-14)
