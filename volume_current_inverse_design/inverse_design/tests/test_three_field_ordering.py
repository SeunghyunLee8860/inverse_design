"""eroded <= nominal <= dilated at every node, every beta."""

import numpy as np
import pytest

from periodic_constrained_mapping import PeriodicConstrainedMapping


@pytest.mark.parametrize("beta", [1.0, 4.0, 16.0, 64.0])
def test_three_field_ordering(beta):
    m = PeriodicConstrainedMapping(241, 241, 13, period_um=6.0, filter_radius_um=0.5)
    lat = np.random.default_rng(int(beta)).random(m.Nux * m.Nuy)
    e, n, d = [np.asarray(a) for a in m.three_fields_unique(lat, beta)]
    assert np.all(e <= n + 1e-12)
    assert np.all(n <= d + 1e-12)


def test_thresholds_bracket_half():
    m = PeriodicConstrainedMapping(241, 241, 13, period_um=6.0, filter_radius_um=0.5)
    assert m.config.eta_dilate < 0.5 < m.config.eta_erode
