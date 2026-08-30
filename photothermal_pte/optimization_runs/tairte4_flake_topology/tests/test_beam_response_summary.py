import numpy as np

from photothermal_pte.optimization_runs.summarize_exact_binary_beam_response import (
    line_metrics,
)


POSITIONS_UM = np.asarray((-10.0, -5.0, 0.0, 5.0, 10.0))


def test_monotonic_line_is_signed_position_candidate():
    metrics = line_metrics(
        POSITIONS_UM,
        np.asarray((10.0, 20.0, 30.0, 40.0, 50.0)),
        center_nA=30.0,
    )
    assert metrics["assessment"] == "promising_1D"
    assert metrics["unsigned_centering_assessment"] == "limited_unsigned_displacement"


def test_center_peak_is_unsigned_but_not_signed_position_candidate():
    metrics = line_metrics(
        POSITIONS_UM,
        np.asarray((10.0, 40.0, 100.0, 45.0, 12.0)),
        center_nA=100.0,
    )
    assert metrics["assessment"] == "limited_or_nonmonotonic_1D"
    assert metrics["negative_half_monotonic"]
    assert metrics["positive_half_monotonic"]
    assert metrics["center_is_extremum"]
    assert metrics["unsigned_centering_assessment"] == "promising_unsigned_displacement"
