from __future__ import annotations

import numpy as np

from photothermal_pte.validation.paper_ir_sanity.audit_paper_ir_beam_contract import (
    BOUNDARY_MAX_GATE,
    BOUNDARY_MEAN_GATE,
    CAPTURE_GATE,
    SELECTED_W0_M,
    SOURCE_SPAN_M,
    SOURCE_Z_M,
    FOCUS_Z_M,
    WAVELENGTH_M,
    required_half_span_ratio,
    square_metrics,
)
from photothermal_pte.validation.paper_ir_sanity.validate_paper_ir_source_only_gpu import (
    fit_gaussian,
    strict_gpu_run,
)


def test_selected_aperture_passes_all_analytic_gates() -> None:
    z_rayleigh = np.pi * SELECTED_W0_M**2 / WAVELENGTH_M
    distance = SOURCE_Z_M - FOCUS_Z_M
    source_radius = SELECTED_W0_M * np.sqrt(
        1.0 + (distance / z_rayleigh) ** 2
    )
    metrics = square_metrics(0.5 * SOURCE_SPAN_M, source_radius)
    assert metrics["square_captured_fraction"] >= CAPTURE_GATE
    assert metrics["boundary_max_intensity_over_peak"] <= BOUNDARY_MAX_GATE
    assert metrics["boundary_mean_intensity_over_peak"] <= BOUNDARY_MEAN_GATE


def test_mean_boundary_gate_governs_required_span() -> None:
    ratios = required_half_span_ratio()
    assert ratios["governing"] == ratios["boundary_mean_gate"]


def test_backward_converging_source_distance_is_negative() -> None:
    assert -(SOURCE_Z_M - FOCUS_Z_M) < 0.0


def test_plane_fit_recovers_gaussian_1e2_radius() -> None:
    x = np.linspace(-25e-6, 25e-6, 251)
    y = np.linspace(-25e-6, 25e-6, 251)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    intensity = np.exp(-2.0 * (xx**2 + yy**2) / SELECTED_W0_M**2)
    fit = fit_gaussian(x, y, intensity)
    assert np.isclose(
        fit["fitted_waist_effective_m"],
        SELECTED_W0_M,
        rtol=1e-10,
    )


def test_gpu_runner_never_requests_cpu_solver() -> None:
    calls: list[tuple[str, str, str]] = []

    class FailedSession:
        def run(self, solver: str, engine: str, resource: str) -> None:
            calls.append((solver, engine, resource))
            raise RuntimeError("expected")

    try:
        strict_gpu_run(FailedSession(), "unit-test")
    except RuntimeError:
        pass
    assert calls
    assert all(engine == "GPU" for _, engine, _ in calls)
