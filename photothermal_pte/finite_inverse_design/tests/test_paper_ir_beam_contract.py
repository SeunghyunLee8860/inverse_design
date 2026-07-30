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
    APPROVED_API,
    APPROVED_ROOT,
    CALIBRATED_SOURCE_OBJECT_W0_M,
    CALIBRATION_BASELINE_REALIZED_W0_M,
    ELLIPTICITY_GATE,
    FIT_RESIDUAL_GATE,
    fit_gaussian,
    strict_gpu_run,
    v261_session_provenance,
    write_json,
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


def test_backward_source_property_uses_negative_distance() -> None:
    assert -(SOURCE_Z_M - FOCUS_Z_M) < 0.0


def test_source_object_calibration_targets_physical_12um_waist() -> None:
    predicted = (
        CALIBRATION_BASELINE_REALIZED_W0_M
        * CALIBRATED_SOURCE_OBJECT_W0_M
        / SELECTED_W0_M
    )
    assert CALIBRATED_SOURCE_OBJECT_W0_M < SELECTED_W0_M
    assert np.isclose(predicted, SELECTED_W0_M, rtol=0.0, atol=1e-18)


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
    assert fit["Gaussian_fit_NRMSE"] < FIT_RESIDUAL_GATE
    assert fit["fitted_xy_ellipticity"] < ELLIPTICITY_GATE


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


def test_v261_session_uses_internal_8_35_version_and_exact_paths() -> None:
    provenance = v261_session_provenance(
        solver_version="8.35.4522",
        loaded_lumapi_path=APPROVED_API / "lumapi.py",
        installation_version_key="v261",
        installation_root=APPROVED_ROOT,
    )
    assert provenance["all"]
    assert all(provenance["checks"].values())


def test_release_label_is_not_expected_from_fdtd_version() -> None:
    provenance = v261_session_provenance(
        solver_version="2026 R1",
        loaded_lumapi_path=APPROVED_API / "lumapi.py",
        installation_version_key="v261",
        installation_root=APPROVED_ROOT,
    )
    assert not provenance["all"]
    assert not provenance["checks"]["solver_version_8_35_series"]


def test_source_result_json_accepts_numpy_scalars(tmp_path) -> None:
    output = tmp_path / "result.json"
    write_json(
        output,
        {
            "gate": np.bool_(True),
            "metric": np.float64(0.25),
            "count": np.int64(3),
        },
    )
    assert output.read_text(encoding="utf-8") == (
        '{\n  "gate": true,\n  "metric": 0.25,\n  "count": 3\n}\n'
    )
