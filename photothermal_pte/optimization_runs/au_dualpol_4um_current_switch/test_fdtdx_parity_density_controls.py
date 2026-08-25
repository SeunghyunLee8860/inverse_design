from __future__ import annotations

import inspect
import math

import numpy as np

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch import (
    fdtdx_parity_density_controls,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_parity_density_controls import (
    SCHEMA_CASE,
    _report_hash,
    aggregate_cases,
    density_gate,
    expected_case_status,
    pointwise_au_epsilon_mismatch,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_parity_design_mapping import (
    control_density,
)


def _metrics() -> dict[str, float | bool]:
    return {
        "finite": True,
        "target_Q_late_W": 1.0,
        "discrete_ADE_Q_late_W": 1.0,
        "closed_td_flux_W": 1.0,
        "closed_phasor_flux_W": 1.0,
        "previous_late_Q_power_mismatch_relative": 1.0e-4,
        "previous_late_Q_spatial_NRMSE": 1.0e-3,
        "target_discrete_Q_mismatch_relative": 1.0e-6,
        "td_phasor_flux_mismatch_relative": 1.0e-3,
        "discrete_Q_td_flux_mismatch_relative": 1.0e-3,
        "discrete_Q_phasor_flux_mismatch_relative": 1.0e-3,
        "pointwise_Au_epsilon_mismatch_relative": 1.0e-6,
    }


def _material(au: float, ta: float = 1.0):
    return {
        basis: {
            window: {"au": au, "tairte4": ta}
            for window in ("previous", "late")
        }
        for basis in ("target", "discrete_ADE")
    }


def test_empty_gate_requires_exact_zero_au_and_positive_tairte4() -> None:
    cells = control_density("empty")["cells"]
    status, gates = density_gate(
        "empty", cells=cells, metrics=_metrics(), by_material=_material(0.0)
    )
    assert status == "PASS_EMPTY_OPTICAL_CONTROL"
    assert all(gates.values())
    assert expected_case_status("empty") == status
    blocked = _material(0.0)
    blocked["discrete_ADE"]["late"]["au"] = np.finfo(float).tiny
    assert density_gate("empty", cells=cells, metrics=_metrics(), by_material=blocked)[0] == "BLOCKED"


def test_nonuniform_gate_requires_strict_gray_range_and_positive_au() -> None:
    cells = control_density("nonuniform_gray")["cells"]
    status, gates = density_gate(
        "nonuniform_gray", cells=cells, metrics=_metrics(), by_material=_material(0.5)
    )
    assert status == "PASS_NONUNIFORM_GRAY_OPTICAL_CONTROL"
    assert all(gates.values())
    assert expected_case_status("nonuniform_gray") == status
    assert density_gate(
        "nonuniform_gray",
        cells=np.full((80, 80), 0.5),
        metrics=_metrics(),
        by_material=_material(0.5),
    )[0] == "BLOCKED"


def test_pointwise_epsilon_gate_covers_empty_and_gray_carriers() -> None:
    assert pointwise_au_epsilon_mismatch(control_density("empty")["cells"]) == 0.0
    error = pointwise_au_epsilon_mismatch(
        control_density("nonuniform_gray")["cells"]
    )
    assert math.isfinite(error)
    assert error < 1.0e-5


def _case(polarization: str, density_case: str) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema": SCHEMA_CASE,
        "status": expected_case_status(density_case),
        "polarization": polarization,
        "density_case": density_case,
        "git_commit": "a" * 40,
        "script_sha256": "b" * 64,
        "density_sha256": {"latent": "c" * 64},
        "mapping": {"coefficient_sha256": "d" * 64},
        "source_calibration": {"report_sha256": "e" * 64},
        "metrics_unscaled": {"target_Q_late_W": 1.0},
        "powers_scaled_to_285uW_incident_W": {"target_Q_late_W": 2.0},
        "raw_artifact": {"path": f"/{polarization}.npz", "file_sha256": "f" * 64},
    }
    payload["report_sha256"] = _report_hash(payload)
    return payload


def test_aggregate_requires_matching_density_and_shared_invariants() -> None:
    ea = _case("Ea", "empty")
    eb = _case("Eb", "empty")
    report = aggregate_cases(ea, eb)
    assert report["status"] == "PASS_EMPTY_EA_EB_OPTICAL_CONTROLS"
    eb["density_case"] = "nonuniform_gray"
    unhashed = dict(eb)
    unhashed.pop("report_sha256")
    eb["report_sha256"] = _report_hash(unhashed)
    try:
        aggregate_cases(ea, eb)
    except RuntimeError as error:
        assert "same case" in str(error)
    else:
        raise AssertionError("aggregate accepted mismatched density cases")


def test_runner_has_no_arbitrary_density_or_legacy_input() -> None:
    source = inspect.getsource(fdtdx_parity_density_controls)
    assert "choices=ALLOWED_CASES" in source
    assert "legacy_scripts" not in source
    assert "historical_checkpoint" not in source
    assert "material_fraction" not in source
    assert "rho**3" not in source and "rho ** 3" not in source
