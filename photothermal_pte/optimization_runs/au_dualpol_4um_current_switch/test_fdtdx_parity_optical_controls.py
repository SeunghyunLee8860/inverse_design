from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_parity_model import (
    model_plan,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_parity_optical_controls import (
    MAX_PREVIOUS_LATE_POWER_MISMATCH,
    SCHEMA_CASE,
    _report_hash,
    aggregate_cases,
    control_gate,
    electric_yee_dual_volumes,
    file_sha256,
    full_au_density,
    load_source_calibration,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_parity_source_calibration import (
    SCHEMA_AGGREGATE as SOURCE_SCHEMA_AGGREGATE,
    SCHEMA_CASE as SOURCE_SCHEMA_CASE,
    TARGET_POWER_W,
)


def _write_report(path: Path, payload: dict[str, object]) -> None:
    payload["report_sha256"] = _report_hash(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _source_artifacts(tmp_path: Path) -> tuple[Path, str]:
    case_paths: dict[str, Path] = {}
    case_reports: dict[str, dict[str, object]] = {}
    powers = {"Ea": 1.0e-12, "Eb": 1.001e-12}
    for polarization in ("Ea", "Eb"):
        case: dict[str, object] = {
            "schema": SOURCE_SCHEMA_CASE,
            "status": "PASS_SOURCE_CASE",
            "polarization": polarization,
            "git_status_porcelain": "",
            "metrics": {"incident_power_late_W": powers[polarization]},
            "model_plan": model_plan(polarization, air_only=True),
        }
        path = tmp_path / f"source_{polarization}.json"
        _write_report(path, case)
        case_paths[polarization] = path
        case_reports[polarization] = case
    aggregate: dict[str, object] = {
        "schema": SOURCE_SCHEMA_AGGREGATE,
        "status": "PASS_SOURCE_CALIBRATION",
        "target_incident_power_W": TARGET_POWER_W,
        "incident_power_W": powers,
        "power_or_Q_scale_to_target": {
            polarization: TARGET_POWER_W / powers[polarization]
            for polarization in ("Ea", "Eb")
        },
        "case_report_sha256": {
            polarization: case_reports[polarization]["report_sha256"]
            for polarization in ("Ea", "Eb")
        },
        "input_files": {
            polarization: {
                "path": str(case_paths[polarization]),
                "file_sha256": file_sha256(case_paths[polarization]),
            }
            for polarization in ("Ea", "Eb")
        },
        "git_commit": "a" * 40,
    }
    aggregate_path = tmp_path / "aggregate.json"
    _write_report(aggregate_path, aggregate)
    return aggregate_path, file_sha256(aggregate_path)


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
    }


def test_full_control_uses_81_nodes_and_exact_four_node_average() -> None:
    nodes, cells = full_au_density()
    assert nodes.shape == (81, 81)
    assert cells.shape == (80, 80)
    assert np.all(nodes == 1.0)
    assert np.all(cells == 1.0)


def test_yee_dual_volumes_are_exact_on_uniform_grid() -> None:
    edges = (
        np.arange(5, dtype=float) * 2.0,
        np.arange(6, dtype=float) * 3.0,
        np.arange(7, dtype=float) * 5.0,
    )
    volumes = electric_yee_dual_volumes(edges, (slice(1, 3), slice(2, 5), slice(3, 6)))
    assert volumes.shape == (3, 2, 3, 3)
    assert np.array_equal(volumes, np.full((3, 2, 3, 3), 30.0))


def test_yee_dual_volumes_use_previous_and_current_widths_off_axis() -> None:
    edges = (
        np.asarray([0.0, 1.0, 3.0]),
        np.asarray([0.0, 2.0, 6.0]),
        np.asarray([0.0, 3.0, 8.0]),
    )
    volumes = electric_yee_dual_volumes(edges, (slice(1, 2), slice(1, 2), slice(1, 2)))
    # widths at cell 1 are (2,4,5), dual metrics are (1.5,3,4).
    assert volumes[:, 0, 0, 0] == pytest.approx([2.0 * 3.0 * 4.0, 1.5 * 4.0 * 4.0, 1.5 * 3.0 * 5.0])


def test_source_calibration_loader_checks_entire_hash_chain(tmp_path: Path) -> None:
    aggregate, digest = _source_artifacts(tmp_path)
    loaded = load_source_calibration(aggregate, expected_file_sha256=digest)
    assert loaded["target_incident_power_W"] == TARGET_POWER_W
    assert set(loaded["power_or_Q_scale_to_target"]) == {"Ea", "Eb"}

    with pytest.raises(RuntimeError, match="file hash mismatch"):
        load_source_calibration(aggregate, expected_file_sha256="0" * 64)

    payload = json.loads(aggregate.read_text(encoding="utf-8"))
    case = Path(payload["input_files"]["Ea"]["path"])
    case.write_text(case.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(RuntimeError, match="file hash mismatch"):
        load_source_calibration(aggregate, expected_file_sha256=digest)


def test_control_gate_fails_closed_one_metric_at_a_time() -> None:
    metrics = _metrics()
    status, gates = control_gate(metrics)
    assert status == "PASS_FULL_AU_OPTICAL_CONTROL"
    assert all(gates.values())

    blocked = dict(metrics)
    blocked["previous_late_Q_power_mismatch_relative"] = MAX_PREVIOUS_LATE_POWER_MISMATCH
    assert control_gate(blocked)[0] == "BLOCKED"
    for key in (
        "previous_late_Q_spatial_NRMSE",
        "target_discrete_Q_mismatch_relative",
        "td_phasor_flux_mismatch_relative",
        "discrete_Q_td_flux_mismatch_relative",
        "discrete_Q_phasor_flux_mismatch_relative",
    ):
        blocked = dict(metrics)
        blocked[key] = math.inf
        assert control_gate(blocked)[0] == "BLOCKED"


def _physical_case(polarization: str) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema": SCHEMA_CASE,
        "status": "PASS_FULL_AU_OPTICAL_CONTROL",
        "polarization": polarization,
        "git_commit": "a" * 40,
        "script_sha256": "b" * 64,
        "density_sha256": {"rho_nodes": "c" * 64, "rho_cells": "d" * 64},
        "source_calibration": {"report_sha256": "e" * 64},
        "metrics_unscaled": {"target_Q_late_W": 1.0},
        "powers_scaled_to_285uW_incident_W": {"target_Q_late_W": 2.0},
        "raw_artifact": {"path": f"/{polarization}.npz", "file_sha256": "f" * 64},
    }
    payload["report_sha256"] = _report_hash(payload)
    return payload


def test_aggregate_requires_both_polarizations_and_shared_invariants() -> None:
    ea = _physical_case("Ea")
    eb = _physical_case("Eb")
    assert aggregate_cases(ea, eb)["status"] == "PASS_FULL_AU_EA_EB_OPTICAL_CONTROLS"

    eb["density_sha256"] = {"rho_nodes": "0" * 64, "rho_cells": "d" * 64}
    unhashed = dict(eb)
    unhashed.pop("report_sha256")
    eb["report_sha256"] = _report_hash(unhashed)
    assert aggregate_cases(ea, eb)["status"] == "BLOCKED"
