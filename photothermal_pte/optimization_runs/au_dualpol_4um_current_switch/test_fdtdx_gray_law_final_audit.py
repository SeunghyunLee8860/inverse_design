from __future__ import annotations

from pathlib import Path

import pytest

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch import (
    fdtdx_gray_law_final_audit as gray_audit,
)


def test_ast_scan_detects_executable_literal_power_three(tmp_path: Path) -> None:
    source = tmp_path / "sample.py"
    source.write_text(
        '"rho ** 3 in a docstring is not executable"\n'
        "label = 'rho ** 3'\n"
        "value = rho ** 3\n",
        encoding="utf-8",
    )
    assert gray_audit.literal_power_three_locations(source) == [
        {"line": 3, "expression": "rho ** 3"}
    ]


def test_active_fdtdx_runtime_has_no_literal_power_three() -> None:
    root = Path(gray_audit.__file__).resolve().parent
    findings = {
        name: gray_audit.literal_power_three_locations(root / name)
        for name in gray_audit.ACTIVE_FDTDX_RUNTIME_FILES
    }
    assert findings == {name: [] for name in gray_audit.ACTIVE_FDTDX_RUNTIME_FILES}


def test_optimizer_entrypoints_gate_before_mutation_and_compile() -> None:
    root = Path(gray_audit.__file__).resolve().parent
    records = {
        name: gray_audit.optimizer_gate_order(root / name)
        for name in gray_audit.OPTIMIZER_ENTRYPOINTS
    }
    assert all(record["ready"] for record in records.values())


def test_exact_binary_path_rejects_float_and_gray() -> None:
    report = gray_audit.exact_binary_audit()
    assert report["ready"] is True
    assert report["material"]["gray_density_allowed"] is False
    assert report["material"]["rho_power"] is None
    assert all(report["checks"].values())


def test_shared_state_discrete_maps_pass_solver_free_derivative_checks() -> None:
    report = gray_audit.shared_state_numeric_audit()
    assert report["ready"] is True
    assert report["cell_map_transpose_relative_error"] < 1.0e-12
    assert max(
        row["relative_error"] for row in report["cell_map_finite_difference"]
    ) < 1.0e-7
    assert max(
        row["complex_relative_error"]
        for row in report["optical_law_finite_difference"]
    ) < 1.0e-9
    assert report["state"]["nodal_shape_xy"] == [81, 81]
    assert report["state"]["pde_cell_shape_xy"] == [80, 80]
    assert report["state"]["optical_rho_power"] is None


def test_certificate_kind_rejects_internal_misuse(tmp_path: Path) -> None:
    certificate = tmp_path / "certificate.json"
    certificate.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected"):
        gray_audit.certificate_audit(
            certificate,
            gray_audit.sha256(certificate),
            kind="unexpected",
        )
