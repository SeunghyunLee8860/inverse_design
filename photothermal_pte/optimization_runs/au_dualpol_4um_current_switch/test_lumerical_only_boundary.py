from __future__ import annotations

import importlib

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.lumerical_only_boundary import (
    audit_production_imports,
    assert_lumerical_only_process,
)


def test_production_import_graph_contains_no_alternative_maxwell_solver() -> None:
    audit = audit_production_imports()
    assert audit["passed"], audit["forbidden_imports"]
    assert audit["source_count"] >= 20
    assert audit["solver_contract"] == {
        "Maxwell_forward": "Lumerical FDTD",
        "Maxwell_adjoint": "Lumerical FDTD",
        "thermal_forward_adjoint": "custom CUDA PDE",
        "electrical_forward_adjoint": "custom CUDA PDE",
        "Lumerical_HEAT": False,
        "Lumerical_CHARGE": False,
        "alternative_Maxwell_solver": False,
    }


def test_live_process_gate_rejects_forbidden_solver_module(monkeypatch) -> None:
    import sys

    monkeypatch.setitem(sys.modules, "fdtdx.synthetic_test", importlib)
    try:
        try:
            assert_lumerical_only_process()
        except RuntimeError as error:
            assert "forbidden Maxwell module" in str(error)
        else:
            raise AssertionError("live-process solver boundary did not fail closed")
    finally:
        monkeypatch.delitem(sys.modules, "fdtdx.synthetic_test", raising=False)
