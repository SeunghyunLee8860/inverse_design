from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch import (
    fdtdx_increment_state_exact_binary_mesh_case as mesh_case,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_case_contract import (
    ANCHOR_CASE,
    case_contract,
)


def _model() -> dict:
    return {
        "fresh_mesh_audit": case_contract(ANCHOR_CASE)["resolved_mesh"],
        "pml_face_parameters": case_contract(ANCHOR_CASE)[
            "resolved_pml_face_parameters"
        ],
        "placement": {"same": True},
        "source_contract": {"polarization": "Ea"},
        "config": SimpleNamespace(
            time_step_duration=1.0,
            time_steps_total=10,
        ),
        "jax": SimpleNamespace(__version__="0.test"),
        "fdtdx": SimpleNamespace(__file__="/tmp/fdtdx/__init__.py"),
    }


def test_source_pair_contract_checks_are_fail_closed(monkeypatch):
    model = _model()
    monkeypatch.setattr(
        mesh_case,
        "realized_time_contract",
        lambda _case, _model: {"same": True},
    )
    source_audit = {"commit": "abc", "ready": True}
    pair = {
        "source_case_contracts": {
            "numerical_case_contract": case_contract(ANCHOR_CASE),
            "mesh": model["fresh_mesh_audit"],
            "time_contract": {"same": True},
            "pml_face_parameters": model["pml_face_parameters"],
            "placement": model["placement"],
            "source_contracts": {"Ea": model["source_contract"]},
            "fdtdx_source": source_audit,
            "runtime_lock": mesh_case._runtime_lock(model),
        }
    }

    checks = mesh_case.source_pair_contract_checks(
        pair, source_audit, model, ANCHOR_CASE, "Ea"
    )
    assert checks
    assert all(checks.values())

    pair["source_case_contracts"]["time_contract"] = {"same": False}
    changed = mesh_case.source_pair_contract_checks(
        pair, source_audit, model, ANCHOR_CASE, "Ea"
    )
    assert changed["time_exact"] is False
    assert all(value for name, value in changed.items() if name != "time_exact")


def test_gpu_wrapper_rejects_busy_device_before_cuda_export():
    wrapper = (
        Path(mesh_case.__file__)
        .with_name("run_fdtdx_increment_state_material_gpu.sh")
        .read_text(encoding="utf-8")
    )
    assert wrapper.index("refusing busy GPU") < wrapper.index(
        "export CUDA_VISIBLE_DEVICES"
    )
    assert "--query-compute-apps" in wrapper
    assert '-v gpu_id="$gpu_index"' in wrapper
    assert "export JAX_PLATFORMS=cuda" in wrapper
    assert "shift 5" in wrapper
    assert '"$@"' in wrapper
    assert "Lumerical" not in wrapper


def test_runner_forbids_optimizer_and_uses_increment_state():
    source = Path(mesh_case.__file__).read_text(encoding="utf-8")
    assert 'dispersive_state_representation="increment"' in source
    assert '"optimizer_start_allowed": False' in source
    assert "_power_evaluation(" in source
    assert "validate_source_pair(" in source
