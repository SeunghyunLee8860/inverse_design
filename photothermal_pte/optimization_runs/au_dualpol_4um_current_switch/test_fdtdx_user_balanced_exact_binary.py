from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_case_contract import (
    TimeSpec,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_user_balanced_exact_binary import (
    source_pair_contract_checks,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_user_balanced_source_only import (
    balanced_case_contract,
    realized_time_contract,
)


def _contracts() -> tuple[dict, dict, dict, TimeSpec]:
    time = TimeSpec(total_periods=24, window_periods=4, courant_factor=0.5)
    source_audit = {"path": "/source", "ready": True}
    model = {
        "fresh_mesh_audit": {"mesh": "balanced"},
        "config": SimpleNamespace(time_step_duration=1.0, time_steps_total=100),
        "source_contract": {"polarization": "Ea", "num_startup_periods": 4},
        "pml_face_parameters": {"pml": "balanced"},
        "placement": {"placement": "balanced"},
        "jax": SimpleNamespace(__version__="test-jax"),
        "fdtdx": SimpleNamespace(__file__=Path("/test/fdtdx/__init__.py")),
    }
    source_pair = {
        "source_case_contracts": {
            "numerical_case_contract": balanced_case_contract(time),
            "mesh": model["fresh_mesh_audit"],
            "time_contract": realized_time_contract(time, model),
            "pml_face_parameters": model["pml_face_parameters"],
            "placement": model["placement"],
            "source_contracts": {
                "Ea": model["source_contract"],
                "Eb": {"polarization": "Eb"},
            },
            "fdtdx_source": source_audit,
            "runtime_lock": {
                "python": __import__("platform").python_version(),
                "jax": "test-jax",
                "fdtdx_import": str(Path("/test/fdtdx/__init__.py").resolve()),
            },
        }
    }
    return source_pair, source_audit, model, time


def test_material_requires_every_source_pair_contract_to_match() -> None:
    source_pair, source_audit, model, time = _contracts()

    checks = source_pair_contract_checks(source_pair, source_audit, model, time, "Ea")

    assert checks
    assert all(checks.values())


def test_material_contract_check_fails_closed_on_mesh_change() -> None:
    source_pair, source_audit, model, time = _contracts()
    source_pair["source_case_contracts"]["mesh"] = {"mesh": "other"}

    checks = source_pair_contract_checks(source_pair, source_audit, model, time, "Ea")

    assert checks["mesh_exact"] is False


def test_balanced_material_wrapper_refuses_busy_gpu_before_cuda_export() -> None:
    wrapper = (
        Path(__file__)
        .with_name("run_fdtdx_user_balanced_material_gpu.sh")
        .read_text(encoding="utf-8")
    )
    assert wrapper.index("refusing busy GPU") < wrapper.index(
        "export CUDA_VISIBLE_DEVICES"
    )
    assert "--query-compute-apps" in wrapper
    assert "fdtdx_user_balanced_exact_binary.py" in wrapper
    assert "Lumerical" not in wrapper
