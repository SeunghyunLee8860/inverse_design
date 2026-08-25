from __future__ import annotations

import json
from pathlib import Path

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_case_contract import (
    TimeSpec,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_user_balanced_source_pair import (
    build_pair,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_user_balanced_z_refinement import (
    case_contract,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.test_fdtdx_user_balanced_source_pair import (
    _report,
    _sha256,
)


def _replace_case(report: Path, numerical_case: dict) -> str:
    payload = json.loads(report.read_text(encoding="utf-8"))
    payload["numerical_case_contract"] = numerical_case
    payload["mesh"] = numerical_case["mesh"]
    report.write_text(json.dumps(payload), encoding="utf-8")
    return _sha256(report)


def test_z2_pair_requires_the_explicit_refinement_contract(tmp_path: Path) -> None:
    ea, _ = _report(tmp_path, "Ea", 1.000e-12)
    eb, _ = _report(tmp_path, "Eb", 1.002e-12)
    expected = case_contract(
        TimeSpec(total_periods=24, window_periods=4, courant_factor=0.5), 2
    )
    ea_hash = _replace_case(ea, expected)
    eb_hash = _replace_case(eb, expected)

    refined = build_pair(
        ea,
        ea_hash,
        eb,
        eb_hash,
        expected_case_contract=expected,
    )
    wrong_baseline = build_pair(ea, ea_hash, eb, eb_hash)

    assert refined["ready"] is True
    assert refined["failed_gates"] == []
    assert wrong_baseline["ready"] is False
    assert "numerical_case_exact_canonical_24_4" in wrong_baseline["failed_gates"]


def test_z_source_wrapper_checks_gpu_before_export() -> None:
    wrapper = (
        Path(__file__)
        .with_name("run_fdtdx_user_balanced_z_source_gpu.sh")
        .read_text(encoding="utf-8")
    )
    assert wrapper.index("refusing busy GPU") < wrapper.index(
        "export CUDA_VISIBLE_DEVICES"
    )
    assert "--query-compute-apps" in wrapper
    assert "fdtdx_user_balanced_z_source_only.py" in wrapper
    assert "Lumerical" not in wrapper
