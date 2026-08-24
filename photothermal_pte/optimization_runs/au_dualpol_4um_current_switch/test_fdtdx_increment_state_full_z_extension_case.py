from __future__ import annotations

from pathlib import Path

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch import (
    fdtdx_increment_state_exact_binary_mesh_case as material_runner,
    fdtdx_increment_state_source_only as source_runner,
)

import pytest

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_case_contract import (
    file_sha256,
    load_case_contract,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_increment_state_full_z_extension_case import (
    EXTENSION_Z_FACTORS,
    expected_extension_case,
    resolve_increment_state_case,
    write_extension_case,
)


def test_extension_changes_only_z_factor_and_keeps_increment_timing():
    cases = {level: expected_extension_case(level) for level in EXTENSION_Z_FACTORS}
    baseline = dict(cases["z16"].mesh.__dict__)
    baseline.pop("z_factor")
    for level, factor in EXTENSION_Z_FACTORS.items():
        mesh = dict(cases[level].mesh.__dict__)
        assert mesh.pop("z_factor") == factor
        assert mesh == baseline
        assert cases[level].time.total_periods == 24
        assert cases[level].time.window_periods == 4
        assert cases[level].time.courant_factor == 0.5


def test_extension_resolver_forbids_ambiguous_or_noncanonical_requests():
    assert resolve_increment_state_case("anchor", 0, 24, 4, "z16") == (
        expected_extension_case("z16")
    )
    with pytest.raises(ValueError, match="cannot be combined"):
        resolve_increment_state_case("full_domain_z", 2, 24, 4, "z16")
    with pytest.raises(ValueError, match="canonical 24/4"):
        resolve_increment_state_case("anchor", 0, 32, 4, "z16")


def test_extension_writer_is_canonical_and_no_overwrite(tmp_path: Path):
    output = (tmp_path / "z16.json").resolve()
    result = write_extension_case("z16", output)
    spec, payload, audit = load_case_contract(output, result["file_sha256"])
    assert spec == expected_extension_case("z16")
    assert audit["ready"] is True
    assert payload["resolved_mesh"]["grid_shape_xyz"] == [196, 196, 640]
    assert result["file_sha256"] == file_sha256(output)
    with pytest.raises(RuntimeError):
        write_extension_case("z16", output)


def test_extension_capable_runners_are_explicit_v2():
    assert source_runner.VERSION.endswith("-v2")
    assert material_runner.VERSION.endswith("-v2")
    for module in (source_runner, material_runner):
        source = Path(module.__file__).read_text(encoding="utf-8")
        assert "--full-z-extension" in source
        assert "resolve_increment_state_case(" in source
