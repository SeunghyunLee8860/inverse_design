from __future__ import annotations

import hashlib
import json

import pytest

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch import (
    fdtdx_increment_state_full_z32_extension_certificate as certificate,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch import (
    fdtdx_increment_state_full_z_extension_certificate as prior_module,
)


def _prior_payload() -> dict:
    return {
        "version": certificate.PRIOR_VERSION,
        "status": certificate.PRIOR_STATUS,
        "ready": False,
        "global_checks": {"artifacts": True},
        "failed_global_checks": [],
        "source_pair_audits": {
            level: {"ready": True} for level in ("z8", "z16")
        },
        "case_audits": {
            level: {pol: {"ready": True} for pol in ("Ea", "Eb")}
            for level in ("z8", "z16")
        },
        "z8_to_z16_comparison": {"pass": False, "error": None},
        "promotion": {
            "z8_to_z16_pass": False,
            "full_domain_z_converged": False,
            "selected_mesh_level": None,
            "requires_two_successive_passing_tail_pairs": True,
        },
        "optimizer_start_allowed": False,
        "certificate_provenance": {
            "repository_dirty_porcelain": "",
            "repository_commit": "a" * 40,
        },
    }


def test_prior_blocked_z16_certificate_is_revalidated_by_bytes(tmp_path):
    path = (tmp_path / "prior.json").resolve()
    path.write_text(json.dumps(_prior_payload()), encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    _, audit = certificate.audit_prior_certificate(path, digest)
    assert audit["ready"] is True

    changed = _prior_payload()
    changed["z8_to_z16_comparison"]["pass"] = True
    path.write_text(json.dumps(changed), encoding="utf-8")
    changed_digest = hashlib.sha256(path.read_bytes()).hexdigest()
    _, changed_audit = certificate.audit_prior_certificate(path, changed_digest)
    assert changed_audit["ready"] is False
    assert "z8_to_z16_was_evaluated_and_failed" in changed_audit["failed_checks"]


def test_z16_z32_cases_change_only_z_factor_and_keep_24_4_timing():
    z16 = certificate.expected_case("z16")
    z32 = certificate.expected_case("z32")
    z16_mesh = dict(z16.mesh.__dict__)
    z32_mesh = dict(z32.mesh.__dict__)
    assert z16_mesh.pop("z_factor") == 16
    assert z32_mesh.pop("z_factor") == 32
    assert z16_mesh == z32_mesh
    for case in (z16, z32):
        assert case.time.total_periods == 24
        assert case.time.window_periods == 4
        assert case.time.courant_factor == 0.5


def test_shared_case_audit_distinguishes_z16_and_z32_labels():
    common = {
        "reference": prior_module.DEFAULT_REFERENCE,
        "polarization": "Ea",
        "mesh_axis": "anchor",
        "mesh_level": 0,
    }
    assert prior_module._case_labels(
        {**common, "full_z_extension": "z16"}, "z16", "Ea"
    )
    assert prior_module._case_labels(
        {**common, "full_z_extension": "z32"}, "z32", "Ea"
    )
    assert not prior_module._case_labels(
        {**common, "full_z_extension": "z16"}, "z32", "Ea"
    )
    with pytest.raises(ValueError, match="z8, z16, or z32"):
        prior_module._case_labels(common, "z64", "Ea")


@pytest.mark.parametrize("comparison_pass", [False, True])
def test_promotion_never_selects_mesh_optimizer_or_z64(comparison_pass):
    result = certificate.promotion(comparison_pass)
    assert result["z8_to_z16_pass"] is False
    assert result["z16_to_z32_pass"] is comparison_pass
    assert result["full_domain_z_converged"] is False
    assert result["selected_mesh_level"] is None
    assert result["z_only_ladder_terminated"] is True
    assert result["z64_run_allowed"] is False
    assert result["optimizer_start_allowed"] is False
