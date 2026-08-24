from __future__ import annotations

import hashlib
import json

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch import (
    fdtdx_increment_state_full_z_extension_certificate as certificate,
)


def _prior_payload() -> dict:
    return {
        "version": certificate.PRIOR_VERSION,
        "status": certificate.PRIOR_STATUS,
        "ready": False,
        "global_checks": {"artifacts": True},
        "failed_global_checks": [],
        "source_pair_audits": {level: {"ready": True} for level in ("z2", "z4", "z8")},
        "case_audits": {
            level: {pol: {"ready": True} for pol in ("Ea", "Eb")}
            for level in ("z2", "z4", "z8")
        },
        "pair_comparisons": {
            "z2_to_z4": {"pass": False},
            "z4_to_z8": {"pass": False},
        },
        "promotion": {"full_domain_z_converged": False},
        "optimizer_start_allowed": False,
        "certificate_provenance": {
            "repository_dirty_porcelain": "",
            "repository_commit": "a" * 40,
        },
    }


def test_prior_blocked_certificate_is_revalidated_by_bytes(tmp_path):
    path = (tmp_path / "prior.json").resolve()
    path.write_text(json.dumps(_prior_payload()), encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    _, audit = certificate.audit_prior_certificate(path, digest)
    assert audit["ready"] is True

    changed = _prior_payload()
    changed["pair_comparisons"]["z4_to_z8"]["pass"] = True
    path.write_text(json.dumps(changed), encoding="utf-8")
    changed_digest = hashlib.sha256(path.read_bytes()).hexdigest()
    _, changed_audit = certificate.audit_prior_certificate(path, changed_digest)
    assert changed_audit["ready"] is False
    assert "both_prior_pairs_retained_and_failed" in changed_audit["failed_checks"]


def test_extension_cases_change_only_z_factor_and_use_24_4_courant_half():
    z8 = certificate.expected_case("z8")
    z16 = certificate.expected_case("z16")
    z8_mesh = dict(z8.mesh.__dict__)
    z16_mesh = dict(z16.mesh.__dict__)
    assert z8_mesh.pop("z_factor") == 8
    assert z16_mesh.pop("z_factor") == 16
    assert z8_mesh == z16_mesh
    for case in (z8, z16):
        assert case.time.total_periods == 24
        assert case.time.window_periods == 4
        assert case.time.courant_factor == 0.5


def test_case_labels_distinguish_prior_v1_and_extension_v2():
    common = {
        "reference": certificate.DEFAULT_REFERENCE,
        "polarization": "Ea",
    }
    assert certificate._case_labels(
        {**common, "mesh_axis": "full_domain_z", "mesh_level": 2}, "z8", "Ea"
    )
    assert certificate._case_labels(
        {
            **common,
            "mesh_axis": "anchor",
            "mesh_level": 0,
            "full_z_extension": "z16",
        },
        "z16",
        "Ea",
    )
    assert not certificate._case_labels(
        {**common, "mesh_axis": "anchor", "mesh_level": 0}, "z16", "Ea"
    )
