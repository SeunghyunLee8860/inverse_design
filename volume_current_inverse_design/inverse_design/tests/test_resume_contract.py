"""Provenance hashes: same inputs -> same hash; any change -> different hash."""

import importlib


def _mod():
    return importlib.import_module("run_constrained_inverse_design")


def test_config_hash_stable_and_sensitive():
    m = _mod()
    base = {"mfs_um": 0.5, "beta_schedule": [2, 4, 8], "rho_step": 0.001}
    h0 = m._config_hash(base)
    assert h0 == m._config_hash(dict(base))
    changed = dict(base); changed["mfs_um"] = 0.4
    assert m._config_hash(changed) != h0


def test_code_hash_stable(tmp_path):
    m = _mod()
    f = tmp_path / "a.py"
    f.write_text("x = 1\n")
    h0 = m._code_hash([f])
    assert h0 == m._code_hash([f])
    f.write_text("x = 2\n")
    assert m._code_hash([f]) != h0


def test_beta_stages_parsing():
    m = _mod()
    assert m._beta_stages("2,4,8,16") == [2.0, 4.0, 8.0, 16.0]
