"""Integration guards for finalizer provenance + checkpoint atomicity + success.

All run WITHOUT Lumerical: final_projection --no-fdtd builds the mapping from env
(no eqc_lib/lumapi) and rejects bad-provenance designs before any solver; the
positive-path test mocks the evaluator to reach SUCCESS.json.
"""

import json
import os
import subprocess
import sys
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "inverse_design"
PY = sys.executable


def _env():
    e = dict(os.environ)
    e.update(MSOPT_MAPPING="periodic_constrained", PERIOD_UM="6.0", TARGET_WL_UM="4.0",
             MFS_UM="0.5", MGS_UM="0.5")
    return e


def _hashes_and_identity():
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "bundle"))
    sys.path.insert(0, str(HERE))
    from run_constrained_inverse_design import _code_hash, production_code_files
    from final_projection import _mapping_from_env, _identity
    os.environ.update(MSOPT_MAPPING="periodic_constrained", PERIOD_UM="6.0",
                      MFS_UM="0.5", MGS_UM="0.5")
    ch = _code_hash(production_code_files(ROOT, HERE))
    m = _mapping_from_env()
    return ch, _identity(m, 0.5, 0.5), m


def _valid_fields(ch, ident, m):
    """A complete, strict-valid provenance set for a normal-runner NPZ."""
    return dict(
        latent=np.full(m.Nux * m.Nuy, 0.5),
        had_feasible=np.array(True),
        code_hash=np.array(ch),
        config_hash=np.array("cfg-test"),
        attempt=np.array(1),
        mapping_identity=np.array(ident),
    )


def _run(design_npz, out):
    r = subprocess.run(
        [PY, str(HERE / "final_projection.py"), str(design_npz),
         "--no-fdtd", "--output", str(out), "--mfs-um", "0.5", "--mgs-um", "0.5"],
        capture_output=True, text=True, env=_env(), cwd=str(ROOT))
    man = out / "final_manifest.json"
    cat = json.loads(man.read_text()).get("category") if man.exists() else None
    return r.returncode, cat, (out / "SUCCESS.json").exists()


def test_checkpoint_atomic_filename(tmp_path):
    # P0-2: temp name must end in .npz so numpy doesn't append and replace works
    tmp = tmp_path / "best_feasible.tmp.npz"
    np.savez_compressed(tmp, latent=np.zeros(4), beta=np.array(2.0), objective=np.array(1.0))
    assert tmp.exists()
    tmp.replace(tmp_path / "best_feasible.npz")
    assert (tmp_path / "best_feasible.npz").exists()


def _reject_case(tmp_path, drop=None, corrupt=None):
    ch, ident, m = _hashes_and_identity()
    fields = _valid_fields(ch, ident, m)
    if drop:
        fields.pop(drop)
    if corrupt:
        fields.update(corrupt)
    np.savez(tmp_path / "d.npz", **fields)
    return _run(tmp_path / "d.npz", tmp_path / "o")


def test_missing_had_feasible_rejected(tmp_path):
    rc, cat, ok = _reject_case(tmp_path, drop="had_feasible")
    assert rc == 2 and cat == "missing_provenance" and not ok


def test_missing_code_hash_rejected(tmp_path):
    rc, cat, ok = _reject_case(tmp_path, drop="code_hash")
    assert rc == 2 and cat == "missing_provenance" and not ok


def test_missing_config_hash_rejected(tmp_path):
    rc, cat, ok = _reject_case(tmp_path, drop="config_hash")
    assert rc == 2 and cat == "missing_provenance" and not ok


def test_missing_attempt_rejected(tmp_path):
    rc, cat, ok = _reject_case(tmp_path, drop="attempt")
    assert rc == 2 and cat == "missing_provenance" and not ok


def test_code_hash_mismatch_rejected(tmp_path):
    rc, cat, ok = _reject_case(tmp_path, corrupt={"code_hash": np.array("deadbeef00000000")})
    assert rc == 2 and cat == "code_hash_mismatch" and not ok


def test_config_identity_mismatch_rejected(tmp_path):
    ch, ident, m = _hashes_and_identity()
    bad = ident.replace('"isolation_gap_um": 0.0', '"isolation_gap_um": 0.5')
    rc, cat, ok = _reject_case(tmp_path, corrupt={"mapping_identity": np.array(bad)})
    assert rc == 2 and cat == "config_mismatch" and not ok


def test_positive_path_reaches_success(tmp_path, monkeypatch):
    # valid provenance + DRC-passing design + (mocked) FDTD -> completed + SUCCESS
    ch, ident, m = _hashes_and_identity()
    import final_projection as FP

    # DRC-passing 3um/3um grating (solid stripe [0,120), void [120,240))
    lat = np.zeros((m.Nux, m.Nuy))
    lat[:120, :] = 0.95
    lat[120:, :] = 0.05
    design = tmp_path / "design.npz"
    fields = _valid_fields(ch, ident, m)
    fields["latent"] = lat.reshape(-1)
    np.savez(design, **fields)

    fake = types.ModuleType("volume_current_evaluator")

    class _Ev:
        def __init__(self, workdir, rho_step, pol):
            self.pol = pol

        def prepare(self, force_rebuild=False):
            pass

        def forward_fom(self, phys, label=None):
            return {"x": 1.0, "y": 2.0}[self.pol]

    fake.VolumeCurrentEvaluator = _Ev
    monkeypatch.setitem(sys.modules, "volume_current_evaluator", fake)
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["final_projection.py", str(design),
                                      "--output", str(out), "--mfs-um", "0.5",
                                      "--mgs-um", "0.5"])
    for k, v in (("MSOPT_MAPPING", "periodic_constrained"), ("PERIOD_UM", "6.0"),
                 ("MFS_UM", "0.5"), ("MGS_UM", "0.5")):
        monkeypatch.setenv(k, v)

    FP.main()   # must NOT raise on the success path

    man = json.loads((out / "final_manifest.json").read_text())
    assert man["status"] == "completed"
    assert man["had_feasible"] is True
    assert man["design_config_hash"] == "cfg-test"
    assert man["design_attempt"] == 1
    assert man["mapping_identity"] == ident

    succ = json.loads((out / "SUCCESS.json").read_text())
    assert succ["exact_binary"] is True
    assert succ["Fx"] == 1.0 and succ["Fy"] == 2.0 and succ["F_sum"] == 3.0
    assert succ["design_config_hash"] == "cfg-test"
    assert succ["design_attempt"] == 1
    assert succ["mapping_identity"] == ident
    # capped flags present + True for the wide 3um grating (measured value is a
    # lower bound, not exact)
    assert succ["minimum_solid_width_capped"] is True
    assert succ["minimum_void_width_capped"] is True
    assert succ["artifact_sha256"]
