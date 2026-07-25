"""Integration guards for finalizer provenance + checkpoint atomicity (review P0-2/5/6).

All run WITHOUT Lumerical: final_projection --no-fdtd builds the mapping from env
(no eqc_lib/lumapi) and rejects bad-provenance designs before any solver.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

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
    ident = _identity(m, 0.5, 0.5)
    return ch, ident, m


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
    assert tmp.exists()                      # numpy did NOT append a second .npz
    tmp.replace(tmp_path / "best_feasible.npz")
    assert (tmp_path / "best_feasible.npz").exists()


def test_missing_had_feasible_rejected(tmp_path):
    ch, ident, m = _hashes_and_identity()
    lat = np.full(m.Nux * m.Nuy, 0.5)
    np.savez(tmp_path / "d.npz", latent=lat, code_hash=np.array(ch),
             mapping_identity=np.array(ident))          # no had_feasible
    rc, cat, ok = _run(tmp_path / "d.npz", tmp_path / "o")
    assert rc == 2 and cat == "missing_provenance" and not ok


def test_missing_code_hash_rejected(tmp_path):
    ch, ident, m = _hashes_and_identity()
    lat = np.full(m.Nux * m.Nuy, 0.5)
    np.savez(tmp_path / "d.npz", latent=lat, had_feasible=np.array(True),
             mapping_identity=np.array(ident))          # no code_hash
    rc, cat, ok = _run(tmp_path / "d.npz", tmp_path / "o")
    assert rc == 2 and cat == "missing_provenance" and not ok


def test_code_hash_mismatch_rejected(tmp_path):
    ch, ident, m = _hashes_and_identity()
    lat = np.full(m.Nux * m.Nuy, 0.5)
    np.savez(tmp_path / "d.npz", latent=lat, had_feasible=np.array(True),
             code_hash=np.array("deadbeef00000000"), mapping_identity=np.array(ident))
    rc, cat, ok = _run(tmp_path / "d.npz", tmp_path / "o")
    assert rc == 2 and cat == "code_hash_mismatch" and not ok


def test_config_identity_mismatch_rejected(tmp_path):
    ch, ident, m = _hashes_and_identity()
    lat = np.full(m.Nux * m.Nuy, 0.5)
    bad = ident.replace('"isolation_gap_um": 0.0', '"isolation_gap_um": 0.5')
    np.savez(tmp_path / "d.npz", latent=lat, had_feasible=np.array(True),
             code_hash=np.array(ch), mapping_identity=np.array(bad))
    rc, cat, ok = _run(tmp_path / "d.npz", tmp_path / "o")
    assert rc == 2 and cat == "config_mismatch" and not ok
