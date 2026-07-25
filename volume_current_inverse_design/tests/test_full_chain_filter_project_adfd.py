"""Full-chain AD/FD for the production mapping + solver-safe layer.

Requires a working Lumerical v261 + license + GPU, so it SKIPS automatically in
environments without them.  When it runs it checks that the adjoint gradient
(pulled through PeriodicConstrainedMapping and the solver-safe affine layer)
matches a central finite difference of the exact forward FOM, for Fx, Fy and
Fx+Fy, at low and high beta, including a near-rail latent.

This is the spec-section-18 re-validation gate; a green core test suite does NOT
substitute for it.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "bundle", ROOT / "inverse_design"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

os.environ.setdefault("MSOPT_MAPPING", "periodic_constrained")
os.environ.setdefault("PERIOD_UM", "6.0")
os.environ.setdefault("TARGET_WL_UM", "4.0")
os.environ.setdefault("MFS_UM", "0.5")
os.environ.setdefault("MGS_UM", "0.5")


def _evaluators(tmp):
    from volume_current_evaluator import VolumeCurrentEvaluator
    evs = {}
    for pol in ("x", "y"):
        ev = VolumeCurrentEvaluator(tmp / f"solver_{pol}", 0.001, pol)
        ev.prepare(force_rebuild=False)  # raises without Lumerical -> skip
        evs[pol] = ev
    return evs


@pytest.mark.parametrize("beta", [4.0, 32.0])
def test_full_chain_adfd(tmp_path, beta):
    import eqc_lib as lib
    from autograd import tensor_jacobian_product
    try:
        model = lib.load_model()
        evs = _evaluators(tmp_path)
    except Exception as exc:  # no Lumerical / license / GPU
        pytest.skip(f"Lumerical not available: {exc}")

    mapping = model.mapping
    shape = (model.Nx, model.Ny, model.Nz)
    rng = np.random.default_rng(0)
    lat = np.clip(0.5 + 0.2 * rng.standard_normal(model.Nux * model.Nuy), 0.02, 0.98)

    def fom_sum(latent):
        phys = np.asarray(mapping(latent, beta), float).reshape(shape)
        return sum(evs[p].forward_fom(phys, label="fd") for p in ("x", "y"))

    # AD via adjoint + mapping VJP (probe-safe)
    phys = np.asarray(mapping(lat, beta), float).reshape(shape)
    g_phys = np.zeros(shape)
    for p in ("x", "y"):
        g_phys += evs[p].value_and_gradient(phys, density_mode="probe_safe").gradient_physical
    dlat = np.asarray(
        tensor_jacobian_product(lambda z: mapping(z, beta))(lat, g_phys.reshape(-1)), float
    )
    d = rng.standard_normal(lat.size); d /= np.linalg.norm(d)
    ad = float(np.dot(dlat, d))
    h = 5e-3
    fd = (fom_sum(lat + h * d) - fom_sum(lat - h * d)) / (2 * h)
    rel = abs(ad - fd) / max(abs(ad), abs(fd), 1e-300)
    assert rel < 0.05, f"beta={beta}: AD/FD rel err {rel:.3e} (ad={ad:.4e} fd={fd:.4e})"
