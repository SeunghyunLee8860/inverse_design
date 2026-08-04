"""Full-chain AD/FD for the production mapping + solver-safe layer.

Requires Lumerical v261 + license + GPU; SKIPS cleanly otherwise (every
Lumerical touch is inside the `evaluators` fixture, which calls pytest.skip on
failure -- nothing Lumerical runs at import/collection time).

Two mathematically-consistent checks (the previous single test compared the
probe-safe AD gradient against an EXACT forward FD -- different functions):

  * safe-vs-safe : AD(density_mode="probe_safe") vs central FD of
    F_safe(rho_geom) = F(delta + (1-2 delta) rho_geom)   [affine applied to FD too]
  * exact-vs-exact : AD(density_mode="exact") vs central FD of forward_fom, at a
    rail-safe interior latent (low beta, gray) so the exact centered probe is valid.
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


@pytest.fixture(scope="module")
def ctx(tmp_path_factory):
    """(model, evaluators dict, delta) or skip if Lumerical is unavailable."""
    try:
        import eqc_lib as lib
        from volume_current_evaluator import VolumeCurrentEvaluator
        model = lib.load_model()
        tmp = tmp_path_factory.mktemp("adfd")
        evs = {}
        for pol in ("x", "y"):
            ev = VolumeCurrentEvaluator(tmp / f"solver_{pol}", 0.001, pol)
            ev.prepare(force_rebuild=False)   # raises without Lumerical/license/GPU
            evs[pol] = ev
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Lumerical not available: {exc}")
    return model, evs, evs["x"].delta


def _dir(n, rng):
    d = rng.standard_normal(n)
    return d / np.linalg.norm(d)


@pytest.mark.parametrize("beta", [4.0, 32.0])
def test_full_chain_safe_vs_safe(ctx, beta):
    from autograd import tensor_jacobian_product
    model, evs, delta = ctx
    mapping = model.mapping
    shape = (model.Nx, model.Ny, model.Nz)
    rng = np.random.default_rng(0)
    lat = np.clip(0.5 + 0.2 * rng.standard_normal(model.Nux * model.Nuy), 0.02, 0.98)

    def F_safe(latent):
        rho_geom = np.asarray(mapping(latent, beta), float)
        rho_solver = (delta + (1.0 - 2.0 * delta) * rho_geom).reshape(shape)
        return sum(evs[p].forward_fom(rho_solver, label="fd") for p in ("x", "y"))

    phys = np.asarray(mapping(lat, beta), float).reshape(shape)
    g_phys = sum(evs[p].value_and_gradient(phys, density_mode="probe_safe").gradient_physical
                 for p in ("x", "y")).reshape(-1)
    dlat = np.asarray(tensor_jacobian_product(lambda z: mapping(z, beta))(lat, g_phys), float)
    d = _dir(lat.size, rng)
    ad = float(np.dot(dlat, d))
    h = 5e-3
    fd = (F_safe(lat + h * d) - F_safe(lat - h * d)) / (2 * h)
    rel = abs(ad - fd) / max(abs(ad), abs(fd), 1e-300)
    print(f"[adfd] safe beta={beta}: rel={rel:.4e} ad={ad:.4e} fd={fd:.4e}", flush=True)
    assert rel < 0.05, f"safe beta={beta}: AD/FD rel err {rel:.3e} (ad={ad:.4e} fd={fd:.4e})"


def test_full_chain_exact_vs_exact(ctx):
    from autograd import tensor_jacobian_product
    model, evs, _ = ctx
    mapping = model.mapping
    shape = (model.Nx, model.Ny, model.Nz)
    beta = 8.0                              # soft enough that the exact probe is rail-safe
    rng = np.random.default_rng(1)
    # Structured latent (NOT near-uniform).  A near-uniform flake makes the
    # coherent FOM ~1e-16 (numerical noise floor) where an AD/FD ratio is
    # meaningless.  std=0.5 at beta=8 keeps the physical density gray (rho in
    # ~[0.36,0.66], exact centered probe safe) but gives a meaningful FOM.
    lat = np.clip(0.5 + 0.5 * rng.standard_normal(model.Nux * model.Nuy), 0.05, 0.95)

    def F_exact(latent):
        phys = np.asarray(mapping(latent, beta), float).reshape(shape)
        return sum(evs[p].forward_fom(phys, label="fd") for p in ("x", "y"))

    phys = np.asarray(mapping(lat, beta), float).reshape(shape)
    g_phys = sum(evs[p].value_and_gradient(phys, density_mode="exact").gradient_physical
                 for p in ("x", "y")).reshape(-1)
    dlat = np.asarray(tensor_jacobian_product(lambda z: mapping(z, beta))(lat, g_phys), float)
    d = _dir(lat.size, rng)
    ad = float(np.dot(dlat, d))
    h = 5e-3
    fd = (F_exact(lat + h * d) - F_exact(lat - h * d)) / (2 * h)
    rel = abs(ad - fd) / max(abs(ad), abs(fd), 1e-300)
    print(f"[adfd] exact beta={beta}: rel={rel:.4e} ad={ad:.4e} fd={fd:.4e}", flush=True)
    assert rel < 0.05, f"exact: AD/FD rel err {rel:.3e} (ad={ad:.4e} fd={fd:.4e})"
