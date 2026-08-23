"""Immutable access to the historical robust density used for diagnostics."""

from __future__ import annotations

import hashlib

import numpy as np

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.contract import (
    CONTRACT,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.paths import (
    raw_path,
)
from photothermal_pte.optimization_runs.legacy_v261_optical_support.production_density_mapping import (
    ProductionDensityMapping,
)


CHECKPOINT = raw_path("robust_projection_ld_mma", "evaluation_0112.npz")
EXPECTED_CHECKPOINT_SHA256 = (
    "ef8b99bec0029588b89f56edc68bd9c747fa9ed0897933def138c787509332e3"
)


def sha256() -> str:
    digest = hashlib.sha256()
    with CHECKPOINT.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_densities() -> dict[str, np.ndarray]:
    actual = sha256()
    if actual != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError(
            "historical checkpoint provenance failure: expected SHA-256 "
            f"{EXPECTED_CHECKPOINT_SHA256}, got {actual}"
        )
    with np.load(CHECKPOINT, allow_pickle=False) as checkpoint:
        latent = np.asarray(checkpoint["latent"], dtype=np.float64)
        nominal = np.asarray(checkpoint["rho_nominal"], dtype=np.float64)
    for name, density in (("latent", latent), ("rho_nominal", nominal)):
        if density.shape != CONTRACT.design_shape:
            raise RuntimeError(f"{name} has unexpected shape {density.shape}")
        if np.any(~np.isfinite(density)) or np.any((density < 0.0) | (density > 1.0)):
            raise RuntimeError(f"{name} is non-finite or outside [0,1]")
    densities = {"eta_0.50_nominal": nominal}
    for eta in (0.35, 0.65):
        mapping = ProductionDensityMapping(
            shape=CONTRACT.design_shape,
            spacing_m=CONTRACT.design_pitch_m,
            radius_m=CONTRACT.filter_radius_m,
            eta=eta,
        )
        densities[f"eta_{eta:.2f}"] = np.asarray(
            mapping.physical(latent, 256.0), dtype=np.float64
        )
    return densities
