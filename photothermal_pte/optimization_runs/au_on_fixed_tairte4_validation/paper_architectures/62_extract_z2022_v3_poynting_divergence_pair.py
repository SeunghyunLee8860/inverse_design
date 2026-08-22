#!/usr/bin/env python3
"""Extract paired, conservative Poynting-divergence Q for corrected Z M2.

The frequency-domain monitor stores the complex Poynting vector without the
time-average factor 1/2.  This script averages each component onto its finite-
volume face and evaluates -1/2 div(Re(P)).  The construction telescopes to the
six-face flux on the same control volume.  Signed local values are retained;
there is no clipping, smoothing, gain, or rescaling.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


RAW_ROOT = Path("/home/seunghyun/tairte4/raw_artifacts")
INPUTS = {
    "Ea": RAW_ROOT / "paper_z2022_m2_v3_Ea_poynting_volume",
    "Eb": RAW_ROOT / "paper_z2022_m2_v3_Eb_poynting_volume",
}
OUTPUT = RAW_ROOT / "paper_z2022_m2_v3_ea_eb_poynting_divergence"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_volume_q(
    px: np.ndarray,
    py: np.ndarray,
    pz: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
) -> np.ndarray:
    """Return cell-centred -1/2 div(Re(P)) on edges x/y/z."""
    # Transverse four-node averages place each vector component on the proper
    # face of the hexahedral control volume before differencing.
    pxf = 0.25 * (px[:, 1:, 1:] + px[:, :-1, 1:] + px[:, 1:, :-1] + px[:, :-1, :-1])
    pyf = 0.25 * (py[1:, :, 1:] + py[:-1, :, 1:] + py[1:, :, :-1] + py[:-1, :, :-1])
    pzf = 0.25 * (pz[1:, 1:, :] + pz[:-1, 1:, :] + pz[1:, :-1, :] + pz[:-1, :-1, :])
    divergence = (
        (pxf[1:, :, :] - pxf[:-1, :, :]) / np.diff(x)[:, None, None]
        + (pyf[:, 1:, :] - pyf[:, :-1, :]) / np.diff(y)[None, :, None]
        + (pzf[:, :, 1:] - pzf[:, :, :-1]) / np.diff(z)[None, None, :]
    )
    return -0.5 * divergence


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    cases: dict[str, object] = {}
    geometry_reference: dict[str, object] | None = None
    for pol, folder in INPUTS.items():
        json_path = folder / "Z2022_M2_selected_Q.json"
        npz_path = folder / "Z2022_M2_selected_Q.npz"
        payload = json.loads(json_path.read_text())
        if payload.get("geometry_variant") != "figure_period_corrected_v3":
            raise RuntimeError(f"{pol}: corrected-v3 geometry is required")
        if not payload.get("volume_poynting_recorded"):
            raise RuntimeError(f"{pol}: volume Poynting vector was not recorded")
        if geometry_reference is None:
            geometry_reference = payload["geometry"]
        elif payload["geometry"] != geometry_reference:
            raise RuntimeError("Ea/Eb geometry mismatch")
        with np.load(npz_path, allow_pickle=False) as data:
            x, y, z = (np.asarray(data[f"P_{axis}_m"], float) for axis in "xyz")
            p = [np.asarray(data[f"P{axis}_complex_W_m2"]).real for axis in "xyz"]
        if any(component.shape != (x.size, y.size, z.size) for component in p):
            raise RuntimeError(f"{pol}: Poynting coordinate/array mismatch")
        q = finite_volume_q(*p, x, y, z)
        volume = np.diff(x)[:, None, None] * np.diff(y)[None, :, None] * np.diff(z)[None, None, :]
        power = float(np.sum(q * volume))
        positive_power = float(np.sum(np.maximum(q, 0.0) * volume))
        negative_power = float(np.sum(np.minimum(q, 0.0) * volume))
        flux_power = float(payload["P_flux_absorbed_W"])
        closure = abs(power - flux_power) / max(abs(flux_power), np.finfo(float).tiny)
        arrays[f"{pol}_Qdiv_W_m3"] = q
        cases[pol] = {
            "source_npz": str(npz_path),
            "source_npz_size_bytes": npz_path.stat().st_size,
            "source_npz_sha256": sha256(npz_path),
            "P_flux_absorbed_W": flux_power,
            "P_Qdiv_W": power,
            "closure_relative": closure,
            "positive_power_W": positive_power,
            "negative_power_W": negative_power,
            "negative_to_positive_power": abs(negative_power) / max(positive_power, np.finfo(float).tiny),
            "negative_cell_count": int(np.count_nonzero(q < 0.0)),
            "cell_count": int(q.size),
            "no_clipping_smoothing_gain_or_rescaling": True,
        }
        arrays.update({"x_edges_m": x, "y_edges_m": y, "z_edges_m": z})
    out_npz = output / "Z2022_M2_EA_EB_POYNTING_DIVERGENCE_Q.npz"
    np.savez_compressed(out_npz, **arrays)
    gates = {
        "both_flux_closure_lt_0p5pct": all(float(cases[p]["closure_relative"]) < 0.005 for p in cases),
        # This is deliberately diagnostic rather than a pass criterion: a
        # numerical divergence can oscillate locally at metal interfaces.
        "both_have_no_nan_or_inf": all(np.isfinite(arrays[f"{p}_Qdiv_W_m3"]).all() for p in cases),
    }
    summary = {
        "status": "DIAGNOSTIC_Z2022_M2_PAIRED_POYNTING_DIVERGENCE_Q",
        "classification": "signed conservative diagnostic heat source; not promoted physical volumetric Q",
        "axis_mapping": "x=b, y=a, z=c",
        "geometry": geometry_reference,
        "discretization": "face-collocated finite-volume -0.5*div(Re(P)); x/y/z coordinates read from monitor",
        "cases": cases,
        "gates": gates,
        "raw_artifact": {
            "path": str(out_npz),
            "size_bytes": out_npz.stat().st_size,
            "sha256": sha256(out_npz),
        },
    }
    (output / "Z2022_M2_EA_EB_POYNTING_DIVERGENCE_Q.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))
    return 0 if all(gates.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
