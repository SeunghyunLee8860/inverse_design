#!/usr/bin/env python3
"""Synthesize the PAPER's IR optical model as a drop-in optical case dir.

Paper contract (AFM 2026 Methods 2.4 / SI Eq. S1-S2 / thesis Eq. 4.3-4.4):
  * total absorbed power fixed by the TMM absorption of the infinite
    TaIrTe4/SiO2/Si stack (polarization enters ONLY through this scalar),
  * in-plane shape = the incident Gaussian, identical for both polarizations,
  * depth shape = Beer-Lambert exp(-beta z) with beta = 4*pi*k/lambda,
  * no edge/full-wave structure of any kind.

Everything downstream (material-overlap remap, expanded 3D FVM, isotropic
Laplace weighting potential, Shockley-Ramo integral) is then the SAME code
that consumed the full-wave Q, so the two runs differ ONLY in the optics.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from matplotlib.path import Path as PolygonPath

WAVELENGTH_M = 11.0e-6
THICKNESS_M = 130.0e-9
# independently validated 11-um TMM absorption of the real stack
TMM_ABSORPTION = {"a": 0.17673296, "b": 0.26328721}
# solver-readback permittivity at 11 um (crystal axes)
EPS = {"a": complex(-42.96623156686853, 204.5326948215291),
       "b": complex(13.268147712480612, 26.18179590766738)}


def beer_lambert_beta(pol: str) -> float:
    k = np.sqrt(EPS[pol]).imag
    return 4.0 * np.pi * k / WAVELENGTH_M


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--template-case-dir", type=Path, required=True,
                   help="full-wave optical case dir supplying grid + contract")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--polarization", choices=("a", "b"), required=True)
    p.add_argument("--beam-center-x-um", type=float, required=True)
    p.add_argument("--beam-center-y-um", type=float, required=True)
    p.add_argument("--waist-um", type=float, required=True,
                   help="1/e^2 intensity radius w0 of the incident Gaussian")
    p.add_argument("--incident-power-w", type=float, default=None,
                   help="common incident power at 1 W/m^2 for BOTH polarizations")
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = json.loads((args.template_case_dir / "case_result.json").read_text())
    with np.load(args.template_case_dir / "finite_q_on_artifact.npz",
                 allow_pickle=False) as raw:
        x = np.asarray(raw["x_m"], float)
        y = np.asarray(raw["y_m"], float)
        z = np.asarray(raw["z_m"], float)

    # The paper's model has ONE beam for both polarizations.  The two full-wave
    # empty references differ by 5% in measured incident power (a realized-beam
    # artifact), so importing them here would smuggle that artifact into the
    # paper model.  Use one common value for both polarizations.
    incident_power_W = (
        args.incident_power_w
        if args.incident_power_w is not None
        else float(result["run_result"]["normalization"]["incident_power_W_at_1_W_m2"])
    )
    total_absorbed_W = TMM_ABSORPTION[args.polarization] * incident_power_W
    beta = beer_lambert_beta(args.polarization)
    w0 = args.waist_um * 1e-6

    # in-plane: normalized Gaussian, integral over the full plane = 1
    xx, yy = np.meshgrid(x, y, indexing="ij")
    r2 = (xx - args.beam_center_x_um * 1e-6) ** 2 + (yy - args.beam_center_y_um * 1e-6) ** 2
    g = (2.0 / (np.pi * w0 ** 2)) * np.exp(-2.0 * r2 / w0 ** 2)

    # depth: Beer-Lambert inside the flake only, integral over depth = 1
    depth = np.clip(-z, 0.0, THICKNESS_M)
    inside = (z <= 0.0 + 1e-15) & (z >= -THICKNESS_M - 1e-15)
    f = np.where(inside, beta * np.exp(-beta * depth) / (1.0 - np.exp(-beta * THICKNESS_M)), 0.0)
    # The flake spans only ~14 z-cells, and 1/beta differs 4x between the two
    # polarizations, so the raw discrete depth integral is biased differently
    # per polarization (a: +9.5%, b: <1%).  Renormalize discretely so the
    # TMM-fixed total power is exact for BOTH and the ratio carries no
    # discretization bias.
    dz_all = np.gradient(z)
    depth_norm = float((f * dz_all).sum())
    f = f / depth_norm

    # Mask to the flake polygon.  The full-wave Q is identically zero off the
    # flake (there is no material there); without this mask the downstream
    # power-conserving remap dumps the whole off-flake beam tail into the
    # flake (observed: Tmax 11 K instead of 0.37 K).  Mask AFTER normalization
    # so the surviving total is A_TMM * P * (on-flake capture fraction), which
    # is exactly the paper's construction.
    flake_xy = PolygonPath(
        np.asarray(result["pre_run_contract"]["geometry"]["flake_vertices_um"], float)
        * 1e-6
    ).contains_points(
        np.column_stack((xx.ravel(), yy.ravel())), radius=1e-15
    ).reshape(xx.shape)

    Q = total_absorbed_W * (g * flake_xy)[:, :, None] * f[None, None, :]

    # verify the discrete integral reproduces the TMM contract
    dx = np.gradient(x); dy = np.gradient(y); dz = np.gradient(z)
    dV = dx[:, None, None] * dy[None, :, None] * dz[None, None, :]
    integral = float((Q * dV).sum())
    print(f"[{args.polarization}] beta={beta:.4e} 1/m  target={total_absorbed_W:.6e} W  "
          f"discrete={integral:.6e} W  rel.err={integral/total_absorbed_W-1:+.3%}")

    np.savez_compressed(
        args.output_dir / "finite_q_on_artifact.npz",
        x_m=x, y_m=y, z_m=z, Q_on_W_m3=Q,
        incident_intensity_W_m2=np.array([1.0]),
        P_abs_volume_W=np.array([integral]),
    )

    geo = result["pre_run_contract"]["geometry"]
    geo["electrodes_in_optical_model"] = False
    geo["geometry_source"] = (
        "paper-model optics: TMM-fixed total absorption x incident Gaussian x "
        "Beer-Lambert depth; no full-wave edge structure (AFM Methods 2.4)")
    result["run_result"]["normalization"]["incident_power_W_at_1_W_m2"] = incident_power_W
    result["optical_model_identity"] = "paper-replication (TMM + Gaussian + Beer-Lambert)"
    result["paper_model_parameters"] = {
        "TMM_absorption": TMM_ABSORPTION[args.polarization],
        "beta_1_per_m": beta,
        "waist_um": args.waist_um,
        "beam_center_um": [args.beam_center_x_um, args.beam_center_y_um],
        "wavelength_m": WAVELENGTH_M,
    }
    (args.output_dir / "case_result.json").write_text(json.dumps(result, indent=1))
    print(f"  wrote {args.output_dir}")


if __name__ == "__main__":
    main()
