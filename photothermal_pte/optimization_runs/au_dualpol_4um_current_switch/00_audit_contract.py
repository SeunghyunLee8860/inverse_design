#!/usr/bin/env python3
"""Publish the pre-solve geometry, sign, material, and DFM contract."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.contract import CONTRACT
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.dfm import exact_500nm_audit


HERE = Path(__file__).resolve().parent
OUT = HERE / "results_preflight"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    c = CONTRACT
    audit = c.audit()
    rho = np.full(c.design_shape, 0.5, dtype=float)
    exact = exact_500nm_audit(rho, c.design_pitch_m, c.minimum_solid_feature_m)
    audit["initial_design"] = {
        "type": "uniform latent/physical preflight only",
        "value": 0.5,
        "note": (
            "the contract imposes no mirror/rotational symmetry; any production "
            "initialization must be explicitly recorded and independently seeded"
        ),
    }
    audit["DFM_audit_definition"] = {
        key: value for key, value in exact.items() if not isinstance(value, np.ndarray)
    }
    audit["status"] = "AUDITED_AU_DUALPOL_4UM_PREFLIGHT_NOT_YET_SOLVED"
    (OUT / "AU_DUALPOL_4UM_CONTRACT.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )

    um = 1.0e-6
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    ax = axes[0, 0]
    half_domain = c.optical_lateral_span_m / (2 * um)
    half_flake = c.flake_span_x_m / (2 * um)
    half_design = c.design_span_x_m / (2 * um)
    half_aperture = c.source_aperture_span_m / (2 * um)
    ax.add_patch(Rectangle((-half_domain, -half_domain), 2*half_domain, 2*half_domain,
                           facecolor="#e8f3ff", edgecolor="#6a1b9a", lw=4, label="optical domain; six PML"))
    ax.add_patch(Rectangle((-half_flake, -half_flake), 2*half_flake, 2*half_flake,
                           facecolor="#d8a6a6", edgecolor="#8e2424", alpha=0.55, label="fixed TaIrTe4"))
    ax.add_patch(Rectangle((-half_design, -half_design), 2*half_design, 2*half_design,
                           facecolor="#f6c344", edgecolor="#8a5a00", alpha=0.85, label="Au design region"))
    ax.add_patch(Rectangle((-half_aperture, -half_aperture), 2*half_aperture, 2*half_aperture,
                           fill=False, edgecolor="#2878b5", ls="--", lw=2, label="Gaussian aperture"))
    ax.axvline(-half_flake, color="black", lw=7, label="psi=0 left terminal BC")
    ax.axvline(+half_flake, color="gold", lw=7, label="psi=1 right terminal BC")
    ax.annotate("+I", (3.0, -6.6), (-1.0, -6.6), arrowprops=dict(arrowstyle="->", lw=2), ha="center")
    ax.set(xlim=(-11, 11), ylim=(-11, 11), aspect="equal", xlabel="Lumerical x = b (um)", ylabel="Lumerical y = a (um)", title="Top view: fixed flake + floating Au design")
    ax.legend(fontsize=8, loc="upper right")

    ax = axes[0, 1]
    x = np.linspace(-half_aperture, half_aperture, 501)
    intensity = np.exp(-2.0 * (x / (c.gaussian_waist_m/um))**2)
    ax.plot(x, intensity, lw=2)
    ax.axvline(-half_flake, color="#8e2424", ls="--")
    ax.axvline(half_flake, color="#8e2424", ls="--")
    ax.set(xlabel="x or y at target plane (um)", ylabel="I / I_peak", title=f"Requested Gaussian w0={c.gaussian_waist_m/um:.1f} um", yscale="log", ylim=(1e-5, 1.2))
    ax.grid(True, which="both", alpha=0.3)

    ax = axes[1, 0]
    layers = [
        (-2.0, -0.385, "Si substrate", "#708090"),
        (-0.385, -0.100, "285 nm SiO2", "#91d4d8"),
        (-0.100, 0.0, "100 nm TaIrTe4", "#e77765"),
        (0.0, 0.050, "0/50 nm floating Au design", "#f6c344"),
        (0.050, 2.5, "air + downward Gaussian", "#e8f3ff"),
    ]
    for z0, z1, label, color in layers:
        ax.add_patch(Rectangle((-8, z0), 16, z1-z0, color=color, ec="black", lw=0.8))
        ax.text(0, 0.5*(z0+z1), label, ha="center", va="center", fontsize=9)
    ax.annotate("normal incidence -z", (0, 0.35), (0, 1.8), arrowprops=dict(arrowstyle="->", lw=3, color="#2878b5"), ha="center", color="#2878b5")
    ax.set(xlim=(-10,10), ylim=(-2.1,2.6), xlabel="x=b or y=a (um)", ylabel="z (um)", title="Cross section (remote z grid/PML audited at solver build)")

    ax = axes[1, 1]
    ax.axis("off")
    text = (
        "Signed detector objective\n\n"
        "+I : solver +x inside flake; x_min -> x_max\n"
        "E||a target: I_a > 0 (x_min -> x_max)\n"
        "E||b target: I_b < 0 (x_max -> x_min)\n\n"
        "maximize t\n"
        "subject to  I_a >= t,  -I_b >= t\n\n"
        "Au is floating: it changes Q, T, and psi,\n"
        "but it is not a terminal electrode.\n\n"
        "500 nm solid AND void: exact binary morphology audit\n"
        "Robust path: filter/projection + per-eta grayness gate\n"
        "Legacy smooth constraints are not the robust production gate"
    )
    ax.text(0.03, 0.96, text, va="top", fontsize=13, family="monospace")
    fig.suptitle("4 um dual-polarization Au/PTE inverse-design preflight", fontsize=16)
    fig.savefig(OUT / "AU_DUALPOL_4UM_GEOMETRY_AND_OBJECTIVE.png", dpi=180)
    plt.close(fig)

    report = f"""# 4 um Au dual-polarization PTE inverse-design contract

Status: **{audit['status']}**

The fixed TaIrTe4 flake is 16 x 16 x 0.1 um and the centered floating Au
design window is 8 x 8 x 0.05 um. The design has {c.design_shape[0]} x
{c.design_shape[1]} physical cells at 100 nm pitch. The optical excitation is
a centered, normally incident scalar Gaussian at 4 um with w0=4 um. At the
flake/source-aperture boundary the requested infinite-Gaussian intensity is
{100*c.flake_boundary_intensity_fraction:.5f}% of its peak.

Lumerical x is crystal b and y is crystal a. The low/high electrical terminal
boundary conditions are imposed on the fixed flake at x_min/x_max. The
implemented positive current is the +x component of internal conventional
current, from x_min to x_max. Therefore the requested switch is

- E||a: I_a > 0 (x_min to x_max),
- E||b: I_b < 0 (x_max to x_min).

Production uses an epigraph objective: maximize t subject to I_a >= t and
-I_b >= t. This prevents one polarization from becoming large while the other
remains weak.

The patterned Au is electrically floating, not an optical model of the
measurement electrodes. It must be included in optical absorption, thermal
spreading/Au-Ta contact, and electrical shunting/weighting-field response.

The robust production path uses a 250 nm-radius density filter, projection,
per-eta grayness constraints, and a separate exact thresholded morphology
audit. The legacy smooth solid/void functions remain tested utilities but are
not the robust optimizer's active constraints. Final promotion requires zero
exact bad cells. This preflight does not claim that the initial uniform
rho=0.5 design is manufacturable.

Au/Ta thermal and electrical contact values are explicitly named numerical
scenarios because direct TaIrTe4/Au values have not been experimentally fixed.
They require sensitivity analysis before an experimental prediction claim.

No Maxwell, thermal, electrical, adjoint, or optimization solve is claimed by
this checkpoint.
"""
    (OUT / "AU_DUALPOL_4UM_CONTRACT.md").write_text(report, encoding="utf-8")
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
