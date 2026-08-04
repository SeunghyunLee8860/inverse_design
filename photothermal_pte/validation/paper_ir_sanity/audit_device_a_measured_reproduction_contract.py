#!/usr/bin/env python3
"""Freeze the evidence and fail-closed limits for Device-A 11-um current.

This audit does not run FDTD, thermal, PTE, adjoint, or optimization.  It
cross-references the paper, Supporting Information, thesis, the frozen
Figure-2 geometry digitization, and the existing Palik substrate readback.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from photothermal_pte.validation.paper_ir_sanity import (
    run_device_a_explicit_thermal_pte as thermal,
)
from photothermal_pte.validation.paper_ir_sanity import (
    run_lumerical_device_a_ir_q as optical,
)


REPOSITORY = Path(__file__).resolve().parents[3]
PAPERS = Path("/home/seunghyun/tairte4/papers")
MAIN_PAPER = PAPERS / (
    "Adv Funct Materials - 2026 - Blevins - Large Transverse "
    "Thermoelectric Effect in Weyl Semimetal TaIrTe4 Engineered for-2.pdf"
)
SUPPLEMENT = PAPERS / "adfm75986-sup-0001-suppmat-2.pdf"
THESIS = PAPERS / "PhD_Thesis_Blevins.pdf"
GEOMETRY = (
    REPOSITORY
    / "photothermal_pte/reports/paper_ir_device_a_end_to_end/"
    "device_a_geometry_digitization.json"
)
FIG3J = (
    REPOSITORY
    / "photothermal_pte/reports/"
    "paper_ir_w12_50nm_maxwell_analytic_explicit3d/"
    "paper_fig3j_11um_current_ratio_digitization.json"
)
PALIK_RESULT = Path(
    "/data/seunghyun/tairte4/artifacts/paper_ir_device_a_boundary_mesh/"
    "finite_acc3_outer200_mid100_h12_edge50_x9_y12_dz10_a_gpu2_retry1_20260802/"
    "case_result.json"
)
EXACT_NK_READBACK = (
    REPOSITORY
    / "photothermal_pte/reports/paper_ir_device_a_measured_reproduction/"
    "kitamura_palik_nk_substrate_readback.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_contract(path: Path, pages: int | None = None) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
        "pages": pages,
    }


def resistance_audit() -> dict[str, Any]:
    contract = optical.load_digitized_device_a_contract(
        GEOMETRY,
        domain_um=60.0,
        source_span_um=50.0,
    )
    thermal.FLAKE_VERTICES_UM = np.asarray(
        contract["flake_vertices_simulation_um"], float
    )
    shift = np.asarray(contract["simulation_origin_shift_um"], float)
    payload = contract["payload"]
    thermal.TOP_CONTACT_SEGMENT_UM = np.asarray(
        payload["top_electrical_contact_segment_code_um"], float
    ) + shift
    thermal.BOTTOM_CONTACT_SEGMENT_UM = np.asarray(
        payload["bottom_electrical_contact_segment_code_um"], float
    ) + shift
    geometry = thermal.build_geometry(
        domain_m=48.0e-6,
        si_depth_m=20.0e-6,
        core_step_m=100.0e-9,
        flake_dz_m=10.0e-9,
    )
    return thermal.audit_two_terminal_resistance(
        geometry.x_edges_m,
        geometry.y_edges_m,
        np.any(geometry.flake_mask, axis=2),
    )


def markdown(summary: dict[str, Any]) -> str:
    kit = summary["optical_substrate_models"]["paper_consistent_scenario"][
        "SiO2_at_11um"
    ]
    palik = summary["optical_substrate_models"]["legacy_comparison"][
        "SiO2_at_11um"
    ]
    resistance = summary["absolute_current_audit"]["two_terminal_resistance"]
    ratio = summary["experimental_targets"]["figure_3J_digitized_ratio"]
    return f"""# Device-A measured 11-µm reproduction contract audit

## Outcome

Status: `{summary['status']}`

The three supplied documents were audited.  A paper-consistent substrate
scenario is now defined as **Kitamura-2007 fused silica for the 285-nm SiO2
film plus Lumerical Palik Si as an explicit closure**.  The paper cites
Kitamura for the silica phonon but does not identify a numerical Si optical
database, so this is not described as an exact hidden author input.

The requested calculation can test the optical/thermal/current trend and the
published polarization ratio.  It cannot certify an exact absolute current
without unpublished beam, CAD, contact-resistance, and scan-position data.
No empirical optical-power or current rescaling is allowed.

## Audited documents

| Document | Pages | SHA-256 |
|---|---:|---|
| Main paper | 10 | `{summary['sources']['main_paper']['sha256']}` |
| Supporting Information | 14 | `{summary['sources']['supporting_information']['sha256']}` |
| Blevins thesis | 164 | `{summary['sources']['thesis']['sha256']}` |

## Fixed published Device-A inputs

- TaIrTe4 thickness: 130 nm.
- Substrate: 285 nm thermally grown SiO2 on Si (main prose also says nominal
  300 nm; AFM/figure value 285 nm is used).
- Electrodes: 5 nm Ti / 50 nm Au.
- Wavelength: 11 µm; normal incidence.
- Time-averaged incident power: 284.40 µW from SI; 285 µW is the rounded
  main-figure caption.
- Objective: 40x reflective, NA=0.4; QCL stated spot range 9–16 µm.
- The documents do not state whether spot size means FWHM, 1/e2 diameter, or
  radius and do not tabulate the 11-µm realized beam profile.
- TaIrTe4 parameters: kappa(a,b,c)=(14.4,3.8,1.0) W/(m K),
  sigma(a,b)=(4.91e5,1.10e5) S/m, S(a,b)=(-6,27) µV/K,
  G(TaIrTe4/thermal-SiO2)=7.37e6 W/(m2 K), and G(TaIrTe4/air)=1 W/(m2 K).

## 11-µm substrate optical constants

| Scenario | SiO2 n+ik | SiO2 epsilon | Interpretation |
|---|---|---|---|
| Paper-consistent Kitamura | {kit['n_real']:.9f} + {kit['n_imag']:.9f}i | {kit['epsilon_real']:.9f} + {kit['epsilon_imag']:.9f}i | Production reproduction scenario |
| Existing Palik fitted readback | {palik['n_real']:.9f} + {palik['n_imag']:.9f}i | {palik['epsilon_real']:.9f} + {palik['epsilon_imag']:.9f}i | Preserved comparison only |

Kitamura loss k is {summary['optical_substrate_models']['SiO2_k_ratio_Kitamura_over_Palik']:.3f} times the existing fitted Palik value at 11 µm, so the prior Palik Maxwell artifact is not reused as the paper-consistent result.
Si remains Palik (`n={summary['optical_substrate_models']['paper_consistent_scenario']['Si_at_11um']['n_real']:.9f}+{summary['optical_substrate_models']['paper_consistent_scenario']['Si_at_11um']['n_imag']:.9g}i`) and is explicitly marked as an unpublished closure.

## Current equation and absolute-current gate

The implementation uses `x=b`, `y=a`,
`Jx=-sigma_b S_b dT/dx`, `Jy=-sigma_a S_a dT/dy`, followed by the
Shockley–Ramo volume integral.  This matches SI Eq. S5–S7.  The reduced
thermal reference uses the paper's top-air and bottom-SiO2 Robin boundaries;
bulk SiO2/Si thermal cells are not silently claimed to be the paper model.

Using the frozen Figure-2 digitization, published conductivities, 130-nm
thickness, and no fitted contact resistance predicts
`R={resistance['predicted_resistance_ohm']:.3f} ohm`, versus measured
`213 ohm` ({100.0*resistance['relative_difference_vs_measured']:.2f}%
difference).  Therefore absolute-current magnitude is fail-closed as
`BLOCKED_DIGITIZED_GEOMETRY_RESISTANCE_MISMATCH`; the computed current is not
renormalized to 213 ohm.

## Experimental comparison targets and paper inconsistency

- Figure 3J digitization: `|Ia|/|Ib|={ratio['value']:.6f} ± {ratio['uncertainty']:.6f}`
  at 11 µm, or `|Ib|/|Ia|={ratio['inverse']:.6f}`.
- Figure 3H and SI Figure S5 map colorbars are in pA (roughly ±200 pA at
  11 µm), while extracted Figure 3I labels its profile axis as nA.  Those
  differ by 1000x and cannot both be literal.  Absolute comparison therefore
  reports both the plotted value and the likely pA interpretation rather than
  silently choosing the nA label.
- SI Figure S5 independently fits the 11-µm off-axis E||a response to
  `I0=129 pA`, `tau=26±3.3 µs`.  At the reported 3675-Hz chopper frequency
  this fit gives `{summary['experimental_targets']['SI_Figure_S5_off_axis_Ea_frequency_fit']['fitted_current_at_measurement_frequency_pA']:.3f} pA`.
  This supports interpreting the Figure-3I minima (visually about 122 and
  143 plot units for a and b) as pA, while retaining the printed nA typo as a
  source inconsistency.
- The SI/thesis emphasize simulated current pattern robustness; the exact
  COMSOL CAD, beam profile, objective transmission, and tabulated scan
  coordinates are absent.

## Approved execution matrix

1. Same-substrate empty-stack references for E||a and E||b.
2. Digitized Device-A off-axis edge for E||a and E||b.
3. Scalar Gaussian, explicit assumed w0=8.75 µm, 50-µm source aperture,
   60-µm six-PML domain, existing boundary-aware local mesh, no CPU fallback.
4. Preserve full raw Q; use only material-overlap-attributed TaIrTe4 power for
   the paper-reduced thermal calculation.  No clipping, smoothing, gain,
   polarization matching, or global rescaling.
5. Use exact 284.40 µW and the paper-reduced Robin thermal operator, then the
   digitized weighting field and Shockley–Ramo current.
6. Judge numerical gates, `|Ia|/|Ib|`, signs/maps, and absolute pA separately.

This is a **paper-like measured Device-A reproduction with explicit closures**,
not an exact paper-certified recreation and not an inverse-design result.
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fig3j = json.loads(FIG3J.read_text())
    palik_result = json.loads(PALIK_RESULT.read_text())
    palik_readback = palik_result["run_result"]["substrate_epsilon_readback"]
    sio2_palik = palik_readback["materials"]["SiO2"]
    exact_readback = json.loads(EXACT_NK_READBACK.read_text())["readback"]
    si_palik = exact_readback["materials"]["Si"]
    epsilon_kitamura = complex(
        optical.kitamura_2007_sio2_epsilon(optical.WAVELENGTH_M)
    )
    n_kitamura = np.sqrt(epsilon_kitamura)
    audit = resistance_audit()
    chopper_frequency_Hz = 3675.0
    off_axis_I0_pA = 129.0
    off_axis_tau_s = 26.0e-6
    fitted_current_at_measurement_frequency_pA = off_axis_I0_pA / np.sqrt(
        1.0 + (2.0 * np.pi * chopper_frequency_Hz * off_axis_tau_s) ** 2
    )
    summary = {
        "status": "READY_DEVICE_A_MEASURED_REPRODUCTION_CONTRACT",
        "scope": "offline evidence, material, equation, and absolute-current audit",
        "sources": {
            "main_paper": file_contract(MAIN_PAPER, 10),
            "supporting_information": file_contract(SUPPLEMENT, 14),
            "thesis": file_contract(THESIS, 164),
            "geometry_digitization": file_contract(GEOMETRY),
            "figure_3J_digitization": file_contract(FIG3J),
            "existing_Palik_case_result": file_contract(PALIK_RESULT),
            "exact_11um_nk_material_readback": file_contract(EXACT_NK_READBACK),
        },
        "optical_substrate_models": {
            "paper_consistent_scenario": {
                "SiO2_model": "Kitamura et al. 2007 Eq.21-24/Table 2",
                "SiO2_at_11um": {
                    "n_real": float(n_kitamura.real),
                    "n_imag": float(n_kitamura.imag),
                    "epsilon_real": float(epsilon_kitamura.real),
                    "epsilon_imag": float(epsilon_kitamura.imag),
                },
                "Si_model": (
                    "Lumerical v261 Palik raw 11-um n,k copied into the "
                    "single-frequency n,k model; explicit closure"
                ),
                "Si_at_11um": {
                    "n_real": si_palik["n_complex"]["real"],
                    "n_imag": si_palik["n_complex"]["imag"],
                    "epsilon_real": si_palik["epsilon_r_complex"]["real"],
                    "epsilon_imag": si_palik["epsilon_r_complex"]["imag"],
                },
                "identity_limit": (
                    "paper cites Kitamura for the SiO2 phonon but does not "
                    "identify an exact Si database or say its RCWA used this exact fit"
                ),
                "FDTD_material_contract": (
                    "exact 11-um Kitamura-SiO2 and Palik-Si n,k readback; "
                    "finite numerical pulse centred in frequency at 11 um"
                ),
            },
            "legacy_comparison": {
                "model": "Lumerical Palik SiO2/Si fitted over 7-13 um",
                "SiO2_at_11um": {
                    "n_real": sio2_palik["n_complex"]["real"],
                    "n_imag": sio2_palik["n_complex"]["imag"],
                    "epsilon_real": sio2_palik["epsilon_r_complex"]["real"],
                    "epsilon_imag": sio2_palik["epsilon_r_complex"]["imag"],
                },
            },
            "SiO2_k_ratio_Kitamura_over_Palik": float(
                n_kitamura.imag / sio2_palik["n_complex"]["imag"]
            ),
        },
        "experimental_targets": {
            "wavelength_m": 11.0e-6,
            "incident_power_W_exact_SI": 284.40e-6,
            "incident_power_W_rounded_main_caption": 285.0e-6,
            "figure_3J_digitized_ratio": {
                "quantity": "|Ia|/|Ib|",
                "value": fig3j["measured_abs_Ia_over_abs_Ib_digitized"],
                "uncertainty": fig3j[
                    "measured_abs_Ia_over_abs_Ib_uncertainty_estimate"
                ],
                "inverse": fig3j["requested_abs_Ib_over_abs_Ia_inverted"],
            },
            "absolute_current_unit_consistency": "FAIL_PAPER_FIG3I_NA_VS_MAPS_PA",
            "SI_Figure_S5_off_axis_Ea_frequency_fit": {
                "I0_pA": off_axis_I0_pA,
                "tau_s": off_axis_tau_s,
                "tau_uncertainty_s": 3.3e-6,
                "measurement_chopper_frequency_Hz": chopper_frequency_Hz,
                "fitted_current_at_measurement_frequency_pA": float(
                    fitted_current_at_measurement_frequency_pA
                ),
            },
            "main_Figure_3I_visual_profile_estimate": {
                "Ia_minimum_plot_units_approx": -122.0,
                "Ib_minimum_plot_units_approx": -143.0,
                "printed_axis_unit": "nA",
                "likely_physical_unit": "pA",
                "basis": (
                    "the 11-um maps use +/-200 pA and the independent SI "
                    "off-axis frequency fit gives I0=129 pA"
                ),
                "not_a_tabulated_measurement": True,
            },
        },
        "absolute_current_audit": {
            "two_terminal_resistance": audit,
            "no_resistance_or_current_rescaling": True,
        },
        "equation_contract": {
            "coordinate_mapping": "x=b, y=a",
            "Jx": "-sigma_b*S_b*dT/dx",
            "Jy": "-sigma_a*S_a*dT/dy",
            "collection": "volume integral J_local dot grad(psi)",
            "thermal_primary": "paper SI Eq.S4 reduced flake-only Robin model",
        },
        "missing_for_exact_reproduction": [
            "exact Device-A CAD and hidden metal/flakes overlap",
            "contact resistance decomposition",
            "11-um beam definition and measured x/y profile",
            "objective transmission eta",
            "exact off-axis scan coordinates",
            "paper's exact Si optical dataset",
            "internally consistent absolute-current units for Figure 3H/I",
        ],
        "prohibited_adjustments": [
            "Q clipping",
            "Q smoothing",
            "optical gain or global rescaling",
            "polarization power matching",
            "current or conductivity fitting to 213 ohm",
        ],
        "FDTD_run": False,
        "thermal_run": False,
        "PTE_run": False,
        "adjoint_run": False,
        "optimization_run": False,
    }
    summary_path = args.output_dir / "device_a_measured_reproduction_contract.json"
    report_path = args.output_dir / "DEVICE_A_MEASURED_REPRODUCTION_CONTRACT.md"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    report_path.write_text(markdown(summary))
    print(json.dumps({"summary": str(summary_path), "report": str(report_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
