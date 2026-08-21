#!/usr/bin/env python3
"""Publish the fail-closed resolution of the v261 Au-gradient investigation.

This script performs no Lumerical, thermal, PTE, adjoint, or optimization
solve.  It consolidates already completed raw controls, preserves their
provenance, and separates demonstrated root causes from proposed remedies.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
RAW = Path("/data/seunghyun/tairte4/raw_artifacts/au_topology_validation")
HOME_RAW = Path("/home/seunghyun/tairte4/raw_artifacts/au_topology_validation")
LUMOPT2_DEPS = Path(
    "/home/seunghyun/lumerical_r12/opt/lumerical/v261/api/python/"
    "lumopt2/parametrization/d_eps_calculator.py"
)
LEGACY_LUMOPT_GEOMETRY = Path(
    "/home/seunghyun/lumerical_r12/opt/lumerical/v261/api/python/"
    "lumopt/geometries/geometry.py"
)

PATHS = {
    "all_metal_dt0p5": RAW
    / "temperature_density_au_50nm_rho1_span1000_conformal1_dt0p5_allmetal_short_gpu2"
    / "case_result.json",
    "pml_dt0p5": RAW
    / "temperature_density_au_50nm_rho1_span1000_conformal1_dt0p5_short_gpu0"
    / "case_result.json",
    "sampled_passive_base": RAW
    / "temperature_density_reverse_ordal_sampled_passive_rho1_conformal1_short_gpu2"
    / "case_result.json",
    "global_cv0_air_base": RAW
    / "temperature_density_au_50nm_rho1_span1000_conformal0_gpu3"
    / "case_result.json",
    "global_cv0_exact_au_base": RAW
    / "temperature_density_reverse_exact_au_rho1_span1000_conformal0_gpu3"
    / "case_result.json",
    "fixed_material_adfd": HOME_RAW
    / "pva5_fixedgrid_material_adjoint_control"
    / "au_fixed_geometry_material_adjoint_result.json",
    "smooth_boundary_adfd": RAW
    / "pva5_smooth3d_ellipsoid_boundary_adjoint_gpu0"
    / "au_smooth3d_ellipsoid_boundary_adjoint_result.json",
    "complex_deps_coarse": RAW
    / "same_session_complex_deps_checkpointed_gpu4"
    / "au_same_session_complex_deps_result.json",
    "complex_deps_subnm": RAW
    / "same_session_complex_deps_subnm_gpu4"
    / "au_same_session_complex_deps_result.json",
    "same_step_local_fd_h1nm": RAW
    / "same_step_local_maxwell_fd_h1nm"
    / "au_same_step_local_maxwell_fd_result.json",
    "same_step_local_fd_h0p5nm": RAW
    / "same_step_local_maxwell_fd_h0p5nm"
    / "au_same_step_local_maxwell_fd_result.json",
}

STATUS = "BLOCKED_AU_PRODUCTION_GRADIENT_REQUIRES_DISPERSIVE_DISCRETE_ADJOINT"
N_AU = 12.1
K_AU = 69.2
WAVELENGTH_M = 10.0e-6
Z0_OHM = 376.730313668


def read(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(role: str, path: Path) -> dict[str, object]:
    return {
        "role": role,
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def recorded_artifact(role: str, record: dict[str, object]) -> dict[str, object]:
    """Forward already verified path/size/SHA metadata without rehashing GB files."""

    return {
        "role": role,
        "path": str(Path(str(record["path"])).resolve()),
        "size_bytes": int(record["size_bytes"]),
        "sha256": str(record["sha256"]),
        "metadata_source": "completed forward case_result.json",
    }


def diverged(result: dict) -> bool:
    return "fields were diverging" in str(result.get("error", ""))


def source_matches(path: Path, needles: list[str]) -> dict[str, object]:
    lines = path.read_text().splitlines()
    matches = {
        needle: [index for index, line in enumerate(lines, start=1) if needle in line]
        for needle in needles
    }
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
        "matches": matches,
        "all_patterns_found": all(bool(indices) for indices in matches.values()),
    }


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    data = {name: read(path) for name, path in PATHS.items()}
    fixed = data["fixed_material_adfd"]
    boundary = data["smooth_boundary_adfd"]
    coarse = data["complex_deps_coarse"]
    subnm = data["complex_deps_subnm"]
    same_step_h1 = data["same_step_local_fd_h1nm"]
    same_step_h0p5 = data["same_step_local_fd_h0p5nm"]

    epsilon_au = complex(N_AU, K_AU) ** 2
    amplitude_skin_depth_m = WAVELENGTH_M / (2.0 * np.pi * K_AU)
    intensity_skin_depth_m = WAVELENGTH_M / (4.0 * np.pi * K_AU)
    surface_impedance = Z0_OHM / complex(N_AU, K_AU)
    lumopt2_audit = source_matches(
        LUMOPT2_DEPS,
        [
            "real(D{label}.index_x^2)",
            "real(d_eps_ctr_sparse.eps_x.data)",
            "eps_real_data = np.where(eps_real_data < 0, 1e-6",
            "n>>k~0",
        ],
    )
    legacy_lumopt_audit = source_matches(
        LEGACY_LUMOPT_GEOMETRY,
        [
            "index_x^2",
            "index_y^2",
            "index_z^2",
            "(eps_data1 - eps_data2) / (2*dx)",
        ],
    )
    strong_fixed = fixed["material_steps"][-1]
    strong_boundary = boundary["finite_difference"]["h_0.05_um"]
    finest = subnm["steps"]["h_0.1_nm"]
    fd_reference = float(strong_boundary["derivative_J_proxy_per_um"])
    finest_ad = float(finest["same_session_complex_deps_AD_J_proxy_per_um"])
    finest_vs_coarse_fd = abs(finest_ad - fd_reference) / max(
        abs(fd_reference), np.finfo(float).tiny
    )
    same_step_fd = float(same_step_h0p5["Maxwell_central_FD_J_proxy_per_um"])
    same_step_ad = float(
        same_step_h0p5["same_session_complex_dEps_AD_J_proxy_per_um"]
    )
    same_step_error = float(same_step_h0p5["AD_FD_relative_error"])
    fd_h1 = float(same_step_h1["Maxwell_central_FD_J_proxy_per_um"])
    fd_plateau_error = abs(same_step_fd - fd_h1) / max(
        abs(same_step_fd), np.finfo(float).tiny
    )

    carrier_names = (
        "all_metal_dt0p5",
        "pml_dt0p5",
        "sampled_passive_base",
        "global_cv0_air_base",
        "global_cv0_exact_au_base",
    )
    carrier_controls = {
        name: {
            "status": data[name].get("status"),
            "diverged": diverged(data[name]),
            "boundary_mode": data[name]
            .get("built_source_contract", {})
            .get("domain", {})
            .get("boundary_mode_requested"),
            "mesh_refinement": data[name]
            .get("built_source_contract", {})
            .get("mesh_contract", {})
            .get("mesh_refinement"),
            "dt_stability_factor": data[name]
            .get("built_source_contract", {})
            .get("mesh_contract", {})
            .get("dt_stability_factor"),
            "base_material_model": data[name]
            .get("material", {})
            .get("base_material_model"),
            "grid_attribute_conformal_control_available": data[name]
            .get("material", {})
            .get("temperature_attribute_conformal_control_available"),
        }
        for name in carrier_names
    }

    summary = {
        "status": STATUS,
        "production_Au_optimization_permitted": False,
        "root_cause_resolution": {
            "PML_is_root_cause": False,
            "Courant_step_is_root_cause": False,
            "global_conformal_variant_1_is_only_root_cause": False,
            "exact_scalar_Au_forward_is_inherently_unstable": False,
            "temperature_grid_density_carrier_is_usable_for_exact_Au": False,
            "index_monitor_complex_diagonal_epsilon_is_complete_conformal_operator": False,
            "demonstrated_cause": (
                "The v261 GPU Index-perturbation/temperature-grid wrapper is "
                "unstable at the exact-Au endpoint even for an exact-Au base "
                "with zero endpoint perturbation. Independently, the bundled "
                "lumopt2 geometry-difference implementation discards Im(epsilon) "
                "and its wavelength-remapping helper assumes n >> k approximately 0, "
                "so that generic implementation is incompatible with lossy Au. "
                "Finally, an equal-step h=1 nm test shows that retaining complex "
                "diagonal epsilon is still incomplete for the conformal moving-Au "
                "operator: equal-step h=1 and 0.5 nm tests have the correct sign "
                "but miss the independently re-solved Maxwell derivatives by "
                "66.98% and 39.20%, while the two Maxwell FDs form a 0.0821% "
                "plateau."
            ),
        },
        "Au_10um": {
            "n": N_AU,
            "k": K_AU,
            "epsilon": [epsilon_au.real, epsilon_au.imag],
            "epsilon_magnitude": abs(epsilon_au),
            "field_amplitude_e_folding_depth_nm": amplitude_skin_depth_m * 1e9,
            "intensity_e_folding_depth_nm": intensity_skin_depth_m * 1e9,
            "semi_infinite_surface_impedance_ohm": [
                surface_impedance.real,
                surface_impedance.imag,
            ],
            "surface_resistance_ohm": surface_impedance.real,
            "interpretation": (
                "The previous 50 nm lateral / 25 nm vertical Au control mesh "
                "is not an interface-converged production mesh for this loss."
            ),
        },
        "temperature_grid_carrier_controls": carrier_controls,
        "fixed_geometry_material_AD_FD": {
            "status": fixed["status"],
            "AD": strong_fixed["official_AD_J_proxy_per_relative_epsilon"],
            "FD": strong_fixed["FD_J_proxy_per_relative_epsilon"],
            "relative_error": strong_fixed["official_relative_error"],
            "passed": fixed["passed"],
        },
        "moving_boundary": {
            "coarse_Maxwell_FD_h_nm": 50.0,
            "coarse_Maxwell_FD": fd_reference,
            "same_session_complex_dEps_steps_nm": sorted(
                {
                    float(row["step_nm"])
                    for row in [*coarse["steps"].values(), *subnm["steps"].values()]
                },
                reverse=True,
            ),
            "finest_dEps_step_nm": 0.1,
            "finest_complex_dEps_AD": finest_ad,
            "finest_sign_matches_coarse_Maxwell_FD": finest_ad * fd_reference > 0,
            "finest_relative_error_vs_coarse_Maxwell_FD": finest_vs_coarse_fd,
            "subnm_tail_change": subnm["tail_max_relative_change"],
            "index_field_max_coordinate_mismatch_m": subnm[
                "index_grid_maximum_coordinate_mismatch_m"
            ],
            "same_step_h_nm": same_step_h0p5["parameter_step"]["h_nm"],
            "same_step_Maxwell_central_FD": same_step_fd,
            "same_step_complex_dEps_AD": same_step_ad,
            "same_step_same_sign": same_step_h0p5["AD_FD_same_sign"],
            "same_step_relative_error": same_step_error,
            "same_step_gate_lt_1pct": same_step_h0p5["passed"],
            "Maxwell_FD_h1nm": fd_h1,
            "Maxwell_FD_h0p5nm": same_step_fd,
            "Maxwell_FD_h1_to_h0p5_relative_change": fd_plateau_error,
            "Maxwell_FD_local_plateau_lt_1pct": fd_plateau_error < 0.01,
            "same_step_controls": {
                "h1nm": {
                    "AD": same_step_h1[
                        "same_session_complex_dEps_AD_J_proxy_per_um"
                    ],
                    "FD": same_step_h1["Maxwell_central_FD_J_proxy_per_um"],
                    "relative_error": same_step_h1["AD_FD_relative_error"],
                    "passed": same_step_h1["passed"],
                },
                "h0p5nm": {
                    "AD": same_step_h0p5[
                        "same_session_complex_dEps_AD_J_proxy_per_um"
                    ],
                    "FD": same_step_h0p5["Maxwell_central_FD_J_proxy_per_um"],
                    "relative_error": same_step_h0p5["AD_FD_relative_error"],
                    "passed": same_step_h0p5["passed"],
                },
            },
            "resolution": (
                "The equal-step control fails. A diagonal volume d-epsilon "
                "contraction is not the complete derivative of v261's conformal "
                "moving-metal update. The exact-binary Au adjoint route is not "
                "promoted."
            ),
        },
        "v261_lumopt2_source_audit": {
            "installed_path": str(LUMOPT2_DEPS),
            "installed_source": lumopt2_audit,
            "real_only_index_square_lines": [897, 898, 899],
            "real_only_sparse_difference_lines": [956, 957, 958],
            "consequence": (
                "The generic v261 lumopt2 d_eps route discards the imaginary "
                "part of epsilon and is therefore not a valid Au-loss shape "
                "Jacobian without an independently validated replacement."
            ),
            "wavelength_remap_audit": (
                "grab_epsilon_material clips negative real epsilon to 1e-6 and "
                "fits a Cauchy refractive-index model documented in the source "
                "as n >> k approximately 0. This is also incompatible with Au."
            ),
        },
        "legacy_lumopt_complex_dEps_source_audit": {
            "installed_source": legacy_lumopt_audit,
            "interpretation": (
                "The legacy geometry dEps route retains complex index squared. "
                "The same-session control reproduces that useful part, but it "
                "still requires the equal-step Maxwell FD gate."
            ),
        },
        "resolution_routes": {
            "near_term_Lumerical_exact_binary_shape": {
                "requirements": [
                    "exact scalar dispersive Au only; no gray temperature-grid carrier",
                    "CV0/CV1 and 10/5/2.5 nm Au-interface mesh convergence on a compact control",
                    "same-step local Maxwell FD versus same-session complex dEps",
                    "few bounded spline/level-set parameters, not per-pixel density",
                    "full coupled PTE AD-FD before optimization",
                ],
                "current_state": (
                    "failed at h=1/0.5 nm (66.98%/39.20% error) despite a "
                    "0.0821% Maxwell-FD plateau; not a valid adjoint route"
                ),
            },
            "immediate_exact_binary_fallback": {
                "method": (
                    "keep Au exact and optimize only a small bounded set of shape "
                    "parameters with independent Maxwell central differences or a "
                    "derivative-free trust-region method"
                ),
                "limitations": (
                    "valid for a few parameters only; it is not a free-form pixel "
                    "topology gradient"
                ),
                "required_gates": [
                    "10/5/2.5 nm Au-interface forward mesh convergence",
                    "central-FD step plateau for every active parameter",
                    "exact-binary thermal/electrical/PTE reevaluation",
                ],
            },
            "production_freeform_metal_topology": {
                "requirements": [
                    "discrete time-domain adjoint",
                    "causal Drude or CCPR auxiliary differential equations",
                    "density interpolation of dispersive pole/residue parameters",
                    "auxiliary-state terms included in the gradient",
                    "binary endpoint cross-validation against exact-scalar Lumerical",
                ],
                "current_state": "not implemented in the v261 GPU carrier path",
            },
            "optional_surface_impedance_approximation": {
                "surface_resistance_ohm": surface_impedance.real,
                "requirements": [
                    "exact-Au versus surface-impedance forward comparison",
                    "finite-thickness and substrate correction",
                    "surface-loss Q and shape-gradient AD-FD",
                ],
                "current_state": "candidate acceleration only; not validated",
            },
        },
        "no_empirical_sign_flip_scale_or_gradient_rescaling": True,
        "no_thermal_PTE_or_optimization_executed": True,
    }

    rows = []
    merged_steps: dict[float, dict] = {}
    for source in (coarse, subnm):
        for row in source["steps"].values():
            merged_steps[float(row["step_nm"])] = row
    for step_nm in sorted(merged_steps, reverse=True):
        row = merged_steps[step_nm]
        rows.append(
            {
                "step_nm": step_nm,
                "complex_dEps_AD_J_proxy_per_um": row[
                    "same_session_complex_deps_AD_J_proxy_per_um"
                ],
                "Maxwell_FD_J_proxy_per_um": row.get("FD_J_proxy_per_um"),
                "sign_matches_available_Maxwell_FD": row.get(
                    "sign_agrees_with_Maxwell_FD"
                ),
                "relative_error_vs_available_Maxwell_FD": row.get(
                    "relative_error_vs_Maxwell_FD"
                ),
            }
        )

    summary_path = RESULTS / "au_differentiable_route_resolution_summary.json"
    csv_path = RESULTS / "au_complex_deps_step_sweep.csv"
    plot_path = RESULTS / "au_complex_deps_step_sweep.png"
    report_path = RESULTS / "AU_DIFFERENTIABLE_ROUTE_RESOLUTION_REPORT.md"
    manifest_path = RESULTS / "AU_DIFFERENTIABLE_ROUTE_RAW_ARTIFACT_MANIFEST.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    steps = np.asarray([row["step_nm"] for row in rows], float)
    derivatives = np.asarray(
        [row["complex_dEps_AD_J_proxy_per_um"] for row in rows], float
    )
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))
    for ax in axes:
        ax.axhline(
            fd_reference,
            color="black",
            linestyle="--",
            label="Maxwell FD, h=50 nm",
        )
        ax.set_xscale("log")
        ax.invert_xaxis()
        ax.set_xlabel("CAD permittivity-difference half-step (nm)")
        ax.grid(True, which="both", alpha=0.3)
        ax.scatter(
            [same_step_h1["parameter_step"]["h_nm"], same_step_h0p5["parameter_step"]["h_nm"]],
            [fd_h1, same_step_fd],
            marker="s",
            s=65,
            color="tab:red",
            zorder=5,
            label="Maxwell FD, h=1/0.5 nm",
        )
    axes[0].plot(
        steps, derivatives, "o-", label="same-session complex $d\\epsilon/dp$"
    )
    axes[0].set_ylabel("directional derivative proxy (J / um)")
    axes[0].set_title("Full step sweep")
    fine = steps <= 2.5
    axes[1].plot(
        steps[fine],
        derivatives[fine],
        "o-",
        label="same-session complex $d\\epsilon/dp$",
    )
    axes[1].set_title("FD plateaus; equal-step complex $d\\epsilon$ still fails")
    axes[0].legend()
    axes[1].legend()
    fig.suptitle("Exact-Au moving-boundary derivative: step dependence")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=220)
    plt.close(fig)

    manifest = {
        "status": STATUS,
        "raw_artifacts_committed_to_git": False,
        "artifacts": [artifact(name, path) for name, path in PATHS.items()]
        + [
            artifact("installed_v261_lumopt2_d_eps_calculator", LUMOPT2_DEPS),
            artifact("installed_v261_legacy_lumopt_geometry", LEGACY_LUMOPT_GEOMETRY),
            *[
                recorded_artifact(
                    f"same_step_{step_name}_{side}_FSP", case["project"]
                )
                for step_name, control in (
                    ("h1nm", same_step_h1),
                    ("h0p5nm", same_step_h0p5),
                )
                for side, case in control["objective_cases"].items()
            ],
            *[
                recorded_artifact(
                    f"same_step_{step_name}_{side}_{Path(str(raw['path'])).suffix.lstrip('.').upper()}",
                    raw,
                )
                for step_name, control in (
                    ("h1nm", same_step_h1),
                    ("h0p5nm", same_step_h0p5),
                )
                for side, case in control["objective_cases"].items()
                for raw in read(Path(str(case["result"]["path"]))).get(
                    "raw_artifacts", []
                )
                if Path(str(raw["path"])).suffix.lower() == ".npz"
            ],
        ],
        "generated_outputs": [
            artifact("published_summary", summary_path),
            artifact("step_sweep_csv", csv_path),
            artifact("step_sweep_plot", plot_path),
            artifact("published_report", report_path)
            if report_path.exists()
            else None,
        ],
    }
    manifest["generated_outputs"] = [
        row for row in manifest["generated_outputs"] if row is not None
    ]

    report = f"""# Au differentiable-route resolution

Status: `{STATUS}`

## What is now resolved

The failure is **not** a PML problem and is **not** repaired by a smaller
Courant factor.  The all-Metal control and the `dt=0.5` PML control both
diverged.  Global Conformal Variant 0 also diverged.  Most decisively, an
exact scalar-Au base with zero endpoint perturbation still diverged as soon as
the v261 Index-perturbation/temperature-grid wrapper was active.

The exact scalar Au forward model itself is stable, and the fixed-geometry
material derivative passed AD--FD with relative error
`{strong_fixed['official_relative_error']:.6%}`.  The blocker is therefore the
**differentiable moving-metal representation**, not Maxwell propagation through
Au in general.

## Why the boundary derivative fails

At 10 um, the frozen Ordal endpoint is

`n+ik = {N_AU}+{K_AU}i`, `epsilon = {epsilon_au.real:.2f}+{epsilon_au.imag:.2f}i`.

Its field-amplitude and intensity e-folding depths are only
`{amplitude_skin_depth_m*1e9:.2f} nm` and `{intensity_skin_depth_m*1e9:.2f} nm`.
The old `50 nm x 50 nm x 25 nm` Au control mesh is therefore not an
interface-converged production mesh.

The same-session complex diagonal-epsilon derivative changes sign only below
about 1 nm.  At the finest tested 0.1 nm CAD step it has the correct sign
relative to the older 50 nm Maxwell FD, but still differs by
`{finest_vs_coarse_fd:.3%}` and its sub-nm tail changes by
`{float(subnm['tail_max_relative_change']):.3%}`.  The coordinate mismatch is
only `{float(subnm['index_grid_maximum_coordinate_mismatch_m']):.3e} m`, so an
ordinary coordinate pairing error is excluded.

The decisive equal-step tests are now complete.  The independently re-solved
Maxwell central FD is `{fd_h1:.6e} J/um` at `h=1 nm` and
`{same_step_fd:.6e} J/um` at `h=0.5 nm`; their relative change is only
`{fd_plateau_error:.4%}`.  The Maxwell local derivative is therefore on a
sub-1% step plateau.  By contrast, the matching complex diagonal-epsilon
contractions miss those FDs by
`{float(same_step_h1['AD_FD_relative_error']):.3%}` and
`{same_step_error:.3%}`, respectively.  Both signs agree, but both magnitude
gates fail.  Comparing unlike parameter steps and an unconverged Maxwell FD
are therefore excluded.  The diagonal volume `d epsilon` term is not the
complete derivative of the conformal moving-metal operator.

Independently, the installed v261
`lumopt2` implementation explicitly applies `real(index_c**2)` and later takes
the real part of the sparse difference.  Its wavelength-remapping helper also
clips negative real epsilon and fits a lossless Cauchy model under the source
assumption `n >> k approximately 0`.  That generic path discards Au loss and
cannot be promoted for Au.

The legacy bundled `lumopt` geometry path, unlike `lumopt2`, retains complex
`index_c**2`.  The same-session control deliberately reproduces that part of
the legacy contract.  It is not promoted merely because the source looks
better: the independent equal-step Maxwell FD remains the deciding numerical
test.

This is consistent with the documented scope rather than a hidden PML setting.
Ansys describes `FunctionDefinedPolygon` as using a *shape derivative
approximation* and documents `eps_in`/`eps_out` as scalar permittivities; the
official examples are ordinary dielectrics, not high-loss Au.  Ansys also
states that Precise Volume Average evaluates dispersive materials at one mesh
frequency and makes the averaged cell non-dispersive.  That mesh operation is
useful for forward geometry sensitivity, but it is not a causal dispersive
material Jacobian with auxiliary states.

The Ansys page discussing an `enable conformal meshing` property explicitly
says its tips do not apply to np-density and Temperature attributes.  The
installed v261 Temperature object was queried directly and exposes no such
property.  Therefore that switch is not an available repair for this carrier;
global CV0 was tested separately and still diverged.

## Resolution, not a rescaling

There are three physically defensible routes:

1. **Immediate few-parameter exact-binary route.** Keep exact scalar Au and
   use independently re-solved central differences (or a derivative-free
   trust-region method) for only a small number of bounded geometric
   parameters.  First converge 10/5/2.5 nm Au-interface meshes.  This is a
   practical exact-Au fallback, but it is not free-form topology optimization.
   The current lumopt2 documentation explicitly supports gradient-free SciPy
   methods and states that the adjoint solve is skipped for those methods.
2. **Fixed-Au coupled inverse design.** Treat the electrode geometry as fixed;
   the validated fixed-geometry material/field chain can then be coupled to
   TaIrTe4/dielectric design variables without differentiating a moving Au
   boundary.  The complete coupled PTE AD--FD must still pass.
3. **Production free-form metal topology route.** Use a discrete dispersive
   FDTD adjoint with Drude/CCPR auxiliary states.  Density must interpolate
   causal dispersive parameters, and the auxiliary-state gradient terms must
   be included.  Exact binary endpoints must be cross-validated against
   Lumerical.  This is the route demonstrated in the plasmonic inverse-design
   literature; it is not provided by the tested v261 GPU carrier and must be
   implemented in a solver that exposes its discrete update equations.

A PEC/surface-impedance model is only an optional reduced approximation.  The
semi-infinite estimate is `Rs = {surface_impedance.real:.4f} ohm`, but it needs
finite-thickness, substrate, forward-field and shape-gradient validation before
use.

No sign flip, empirical normalization, or gradient scaling was used.  No
thermal, PTE, or optimization stage was run.

## Primary references

- [Ansys conformal-mesh selection](https://optics.ansys.com/hc/en-us/articles/360034382614-Selecting-the-best-mesh-refinement-option-in-the-FDTD-simulation-object)
- [Ansys grid-attribute limitations](https://optics.ansys.com/hc/en-us/articles/360034915193-Tips-and-background-information-when-using-grid-attributes)
- [Ansys GPU material limitations](https://optics.ansys.com/hc/en-us/articles/17518942465811-Getting-started-with-running-FDTD-on-GPU)
- [Ansys optimizable-geometry d-epsilon contract](https://optics.ansys.com/hc/en-us/articles/360052044913-Optimizable-Geometry-Python-API)
- [Ansys lumopt2 optimization and gradient-free fallback](https://lumerical.docs.pyansys.com/version/dev/user_guide/lumopt2/optimization_session.html)
- [Zeng et al., discrete plasmonic FDTD adjoint](https://arxiv.org/abs/2007.11442)
- [Hassan and Calà Lesina, dispersive Drude-ADE topology](https://arxiv.org/abs/2203.01462)
- [Gedeon et al., CCPR-ADE power-dissipation topology](https://arxiv.org/abs/2407.05994)
"""
    report_path.write_text(report)
    manifest["generated_outputs"] = [
        artifact("published_summary", summary_path),
        artifact("step_sweep_csv", csv_path),
        artifact("step_sweep_plot", plot_path),
        artifact("published_report", report_path),
    ]
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
