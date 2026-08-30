#!/usr/bin/env python3
"""Freeze the finite T/Z Maxwell -> thermal -> electrical/PTE contract.

This file performs no solver call.  It makes the geometry, axes, boundaries,
material scenarios, source aperture and terminal definitions reviewable before
the expensive finite Maxwell jobs are launched.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[3]
OUTPUT = HERE / "results_finite_T_Z_thermal_electrical_contract"
GEOMETRY_FILE = HERE / "05_actual_metasurface_geometry.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _architecture_contracts() -> dict[str, dict[str, object]]:
    geometry = _load(GEOMETRY_FILE, "finite_t_z_contract_geometry")
    t = geometry.inverse_t_mir_4750nm()
    z = geometry.z_m2_5300nm_centered_expanded_supercell_v4("LH")
    return {
        "T": {
            "wavelength_um": 4.75,
            "Lumerical_source_object_w0_um": 3.9444817826057172,
            "top_Au_thickness_nm": 33.0,
            "Al2O3_thickness_nm": 35.0,
            "Au_mirror_thickness_nm": 200.0,
            "SiO2_thickness_nm": 1500.0,
            "optical_z_bounds_um": [-2.5, 1.5],
            "loss_control_z_bounds_um": [-2.0, 0.60],
            "geometry": t.as_dict(),
            "identity": (
                "2024 inverse-T figure-digitized project scenario; 100-nm "
                "TaIrTe4 substitution; not exact author CAD"
            ),
        },
        "Z": {
            "wavelength_um": 5.30,
            "Lumerical_source_object_w0_um": 3.936280659072623,
            "top_Au_thickness_nm": 50.0,
            "Al2O3_thickness_nm": 200.0,
            "Au_mirror_thickness_nm": 200.0,
            "SiO2_thickness_nm": 285.0,
            "optical_z_bounds_um": [-1.8, 1.5],
            "loss_control_z_bounds_um": [-1.30, 0.60],
            "geometry": z.as_dict(),
            "identity": (
                "2022 M2 centered-expanded v4 project supercell scenario; "
                "published scalar dimensions but not exact author CAD"
            ),
        },
    }


def _contract() -> dict[str, object]:
    w0_um = 4.0
    aperture_half_um = 9.0
    flake_half_um = 10.0
    pml_half_um = 12.0
    return {
        "status": "FROZEN_FINITE_T_Z_THERMAL_ELECTRICAL_CONTRACT",
        "scope": (
            "finite nonperiodic Maxwell Q followed by explicit 3-D thermal and "
            "two-terminal electrical/PTE forwards; no adjoint or optimization"
        ),
        "axes": {
            "Lumerical_x": "TaIrTe4 b",
            "Lumerical_y": "TaIrTe4 a",
            "Lumerical_z": "TaIrTe4 c with epsilon_c=epsilon_b optical closure",
        },
        "architectures": _architecture_contracts(),
        "optical": {
            "finite_TaIrTe4_flake_um": [20.0, 20.0, 0.10],
            "FDTD_lateral_span_um": [24.0, 24.0],
            "boundaries": {
                face: "PML"
                for face in ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")
            },
            "PML_layers": 24,
            "source": {
                "type": "scalar Gaussian; waist-size-and-position",
                "propagation": "-z; normal incidence",
                "physical_target_waist_um": w0_um,
                "source_span_um": [18.0, 18.0],
                "source_z_um": 0.5,
                "waist_plane_z_um": 0.10,
                "polarizations_for_finite_thermal": ["E||a (Ey)", "E||b (Ex)"],
                "incident_power_normalization": (
                    "literal Lumerical source power first; optional 285-uW linear "
                    "response reported separately, never by changing raw Q"
                ),
            },
            "analytic_intensity_ratios": {
                "square_aperture_edge_over_peak": float(
                    np.exp(-2.0 * (aperture_half_um / w0_um) ** 2)
                ),
                "flake_edge_over_peak": float(
                    np.exp(-2.0 * (flake_half_um / w0_um) ** 2)
                ),
                "nearest_PML_over_peak": float(
                    np.exp(-2.0 * (pml_half_um / w0_um) ** 2)
                ),
            },
            "mesh": {
                "auto_nonuniform": True,
                "conformal": "conformal variant 1",
                "accuracy": 3,
                "flake_and_Q_region_dx_dy_nm": [100.0, 100.0],
                "top_Au_edge_local_dx_dy_nm": [25.0, 25.0],
                "structure_dz_nm": 5.0,
                "far_xy_nominal_override_nm": 250.0,
            },
            "gates": {
                "source_realized_waist_relative": 0.005,
                "Gaussian_fit_NRMSE": 0.005,
                "realized_xy_ellipticity": 0.01,
                "source_center_displacement_nm": 50.0,
                "auto_shutoff": 1.0e-5,
                "six_face_closure_relative": 0.005,
                "Q_mapping_relative": 0.005,
            },
            "ellipticity_note": (
                "The finite beam has lambda/w0 near unity.  The source-object "
                "profile is circular, while Maxwell transversality produces a "
                "small polarization-oriented target-plane ellipticity and a "
                "nonzero Ez component.  This realized effect is reported, not "
                "numerically symmetrized."
            ),
            "forbidden": [
                "periodic tiling", "old-Q cropping", "Q clipping", "Q smoothing",
                "gain", "global Q rescaling", "source deletion",
            ],
        },
        "thermal": {
            "domain_lateral_um": [32.0, 32.0],
            "Si_depth_um": 20.0,
            "core_xy_cell_nm": 100.0,
            "TaIrTe4_dz_nm": 10.0,
            "boundaries": {
                "x_min_x_max": "DeltaT=0 at far numerical bath",
                "y_min_y_max": "DeltaT=0 at far numerical bath",
                "z_min": "DeltaT=0 at bottom Si bath",
                "z_max": "exposed convection h=10 W/(m2 K) to DeltaT=0 bath",
            },
            "materials_W_mK": {
                "TaIrTe4_xyz_xb_ya_z": [3.8, 14.4, 1.0],
                "Au_bulk_reference": 317.0,
                "Al2O3_numerical_scenario": 30.0,
                "SiO2": 1.38,
                "Si": 145.0,
                "air": 0.026,
            },
            "interfaces_W_m2K": {
                "SiO2_Si": 1.1e9,
                "Au_TaIrTe4_numerical_analogue": 1.0 / 5.8e-8,
                "TaIrTe4_air": 1.0,
                "TaIrTe4_Al2O3_sensitivity": [7.37e4, 7.37e6, "perfect_contact"],
            },
            "interface_note": (
                "Au/TaIrTe4 and TaIrTe4/Al2O3 are named numerical scenarios, not "
                "measured TaIrTe4 interface data"
            ),
            "gates": {"linear_residual": 1.0e-8, "energy_balance_relative": 0.01},
        },
        "electrical": {
            "measurement_terminals": {
                "top_bottom": "TaIrTe4 y=-10 um: psi=0; y=+10 um: psi=1",
                "left_right": "TaIrTe4 x=-10 um: psi=0; x=+10 um: psi=1",
            },
            "top_Au_role": (
                "floating nanostructure with finite vertical Au/TaIrTe4 contact; "
                "not a measurement terminal"
            ),
            "TaIrTe4_sigma_xy_S_m": [1.10e5, 4.91e5],
            "TaIrTe4_Seebeck_xy_V_K": [27.0e-6, -6.0e-6],
            "Au_sigma_bulk_reference_S_m": 1.0 / 2.43e-8,
            "Au_Seebeck_control_V_K": 0.0,
            "Au_TaIrTe4_contact_S_m2": [1.0e8, 1.0e10, 1.0e12],
            "contact_note": "named numerical scenarios; no device-specific contact data",
            "outputs": [
                "weighting potential psi", "Ew_x=-dpsi/dx", "Ew_y=-dpsi/dy",
                "short-circuit V", "J_Ta_x", "J_Ta_y", "J_Au_x", "J_Au_y",
                "local thermoelectric source", "current-integrand x/y components",
                "low/high terminal currents", "terminal balance",
            ],
        },
        "Au_effect_decomposition": [
            "full Au-on minus matched Au-off",
            "optical-only: Au-on Q with Au-disabled thermal/electrical operator",
            "thermal-only: matched Q with Au enabled/disabled as heat path",
            "electrical-only: same T with floating Au contact enabled/disabled",
        ],
    }


def _draw_xy(ax, arch: dict[str, object], title: str) -> None:
    ax.add_patch(Rectangle((-12, -12), 24, 24, fill=False, lw=5, ec="#78298e", label="PML boundary"))
    ax.add_patch(Rectangle((-10, -10), 20, 20, fc="#e9b8b8", ec="#bd6868", alpha=0.55, label="finite TaIrTe4"))
    for polygon in arch["geometry"]["polygons"]:
        vertices = np.asarray(polygon["vertices_nm"], float) * 1e-3
        ax.fill(vertices[:, 0], vertices[:, 1], fc="#f7bd3e", ec="#825900", lw=2, label="top Au")
    circle = plt.Circle((0, 0), 4, fill=False, ls="--", lw=2, ec="#2878b5", label="$w_0=4$ um")
    ax.add_patch(circle)
    ax.add_patch(Rectangle((-9, -9), 18, 18, fill=False, ls=":", lw=2, ec="#3a9c64", label="source aperture"))
    ax.set(xlim=(-12.5, 12.5), ylim=(-12.5, 12.5), xlabel="x=b (um)", ylabel="y=a (um)", title=title)
    ax.set_aspect("equal")


def _draw_xz(ax, arch: dict[str, object], title: str) -> None:
    al = float(arch["Al2O3_thickness_nm"]) * 1e-3
    mirror = float(arch["Au_mirror_thickness_nm"]) * 1e-3
    oxide = float(arch["SiO2_thickness_nm"]) * 1e-3
    top = float(arch["top_Au_thickness_nm"]) * 1e-3
    z0 = -al - mirror - oxide
    ax.add_patch(Rectangle((-12, z0 - 0.5), 24, 0.5, fc="#6c85aa", label="Si"))
    ax.add_patch(Rectangle((-12, z0), 24, oxide, fc="#9fd8d2", label="SiO2"))
    ax.add_patch(Rectangle((-12, -al - mirror), 24, mirror, fc="#ba8700", label="Au mirror"))
    ax.add_patch(Rectangle((-12, -al), 24, al, fc="#b8dcff", label="Al2O3"))
    ax.add_patch(Rectangle((-10, 0), 20, 0.1, fc="#d96b6b", label="TaIrTe4"))
    ax.add_patch(Rectangle((-0.7, 0.1), 1.4, top, fc="#f7bd3e", label="top Au (section)"))
    ax.annotate("Gaussian -z", xy=(0, 0.25), xytext=(0, 0.65), ha="center", arrowprops={"arrowstyle": "->", "lw": 2, "color": "#2878b5"}, color="#2878b5")
    ax.axvline(-12, color="#78298e", lw=4)
    ax.axvline(12, color="#78298e", lw=4)
    ax.set(xlim=(-12.5, 12.5), ylim=(z0 - 0.5, 1.5), xlabel="x=b (um)", ylabel="z (um)", title=title)


def _draw_terminals(ax, orientation: str) -> None:
    ax.add_patch(Rectangle((-10, -10), 20, 20, fc="#e9b8b8", ec="#855", alpha=0.55))
    if orientation == "top-bottom":
        ax.add_patch(Rectangle((-10, 9.4), 20, 0.6, fc="#dda300"))
        ax.add_patch(Rectangle((-10, -10), 20, 0.6, fc="#826800"))
        ax.arrow(0, -6, 0, 12, width=0.12, color="#3e339b", length_includes_head=True)
    else:
        ax.add_patch(Rectangle((9.4, -10), 0.6, 20, fc="#dda300"))
        ax.add_patch(Rectangle((-10, -10), 0.6, 20, fc="#826800"))
        ax.arrow(-6, 0, 12, 0, width=0.12, color="#3e339b", length_includes_head=True)
    ax.text(0.5, 0.5, r"$\mathbf{E}_w=-\nabla\psi$", transform=ax.transAxes, ha="center", fontsize=13, color="#3e339b")
    ax.set(xlim=(-10.5, 10.5), ylim=(-10.5, 10.5), xlabel="x=b (um)", ylabel="y=a (um)", title=orientation)
    ax.set_aspect("equal")


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    contract = _contract()
    summary = OUTPUT / "FINITE_T_Z_THERMAL_ELECTRICAL_CONTRACT.json"
    summary.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(2, 2, figsize=(13, 11), constrained_layout=True)
    for row, key in enumerate(("T", "Z")):
        arch = contract["architectures"][key]
        _draw_xy(axes[row, 0], arch, f"finite {key}: Maxwell x-y")
        _draw_xz(axes[row, 1], arch, f"finite {key}: Maxwell x-z (schematic)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    unique = dict(zip(labels, handles, strict=False))
    fig.legend(unique.values(), unique.keys(), loc="outside upper center", ncol=5)
    optical_plot = OUTPUT / "finite_T_Z_optical_geometry_xy_xz.png"
    fig.savefig(optical_plot, dpi=190)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), constrained_layout=True)
    _draw_terminals(axes[0], "top-bottom")
    _draw_terminals(axes[1], "left-right")
    fig.suptitle("Finite electrical domains; top Au is floating, not a terminal")
    terminal_plot = OUTPUT / "finite_top_bottom_left_right_terminal_contract.png"
    fig.savefig(terminal_plot, dpi=190)
    plt.close(fig)

    report = OUTPUT / "FINITE_T_Z_THERMAL_ELECTRICAL_CONTRACT.md"
    report.write_text(
        """# Finite T/Z thermal-electrical contract

Status: `FROZEN_FINITE_T_Z_THERMAL_ELECTRICAL_CONTRACT`

The preceding periodic calculations certify optical per-cell Q only. They are
not tiled, cropped, or interpreted as periodic temperature/PTE. Each T/Z
antenna is now placed once at the center of a finite 20 x 20 um TaIrTe4 flake
and is illuminated by a finite scalar Gaussian.

The Maxwell domain uses six PML boundaries. The thermal domain is a separate
explicit 3-D 32 x 32 um domain with 20-um Si depth, fixed far x/y and bottom
bath boundaries, and top convection. Optical PML is never used as a thermal or
electrical boundary.

Both top-bottom and left-right TaIrTe4 terminal pairs are solved. The top Au T/Z
structure is electrically floating and may alter the weighting field through a
finite vertical contact, but it is not a readout electrode.

Au/TaIrTe4 thermal and electrical contacts and TaIrTe4/Al2O3 thermal contact are
explicitly named numerical scenarios because device-specific measured values
are unavailable. No single scenario is promoted as experimental truth.
""",
        encoding="utf-8",
    )
    manifest = {
        "status": contract["status"],
        "published": [
            {"path": str(path), "size_bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in (summary, report, optical_plot, terminal_plot)
        ],
    }
    (OUTPUT / "ARTIFACT_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": contract["status"], "output": str(OUTPUT)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
