#!/usr/bin/env python3
"""Audit the digitized Device-A weighting potential without thermal/PTE solves."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from photothermal_pte.validation.paper_ir_sanity import (
    run_device_a_explicit_thermal_pte as thermal,
)
from photothermal_pte.validation.paper_ir_sanity.run_lumerical_device_a_ir_q import (
    load_digitized_device_a_contract,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def configure_geometry(contract_path: Path, thermal_domain_um: float) -> dict:
    contract = load_digitized_device_a_contract(
        contract_path,
        # Preserve the coordinate translation frozen by the optical setup.
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
    if np.max(np.abs(thermal.FLAKE_VERTICES_UM)) >= 0.5 * thermal_domain_um:
        raise ValueError(
            "thermal domain does not contain the frozen translated "
            "Device-A flake polygon"
        )
    return contract


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry-contract-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--thermal-domain-um", type=float, default=48.0)
    parser.add_argument("--si-depth-um", type=float, default=20.0)
    parser.add_argument("--core-step-nm", type=float, default=100.0)
    parser.add_argument("--flake-dz-nm", type=float, default=10.0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)

    contract = configure_geometry(
        args.geometry_contract_json,
        args.thermal_domain_um,
    )
    geometry = thermal.build_geometry(
        domain_m=args.thermal_domain_um * 1e-6,
        si_depth_m=args.si_depth_um * 1e-6,
        core_step_m=args.core_step_nm * 1e-9,
        flake_dz_m=args.flake_dz_nm * 1e-9,
    )
    flake_xy = np.any(geometry.flake_mask, axis=2)
    psi, grad_x, grad_y, diagnostics = thermal.solve_weighting_potential(
        geometry.x_edges_m,
        geometry.y_edges_m,
        flake_xy,
    )
    magnitude = np.hypot(grad_x, grad_y)
    finite_psi = psi[flake_xy]
    finite_gradient = magnitude[flake_xy]
    gate = {
        "nonempty_top_contact": diagnostics["top_contact_cells"] > 0,
        "nonempty_bottom_contact": diagnostics["bottom_contact_cells"] > 0,
        "psi_reaches_collecting_contact": float(np.max(finite_psi)) > 0.95,
        "psi_reaches_opposite_contact": float(np.min(finite_psi)) < 0.05,
        "linear_residual_lt_1e-8": diagnostics["linear_residual_relative"] < 1e-8,
        "finite_fields": bool(
            np.all(np.isfinite(finite_psi))
            and np.all(np.isfinite(finite_gradient))
        ),
    }
    payload = {
        "status": (
            "VALIDATED_DEVICE_A_WEIGHTING_POTENTIAL_AUDIT"
            if all(gate.values())
            else "FAILED_DEVICE_A_WEIGHTING_POTENTIAL_AUDIT"
        ),
        "model": (
            "paper-SI Eq. S7 finite-volume weighting potential on the frozen "
            "Figure-2 digitized flake/contact geometry; psi=1 on the top "
            "collecting segment, psi=0 on the opposite segment, and zero "
            "normal flux on all remaining flake-air edges"
        ),
        "geometry_provenance": {
            "contract_path": str(args.geometry_contract_json.resolve()),
            "contract_sha256": sha256(args.geometry_contract_json),
            "status": "FIGURE_DIGITIZED_APPROXIMATION_NOT_CAD",
            "axis_mapping": "code x=b, code y=a",
            "thermal_domain_um": args.thermal_domain_um,
            "core_step_nm": args.core_step_nm,
            "flake_dz_nm": args.flake_dz_nm,
            "grid_shape_3d": list(geometry.material_id.shape),
            "flake_xy_cell_count": int(np.count_nonzero(flake_xy)),
            "simulation_origin_shift_um": np.asarray(
                contract["simulation_origin_shift_um"], float
            ).tolist(),
            "top_contact_segment_um": thermal.TOP_CONTACT_SEGMENT_UM.tolist(),
            "bottom_contact_segment_um": thermal.BOTTOM_CONTACT_SEGMENT_UM.tolist(),
        },
        "diagnostics": {
            **diagnostics,
            "psi_min": float(np.min(finite_psi)),
            "psi_max": float(np.max(finite_psi)),
            "weighting_gradient_max_m_inv": float(np.max(finite_gradient)),
            "weighting_gradient_p99_m_inv": float(
                np.percentile(finite_gradient, 99.0)
            ),
        },
        "gates": gate,
        "qualitative_Fig2G_comparison": (
            "geometry and streamline topology can be compared qualitatively; "
            "pixelwise certification is impossible without device CAD and the "
            "raw paper weighting-potential field"
        ),
        "thermal_run": False,
        "PTE_run": False,
        "adjoint_run": False,
        "optimization_run": False,
    }
    (args.output_dir / "device_a_weighting_potential_audit.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )

    x = 0.5 * (geometry.x_edges_m[:-1] + geometry.x_edges_m[1:]) * 1e6
    y = 0.5 * (geometry.y_edges_m[:-1] + geometry.y_edges_m[1:]) * 1e6
    masked_psi = np.ma.masked_where(~flake_xy, psi)
    masked_gradient = np.ma.masked_where(~flake_xy, magnitude)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), constrained_layout=True)
    im0 = axes[0].pcolormesh(
        geometry.x_edges_m * 1e6,
        geometry.y_edges_m * 1e6,
        masked_psi.T,
        shading="flat",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
    )
    fig.colorbar(im0, ax=axes[0], label="weighting potential psi")
    im1 = axes[1].pcolormesh(
        geometry.x_edges_m * 1e6,
        geometry.y_edges_m * 1e6,
        masked_gradient.T,
        shading="flat",
        cmap="magma",
    )
    fig.colorbar(im1, ax=axes[1], label="|grad psi| [1/m]")
    stride = max(1, int(round(1.0e3 / args.core_step_nm)))
    sampled = np.s_[::stride, ::stride]
    valid = flake_xy[sampled]
    qx = np.where(valid, grad_x[sampled], np.nan)
    qy = np.where(valid, grad_y[sampled], np.nan)
    axes[1].quiver(
        x[::stride],
        y[::stride],
        qx.T,
        qy.T,
        color="white",
        alpha=0.7,
        scale=None,
        width=0.0025,
    )
    for ax in axes:
        ax.plot(
            *np.vstack((thermal.FLAKE_VERTICES_UM, thermal.FLAKE_VERTICES_UM[0])).T,
            color="white",
            linewidth=1.2,
        )
        ax.plot(*thermal.TOP_CONTACT_SEGMENT_UM.T, color="cyan", linewidth=4)
        ax.plot(*thermal.BOTTOM_CONTACT_SEGMENT_UM.T, color="lime", linewidth=4)
        ax.set(
            xlabel="code x = crystal b [um]",
            ylabel="code y = crystal a [um]",
            aspect="equal",
        )
    axes[0].set_title("Device A weighting potential")
    axes[1].set_title("Weighting-field magnitude and direction")
    fig.savefig(args.output_dir / "DEVICE_A_WEIGHTING_POTENTIAL_AUDIT.png", dpi=220)
    plt.close(fig)
    print(json.dumps({"status": payload["status"], **diagnostics}, indent=2))
    return 0 if all(gate.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
