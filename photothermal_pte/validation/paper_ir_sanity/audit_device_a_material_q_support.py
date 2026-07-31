#!/usr/bin/env python3
"""Reclassify immutable Device-A Q by explicit lateral/z material support."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path as PolygonPath
import numpy as np

from photothermal_pte.validation.paper_ir_sanity import (
    run_lumerical_device_a_ir_q as runner,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    artifact_path = args.case_dir / "finite_q_on_artifact.npz"
    result_path = args.case_dir / "case_result.json"
    result = json.loads(result_path.read_text())
    if result.get("status") != "COMPLETED":
        raise RuntimeError("material-Q audit requires a completed optical case")
    geometry = result["pre_run_contract"]["geometry"]
    electrode = geometry["electrode_material_contract"]
    with np.load(artifact_path, allow_pickle=False) as raw:
        x = np.asarray(raw["x_m"], float)
        y = np.asarray(raw["y_m"], float)
        z = np.asarray(raw["z_m"], float)
        q = np.asarray(raw["Q_on_W_m3"], float)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    points = np.column_stack((xx.ravel(), yy.ravel()))

    def xy_mask(vertices_um: list[list[float]]) -> np.ndarray:
        return PolygonPath(np.asarray(vertices_um, float) * 1e-6).contains_points(
            points, radius=1e-15
        ).reshape(xx.shape)

    flake_xy = xy_mask(geometry["flake_vertices_um"])
    metal_xy = xy_mask(electrode["top_polygon_simulation_um"]) | xy_mask(
        electrode["bottom_polygon_simulation_um"]
    )
    at_flake_bottom = np.isclose(
        z, -runner.FLAKE_THICKNESS_M, rtol=0.0, atol=1e-15
    )
    at_flake_top = np.isclose(z, 0.0, rtol=0.0, atol=1e-15)
    flake_z = (
        ((z >= -runner.FLAKE_THICKNESS_M) | at_flake_bottom)
        & ((z <= 0.0) | at_flake_top)
    )
    flake = flake_xy[:, :, None] & flake_z[None, None, :]
    at_ti_top = np.isclose(z, runner.TI_THICKNESS_M, rtol=0.0, atol=1e-15)
    ti = metal_xy[:, :, None] & (
        ((z[None, None, :] >= 0.0) | at_flake_top[None, None, :])
        & (
            (z[None, None, :] <= runner.TI_THICKNESS_M)
            | at_ti_top[None, None, :]
        )
    )
    ti &= ~flake
    at_au_top = np.isclose(
        z,
        runner.TI_THICKNESS_M + runner.AU_THICKNESS_M,
        rtol=0.0,
        atol=1e-15,
    )
    au = metal_xy[:, :, None] & (
        (z[None, None, :] > runner.TI_THICKNESS_M)
        & ~at_ti_top[None, None, :]
        & (
            (z[None, None, :] <= runner.TI_THICKNESS_M + runner.AU_THICKNESS_M)
            | at_au_top[None, None, :]
        )
    )
    assigned = flake | ti | au

    def power(mask: np.ndarray) -> float:
        return float(runner.integrate_xyz_bounded(
            q * mask,
            {"x": x, "y": y, "z": z},
            geometry["pabs_nominal_control_volume_bounds_m"],
        ))

    all_cells = np.ones_like(assigned, dtype=bool)
    common_grid_total = power(all_cells)
    powers = {
        "TaIrTe4_W": power(flake),
        "Ti_W": power(ti),
        "Au_W": power(au),
        "conformal_interface_ambiguous_W": power(~assigned),
    }
    powers["sum_W"] = sum(powers.values())
    native_total = float(result["run_result"]["P_Q_W"])
    powers["common_grid_total_W"] = common_grid_total
    powers["native_component_total_P_Q_W"] = native_total
    powers["signed_partition_residual_W"] = common_grid_total - powers["sum_W"]
    powers["relative_partition_residual"] = (
        abs(powers["signed_partition_residual_W"]) / abs(common_grid_total)
    )
    powers["signed_common_minus_native_W"] = common_grid_total - native_total
    powers["relative_common_native_difference"] = (
        abs(powers["signed_common_minus_native_W"]) / abs(native_total)
    )
    powers["ambiguous_fraction_of_common_grid_power"] = (
        powers["conformal_interface_ambiguous_W"] / common_grid_total
    )

    ambiguous_positive = (~assigned) & (q > 0.0)
    if np.any(ambiguous_positive):
        ix, iy, iz = np.nonzero(ambiguous_positive)
        ambiguous_bounds = {
            "x_m": [float(np.min(x[ix])), float(np.max(x[ix]))],
            "y_m": [float(np.min(y[iy])), float(np.max(y[iy]))],
            "z_m": [float(np.min(z[iz])), float(np.max(z[iz]))],
            "positive_sample_count": int(ix.size),
        }
    else:
        ambiguous_bounds = {
            "x_m": None,
            "y_m": None,
            "z_m": None,
            "positive_sample_count": 0,
        }
    z_only = np.ones((1, 1, z.size), dtype=bool)
    z_layer_power = []
    for index, z_value in enumerate(z):
        layer = np.zeros_like(z_only)
        layer[0, 0, index] = True
        layer_mask = np.broadcast_to(layer, q.shape)
        layer_power = power(layer_mask)
        ambiguous_layer_power = power(layer_mask & ~assigned)
        if layer_power != 0.0 or ambiguous_layer_power != 0.0:
            z_layer_power.append(
                {
                    "z_m": float(z_value),
                    "total_W": layer_power,
                    "ambiguous_W": ambiguous_layer_power,
                }
            )
    incident_unit = float(
        result["run_result"]["normalization"]["incident_power_W_at_1_W_m2"]
    )
    physical_scale = 285.0e-6 / incident_unit
    physical = {
        key: value * physical_scale
        for key, value in powers.items()
        if key.endswith("_W")
    }
    has_ambiguity = powers["ambiguous_fraction_of_common_grid_power"] > 1e-6
    payload = {
        "status": (
            (
                "DEVICE_A_MATERIAL_Q_SUPPORT_AUDITED_WITH_"
                "CONFORMAL_INTERFACE_AMBIGUITY"
                if has_ambiguity
                else "DEVICE_A_MATERIAL_Q_SUPPORT_AUDITED"
            )
            if powers["relative_partition_residual"] < 5e-12
            else "FAILED_DEVICE_A_MATERIAL_Q_SUPPORT_CLOSURE"
        ),
        "input": {
            "artifact_path": str(artifact_path.resolve()),
            "artifact_size_bytes": artifact_path.stat().st_size,
            "artifact_sha256": sha256(artifact_path),
            "case_result_path": str(result_path.resolve()),
            "case_result_sha256": sha256(result_path),
        },
        "support_contract": (
            "analytic material supports are reported separately; common-grid "
            "Q outside those exact supports is retained as conformal-interface "
            "ambiguity and is not silently assigned, clipped, smoothed, gained, "
            "rescaled, or deleted"
        ),
        "ambiguity": {
            "interpretation": (
                "Yee/conformal interface samples cannot be assigned to a bulk "
                "thermal material from array indices alone; index-detail or a "
                "declared conservative interface rule is required before full "
                "metal-inclusive thermal propagation"
            ),
            "positive_Q_bounds": ambiguous_bounds,
            "nonzero_z_layer_power": z_layer_power,
        },
        "power_at_unit_central_intensity_W": powers,
        "incident_power_normalization": {
            "target_incident_power_W": 285e-6,
            "incident_power_at_unit_central_intensity_W": incident_unit,
            "single_common_linear_scale": physical_scale,
            "polarization_dependent_Q_matching": False,
        },
        "power_at_285uW_incident_W": physical,
        "Q_modified": False,
    }
    (args.output_dir / "material_q_support_audit.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    areal = {
        "TaIrTe4": np.trapezoid(q * flake, z, axis=2),
        "Ti": np.trapezoid(q * ti, z, axis=2),
        "Au": np.trapezoid(q * au, z, axis=2),
        "Conformal/interface ambiguous": np.trapezoid(q * ~assigned, z, axis=2),
    }
    common_vmax = max(float(np.max(values)) for values in areal.values())
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.5), constrained_layout=True)
    extent = [x[0] * 1e6, x[-1] * 1e6, y[0] * 1e6, y[-1] * 1e6]
    for ax, (name, values) in zip(axes, areal.items()):
        image = ax.imshow(
            values.T,
            origin="lower",
            extent=extent,
            cmap="inferno",
            vmin=0.0,
            vmax=common_vmax,
            aspect="equal",
        )
        ax.set(title=f"{name} depth-integrated Q", xlabel="x=b (um)", ylabel="y=a (um)")
        fig.colorbar(image, ax=ax, label="W/m2")
    fig.savefig(args.output_dir / "device_a_material_q_maps.png", dpi=200)
    plt.close(fig)
    print(json.dumps(payload["power_at_unit_central_intensity_W"], indent=2))
    return 0 if payload["status"].startswith("DEVICE_A") else 2


if __name__ == "__main__":
    raise SystemExit(main())
