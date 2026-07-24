"""FDTD-only metadata and epsilon-slice helpers for Pabs diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


MATERIAL_ID_LEGEND = {
    0: "air_or_background",
    1: "SiO2",
    2: "TaIrTe4",
    3: "design_n4",
}


def safe_getnamed(fdtd: Any, object_name: str, property_name: str) -> Any:
    try:
        value = fdtd.getnamed(object_name, property_name)
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        return value
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def geometry_material_id(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    geometry: str,
    disk_radius_m: float,
    square_half_width_m: float,
    design_gap_m: float,
    design_height_m: float,
) -> np.ndarray:
    """Return an analytic geometry ID array on arbitrary broadcastable grids."""
    xx, yy, zz = np.broadcast_arrays(
        np.asarray(x, float), np.asarray(y, float), np.asarray(z, float)
    )
    result = np.zeros(xx.shape, dtype=np.uint8)
    result[zz < -100e-9] = 1
    result[(zz >= -100e-9) & (zz <= 0.0)] = 2
    in_design_z = (
        (zz > design_gap_m)
        & (zz <= design_gap_m + design_height_m)
    )
    if geometry in {"centered-disk", "centered-disk-gap"}:
        in_design_xy = xx**2 + yy**2 <= disk_radius_m**2
    elif geometry == "centered-square":
        in_design_xy = (
            (np.abs(xx) <= square_half_width_m)
            & (np.abs(yy) <= square_half_width_m)
        )
    elif geometry == "full-film":
        in_design_xy = np.ones(xx.shape, dtype=bool)
    elif geometry == "none":
        in_design_xy = np.zeros(xx.shape, dtype=bool)
    else:
        raise ValueError(f"unknown geometry {geometry!r}")
    result[in_design_z & in_design_xy] = 3
    return result


def save_epsilon_and_material_slices(
    fdtd: Any,
    output_dir: Path,
    *,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    geometry: str,
    disk_radius_m: float,
    square_half_width_m: float,
    design_gap_m: float,
    design_height_m: float,
) -> dict[str, Any]:
    """Save raw solver epsilon and analytic material-ID slices on the Pabs grid."""
    index_name = "stage1_pabs_adv::index"
    raw_x = np.asarray(fdtd.getdata(index_name, "x", 1), float).reshape(-1)
    raw_y = np.asarray(fdtd.getdata(index_name, "y", 1), float).reshape(-1)
    raw_z = np.asarray(fdtd.getdata(index_name, "z", 1), float).reshape(-1)
    if not (
        np.allclose(raw_x, x, rtol=0.0, atol=1e-15)
        and np.allclose(raw_y, y, rtol=0.0, atol=1e-15)
        and np.allclose(raw_z, z, rtol=0.0, atol=1e-15)
    ):
        raise RuntimeError("raw index monitor coordinates differ from Pabs coordinates")
    epsilon = {}
    for axis in "xyz":
        index = np.asarray(fdtd.getdata(index_name, f"index_{axis}", 1)).squeeze()
        epsilon[axis] = index**2
    ix = int(np.argmin(np.abs(x)))
    iy = int(np.argmin(np.abs(y)))
    iz = int(np.argmin(np.abs(z + 50e-9)))

    id_xy = geometry_material_id(
        x[:, None], y[None, :], np.array(z[iz]),
        geometry=geometry, disk_radius_m=disk_radius_m,
        square_half_width_m=square_half_width_m,
        design_gap_m=design_gap_m, design_height_m=design_height_m,
    )
    id_xz = geometry_material_id(
        x[:, None], np.array(0.0), z[None, :],
        geometry=geometry, disk_radius_m=disk_radius_m,
        square_half_width_m=square_half_width_m,
        design_gap_m=design_gap_m, design_height_m=design_height_m,
    )
    id_yz = geometry_material_id(
        np.array(0.0), y[:, None], z[None, :],
        geometry=geometry, disk_radius_m=disk_radius_m,
        square_half_width_m=square_half_width_m,
        design_gap_m=design_gap_m, design_height_m=design_height_m,
    )
    path = output_dir / "epsilon_material_slices.npz"
    np.savez_compressed(
        path,
        x_m=x, y_m=y, z_m=z,
        xy_z_m=z[iz], xz_y_m=y[iy], yz_x_m=x[ix],
        epsilon_x_xy=epsilon["x"][:, :, iz],
        epsilon_y_xy=epsilon["y"][:, :, iz],
        epsilon_z_xy=epsilon["z"][:, :, iz],
        epsilon_x_xz=epsilon["x"][:, iy, :],
        epsilon_y_xz=epsilon["y"][:, iy, :],
        epsilon_z_xz=epsilon["z"][:, iy, :],
        epsilon_x_yz=epsilon["x"][ix, :, :],
        epsilon_y_yz=epsilon["y"][ix, :, :],
        epsilon_z_yz=epsilon["z"][ix, :, :],
        material_id_xy=id_xy,
        material_id_xz=id_xz,
        material_id_yz=id_yz,
    )
    return {
        "path": str(path),
        "material_id_legend": MATERIAL_ID_LEGEND,
        "xy_z_m": float(z[iz]),
        "xz_y_m": float(y[iy]),
        "yz_x_m": float(x[ix]),
        "epsilon_imaginary_ranges": {
            axis: [
                float(np.min(np.imag(values))),
                float(np.max(np.imag(values))),
            ]
            for axis, values in epsilon.items()
        },
    }


def mesh_and_material_metadata(
    fdtd: Any,
    model: Any,
    *,
    geometry: str,
    design_gap_m: float,
) -> dict[str, Any]:
    has_design = geometry != "none"
    design_z_min = design_gap_m if has_design else None
    design_z_max = design_gap_m + float(model.design_h) * 1e-6 if has_design else None
    flake_z_min = -float(model.flake_h) * 1e-6
    flake_z_max = 0.0
    eps_a = model.eps_flake(4000.0, "a")
    eps_b = model.eps_flake(4000.0, "b")
    n_a = np.sqrt(eps_a)
    n_b = np.sqrt(eps_b)
    return {
        "geometry": geometry,
        "z_contract": {
            "TaIrTe4_z_min_m": flake_z_min,
            "TaIrTe4_z_max_m": flake_z_max,
            "design_z_min_m": design_z_min,
            "design_z_max_m": design_z_max,
            "signed_gap_m": design_z_min - flake_z_max if has_design else None,
            "volume_overlap_m": max(
                0.0,
                min(design_z_max, flake_z_max)
                - max(design_z_min, flake_z_min),
            ) if has_design else 0.0,
            "touching_at_single_interface": bool(
                has_design and np.isclose(design_z_min, flake_z_max, atol=1e-15)
            ),
        },
        "mesh": {
            "FDTD_mesh_refinement": safe_getnamed(
                fdtd, "FDTD", "mesh refinement"
            ),
            "FDTD_mesh_type": safe_getnamed(fdtd, "FDTD", "mesh type"),
            "global_resolution_cells_per_um": int(model.GLOBAL_RESOLUTION),
            "global_nominal_step_m": 1e-6 / float(model.GLOBAL_RESOLUTION),
            "flake_override_dz_m": float(model.FLAKE_DZ_NM) * 1e-9,
            "design_grid_step_xy_m": float(model.design_grid_steps[0]),
            "design_grid_step_z_m": float(model.design_grid_steps[2]),
            "design_mesh_order": safe_getnamed(fdtd, "design", "mesh order"),
            "design_override_mesh_order": safe_getnamed(
                fdtd, "design", "override mesh order from material database"
            ),
            "TaIrTe4_mesh_order": safe_getnamed(
                fdtd, "TaIrTe4_flake", "mesh order"
            ),
            "TaIrTe4_override_mesh_order": safe_getnamed(
                fdtd, "TaIrTe4_flake",
                "override mesh order from material database",
            ),
        },
        "material_at_4um": {
            "TaIrTe4_epsilon_diagonal": [
                {"real": float(np.real(eps_a)), "imag": float(np.imag(eps_a))},
                {"real": float(np.real(eps_b)), "imag": float(np.imag(eps_b))},
                {"real": float(model.eps_c_flake), "imag": 0.0},
            ],
            "TaIrTe4_refractive_index_diagonal": [
                {"real": float(np.real(n_a)), "imag": float(np.imag(n_a))},
                {"real": float(np.real(n_b)), "imag": float(np.imag(n_b))},
                {"real": float(np.sqrt(model.eps_c_flake)), "imag": 0.0},
            ],
            "design_index": {"real": float(model.DESIGN_N), "imag": 0.0},
            "SiO2_index": {"real": float(model.sio2_index[0]), "imag": 0.0},
            "Si_index": {"real": float(model.si_index[0]), "imag": 0.0},
        },
    }
