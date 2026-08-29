"""Compact nonperiodic Lumerical geometry for TaIrTe4-flake topology."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from photothermal_pte.finite_inverse_design.probe_v261_cpu_tfsf_device import (
    PABS_FIELD,
    PABS_GROUP,
    PABS_INDEX,
)
from photothermal_pte.optimization_runs.tairte4_flake_topology.contract import (
    CONTRACT,
)
from photothermal_pte.optimization_runs.tairte4_flake_topology.rotated_device import (
    device_to_crystal_field,
)


REPOSITORY = Path(__file__).resolve().parents[3]
RUN002 = REPOSITORY / "photothermal_pte" / "optimization_runs" / "legacy_v261_optical_support"
TAIRTE4_MATERIAL = "run010_TaIrTe4_paper_abc"
SIO2_MATERIAL = "run010_Kitamura_SiO2_10um"
SI_MATERIAL = "run010_Palik_Si_10um"
DESIGN_OBJECT = "run010_TaIrTe4_void_anisotropic_design"
PERMITTIVITY_ROTATION_OBJECT = "run064_fixed_crystal_axes_in_device_coordinates"
SOURCE_NAME = "run010_gaussian10_w8p5_source"
ROTATED_DEVICE_ANGLE_DEG = 45.0
Q_HALF_SPAN_XY_M = (
    18.0e-6
    if CONTRACT.geometry_mode == "diagonal_45_contact_anchored"
    else 14.0e-6
)
Q_BOUNDS = {
    "x": (-Q_HALF_SPAN_XY_M, Q_HALF_SPAN_XY_M),
    "y": (-Q_HALF_SPAN_XY_M, Q_HALF_SPAN_XY_M),
    # pabs_adv's child monitors remain centred on z=0.  Use the audited
    # symmetric control volume and match every six-face monitor to it.
    "z": (-1.25e-6, 1.25e-6),
}


def material_epsilon() -> dict[str, complex]:
    data = np.loadtxt(REPOSITORY / "photothermal_pte" / "bundle" / "perm_data.txt")
    order = np.argsort(data[:, 0])
    wavelength_nm = data[order, 0]
    eps_a = (data[:, 1] + 1j * data[:, 2])[order]
    eps_b = (data[:, 3] + 1j * data[:, 4])[order]

    def sample(values: np.ndarray) -> complex:
        return complex(
            np.interp(CONTRACT.wavelength_m * 1e9, wavelength_nm, values.real),
            np.interp(CONTRACT.wavelength_m * 1e9, wavelength_nm, values.imag),
        )

    sampled_a = sample(eps_a)
    sampled_b = sample(eps_b)
    return {"x": sampled_b, "y": sampled_a, "z": sampled_b}


def passive_sqrt(epsilon: np.ndarray) -> np.ndarray:
    index = np.sqrt(np.asarray(epsilon, dtype=np.complex128))
    return np.where(index.imag < 0.0, -index, index)


def design_nodes() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Local device nodes used by the optimization variables."""

    bounds = CONTRACT.design_bounds_m
    return (
        np.linspace(*bounds["x"], CONTRACT.design_node_shape[0]),
        np.linspace(*bounds["y"], CONTRACT.design_node_shape[1]),
        np.linspace(-CONTRACT.flake_thickness_m, 0.0, int(round(CONTRACT.flake_thickness_m / CONTRACT.flake_dz_m)) + 1),
    )


def import_nodes() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Global import nodes used by the selected optical geometry contract."""

    x, y, z = design_nodes()
    if CONTRACT.geometry_mode == "diagonal_45_contact_anchored":
        if CONTRACT.rotated_optical_mode == "physical_crystal_grid":
            half = 0.5 * CONTRACT.crystal_bounding_span_m
            count = CONTRACT.crystal_bounding_node_shape[0]
            x = np.linspace(-half, half, count)
            y = x.copy()
        elif CONTRACT.rotated_optical_mode == "run58_proxy":
            inset = int(round(CONTRACT.fixed_contact_depth_m / CONTRACT.design_step_m))
            x = x[inset:-inset]
    return x, y, z


def anisotropic_index(rho_xy: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    x, y, z = import_nodes()
    rho = CONTRACT.apply_fixed_contact_density(
        np.asarray(rho_xy, dtype=np.float64)
    )
    if rho.shape != CONTRACT.design_node_shape:
        raise ValueError(f"rho shape {rho.shape} != {CONTRACT.design_node_shape}")
    if np.any((rho < 0.0) | (rho > 1.0)) or not np.all(np.isfinite(rho)):
        raise ValueError("rho must be finite in [0,1]")
    if CONTRACT.geometry_mode == "diagonal_45_contact_anchored":
        if CONTRACT.rotated_optical_mode == "physical_crystal_grid":
            rho = device_to_crystal_field(rho)
        elif CONTRACT.rotated_optical_mode == "run58_proxy":
            inset = int(round(CONTRACT.fixed_contact_depth_m / CONTRACT.design_step_m))
            rho = rho[inset:-inset, :]
    if rho.shape != (x.size, y.size):
        raise RuntimeError("optical import density/node shape mismatch")
    endpoint = material_epsilon()
    endpoint_xyz = np.asarray([endpoint[axis] for axis in "xyz"], dtype=np.complex128)
    epsilon_xy = 1.0 + rho[:, :, None] * (endpoint_xyz[None, None, :] - 1.0)
    epsilon = np.repeat(epsilon_xy[:, :, None, :], z.size, axis=2)
    index = passive_sqrt(epsilon)
    return index, {
        "shape_xy_z_components": list(index.shape),
        "rho_range": [float(np.min(rho)), float(np.max(rho))],
        "epsilon_air": [1.0, 0.0],
        "epsilon_TaIrTe4_xyz": {
            axis: [float(endpoint[axis].real), float(endpoint[axis].imag)]
            for axis in "xyz"
        },
        "interpolation": "epsilon_c(rho)=1+rho*(epsilon_TaIrTe4,c-1), then passive complex sqrt",
        "axis_mapping": "Lumerical x=b, y=a, z=c=b closure",
    }


def configure_source(
    audit: Any,
    *,
    optical_lateral_span_m: float | None = None,
) -> None:
    lateral_span = (
        CONTRACT.optical_lateral_span_m
        if optical_lateral_span_m is None
        else float(optical_lateral_span_m)
    )
    if lateral_span <= CONTRACT.source_span_m:
        raise ValueError("optical domain must exceed the finite source span")
    audit.contract = SimpleNamespace(
        WAVELENGTH_M=CONTRACT.wavelength_m,
        SELECTED_W0_M=CONTRACT.target_waist_m,
        SOURCE_SPAN_M=CONTRACT.source_span_m,
        LATERAL_DOMAIN_M=lateral_span,
        SOURCE_Z_M=CONTRACT.source_z_m,
        FOCUS_Z_M=CONTRACT.focus_z_m,
        FDTD_Z_MIN_M=CONTRACT.optical_z_min_m,
        FDTD_Z_MAX_M=CONTRACT.optical_z_max_m,
    )
    audit.TARGET_FREQUENCY_HZ = audit.C0 / CONTRACT.wavelength_m
    audit.SOURCE_START_M = 8.5e-6
    audit.SOURCE_STOP_M = 12.142857142857142e-6
    audit.PML_LAYERS = CONTRACT.pml_layers
    audit.MESH_ACCURACY = CONTRACT.mesh_accuracy
    audit.SOURCE_NAME = SOURCE_NAME
    audit.MONITORS = {
        "source_plane": CONTRACT.source_z_m - 0.5e-6,
        "flake_target_plane": CONTRACT.focus_z_m,
        "downstream_plane": -2.5e-6,
    }


def add_rect(fdtd: Any, name: str, material: str, bounds: dict[str, tuple[float, float]]) -> None:
    item = fdtd.addrect()
    item["name"] = name
    item["material"] = material
    for axis in "xyz":
        item[f"{axis} min"], item[f"{axis} max"] = bounds[axis]


def add_fixed_frame(fdtd: Any) -> list[str]:
    flake = 0.5 * CONTRACT.flake_span_m
    x_design = 0.5 * CONTRACT.design_span_x_m
    y_design = 0.5 * CONTRACT.design_span_y_m
    z = (-CONTRACT.flake_thickness_m, 0.0)
    if (
        CONTRACT.geometry_mode == "diagonal_45_contact_anchored"
        and CONTRACT.rotated_optical_mode
        in {"physical_crystal_grid", "rotated_tensor_local_grid"}
    ):
        return []
    if CONTRACT.geometry_mode == "contact_anchored":
        pieces = {
            "bottom_contact": {"x": (-flake, flake), "y": (-flake, -y_design), "z": z},
            "top_contact": {"x": (-flake, flake), "y": (y_design, flake), "z": z},
        }
    elif CONTRACT.geometry_mode == "left_right_contact_anchored":
        pieces = {
            "left_contact": {"x": (-flake, -x_design), "y": (-flake, flake), "z": z},
            "right_contact": {"x": (x_design, flake), "y": (-flake, flake), "z": z},
        }
    elif CONTRACT.geometry_mode == "diagonal_45_contact_anchored":
        pieces = {
            "left_contact": {
                "x": (-flake, -flake + CONTRACT.fixed_contact_depth_m),
                "y": (-flake, flake),
                "z": z,
            },
            "right_contact": {
                "x": (flake - CONTRACT.fixed_contact_depth_m, flake),
                "y": (-flake, flake),
                "z": z,
            },
        }
    else:
        pieces = {
            "left": {"x": (-flake, -x_design), "y": (-flake, flake), "z": z},
            "right": {"x": (x_design, flake), "y": (-flake, flake), "z": z},
            "bottom": {"x": (-x_design, x_design), "y": (-flake, -y_design), "z": z},
            "top": {"x": (-x_design, x_design), "y": (y_design, flake), "z": z},
        }
    names = []
    for label, bounds in pieces.items():
        name = f"run010_fixed_TaIrTe4_frame_{label}"
        add_rect(fdtd, name, TAIRTE4_MATERIAL, bounds)
        names.append(name)
    return names


def add_design(fdtd: Any, rho_xy: np.ndarray) -> dict[str, object]:
    x, y, z = import_nodes()
    index, metadata = anisotropic_index(rho_xy)
    fdtd.addimport({"name": DESIGN_OBJECT, "x": 0.0, "y": 0.0, "z": 0.0})
    if int(fdtd.importnk2(index, x, y, z)) != 1:
        raise RuntimeError("anisotropic importnk2 returned failure")
    tensor_rotation = None
    if (
        CONTRACT.geometry_mode == "diagonal_45_contact_anchored"
        and CONTRACT.rotated_optical_mode == "rotated_tensor_local_grid"
    ):
        rotation = fdtd.addgridattribute("permittivity rotation")
        rotation["name"] = PERMITTIVITY_ROTATION_OBJECT
        rotation["angle convention"] = "Euler (Z-Y'-Z'')"
        rotation["phi"] = -ROTATED_DEVICE_ANGLE_DEG
        rotation["theta"] = 0.0
        rotation["psi"] = 0.0
        rotation["enable conformal meshing"] = 1
        tensor_rotation = {
            "name": PERMITTIVITY_ROTATION_OBJECT,
            "angle_convention": "Euler (Z-Y'-Z'')",
            "phi_deg": -ROTATED_DEVICE_ANGLE_DEG,
            "theta_deg": 0.0,
            "psi_deg": 0.0,
            "scope": (
                "uniform global attribute; air, SiO2, and Si are isotropic, "
                "so only anisotropic TaIrTe4 is changed"
            ),
        }
    # The physical mode rasterizes the diamond directly on the fixed crystal
    # grid, so the imported primitive itself remains unrotated and the
    # anisotropic material axes stay x=b, y=a, z=c.
    return {
        "name": DESIGN_OBJECT,
        "nodes_m": {"x": x, "y": y, "z": z},
        "permittivity_rotation": tensor_rotation,
        **metadata,
    }


def polarization_angle_deg(polarization: str) -> float:
    """Return the source angle in the active optical coordinate frame."""

    if polarization not in {"a", "b"}:
        raise ValueError("polarization must be 'a' or 'b'")
    if (
        CONTRACT.geometry_mode == "diagonal_45_contact_anchored"
        and CONTRACT.rotated_optical_mode == "rotated_tensor_local_grid"
    ):
        return 45.0 if polarization == "a" else -45.0
    return 90.0 if polarization == "a" else 0.0


def add_mesh(fdtd: Any, name: str, bounds: dict[str, tuple[float, float]], **steps: float) -> None:
    mesh = fdtd.addmesh()
    mesh["name"] = name
    for axis in "xyz":
        mesh[f"{axis} min"], mesh[f"{axis} max"] = bounds[axis]
        mesh[f"override {axis} mesh"] = axis in steps
        if axis in steps:
            mesh[f"d{axis}"] = steps[axis]


def add_mesh_hierarchy(
    fdtd: Any,
    *,
    interface_xy_step_m: float | None = None,
    optical_lateral_span_m: float | None = None,
) -> list[str]:
    interface_step = (
        CONTRACT.interface_xy_step_m
        if interface_xy_step_m is None
        else float(interface_xy_step_m)
    )
    if interface_step <= 0.0:
        raise ValueError("interface xy step must be positive")
    lateral_span = (
        CONTRACT.optical_lateral_span_m
        if optical_lateral_span_m is None
        else float(optical_lateral_span_m)
    )
    half_domain = 0.5 * lateral_span
    names = []
    add_mesh(
        fdtd,
        "run010_outer_coarse_xy_mesh",
        {"x": (-half_domain, half_domain), "y": (-half_domain, half_domain), "z": (CONTRACT.optical_z_min_m, CONTRACT.optical_z_max_m)},
        x=CONTRACT.outer_xy_max_step_m,
        y=CONTRACT.outer_xy_max_step_m,
    )
    names.append("run010_outer_coarse_xy_mesh")
    physical_rotated = (
        CONTRACT.geometry_mode == "diagonal_45_contact_anchored"
        and CONTRACT.rotated_optical_mode == "physical_crystal_grid"
    )
    optical_flake_half = (
        CONTRACT.flake_bounding_half_span_m
        if physical_rotated
        else 0.5 * CONTRACT.flake_span_m
    )
    stack_half = max(14.0e-6, optical_flake_half + 0.1e-6)
    add_mesh(
        fdtd,
        "run010_illuminated_stack_xy_mesh",
        {"x": (-stack_half, stack_half), "y": (-stack_half, stack_half), "z": (-0.5e-6, 0.1e-6)},
        x=250e-9,
        y=250e-9,
    )
    names.append("run010_illuminated_stack_xy_mesh")
    flake = optical_flake_half + CONTRACT.design_step_m
    add_mesh(
        fdtd,
        "run010_flake_xy_z_mesh",
        {
            "x": (-flake, flake),
            "y": (-flake, flake),
            "z": (
                -CONTRACT.flake_thickness_m - CONTRACT.flake_dz_m,
                CONTRACT.flake_dz_m,
            ),
        },
        x=interface_step,
        y=interface_step,
        z=CONTRACT.flake_dz_m,
    )
    names.append("run010_flake_xy_z_mesh")
    return names


def add_absorption_and_flux(fdtd: Any) -> list[str]:
    pabs = fdtd.addobject("pabs_adv")
    pabs["name"] = PABS_GROUP
    for axis in "xyz":
        low, high = Q_BOUNDS[axis]
        pabs[axis] = 0.5 * (low + high)
        pabs[f"{axis} span"] = high - low
    names = []
    for axis in "xyz":
        for side, position in zip(("min", "max"), Q_BOUNDS[axis]):
            name = f"run010_flux_{axis}_{side}"
            monitor = fdtd.addpower()
            monitor["name"] = name
            monitor["monitor type"] = f"2D {axis.upper()}-normal"
            monitor[axis] = position
            for transverse in "xyz":
                if transverse != axis:
                    monitor[f"{transverse} min"] = Q_BOUNDS[transverse][0]
                    monitor[f"{transverse} max"] = Q_BOUNDS[transverse][1]
            monitor["override global monitor settings"] = True
            monitor["use source limits"] = False
            monitor["use wavelength spacing"] = True
            monitor["wavelength center"] = CONTRACT.wavelength_m
            monitor["wavelength span"] = 0.0
            monitor["frequency points"] = 1
            names.append(name)
    return names


def named_bounds(fdtd: Any, name: str) -> dict[str, list[float]]:
    return {
        axis: [
            float(fdtd.getnamed(name, f"{axis} min")),
            float(fdtd.getnamed(name, f"{axis} max")),
        ]
        for axis in "xyz"
    }
