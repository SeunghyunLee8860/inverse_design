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


REPOSITORY = Path(__file__).resolve().parents[3]
RUN002 = REPOSITORY / "photothermal_pte" / "optimization_runs" / "legacy_v261_optical_support"
TAIRTE4_MATERIAL = "run010_TaIrTe4_paper_abc"
SIO2_MATERIAL = "run010_Kitamura_SiO2_10um"
SI_MATERIAL = "run010_Palik_Si_10um"
DESIGN_OBJECT = "run010_TaIrTe4_void_anisotropic_design"
SOURCE_NAME = "run010_gaussian10_w8p5_source"
Q_BOUNDS = {
    "x": (-14.0e-6, 14.0e-6),
    "y": (-14.0e-6, 14.0e-6),
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
    bounds = CONTRACT.design_bounds_m
    return (
        np.linspace(*bounds["x"], CONTRACT.design_node_shape[0]),
        np.linspace(*bounds["y"], CONTRACT.design_node_shape[1]),
        np.linspace(-CONTRACT.flake_thickness_m, 0.0, int(round(CONTRACT.flake_thickness_m / CONTRACT.flake_dz_m)) + 1),
    )


def anisotropic_index(rho_xy: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    x, y, z = design_nodes()
    rho = np.asarray(rho_xy, dtype=np.float64)
    if rho.shape != (x.size, y.size):
        raise ValueError(f"rho shape {rho.shape} != {(x.size, y.size)}")
    if np.any((rho < 0.0) | (rho > 1.0)) or not np.all(np.isfinite(rho)):
        raise ValueError("rho must be finite in [0,1]")
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
    if CONTRACT.geometry_mode == "contact_anchored":
        pieces = {
            "bottom_contact": {"x": (-flake, flake), "y": (-flake, -y_design), "z": z},
            "top_contact": {"x": (-flake, flake), "y": (y_design, flake), "z": z},
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
    x, y, z = design_nodes()
    index, metadata = anisotropic_index(rho_xy)
    fdtd.addimport({"name": DESIGN_OBJECT, "x": 0.0, "y": 0.0, "z": 0.0})
    if int(fdtd.importnk2(index, x, y, z)) != 1:
        raise RuntimeError("anisotropic importnk2 returned failure")
    return {
        "name": DESIGN_OBJECT,
        "nodes_m": {"x": x, "y": y, "z": z},
        **metadata,
    }


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
    add_mesh(
        fdtd,
        "run010_illuminated_stack_xy_mesh",
        {"x": (-14e-6, 14e-6), "y": (-14e-6, 14e-6), "z": (-0.5e-6, 0.1e-6)},
        x=250e-9,
        y=250e-9,
    )
    names.append("run010_illuminated_stack_xy_mesh")
    flake = 0.5 * CONTRACT.flake_span_m + CONTRACT.design_step_m
    add_mesh(
        fdtd,
        "run010_flake_xy_z_mesh",
        {"x": (-flake, flake), "y": (-flake, flake), "z": (-CONTRACT.flake_thickness_m - CONTRACT.flake_dz_m, CONTRACT.flake_dz_m)},
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
