"""Minimal TaIrTe4 Case-X geometry and periodic design mapping.

This module only builds the forward structure.  Adjoint construction is owned
by ``volume_current_evaluator.py`` and never enters the legacy msopt adjoint.
"""

from __future__ import annotations

import os
from pathlib import Path

import autograd.numpy as npa
import numpy as np

from msopt.Lumerical_utill import LumericalFDTDSimulator
from msopt.Mapping import Mapping

HERE = Path(__file__).resolve().parent
c0 = 299792458.0
seed = int(os.environ.get("ID_SEED", "240"))

TARGET_WL_UM = float(os.environ.get("TARGET_WL_UM", "4.0"))
SOURCE_WL_START_UM = float(os.environ.get("SOURCE_WL_START_UM", "3.0"))
SOURCE_WL_STOP_UM = float(os.environ.get("SOURCE_WL_STOP_UM", "6.0"))
MATERIAL_FIT_START_UM = float(os.environ.get("MATERIAL_FIT_START_UM", "2.7"))
MATERIAL_FIT_STOP_UM = float(os.environ.get("MATERIAL_FIT_STOP_UM", "13.2"))
MATERIAL_SAMPLE_COUNT = int(os.environ.get("MATERIAL_SAMPLE_COUNT", "600"))
if not (0.0 < SOURCE_WL_START_UM < TARGET_WL_UM < SOURCE_WL_STOP_UM):
    raise ValueError("source range must contain the single analysis wavelength")
if not (0.0 < MATERIAL_FIT_START_UM < SOURCE_WL_START_UM):
    raise ValueError("material fit must start below the source range")
if not (SOURCE_WL_STOP_UM < MATERIAL_FIT_STOP_UM):
    raise ValueError("material fit must stop above the source range")
GLOBAL_RESOLUTION = int(os.environ.get("RES", "20"))
DESIGN_RESOLUTION_XY = int(os.environ.get("DESIGN_RES_XY", "40"))
FLAKE_DZ_NM = float(os.environ.get("FILM_DZ_NM", "5.0"))
FLAKE_MARGIN_BELOW_CELLS = int(os.environ.get("FILM_MARGIN_CELLS", "10"))
MIN_MESH_STEP_NM = float(os.environ.get("MIN_MESH_STEP_NM", "1.0"))
period_xy = float(os.environ.get("PERIOD_UM", "6.0"))
design_h = float(os.environ.get("DESIGN_H_UM", "0.6"))
MFS_UM = float(os.environ.get("MFS_UM", "0.2"))
MGS_UM = float(os.environ.get("MGS_UM", "0.2"))
DESIGN_N = float(os.environ.get("DESIGN_N", "4.0"))
ID_SEED_AMP = float(os.environ.get("ID_SEED_AMP", "0.15"))

sio2_h = 0.285
flake_h = 0.100
PML_LAYERS = int(os.environ.get("PML_LAYERS", "24"))
BULK_MESH_MODE = os.environ.get("BULK_MESH_MODE", "auto").strip().lower()
MESH_ACCURACY = int(os.environ.get("MESH_ACCURACY", "5"))
SOURCE_MESH_HALFSPAN_UM = float(os.environ.get("SOURCE_MESH_HALFSPAN_UM", "0.1"))
AIR_BULK_DZ_NM = float(os.environ.get("AIR_BULK_DZ_NM", "100.0"))
SI_BULK_DZ_NM = float(os.environ.get("SI_BULK_DZ_NM", "100.0"))
if BULK_MESH_MODE not in ("auto", "uniform", "regional_uniform"):
    raise ValueError("BULK_MESH_MODE must be auto, uniform, or regional_uniform")
si_h = float(os.environ.get("SI_H_UM", "2.0"))
air_top = float(os.environ.get("AIR_TOP_UM", "2.0"))
source_gap = float(os.environ.get("SOURCE_GAP_UM", "0.5"))
# Legacy diagnostic only.  The paper-like 11-um finite-edge implementation
# closes the unpublished out-of-plane response with epsilon_c=epsilon_b from
# the duplicated third complex pair in perm_data.txt.  Keep this constant so
# historical 4-um artifacts remain reproducible; do not use it as the
# production paper-IR c-axis response.
eps_c_flake = 16.0
eps_c_flake_legacy_diagnostic = eps_c_flake
MAX_COEFFS = 20

flake_c = [0, 0, -0.5 * flake_h]
sio2_c = [0, 0, -flake_h - 0.5 * sio2_h]
si_c = [0, 0, -flake_h - sio2_h - 0.5 * si_h]
design_c = [0, 0, +0.5 * design_h]
Z_min = -flake_h - sio2_h - si_h
Z_max = design_h + air_top
# regional_uniform only: depth where the fine 50 nm stack mesh hands over to the
# coarse Si bulk mesh.  Default = SiO2/Si interface (legacy).  Moving it deeper
# into Si (e.g. -0.597, an exact baseline mesh plane) keeps every SiO2- and
# interface-adjacent cell identical to the uniform-mesh baseline while the Si
# bulk below is coarsened.  Must lie inside the Si slab.
SI_COARSE_Z_MAX_UM = float(
    os.environ.get("SI_COARSE_Z_MAX_UM", str(-flake_h - sio2_h))
)
if not (Z_min < SI_COARSE_Z_MAX_UM <= -flake_h - sio2_h):
    raise ValueError(
        f"SI_COARSE_Z_MAX_UM={SI_COARSE_Z_MAX_UM} must lie in "
        f"({Z_min}, {-flake_h - sio2_h}]"
    )
Sx, Sy, Sz = period_xy, period_xy, Z_max - Z_min
sim_center_z = 0.5 * (Z_min + Z_max)
src_c = [0, 0, design_h + source_gap]
fom_c = [0, 0, -0.5 * flake_h]
fom_s = [Sx, Sy, flake_h]
lateral_pad = 1.2
si_s = [Sx * lateral_pad, Sy * lateral_pad, si_h]
sio2_s = [Sx * lateral_pad, Sy * lateral_pad, sio2_h]
flake_s = [Sx * lateral_pad, Sy * lateral_pad, flake_h]
design_s = [Sx, Sy, design_h]

Nx = int(round(design_s[0] * DESIGN_RESOLUTION_XY)) + 1
Ny = int(round(design_s[1] * DESIGN_RESOLUTION_XY)) + 1
Nz = int(round(design_s[2] * GLOBAL_RESOLUTION)) + 1
design_grids = [Nx, Ny, Nz]
design_grid_steps = [
    1e-6 / DESIGN_RESOLUTION_XY,
    1e-6 / DESIGN_RESOLUTION_XY,
    1e-6 / GLOBAL_RESOLUTION,
]

target_wl = np.asarray([TARGET_WL_UM], dtype=float)
si_index = [3.425]
sio2_index = [1.38]
design_high_index = [DESIGN_N]
design_low_index = [1.0]

_data = np.loadtxt(HERE / "perm_data.txt")
_order = np.argsort(_data[:, 0])
_lam = _data[_order, 0]
_eps_a = (_data[:, 1] + 1j * _data[:, 2])[_order]
_eps_b = (_data[:, 3] + 1j * _data[:, 4])[_order]
_eps_c_table = (_data[:, 5] + 1j * _data[:, 6])[_order]
if not np.array_equal(_eps_b, _eps_c_table):
    raise RuntimeError(
        "paper-derived perm_data.txt no longer satisfies the approved "
        "epsilon_c=epsilon_b 3D closure"
    )


def eps_flake(lam_nm, axis):
    if axis == "a":
        source = _eps_a
    elif axis == "b":
        source = _eps_b
    elif axis == "c":
        source = _eps_c_table
    else:
        raise ValueError("TaIrTe4 optical axis must be a, b, or c")
    return np.interp(lam_nm, _lam, source.real) + 1j * np.interp(
        lam_nm, _lam, source.imag
    )


def add_flake_material(sim):
    # perm_data.txt uses nanometres; eps_flake() therefore receives nm.
    lam_s_nm = np.linspace(
        MATERIAL_FIT_START_UM * 1e3,
        MATERIAL_FIT_STOP_UM * 1e3,
        MATERIAL_SAMPLE_COUNT,
    )
    f_s = c0 / (lam_s_nm * 1e-9)
    ea, eb = eps_flake(lam_s_nm, "a"), eps_flake(lam_s_nm, "b")
    ec = np.full_like(ea, eps_c_flake)
    material = sim.fdtd.addmaterial("Sampled 3D data")
    sim.fdtd.setmaterial(material, "name", "TaIrTe4_ani")
    sim.fdtd.setmaterial("TaIrTe4_ani", "anisotropy", 1)
    sim.fdtd.setmaterial("TaIrTe4_ani", "max coefficients", MAX_COEFFS)
    sim.fdtd.setmaterial(
        "TaIrTe4_ani", "sampled data", np.column_stack((f_s, ea, eb, ec))
    )


def build_case(incident_polarization="x"):
    if incident_polarization not in ("x", "y"):
        raise ValueError("incident_polarization must be x or y")
    polarization_angle = 0.0 if incident_polarization == "x" else 90.0
    sim = LumericalFDTDSimulator(
        sim_size=[Sx, Sy, Sz], resolution=GLOBAL_RESOLUTION, unit=1e-6,
        background_index=1.0, center_wl=float(np.mean(target_wl)),
        N_f=len(target_wl), bc_x="Periodic", bc_y="Periodic", bc_z="PML",
        create_global_uniform_mesh=(BULK_MESH_MODE == "uniform"),
    )
    sim.fdtd.setnamed("FDTD", "z", sim_center_z * 1e-6)
    try:
        sim.fdtd.setnamed("FDTD", "pml layers", PML_LAYERS)
    except Exception:
        pass
    if BULK_MESH_MODE != "uniform":
        if sim.fdtd.getnamednumber("global_uniform_mesh") > 0:
            raise RuntimeError("global_uniform_mesh was unexpectedly created")
        sim.fdtd.setnamed("FDTD", "mesh type", "auto non-uniform")
        sim.fdtd.setnamed("FDTD", "mesh accuracy", MESH_ACCURACY)
    else:
        sim.fdtd.setnamed("global_uniform_mesh", "z", sim_center_z * 1e-6)
    add_flake_material(sim)
    sim.add_geo(center=si_c, size=si_s, index=si_index, name="Si_substrate")
    sim.add_geo(center=sio2_c, size=sio2_s, index=sio2_index, name="SiO2_spacer")
    sim.add_geo(center=flake_c, size=flake_s, index="TaIrTe4_ani", name="TaIrTe4_flake")
    if BULK_MESH_MODE == "regional_uniform":
        sim.fdtd.addmesh()
        sim.fdtd.set("name", "air_bulk_z_mesh")
        sim.fdtd.set("x", 0.0); sim.fdtd.set("x span", Sx*1e-6)
        sim.fdtd.set("y", 0.0); sim.fdtd.set("y span", Sy*1e-6)
        sim.fdtd.set("z min", design_h*1e-6); sim.fdtd.set("z max", Z_max*1e-6)
        sim.fdtd.set("override x mesh", 0); sim.fdtd.set("override y mesh", 0)
        sim.fdtd.set("override z mesh", 1); sim.fdtd.set("dz", AIR_BULK_DZ_NM*1e-9)
        sim.fdtd.addmesh()
        sim.fdtd.set("name", "si_bulk_z_mesh")
        sim.fdtd.set("x", 0.0); sim.fdtd.set("x span", Sx*1e-6)
        sim.fdtd.set("y", 0.0); sim.fdtd.set("y span", Sy*1e-6)
        sim.fdtd.set("z min", Z_min*1e-6)
        sim.fdtd.set("z max", SI_COARSE_Z_MAX_UM*1e-6)
        sim.fdtd.set("override x mesh", 0); sim.fdtd.set("override y mesh", 0)
        sim.fdtd.set("override z mesh", 1); sim.fdtd.set("dz", SI_BULK_DZ_NM*1e-9)
    if FLAKE_DZ_NM > 0:
        dz = FLAKE_DZ_NM * 1e-9
        sim.fdtd.addmesh()
        sim.fdtd.set("name", "flake_mesh")
        sim.fdtd.set("x", 0.0); sim.fdtd.set("x span", Sx*lateral_pad*1e-6)
        sim.fdtd.set("y", 0.0); sim.fdtd.set("y span", Sy*lateral_pad*1e-6)
        sim.fdtd.set("z min", -flake_h*1e-6-FLAKE_MARGIN_BELOW_CELLS*dz)
        sim.fdtd.set("z max", 0.0)
        sim.fdtd.set("override x mesh", 0); sim.fdtd.set("override y mesh", 0)
        sim.fdtd.set("override z mesh", 1); sim.fdtd.set("dz", dz)
    sim.add_source(
        mode="plane", name="source", center=src_c, size=[Sx, Sy, 0],
        direction="backward", src_wl=[SOURCE_WL_START_UM, SOURCE_WL_STOP_UM],
        bandwidth=0.0, pol=polarization_angle, single=True,
    )
    sim.min_mesh_step_m = MIN_MESH_STEP_NM * 1e-9
    sim.add_design_grid(
        name="design", center=design_c, size=design_s,
        index1=design_high_index, index2=design_low_index,
        design_grids=design_grids, density=0.5*np.ones(design_grids),
        wavelength=float(np.mean(target_wl)), grid_steps=design_grid_steps,
    )
    sim.add_design_monitor()
    actual_min = float(sim.fdtd.getnamed("FDTD", "min mesh step"))
    if actual_min > MIN_MESH_STEP_NM * 1e-9 * 1.01:
        raise RuntimeError("min mesh step clamped; 5 nm flake override is disabled")
    return sim


def build_case_x():
    return build_case("x")


def build_case_y():
    return build_case("y")


_base_mapping = Mapping(
    Symmetry_sim=False, Sym_geo_width=False, Sym_geo_length=False,
    Sym_geo_C2=False, Sym_geo_C8=False,
    DR_info=[design_s[0], design_s[1], design_s[2], 0, 1, 2],
    DR_N_info=[Nx, Ny, Nz, DESIGN_RESOLUTION_XY], Mask_pixels=0,
    MFS=MFS_UM, MGS=MGS_UM, periodic_filter_axes=[0, 1],
)


class SeamWrappedMapping:
    def __init__(self, base):
        self.base = base
        self.Is_freeform = base.Is_freeform

    def __call__(self, x, beta, Is_opt=True):
        value = self.base(x, beta, Is_opt)
        value = npa.reshape(value, (Nx, Ny, Nz))
        value = npa.concatenate([value[:Nx-1], value[:1]], axis=0)
        value = npa.concatenate([value[:, :Ny-1], value[:, :1]], axis=1)
        return npa.reshape(value, (-1,))


# Mapping selection (opt-in; default = legacy morphology cascade).
# MSOPT_MAPPING=filter_project -> standard unbiased conic-filter + tanh(0.5)
# projection: uniform gray latent maps to 0.5, binarizes cleanly, MFS/MGS
# softly enforced at ~radius; strict length scale applied at final projection.
_MAPPING_MODE = os.environ.get("MSOPT_MAPPING", "morphology").strip().lower()
if _MAPPING_MODE == "filter_project":
    from symmetric_mapping import FilterProjectMapping
    mapping = FilterProjectMapping(
        Nx, Ny, Nz, DESIGN_RESOLUTION_XY, period_xy, MFS_UM, MGS_UM
    )
elif _MAPPING_MODE == "morphology":
    mapping = SeamWrappedMapping(_base_mapping)
else:
    raise ValueError(f"unknown MSOPT_MAPPING={_MAPPING_MODE!r}")


def make_x0():
    rng = np.random.default_rng(seed)
    n_ctrl = max(4, int(round(period_xy/max(MFS_UM, 1e-3)))+1)
    ctrl = rng.uniform(-ID_SEED_AMP, ID_SEED_AMP, size=(n_ctrl, n_ctrl))
    xi, yi = np.linspace(0, n_ctrl-1, Nx), np.linspace(0, n_ctrl-1, Ny)
    x0i, y0i = np.floor(xi).astype(int), np.floor(yi).astype(int)
    x1i, y1i = np.minimum(x0i+1, n_ctrl-1), np.minimum(y0i+1, n_ctrl-1)
    fx, fy = xi-x0i, yi-y0i
    up = (
        ctrl[np.ix_(x0i,y0i)]*np.outer(1-fx,1-fy)
        + ctrl[np.ix_(x1i,y0i)]*np.outer(fx,1-fy)
        + ctrl[np.ix_(x0i,y1i)]*np.outer(1-fx,fy)
        + ctrl[np.ix_(x1i,y1i)]*np.outer(fx,fy)
    )
    return np.clip(0.5+up, 0.05, 0.95).reshape(-1)


x0 = make_x0()
