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
eps_c_flake = 16.0
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
# Unique periodic latent grid: physical uses Nx x Ny nodes (fencepost duplicated
# on the seam), but the optimisation variable lives on the Nux x Nuy = 240 x 240
# UNIQUE nodes so the filter/constraints see exactly one period interval.
Nux = Nx - 1
Nuy = Ny - 1
latent_shape = (Nux, Nuy)
physical_shape = (Nx, Ny, Nz)
dx_um = period_xy / Nux
dy_um = period_xy / Nuy
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


def eps_flake(lam_nm, axis):
    source = _eps_a if axis == "a" else _eps_b
    return np.interp(lam_nm, _lam, source.real) + 1j * np.interp(
        lam_nm, _lam, source.imag
    )


def add_flake_material(sim):
    # perm_data.txt uses nanometres. Keep that unit explicit so a µm-range
    # variable can never be passed into eps_flake without conversion.
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


# Conic filter radius for the length-scale mapping/constraints.  Defaults to the
# requested minimum feature size; the constraints enforce the length scale and
# the independent DRC verifies it.
FILTER_RADIUS_UM = float(os.environ.get("VC_FILTER_RADIUS_UM", str(MFS_UM)))
ETA_ERODE = float(os.environ.get("VC_ETA_ERODE", "0.75"))
ETA_DILATE = float(os.environ.get("VC_ETA_DILATE", "0.25"))

# Mapping selection.
#   periodic_constrained (P0)  -> 240x240 unique three-field length-scale mapping
#                                 (PeriodicConstrainedMapping); latent is 57,600.
#   filter_project / morphology (LEGACY) -> reproduce old runs only; they do NOT
#                                 enforce a minimum length scale (see docs).
_MAPPING_MODE = os.environ.get("MSOPT_MAPPING", "periodic_constrained").strip().lower()
_LEGACY_MAPPINGS = ("filter_project", "morphology")
if _MAPPING_MODE == "periodic_constrained":
    from periodic_constrained_mapping import PeriodicConstrainedMapping
    mapping = PeriodicConstrainedMapping(
        Nx, Ny, Nz, period_xy, FILTER_RADIUS_UM, ETA_DILATE, ETA_ERODE,
    )
    _LATENT_SIZE = Nux * Nuy
elif _MAPPING_MODE in _LEGACY_MAPPINGS:
    import warnings
    warnings.warn(
        f"MSOPT_MAPPING={_MAPPING_MODE!r} is LEGACY (no minimum-length-scale "
        "guarantee); use periodic_constrained for production.",
        DeprecationWarning, stacklevel=2,
    )
    if _MAPPING_MODE == "filter_project":
        from symmetric_mapping import FilterProjectMapping
        mapping = FilterProjectMapping(
            Nx, Ny, Nz, DESIGN_RESOLUTION_XY, period_xy, MFS_UM, MGS_UM
        )
    else:
        mapping = SeamWrappedMapping(_base_mapping)
    _LATENT_SIZE = Nx * Ny
else:
    raise ValueError(f"unknown MSOPT_MAPPING={_MAPPING_MODE!r}")


def make_periodic_latent_seed():
    """Truly periodic 240x240 latent from low-frequency Fourier modes.

    Deterministic in ``seed``; energy only in low |k| so the seed varies on the
    ~MFS length scale (not per-pixel), is exactly periodic (no fencepost bias),
    and averages to ~0.5.  Only an initialisation -- length scale is enforced by
    the constraints, not by the seed.
    """
    rng = np.random.default_rng(seed)
    kmax = max(1, int(round(period_xy / max(MFS_UM, 1e-3))))
    spectrum = np.zeros((Nux, Nuy), dtype=complex)
    for kx in range(-kmax, kmax + 1):
        for ky in range(-kmax, kmax + 1):
            if kx == 0 and ky == 0:
                continue
            amp = 1.0 / (1.0 + kx * kx + ky * ky)
            spectrum[kx % Nux, ky % Nuy] = amp * (
                rng.standard_normal() + 1j * rng.standard_normal()
            )
    field = np.fft.ifft2(spectrum).real
    field = field / (np.std(field) + 1e-12)
    latent = 0.5 + ID_SEED_AMP * field
    return np.clip(latent, 0.05, 0.95).reshape(-1)


def make_legacy_x0():
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


def make_x0():
    if _MAPPING_MODE == "periodic_constrained":
        return make_periodic_latent_seed()
    return make_legacy_x0()


x0 = make_x0()
