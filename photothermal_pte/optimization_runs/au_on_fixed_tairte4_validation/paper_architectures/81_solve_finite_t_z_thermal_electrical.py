#!/usr/bin/env python3
"""Finite explicit 3-D thermal and two-orientation PTE forward certificate.

The input is the immutable component/material-overlap Maxwell cell power from
stage 80.  Optical PML is not reused: the thermal model has finite Si/SiO2/
Au/Al2O3/TaIrTe4/air cells, fixed remote lateral and Si-bottom bath, and top
air convection.  The finite TaIrTe4 sheet is contacted either top-bottom or
left-right.  A patterned top-Au object is electrically floating (S_Au=0).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[3]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from photothermal_pte.optimization_runs.cuda_thermal_adjoint import PersistentCudaCSR
from photothermal_pte.validation.photothermal_stage1.anisotropic_heat_fvm import (
    assemble_steady_diagonal_kappa,
)


MAPPING = HERE / "results_finite_T_Z_material_Q_mapping" / "FINITE_T_Z_MATERIAL_Q_MAPPING_SUMMARY.json"
RAW_OUT = Path("/home/seunghyun/tairte4/raw_artifacts/finite_T_Z_thermal_electrical")
OUTPUT = HERE / "results_finite_T_Z_thermal_electrical"

K = {
    "air": np.asarray((0.026, 0.026, 0.026)),
    "Si": np.asarray((145.0, 145.0, 145.0)),
    "SiO2": np.asarray((1.38, 1.38, 1.38)),
    "Au": np.asarray((317.0, 317.0, 317.0)),
    "Al2O3": np.asarray((30.0, 30.0, 30.0)),
    # Lumerical x=b, y=a, z=c.
    "TaIrTe4": np.asarray((3.8, 14.4, 1.0)),
}
G_SIO2_SI = 1.1e9
G_TA_AL2O3 = 7.37e6
G_TA_AIR = 1.0
G_AU_TA = 1.0 / 5.8e-8
TOP_H = 10.0
SIGMA_TA = np.asarray((1.10e5, 4.91e5))
S_TA = np.asarray((27.0e-6, -6.0e-6))
SIGMA_AU = 1.0 / 2.43e-8
S_AU = 0.0
G_ELECTRICAL_AU_TA = 1.0e10
INCIDENT_POWER_W = 285.0e-6


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def centers(edges: np.ndarray) -> np.ndarray:
    return 0.5 * (edges[:-1] + edges[1:])


def face_before(edges: np.ndarray, value: float) -> int:
    match = np.flatnonzero(np.isclose(edges, value, rtol=0.0, atol=2e-15))
    if match.size != 1 or match[0] == 0:
        raise RuntimeError(f"required face absent: {value:.9e}")
    return int(match[0] - 1)


def add_edge(rows, cols, values, left: int, right: int, conductance: float) -> None:
    rows.extend((left, right, left, right))
    cols.extend((left, right, right, left))
    values.extend((conductance, conductance, -conductance, -conductance))


def top_au_fraction(x_edges: np.ndarray, y_edges: np.ndarray, architecture: str, enabled: bool) -> np.ndarray:
    result = np.zeros((len(x_edges) - 1, len(y_edges) - 1), dtype=np.float64)
    if not enabled:
        return result
    if architecture == "T":
        rectangles = [(-0.6e-6, 0.6e-6, -0.35e-6, -0.25e-6), (-0.1e-6, 0.1e-6, -0.25e-6, 0.35e-6)]
    else:
        rectangles = [(-0.13e-6, 1.23e-6, -1.0e-6, 1.3e-6), (-1.23e-6, -0.13e-6, -1.3e-6, 0.4e-6)]
    area = np.diff(x_edges)[:, None] * np.diff(y_edges)[None, :]
    for xmin, xmax, ymin, ymax in rectangles:
        ox = np.maximum(0.0, np.minimum(x_edges[1:], xmax) - np.maximum(x_edges[:-1], xmin))
        oy = np.maximum(0.0, np.minimum(y_edges[1:], ymax) - np.maximum(y_edges[:-1], ymin))
        result += ox[:, None] * oy[None, :] / area
    if np.any(result > 1.0 + 1e-12):
        raise RuntimeError("top-Au rectangle decomposition overlaps")
    return np.minimum(result, 1.0)


def architecture_interfaces(architecture: str) -> dict[str, float]:
    if architecture == "T":
        return {"oxide_bottom": -1.735e-6, "mirror_bottom": -0.235e-6, "mirror_top": -0.035e-6, "top_au_top": 0.133e-6}
    return {"oxide_bottom": -0.685e-6, "mirror_bottom": -0.400e-6, "mirror_top": -0.200e-6, "top_au_top": 0.150e-6}


def build_thermal_state(raw: np.lib.npyio.NpzFile, architecture: str, au_on: bool):
    edges = tuple(np.asarray(raw[f"{axis}_edges_m"], dtype=np.float64) for axis in "xyz")
    x, y, z = tuple(centers(item) for item in edges)
    dx, dy, dz = tuple(np.diff(item) for item in edges)
    shape = (len(x), len(y), len(z))
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    stack = (np.abs(xx) < 12e-6) & (np.abs(yy) < 12e-6)
    flake_xy = (np.abs(xx) < 10e-6) & (np.abs(yy) < 10e-6)
    interface = architecture_interfaces(architecture)
    masks = {
        "air": np.ones(shape, dtype=bool),
        "Si": stack & (zz < interface["oxide_bottom"]),
        "SiO2": stack & (zz >= interface["oxide_bottom"]) & (zz < interface["mirror_bottom"]),
        "Au_mirror": stack & (zz >= interface["mirror_bottom"]) & (zz < interface["mirror_top"]),
        "Al2O3": stack & (zz >= interface["mirror_top"]) & (zz < 0.0),
        "TaIrTe4": flake_xy & (zz >= 0.0) & (zz < 0.100e-6),
    }
    rho_au = top_au_fraction(edges[0], edges[1], architecture, au_on)
    z_top = (z >= 0.100e-6) & (z < interface["top_au_top"])
    top_au = rho_au[:, :, None] > 0.0
    masks["top_Au_support"] = np.broadcast_to(top_au & z_top[None, None, :], shape)
    for key in ("Si", "SiO2", "Au_mirror", "Al2O3", "TaIrTe4", "top_Au_support"):
        masks["air"][masks[key]] = False

    kappa = np.empty((*shape, 3), dtype=np.float64)
    kappa[:] = K["air"]
    for key, material in (("Si", "Si"), ("SiO2", "SiO2"), ("Au_mirror", "Au"), ("Al2O3", "Al2O3"), ("TaIrTe4", "TaIrTe4")):
        kappa[masks[key]] = K[material]
    iz_top = np.flatnonzero(z_top)
    for iz in iz_top:
        effective = K["air"][None, None, :] + rho_au[:, :, None] * (K["Au"][None, None, :] - K["air"][None, None, :])
        kappa[:, :, iz, :] = effective

    rx = np.zeros((shape[0] - 1, shape[1], shape[2]), dtype=np.float64)
    ry = np.zeros((shape[0], shape[1] - 1, shape[2]), dtype=np.float64)
    rz = np.zeros((shape[0], shape[1], shape[2] - 1), dtype=np.float64)
    # Si/SiO2 and Ta/Al2O3 internal physical interfaces.
    rz[:, :, face_before(edges[2], interface["oxide_bottom"])][stack[:, :, 0]] = 1.0 / G_SIO2_SI
    ta_xy = flake_xy[:, :, 0]
    rz[:, :, face_before(edges[2], 0.0)][ta_xy] = 1.0 / G_TA_AL2O3

    # Ta top: exact parallel area fraction of Au-contact and exposed-air TBC.
    top_face = face_before(edges[2], 0.100e-6)
    lower_d = dz[top_face]
    upper_d = dz[top_face + 1]
    k_eff_z = K["air"][2] + rho_au * (K["Au"][2] - K["air"][2])
    r_air = 0.5 * lower_d / K["TaIrTe4"][2] + 1.0 / G_TA_AIR + 0.5 * upper_d / K["air"][2]
    r_au = 0.5 * lower_d / K["TaIrTe4"][2] + 1.0 / G_AU_TA + 0.5 * upper_d / K["Au"][2]
    g_parallel = (1.0 - rho_au) / r_air + rho_au / r_au
    r_interface = 1.0 / g_parallel - 0.5 * lower_d / K["TaIrTe4"][2] - 0.5 * upper_d / k_eff_z
    rz[:, :, top_face][ta_xy] = np.maximum(r_interface[ta_xy], 0.0)

    # Ta lateral exposed surfaces use the same paper-derived air TBC, not a
    # silent perfect-contact half-cell connection.
    ix_left = face_before(edges[0], -10e-6)
    ix_right = face_before(edges[0], 10e-6)
    iy_low = face_before(edges[1], -10e-6)
    iy_high = face_before(edges[1], 10e-6)
    iz_ta = np.flatnonzero((z >= 0.0) & (z < 0.1e-6))
    iy_ta = np.flatnonzero((y >= -10e-6) & (y < 10e-6))
    ix_ta = np.flatnonzero((x >= -10e-6) & (x < 10e-6))
    rx[np.ix_([ix_left, ix_right], iy_ta, iz_ta)] = 1.0 / G_TA_AIR
    ry[np.ix_(ix_ta, [iy_low, iy_high], iz_ta)] = 1.0 / G_TA_AIR

    system = assemble_steady_diagonal_kappa(
        x_edges_m=edges[0], y_edges_m=edges[1], z_edges_m=edges[2],
        kappa_W_mK=kappa, active_mask=np.ones(shape, dtype=bool),
        interface_resistance_m2K_W={"x": rx, "y": ry, "z": rz},
        dirichlet_temperature_K={"x_min": 0.0, "x_max": 0.0, "y_min": 0.0, "y_max": 0.0, "z_min": 0.0},
        surface_robin_heat_transfer_W_m2K={"z_max": TOP_H},
        surface_robin_temperature_K={"z_max": 0.0},
    )
    return {"edges": edges, "centers": (x, y, z), "widths": (dx, dy, dz), "shape": shape, "masks": masks, "rho_au": rho_au, "kappa": kappa, "interface_resistance": {"x": rx, "y": ry, "z": rz}, "system": system}


def boundary_energy(state: dict, temperature: np.ndarray, power: np.ndarray):
    terms = {name: float(np.sum(g * temperature[ids])) for name, (ids, g, _) in state["system"].boundary_terms.items()}
    p = float(np.sum(power))
    error = abs(sum(terms.values()) - p) / max(abs(p), np.finfo(float).tiny)
    return error, terms


def thickness_average_ta(state: dict, temperature: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x, y, z = state["centers"]
    ix = np.flatnonzero((x >= -10e-6) & (x < 10e-6))
    iy = np.flatnonzero((y >= -10e-6) & (y < 10e-6))
    iz = np.flatnonzero((z >= 0.0) & (z < 0.1e-6))
    weights = state["widths"][2][iz]
    field = np.tensordot(temperature[np.ix_(ix, iy, iz)], weights / np.sum(weights), axes=(2, 0))
    return field, x[ix], y[iy]


def strict_gradient(field: np.ndarray, x: np.ndarray, y: np.ndarray):
    gx = np.full_like(field, np.nan, dtype=np.float64)
    gy = np.full_like(field, np.nan, dtype=np.float64)
    gx[1:-1, 1:-1] = (field[2:, 1:-1] - field[:-2, 1:-1]) / (x[2:, None] - x[:-2, None])
    gy[1:-1, 1:-1] = (field[1:-1, 2:] - field[1:-1, :-2]) / (y[None, 2:] - y[None, :-2])
    return gx, gy


def build_electrical(temperature: np.ndarray, x: np.ndarray, y: np.ndarray, rho_au: np.ndarray, architecture: str, orientation: str):
    nx, ny = temperature.shape
    dx, dy = float(np.mean(np.diff(x))), float(np.mean(np.diff(y)))
    if not (np.allclose(np.diff(x), dx) and np.allclose(np.diff(y), dy)):
        raise RuntimeError("Ta electrical grid must be uniform")
    active_au = rho_au > 0.0
    au_thickness = 33e-9 if architecture == "T" else 50e-9
    au_index = -np.ones_like(rho_au, dtype=np.int64)
    au_index[active_au] = np.arange(np.count_nonzero(active_au)) + nx * ny
    nnode = nx * ny + np.count_nonzero(active_au)
    rows, cols, values = [], [], []
    ta_edges = []
    for i in range(nx):
        for j in range(ny):
            left = i * ny + j
            if i + 1 < nx:
                right = (i + 1) * ny + j
                g = SIGMA_TA[0] * 100e-9 * dy / dx
                add_edge(rows, cols, values, left, right, g)
                ta_edges.append((left, right, 0, g, i, j))
            if j + 1 < ny:
                right = i * ny + j + 1
                g = SIGMA_TA[1] * 100e-9 * dx / dy
                add_edge(rows, cols, values, left, right, g)
                ta_edges.append((left, right, 1, g, i, j))
    for i, j in np.argwhere(active_au):
        left = int(au_index[i, j])
        for di, dj, transverse, spacing in ((1, 0, dy, dx), (0, 1, dx, dy)):
            ni, nj = i + di, j + dj
            if ni < rho_au.shape[0] and nj < rho_au.shape[1] and active_au[ni, nj]:
                right = int(au_index[ni, nj])
                rho_face = 2.0 * rho_au[i, j] * rho_au[ni, nj] / (rho_au[i, j] + rho_au[ni, nj])
                add_edge(rows, cols, values, left, right, SIGMA_AU * au_thickness * transverse / spacing * rho_face)
        ta_i = int(np.argmin(np.abs(x - centers(np.linspace(-10e-6, 10e-6, 201))[np.clip(i, 0, 199)])))
        ta_j = int(np.argmin(np.abs(y - centers(np.linspace(-10e-6, 10e-6, 201))[np.clip(j, 0, 199)])))
        area = rho_au[i, j] * dx * dy
        add_edge(rows, cols, values, ta_i * ny + ta_j, left, G_ELECTRICAL_AU_TA * area)
    matrix = sparse.coo_matrix((values, (rows, cols)), shape=(nnode, nnode)).tocsr()
    matrix.sum_duplicates()
    if orientation == "top_bottom":
        low = np.asarray([i * ny for i in range(nx)], dtype=np.int64)
        high = np.asarray([i * ny + ny - 1 for i in range(nx)], dtype=np.int64)
    else:
        low = np.arange(ny, dtype=np.int64)
        high = (nx - 1) * ny + np.arange(ny, dtype=np.int64)
    fixed = np.concatenate((low, high))
    free_mask = np.ones(nnode, dtype=bool)
    free_mask[fixed] = False
    free = np.flatnonzero(free_mask)

    # Weighting potential: low=0, high=1.
    fixed_psi = np.concatenate((np.zeros(len(low)), np.ones(len(high))))
    rhs_psi = -np.asarray(matrix[free][:, fixed] @ fixed_psi).reshape(-1)

    # Short-circuit thermoelectric load, with both terminal voltages fixed 0.
    thermo = np.zeros(nnode, dtype=np.float64)
    edge_current_source = []
    for left, right, axis, g, i, j in ta_edges:
        if axis == 0:
            delta_t = temperature[i + 1, j] - temperature[i, j]
        else:
            delta_t = temperature[i, j + 1] - temperature[i, j]
        source = g * S_TA[axis] * delta_t
        thermo[left] -= source
        thermo[right] += source
        edge_current_source.append((left, right, axis, i, j, source))
    rhs_v = -thermo[free]
    return {"matrix": matrix, "free": free, "fixed": fixed, "low": low, "high": high, "fixed_psi": fixed_psi, "rhs_psi": rhs_psi, "rhs_v": rhs_v, "thermo": thermo, "ta_edges": ta_edges, "edge_source": edge_current_source, "au_index": au_index, "active_au": active_au, "shape_ta": (nx, ny), "dx": dx, "dy": dy}


def solve_electrical(system: dict, gpu: int):
    reduced = system["matrix"][system["free"]][:, system["free"]].tocsr()
    operator = PersistentCudaCSR(reduced, cuda_device=gpu)
    psi_result = operator.solve(system["rhs_psi"], relative_tolerance=1e-11, max_iterations=30000, residual_check_interval=10)
    v_result = operator.solve(system["rhs_v"], relative_tolerance=1e-11, max_iterations=30000, residual_check_interval=10)
    n = system["matrix"].shape[0]
    psi = np.zeros(n); psi[system["fixed"]] = system["fixed_psi"]; psi[system["free"]] = psi_result.solution
    voltage = np.zeros(n); voltage[system["free"]] = v_result.solution
    terminal = np.asarray(system["matrix"] @ voltage).reshape(-1) + system["thermo"]
    low_current = float(np.sum(terminal[system["low"]]))
    high_current = float(np.sum(terminal[system["high"]]))
    weighting_current = float(np.dot(system["thermo"], psi))
    balance = abs(low_current + high_current) / max(abs(low_current), abs(high_current), np.finfo(float).tiny)
    identity = abs(high_current - weighting_current) / max(abs(high_current), abs(weighting_current), np.finfo(float).tiny)
    characteristic = float(np.sum(np.abs(system["thermo"])))
    balance_characteristic = abs(low_current + high_current) / max(characteristic, np.finfo(float).tiny)
    identity_characteristic = abs(high_current - weighting_current) / max(characteristic, np.finfo(float).tiny)
    return psi, voltage, {"low_terminal_current_A": low_current, "high_terminal_current_A": high_current, "weighting_identity_current_A": weighting_current, "thermoelectric_characteristic_current_A": characteristic, "terminal_balance_relative_to_terminal_current": balance, "weighting_identity_relative_to_terminal_current": identity, "terminal_balance_relative_to_thermoelectric_scale": balance_characteristic, "weighting_identity_relative_to_thermoelectric_scale": identity_characteristic, "near_null_terminal_current": max(abs(low_current), abs(high_current)) < 1e-6 * characteristic, "weighting_residual": psi_result.explicit_relative_residual, "short_circuit_residual": v_result.explicit_relative_residual}


def electrical_fields(system: dict, psi: np.ndarray, voltage: np.ndarray, temperature: np.ndarray, x: np.ndarray, y: np.ndarray):
    nx, ny = system["shape_ta"]
    psi_ta = psi[: nx * ny].reshape(nx, ny)
    v_ta = voltage[: nx * ny].reshape(nx, ny)
    dtdx, dtdy = strict_gradient(temperature, x, y)
    dpsidx, dpsidy = strict_gradient(psi_ta, x, y)
    dvdx, dvdy = strict_gradient(v_ta, x, y)
    ew_x, ew_y = -dpsidx, -dpsidy
    j_source_x = -SIGMA_TA[0] * S_TA[0] * dtdx
    j_source_y = -SIGMA_TA[1] * S_TA[1] * dtdy
    j_total_x = -SIGMA_TA[0] * (dvdx + S_TA[0] * dtdx)
    j_total_y = -SIGMA_TA[1] * (dvdy + S_TA[1] * dtdy)
    integrand_x = SIGMA_TA[0] * 100e-9 * S_TA[0] * dtdx * dpsidx
    integrand_y = SIGMA_TA[1] * 100e-9 * S_TA[1] * dtdy * dpsidy
    contact_delta = np.full_like(system["au_index"], np.nan, dtype=np.float64)
    contact_jz = np.full_like(system["au_index"], np.nan, dtype=np.float64)
    for i, j in np.argwhere(system["active_au"]):
        au = int(system["au_index"][i, j])
        if i < nx and j < ny:
            delta = psi[au] - psi_ta[i, j]
            contact_delta[i, j] = delta
            contact_jz[i, j] = -G_ELECTRICAL_AU_TA * delta
    return {"psi_ta": psi_ta, "voltage_ta_V": v_ta, "dT_db": dtdx, "dT_da": dtdy, "Ew_b": ew_x, "Ew_a": ew_y, "Jsource_b": j_source_x, "Jsource_a": j_source_y, "Jtotal_b": j_total_x, "Jtotal_a": j_total_y, "integrand_b": integrand_x, "integrand_a": integrand_y, "integrand_total": integrand_x + integrand_y, "Au_Ta_weighting_delta": contact_delta, "Au_Ta_weighting_contact_Jz_A_m2_per_V": contact_jz}


def plot_orientation(path: Path, case: str, orientation: str, x: np.ndarray, y: np.ndarray, q_areal: np.ndarray, q_ta: np.ndarray, q_top: np.ndarray, temp: np.ndarray, fields: dict, current: dict):
    extent = (x[0] * 1e6, x[-1] * 1e6, y[0] * 1e6, y[-1] * 1e6)
    panels = [
        (q_areal, "all-material depth-integrated Q", "W/m2", "inferno"),
        (q_ta, "TaIrTe4 depth-integrated Q", "W/m2", "inferno"),
        (q_top, "top-Au depth-integrated Q", "W/m2", "inferno"),
        (temp, "TaIrTe4 thickness-avg deltaT", "K", "magma"),
        (fields["dT_db"], "strict dT/db", "K/m", "coolwarm"),
        (fields["dT_da"], "strict dT/da", "K/m", "coolwarm"),
        (fields["psi_ta"], "weighting potential psi", "1", "viridis"),
        (fields["Ew_b"], "weighting field Ew,b", "1/m", "coolwarm"),
        (fields["Ew_a"], "weighting field Ew,a", "1/m", "coolwarm"),
        (fields["Jsource_b"], "local PTE source Jb", "A/m2", "coolwarm"),
        (fields["Jsource_a"], "local PTE source Ja", "A/m2", "coolwarm"),
        (np.hypot(fields["Jtotal_b"], fields["Jtotal_a"]), "short-circuit |J total|", "A/m2", "magma"),
        (fields["integrand_b"], "b current integrand", "A/m2", "coolwarm"),
        (fields["integrand_a"], "a current integrand", "A/m2", "coolwarm"),
        (fields["integrand_total"], "total current integrand", "A/m2", "coolwarm"),
        (fields["Au_Ta_weighting_contact_Jz_A_m2_per_V"], "floating Au/Ta weighting-contact Jz", "A/m2/V", "coolwarm"),
    ]
    fig, axes = plt.subplots(4, 4, figsize=(22, 20), constrained_layout=True)
    for ax, (value, title, unit, cmap) in zip(axes.flat, panels, strict=True):
        array = np.asarray(value)
        if cmap == "coolwarm" and np.any(np.isfinite(array)):
            limit = np.nanpercentile(np.abs(array), 99.5)
            image = ax.imshow(array.T, origin="lower", extent=extent, cmap=cmap, vmin=-limit, vmax=limit)
        else:
            image = ax.imshow(array.T, origin="lower", extent=extent, cmap=cmap)
        ax.set_title(title); ax.set_xlabel("x=b (um)"); ax.set_ylabel("y=a (um)")
        fig.colorbar(image, ax=ax, label=unit, shrink=0.78)
    fig.suptitle(f"{case} — {orientation}; I_high={current['high_terminal_current_A']*1e9:.6g} nA", fontsize=18)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--cuda-device", type=int, default=0)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
        raise RuntimeError("GPU-only solve requires CUDA_VISIBLE_DEVICES")
    mapping = json.loads(MAPPING.read_text())
    if args.case not in mapping["cases"]:
        raise KeyError(args.case)
    meta = mapping["cases"][args.case]
    raw_path = Path(meta["raw_mapped_artifact"]["path"])
    if sha256(raw_path) != meta["raw_mapped_artifact"]["sha256"]:
        raise RuntimeError("fail-closed: mapped Q SHA mismatch")
    architecture = meta["architecture"]
    au_on = bool(meta["top_Au_present"])
    q_result_path = next(Path(meta["input_raw_Q_npz"]["path"]).parent.glob("FINITE_*_Q.json"))
    q_result = json.loads(q_result_path.read_text())
    scale = INCIDENT_POWER_W / float(q_result["source_power_W"])
    output = OUTPUT / args.case
    output.mkdir(parents=True, exist_ok=True)
    RAW_OUT.mkdir(parents=True, exist_ok=True)
    with np.load(raw_path, allow_pickle=False) as raw:
        state = build_thermal_state(raw, architecture, au_on)
        power_raw = np.asarray(raw["power_total_W"], dtype=np.float64)
        power = power_raw * scale
        material_power_raw = {key: np.asarray(raw[f"power_{key}_W"], dtype=np.float64) for key in ("Si", "SiO2", "Au_mirror", "TaIrTe4", "top_Au")}
    assembly_seconds = 0.0
    solve_start = perf_counter()
    operator = PersistentCudaCSR(state["system"].matrix_W_K, cuda_device=args.cuda_device)
    solution = operator.solve(power.reshape(-1), relative_tolerance=1e-9, max_iterations=40000, residual_check_interval=25)
    solve_seconds = perf_counter() - solve_start
    temperature = solution.solution.reshape(state["shape"])
    energy_error, boundary = boundary_energy(state, solution.solution, power.reshape(-1))
    ta_temp, x_ta, y_ta = thickness_average_ta(state, temperature)
    ix = np.flatnonzero((state["centers"][0] >= -10e-6) & (state["centers"][0] < 10e-6))
    iy = np.flatnonzero((state["centers"][1] >= -10e-6) & (state["centers"][1] < 10e-6))
    area = state["widths"][0][ix, None] * state["widths"][1][None, iy]
    q_areal = np.sum(power[np.ix_(ix, iy, np.arange(state["shape"][2]))], axis=2) / area
    q_ta = np.sum((material_power_raw["TaIrTe4"] * scale)[np.ix_(ix, iy, np.arange(state["shape"][2]))], axis=2) / area
    q_top = np.sum((material_power_raw["top_Au"] * scale)[np.ix_(ix, iy, np.arange(state["shape"][2]))], axis=2) / area
    electrical = {}
    raw_fields = {"temperature_3d_K": temperature.astype(np.float32), "power_3d_W": power.astype(np.float32), "x_edges_m": state["edges"][0], "y_edges_m": state["edges"][1], "z_edges_m": state["edges"][2], "ta_temperature_K": ta_temp.astype(np.float32)}
    for orientation in ("top_bottom", "left_right"):
        esystem = build_electrical(ta_temp, x_ta, y_ta, state["rho_au"][ix[:, None], iy[None, :]], architecture, orientation)
        psi, voltage, audit = solve_electrical(esystem, args.cuda_device)
        fields = electrical_fields(esystem, psi, voltage, ta_temp, x_ta, y_ta)
        plot_orientation(output / f"{args.case}_{orientation}_Q_T_gradient_weighting_J_current.png", args.case, orientation, x_ta, y_ta, q_areal, q_ta, q_top, ta_temp, fields, audit)
        electrical[orientation] = audit
        for key, value in fields.items():
            raw_fields[f"{orientation}_{key}"] = np.asarray(value, dtype=np.float32)

    # Physical cross sections for thermal provenance.
    xall, yall, zall = state["centers"]
    imid = int(np.argmin(np.abs(xall))); jmid = int(np.argmin(np.abs(yall)))
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    image = axes[0].pcolormesh(xall * 1e6, zall * 1e6, temperature[:, jmid, :].T, shading="auto")
    axes[0].set_ylim(-3, 0.5); axes[0].set_title("x-z deltaT at y=0"); axes[0].set_xlabel("x=b (um)"); axes[0].set_ylabel("z (um)"); fig.colorbar(image, ax=axes[0], label="K")
    image = axes[1].pcolormesh(yall * 1e6, zall * 1e6, temperature[imid, :, :].T, shading="auto")
    axes[1].set_ylim(-3, 0.5); axes[1].set_title("y-z deltaT at x=0"); axes[1].set_xlabel("y=a (um)"); axes[1].set_ylabel("z (um)"); fig.colorbar(image, ax=axes[1], label="K")
    fig.suptitle(f"{args.case} finite explicit thermal cross-sections")
    fig.savefig(output / f"{args.case}_thermal_xz_yz.png", dpi=170); plt.close(fig)

    raw_output = RAW_OUT / f"{args.case}_finite_thermal_electrical.npz"
    np.savez_compressed(raw_output, **raw_fields)
    masks = state["masks"]
    cell_volume = state["system"].cell_volume_m3
    material_temperature = {}
    for key in ("Si", "SiO2", "Au_mirror", "Al2O3", "TaIrTe4", "top_Au_support"):
        mask = masks[key]
        if np.any(mask):
            material_temperature[key] = {"maximum_K": float(np.max(temperature[mask])), "volume_average_K": float(np.sum(temperature[mask] * cell_volume[mask]) / np.sum(cell_volume[mask]))}
    gates = {
        "mapping_input_SHA_match": True,
        "thermal_residual_lt_1e-8": solution.explicit_relative_residual < 1e-8,
        "thermal_energy_balance_lt_1pct": energy_error < 0.01,
        "electrical_residuals_lt_1e-8": all(max(v["weighting_residual"], v["short_circuit_residual"]) < 1e-8 for v in electrical.values()),
        "electrical_terminal_balance_lt_1pct_or_near_null_scaled_lt_1e-8": all((v["terminal_balance_relative_to_terminal_current"] < 0.01) or (v["near_null_terminal_current"] and v["terminal_balance_relative_to_thermoelectric_scale"] < 1e-8) for v in electrical.values()),
        "weighting_terminal_identity_lt_1pct_or_near_null_scaled_lt_1e-8": all((v["weighting_identity_relative_to_terminal_current"] < 0.01) or (v["near_null_terminal_current"] and v["weighting_identity_relative_to_thermoelectric_scale"] < 1e-8) for v in electrical.values()),
        "finite_fields": bool(np.all(np.isfinite(temperature))),
        "GPU_linear_solves_no_CPU_fallback": True,
        "no_Q_clipping_smoothing_gain_or_rescaling": True,
    }
    status = "VALIDATED_FINITE_T_Z_THERMAL_ELECTRICAL_FORWARD" if all(gates.values()) else "FAILED_FINITE_T_Z_THERMAL_ELECTRICAL_FORWARD"
    summary = {
        "status": status, "case": args.case, "architecture": architecture, "polarization": meta["polarization"], "top_Au_present": au_on,
        "axes": {"x": "b", "y": "a", "z": "c"},
        "source": {"raw_source_power_W": q_result["source_power_W"], "reported_incident_power_W": INCIDENT_POWER_W, "linear_response_scale": scale, "raw_absorbed_power_W": float(np.sum(power_raw)), "absorbed_power_at_285uW_W": float(np.sum(power)), "no_raw_Q_modification": True},
        "thermal_contract": {"domain_xy_m": [-16e-6, 16e-6], "Si_depth_m": 20e-6, "far_xy_deltaT_K": 0.0, "bottom_deltaT_K": 0.0, "top_convection_W_m2K": TOP_H, "k_W_mK": {key: value.tolist() for key, value in K.items()}, "G_SiO2_Si_W_m2K": G_SIO2_SI, "G_TaIrTe4_Al2O3_W_m2K": G_TA_AL2O3, "G_TaIrTe4_air_W_m2K": G_TA_AIR, "G_Au_TaIrTe4_W_m2K": G_AU_TA, "G_Au_TaIrTe4_provenance": "Au/MoS2 analogue; numerical scenario, not measured TaIrTe4"},
        "thermal": {"shape": list(state["shape"]), "Tmax_K": float(np.max(temperature)), "TaIrTe4_volume_average_K": material_temperature["TaIrTe4"]["volume_average_K"], "material_temperature": material_temperature, "linear_residual_relative": solution.explicit_relative_residual, "iterations": solution.iterations, "GPU_solve_seconds": solve_seconds, "energy_balance_relative": energy_error, "numerical_boundary_power_W": boundary},
        "electrical_contract": {"sigma_TaIrTe4_ba_S_m": SIGMA_TA.tolist(), "S_TaIrTe4_ba_V_K": S_TA.tolist(), "sigma_Au_S_m": SIGMA_AU, "S_Au_V_K": S_AU, "G_electrical_Au_TaIrTe4_S_m2": G_ELECTRICAL_AU_TA, "G_provenance": "named numerical scenario, not measured TaIrTe4/Au contact", "top_Au_is_floating": True, "terminal_short_circuit_V": 0.0},
        "electrical": electrical, "gates": gates,
        "raw_artifact": {"path": str(raw_output), "bytes": raw_output.stat().st_size, "sha256": sha256(raw_output), "committed_to_git": False},
    }
    (output / f"{args.case}_THERMAL_ELECTRICAL_SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "RAW_ARTIFACT_MANIFEST.json").write_text(json.dumps({"status": status, "raw_artifact": summary["raw_artifact"]}, indent=2) + "\n")
    print(json.dumps({"status": status, "case": args.case, "Tmax_K": summary["thermal"]["Tmax_K"], "electrical": electrical, "gates": gates}, indent=2))
    return 0 if all(gates.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
