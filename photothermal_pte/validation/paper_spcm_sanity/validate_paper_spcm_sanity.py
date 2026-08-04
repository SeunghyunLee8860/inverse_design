#!/usr/bin/env python3
"""Minimal independent reproduction of the paper's 2-D SPCM mechanism.

This is intentionally a paper sanity check, not the inverse-design model.
It follows Fig. 1F/I and Supplementary Note S5:

* a thin TaIrTe4 rectangle with insulating lateral crystal edges;
* anisotropic in-plane thermal conduction;
* explicit top-air and bottom-thermal-SiO2 Robin heat loss;
* a scanned Gaussian absorbed-power source;
* a Shockley--Ramo weighting potential with full-width electrodes; and
* the paper's sigma, Seebeck, kappa, and interface-G values.

The paper does not publish the exact Fig. 1 simulation dimensions, mesh,
635-nm absorbed fraction, or a numerical beam radius.  Consequently the
comparison is performed in absorbed-power-normalized units and emphasizes
the reported symmetry/sign and longitudinal-to-transverse mechanism.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from scipy.interpolate import RegularGridInterpolator
from scipy.signal import fftconvolve
from scipy.sparse import linalg as spla


PAPER = Path(
    "/home/seunghyun/tairte4/papers/"
    "Adv Funct Materials - 2026 - Blevins - Large Transverse "
    "Thermoelectric Effect in Weyl Semimetal TaIrTe4 Engineered for-2.pdf"
)
SUPPLEMENT = Path(
    "/home/seunghyun/tairte4/papers/adfm75986-sup-0001-suppmat-2.pdf"
)

KAPPA_A_W_MK = 14.4
KAPPA_B_W_MK = 3.8
SIGMA_A_S_M = 4.91e5
SIGMA_B_S_M = 1.10e5
SEEBECK_A_V_K = -6.0e-6
SEEBECK_B_V_K = 27.0e-6
G_TOP_AIR_W_M2K = 1.0
G_BOTTOM_THERMAL_SIO2_W_M2K = 7.37e6
T_BATH_K = 300.0

# The exact simplified-Fig.-1 dimensions are not published.  These values
# preserve the approximate aspect ratio in Fig. 1I and make all assumptions
# explicit instead of reverse-engineering pixels from the illustration.
WIDTH_M = 6.0e-6
HEIGHT_M = 8.0e-6
THICKNESS_M = 130.0e-9
BEAM_RADIUS_M = 0.50e-6


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rotation(angle_deg: float) -> np.ndarray:
    angle = np.deg2rad(angle_deg)
    return np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )


def rotated_tensor(a_value: float, b_value: float, angle_deg: float) -> np.ndarray:
    rot = rotation(angle_deg)
    return rot @ np.diag([a_value, b_value]) @ rot.T


@dataclass
class Mesh:
    x_m: np.ndarray
    y_m: np.ndarray
    nodes_m: np.ndarray
    triangles: np.ndarray
    triangle_area_m2: np.ndarray
    gradients_m_inv: np.ndarray
    lumped_area_m2: np.ndarray

    @property
    def shape(self) -> tuple[int, int]:
        return (self.x_m.size, self.y_m.size)


def build_mesh(step_m: float) -> Mesh:
    nx = int(round(WIDTH_M / step_m)) + 1
    ny = int(round(HEIGHT_M / step_m)) + 1
    x = np.linspace(-0.5 * WIDTH_M, 0.5 * WIDTH_M, nx)
    y = np.linspace(-0.5 * HEIGHT_M, 0.5 * HEIGHT_M, ny)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    nodes = np.column_stack([xx.ravel(), yy.ravel()])
    ids = np.arange(nx * ny).reshape(nx, ny)
    lower = np.column_stack(
        [
            ids[:-1, :-1].ravel(),
            ids[1:, :-1].ravel(),
            ids[1:, 1:].ravel(),
        ]
    )
    upper = np.column_stack(
        [
            ids[:-1, :-1].ravel(),
            ids[1:, 1:].ravel(),
            ids[:-1, 1:].ravel(),
        ]
    )
    triangles = np.vstack([lower, upper]).astype(np.int64)
    coordinates = nodes[triangles]
    twice_area = (
        (coordinates[:, 1, 0] - coordinates[:, 0, 0])
        * (coordinates[:, 2, 1] - coordinates[:, 0, 1])
        - (coordinates[:, 2, 0] - coordinates[:, 0, 0])
        * (coordinates[:, 1, 1] - coordinates[:, 0, 1])
    )
    if np.any(twice_area <= 0.0):
        raise RuntimeError("triangles must have positive orientation")
    area = 0.5 * twice_area
    gradients = np.empty((triangles.shape[0], 2, 3), float)
    gradients[:, 0, :] = np.column_stack(
        [
            coordinates[:, 1, 1] - coordinates[:, 2, 1],
            coordinates[:, 2, 1] - coordinates[:, 0, 1],
            coordinates[:, 0, 1] - coordinates[:, 1, 1],
        ]
    ) / twice_area[:, None]
    gradients[:, 1, :] = np.column_stack(
        [
            coordinates[:, 2, 0] - coordinates[:, 1, 0],
            coordinates[:, 0, 0] - coordinates[:, 2, 0],
            coordinates[:, 1, 0] - coordinates[:, 0, 0],
        ]
    ) / twice_area[:, None]
    lumped = np.zeros(nodes.shape[0], float)
    np.add.at(lumped, triangles.ravel(), np.repeat(area / 3.0, 3))
    return Mesh(x, y, nodes, triangles, area, gradients, lumped)


def assemble_matrix(
    mesh: Mesh,
    tensor: np.ndarray,
    *,
    thickness_m: float,
    surface_sink_W_m2K: float,
) -> sparse.csr_matrix:
    gradients = mesh.gradients_m_inv
    conduction = thickness_m * mesh.triangle_area_m2[:, None, None] * np.einsum(
        "eai,ab,ebj->eij", gradients, tensor, gradients
    )
    mass_pattern = np.ones((3, 3), float) + np.eye(3)
    sink = (
        surface_sink_W_m2K
        * mesh.triangle_area_m2[:, None, None]
        * mass_pattern[None, :, :]
        / 12.0
    )
    element = conduction + sink
    rows = np.repeat(mesh.triangles, 3, axis=1).ravel()
    columns = np.tile(mesh.triangles, (1, 3)).ravel()
    return sparse.coo_matrix(
        (element.ravel(), (rows, columns)),
        shape=(mesh.nodes_m.shape[0], mesh.nodes_m.shape[0]),
    ).tocsr()


def solve_weighting_potential(mesh: Mesh) -> tuple[np.ndarray, np.ndarray, float]:
    """Solve paper Eq. S7, Laplace psi=0/1 with insulating side edges."""
    laplace = assemble_matrix(
        mesh,
        np.eye(2),
        thickness_m=1.0,
        surface_sink_W_m2K=0.0,
    )
    nx, ny = mesh.shape
    node_ids = np.arange(nx * ny).reshape(nx, ny)
    bottom = node_ids[:, 0]
    top = node_ids[:, -1]
    fixed = np.concatenate([bottom, top])
    fixed_values = np.concatenate(
        [np.zeros(bottom.size), np.ones(top.size)]
    )
    free_mask = np.ones(nx * ny, bool)
    free_mask[fixed] = False
    free = np.flatnonzero(free_mask)
    rhs = -(laplace[free][:, fixed] @ fixed_values)
    psi = np.zeros(nx * ny, float)
    psi[fixed] = fixed_values
    psi[free] = spla.spsolve(laplace[free][:, free].tocsc(), rhs)
    grad_psi = np.einsum(
        "eai,ei->ea",
        mesh.gradients_m_inv,
        psi[mesh.triangles],
    )
    expected = np.array([0.0, 1.0 / HEIGHT_M])
    mismatch = np.max(np.linalg.norm(grad_psi - expected, axis=1))
    relative_mismatch = float(mismatch / np.linalg.norm(expected))
    return psi, grad_psi, relative_mismatch


def build_current_vector(
    mesh: Mesh,
    grad_psi: np.ndarray,
    angle_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    # sigma*S is diagonal in the a/b crystal frame.  Rotating the product is
    # equivalent to rotating sigma and S separately because they share axes.
    current_tensor = rotated_tensor(
        SIGMA_A_S_M * SEEBECK_A_V_K,
        SIGMA_B_S_M * SEEBECK_B_V_K,
        angle_deg,
    )
    coefficients = np.empty((mesh.triangles.shape[0], 3), float)
    for local in range(3):
        grad_basis = mesh.gradients_m_inv[:, :, local]
        local_current_per_K = -(grad_basis @ current_tensor.T)
        coefficients[:, local] = (
            THICKNESS_M
            * mesh.triangle_area_m2
            * np.einsum("ea,ea->e", local_current_per_K, grad_psi)
        )
    vector = np.zeros(mesh.nodes_m.shape[0], float)
    np.add.at(vector, mesh.triangles.ravel(), coefficients.ravel())
    return vector, current_tensor


def gaussian_kernel(mesh: Mesh, beam_radius_m: float) -> np.ndarray:
    xx, yy = np.meshgrid(mesh.x_m, mesh.y_m, indexing="ij")
    # Unit total absorbed power on an unbounded plane, paper Eq. S1.
    return 2.0 / (np.pi * beam_radius_m**2) * np.exp(
        -2.0 * (xx**2 + yy**2) / beam_radius_m**2
    )


def map_metrics(
    current_A_per_W: np.ndarray,
    x_m: np.ndarray,
    y_m: np.ndarray,
    *,
    angle_deg: float,
) -> dict[str, float]:
    xx, yy = np.meshgrid(x_m, y_m, indexing="ij")
    strip = 0.75e-6
    corner_exclusion = 1.0e-6
    side = (
        (np.abs(xx) >= 0.5 * WIDTH_M - strip)
        & (np.abs(yy) <= 0.5 * HEIGHT_M - corner_exclusion)
    )
    electrode = (
        (np.abs(yy) >= 0.5 * HEIGHT_M - strip)
        & (np.abs(xx) <= 0.5 * WIDTH_M - corner_exclusion)
    )
    side_peak = float(np.max(np.abs(current_A_per_W[side])))
    electrode_peak = float(np.max(np.abs(current_A_per_W[electrode])))
    centre_x = int(np.argmin(np.abs(x_m)))
    centre_y = int(np.argmin(np.abs(y_m)))
    left_mid = float(current_A_per_W[0, centre_y])
    right_mid = float(current_A_per_W[-1, centre_y])
    bottom_mid = float(current_A_per_W[centre_x, 0])
    top_mid = float(current_A_per_W[centre_x, -1])
    if angle_deg == 0.0:
        symmetry_residual = np.linalg.norm(
            current_A_per_W + current_A_per_W[:, ::-1]
        ) / np.linalg.norm(current_A_per_W)
    else:
        symmetry_residual = np.linalg.norm(
            current_A_per_W + current_A_per_W[::-1, :]
        ) / np.linalg.norm(current_A_per_W)
    return {
        "map_peak_abs_A_per_Wabsorbed": float(
            np.max(np.abs(current_A_per_W))
        ),
        "side_edge_peak_abs_A_per_Wabsorbed": side_peak,
        "electrode_peak_abs_A_per_Wabsorbed": electrode_peak,
        "side_to_electrode_peak_ratio": side_peak / max(electrode_peak, 1e-300),
        "expected_odd_symmetry_relative_residual": float(symmetry_residual),
        "left_mid_A_per_Wabsorbed": left_mid,
        "right_mid_A_per_Wabsorbed": right_mid,
        "bottom_mid_A_per_Wabsorbed": bottom_mid,
        "top_mid_A_per_Wabsorbed": top_mid,
        "left_right_opposite_sign": bool(left_mid * right_mid < 0.0),
        "bottom_top_opposite_sign": bool(bottom_mid * top_mid < 0.0),
    }


def run_case(step_m: float, angle_deg: float, beam_radius_m: float) -> dict[str, Any]:
    mesh = build_mesh(step_m)
    kappa = rotated_tensor(KAPPA_A_W_MK, KAPPA_B_W_MK, angle_deg)
    thermal = assemble_matrix(
        mesh,
        kappa,
        thickness_m=THICKNESS_M,
        surface_sink_W_m2K=(
            G_TOP_AIR_W_M2K + G_BOTTOM_THERMAL_SIO2_W_M2K
        ),
    )
    psi, grad_psi, weighting_error = solve_weighting_potential(mesh)
    current_vector, current_tensor = build_current_vector(
        mesh, grad_psi, angle_deg
    )
    factor = spla.factorized(thermal.tocsc())
    adjoint = factor(current_vector)
    kernel = gaussian_kernel(mesh, beam_radius_m)
    current_map = fftconvolve(
        (adjoint * mesh.lumped_area_m2).reshape(mesh.shape),
        kernel,
        mode="same",
    )

    # One direct forward solve near the right edge checks energy balance,
    # residual, and the current-vector/adjoint identity.
    centre = np.array([0.5 * WIDTH_M - 0.25e-6, 0.0])
    radius2 = np.sum((mesh.nodes_m - centre[None, :]) ** 2, axis=1)
    source_density = 2.0 / (np.pi * beam_radius_m**2) * np.exp(
        -2.0 * radius2 / beam_radius_m**2
    )
    load = mesh.lumped_area_m2 * source_density
    temperature_rise = factor(load)
    direct_current = float(np.dot(current_vector, temperature_rise))
    adjoint_current = float(np.dot(adjoint, load))
    residual = thermal @ temperature_rise - load
    residual_relative = float(
        np.linalg.norm(residual) / max(np.linalg.norm(load), 1e-300)
    )
    sink_power = float(
        (G_TOP_AIR_W_M2K + G_BOTTOM_THERMAL_SIO2_W_M2K)
        * np.dot(mesh.lumped_area_m2, temperature_rise)
    )
    input_power = float(np.sum(load))
    metrics = map_metrics(
        current_map, mesh.x_m, mesh.y_m, angle_deg=angle_deg
    )
    metrics.update(
        {
            "step_nm": step_m * 1e9,
            "angle_deg": angle_deg,
            "beam_radius_um": beam_radius_m * 1e6,
            "nodes": int(mesh.nodes_m.shape[0]),
            "triangles": int(mesh.triangles.shape[0]),
            "weighting_gradient_relative_error": weighting_error,
            "right_edge_absorbed_fraction": input_power,
            "right_edge_temperature_rise_max_K_per_Wabsorbed": float(
                np.max(temperature_rise)
            ),
            "right_edge_direct_current_A_per_Wabsorbed": direct_current,
            "right_edge_adjoint_current_A_per_Wabsorbed": adjoint_current,
            "forward_adjoint_relative_difference": abs(
                direct_current - adjoint_current
            )
            / max(abs(direct_current), abs(adjoint_current), 1e-300),
            "linear_residual_relative": residual_relative,
            "energy_balance_relative": abs(input_power - sink_power)
            / max(abs(input_power), abs(sink_power), 1e-300),
            "input_power_fraction_inside_flake": input_power,
            "sink_power_fraction": sink_power,
            "current_tensor_xx_A_mK": float(current_tensor[0, 0]),
            "current_tensor_xy_A_mK": float(current_tensor[0, 1]),
            "current_tensor_yx_A_mK": float(current_tensor[1, 0]),
            "current_tensor_yy_A_mK": float(current_tensor[1, 1]),
        }
    )
    return {
        "mesh": mesh,
        "temperature_rise": temperature_rise.reshape(mesh.shape),
        "psi": psi.reshape(mesh.shape),
        "current_map": current_map,
        "adjoint": adjoint.reshape(mesh.shape),
        "metrics": metrics,
    }


def normalized_nrmse(reference: np.ndarray, candidate: np.ndarray) -> float:
    reference = np.asarray(reference, float)
    candidate = np.asarray(candidate, float)
    scale = max(float(np.max(reference) - np.min(reference)), 1e-300)
    return float(np.sqrt(np.mean((candidate - reference) ** 2)) / scale)


def compare_grids(coarse: dict[str, Any], fine: dict[str, Any]) -> dict[str, float]:
    coarse_mesh: Mesh = coarse["mesh"]
    fine_mesh: Mesh = fine["mesh"]
    interpolator = RegularGridInterpolator(
        (coarse_mesh.x_m, coarse_mesh.y_m),
        coarse["current_map"],
        bounds_error=True,
    )
    xx, yy = np.meshgrid(fine_mesh.x_m, fine_mesh.y_m, indexing="ij")
    coarse_on_fine = interpolator(np.column_stack([xx.ravel(), yy.ravel()])).reshape(
        fine_mesh.shape
    )
    return {
        "map_nrmse_range_normalized": normalized_nrmse(
            fine["current_map"], coarse_on_fine
        ),
        "map_peak_relative_difference": abs(
            coarse["metrics"]["map_peak_abs_A_per_Wabsorbed"]
            - fine["metrics"]["map_peak_abs_A_per_Wabsorbed"]
        )
        / fine["metrics"]["map_peak_abs_A_per_Wabsorbed"],
        "side_to_electrode_ratio_relative_difference": abs(
            coarse["metrics"]["side_to_electrode_peak_ratio"]
            - fine["metrics"]["side_to_electrode_peak_ratio"]
        )
        / max(fine["metrics"]["side_to_electrode_peak_ratio"], 1e-300),
    }


def plot_maps(
    aligned: dict[str, Any],
    transverse: dict[str, Any],
    output: Path,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15.0, 9.0), constrained_layout=True)
    for row, (case, title) in enumerate(
        [(aligned, "a-axis aligned (longitudinal PTE)"), (transverse, "a-axis at 45° (transverse PTE)")]
    ):
        mesh: Mesh = case["mesh"]
        extent = [
            mesh.x_m[0] * 1e6,
            mesh.x_m[-1] * 1e6,
            mesh.y_m[0] * 1e6,
            mesh.y_m[-1] * 1e6,
        ]
        temp = case["temperature_rise"]
        pte = case["current_map"]
        norm = pte / np.max(np.abs(pte))
        im0 = axes[row, 0].imshow(
            temp.T,
            origin="lower",
            extent=extent,
            aspect="equal",
            cmap="inferno",
        )
        fig.colorbar(im0, ax=axes[row, 0], label=r"$\Delta T/P_{\rm abs}$ (K/W)")
        axes[row, 0].set_title(f"{title}\nright-edge heating")
        im1 = axes[row, 1].imshow(
            case["psi"].T,
            origin="lower",
            extent=extent,
            aspect="equal",
            cmap="viridis",
            vmin=0,
            vmax=1,
        )
        fig.colorbar(im1, ax=axes[row, 1], label=r"weighting potential $\psi$")
        axes[row, 1].set_title("Shockley–Ramo weighting potential")
        im2 = axes[row, 2].imshow(
            norm.T,
            origin="lower",
            extent=extent,
            aspect="equal",
            cmap="RdBu_r",
            vmin=-1,
            vmax=1,
        )
        fig.colorbar(im2, ax=axes[row, 2], label="normalized photocurrent")
        axes[row, 2].set_title(
            "scanned SPCM map\n"
            f"side/electrode peak = "
            f"{case['metrics']['side_to_electrode_peak_ratio']:.3g}"
        )
        for ax in axes[row]:
            ax.set_xlabel("x (µm)")
            ax.set_ylabel("y (µm)")
    fig.suptitle(
        "Minimal reproduction of Blevins et al. Fig. 1 PTE mechanisms",
        fontsize=16,
    )
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_convergence(
    cases: dict[tuple[float, float], dict[str, Any]],
    comparisons: dict[str, dict[str, float]],
    output: Path,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15.0, 8.5), constrained_layout=True)
    for row, angle in enumerate((0.0, 45.0)):
        coarse = cases[(100.0, angle)]
        fine = cases[(50.0, angle)]
        mesh_c: Mesh = coarse["mesh"]
        mesh_f: Mesh = fine["mesh"]
        if angle == 0.0:
            coordinate_c = mesh_c.y_m * 1e6
            coordinate_f = mesh_f.y_m * 1e6
            jc = coarse["current_map"][mesh_c.x_m.size // 2, :]
            jf = fine["current_map"][mesh_f.x_m.size // 2, :]
            coordinate_label = "y (µm)"
            profile_label = "vertical centerline"
        else:
            coordinate_c = mesh_c.x_m * 1e6
            coordinate_f = mesh_f.x_m * 1e6
            jc = coarse["current_map"][:, mesh_c.y_m.size // 2]
            jf = fine["current_map"][:, mesh_f.y_m.size // 2]
            coordinate_label = "x (µm)"
            profile_label = "horizontal centerline"
        axes[row, 0].plot(
            coordinate_c, jc, "o-", ms=2.5, label="100 nm"
        )
        axes[row, 0].plot(coordinate_f, jf, "-", lw=2, label="50 nm")
        axes[row, 0].set_title(f"α={angle:g}° {profile_label}")
        axes[row, 0].set_xlabel(coordinate_label)
        axes[row, 0].set_ylabel("I / absorbed power (A/W)")
        axes[row, 0].legend()

        values = [
            coarse["metrics"]["map_peak_abs_A_per_Wabsorbed"],
            fine["metrics"]["map_peak_abs_A_per_Wabsorbed"],
        ]
        axes[row, 1].bar(["100 nm", "50 nm"], values)
        axes[row, 1].set_title("peak |I| / absorbed power")
        axes[row, 1].set_ylabel("A/W")

        compare = comparisons[f"alpha_{angle:g}deg"]
        names = ["map NRMSE", "peak Δ", "edge-ratio Δ"]
        vals = [
            compare["map_nrmse_range_normalized"],
            compare["map_peak_relative_difference"],
            compare["side_to_electrode_ratio_relative_difference"],
        ]
        axes[row, 2].bar(names, np.asarray(vals) * 100)
        axes[row, 2].axhline(1.0, color="k", ls="--", lw=1)
        axes[row, 2].set_ylabel("relative difference (%)")
        axes[row, 2].tick_params(axis="x", rotation=15)
        axes[row, 2].set_title("100 → 50 nm convergence")
    fig.suptitle("Paper SPCM sanity-check numerical convergence", fontsize=16)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_beam_sensitivity(
    cases: dict[tuple[float, float], dict[str, Any]],
    output: Path,
) -> None:
    radii = sorted(
        {
            key[0]
            for key in cases
            if key[1] == 45.0 and key[0] not in (50.0, 100.0)
        }
    )
    selected = [cases[(radius, 45.0)] for radius in radii]
    fig, axes = plt.subplots(
        1, len(selected), figsize=(5.0 * len(selected), 4.5), constrained_layout=True
    )
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes])
    for ax, radius, case in zip(axes, radii, selected):
        mesh: Mesh = case["mesh"]
        normalized = case["current_map"] / np.max(np.abs(case["current_map"]))
        image = ax.imshow(
            normalized.T,
            origin="lower",
            extent=[
                mesh.x_m[0] * 1e6,
                mesh.x_m[-1] * 1e6,
                mesh.y_m[0] * 1e6,
                mesh.y_m[-1] * 1e6,
            ],
            aspect="equal",
            cmap="RdBu_r",
            vmin=-1,
            vmax=1,
        )
        ax.set_title(
            f"$w_0$={radius:.2f} µm\n"
            f"side/electrode={case['metrics']['side_to_electrode_peak_ratio']:.2g}"
        )
        ax.set_xlabel("x (µm)")
        ax.set_ylabel("y (µm)")
    fig.colorbar(image, ax=axes.tolist(), label="normalized photocurrent")
    fig.suptitle(
        "45° transverse-PTE pattern versus unpublished beam-radius assumption",
        fontsize=15,
    )
    fig.savefig(output, dpi=220)
    plt.close(fig)


def write_report(
    output: Path,
    summary: dict[str, Any],
    manifest_name: str,
) -> None:
    aligned = summary["fine_cases"]["alpha_0deg"]
    transverse = summary["fine_cases"]["alpha_45deg"]
    coupling = summary["analytic_coupling"]
    convergence = summary["grid_convergence"]
    status = summary["status"]
    text = f"""# Paper SPCM minimal reproduction sanity check

Status: `{status}`

## Scope

This is a separate sanity check of the simplified two-dimensional mechanisms
in Blevins *et al.*, Fig. 1F/I and Supplementary Note S5. It does **not**
replace or modify the non-periodic inverse-design optical/thermal contract,
and it is not a pixel-for-pixel reproduction of Device A or Device B.

The paper does not publish the exact simplified-Fig.-1 rectangle dimensions,
mesh, numerical 635-nm beam radius, or absorbed fraction. We therefore use an
explicit 6 µm × 8 µm × 130 nm canonical rectangle, report current per absorbed
power, and compare the predicted symmetry, sign, and mechanism. The assumed
beam radius is 0.50 µm; 0.40 and 0.75 µm are included as assumption
sensitivity cases. No fitted gain or map rescaling is used for numerical
metrics (normalization is used only for visualization).

## Paper equations and parameters used

- Gaussian source: Supplement Eq. S1.
- Anisotropic steady heat equation: Supplement Eq. S3.
- Explicit Robin boundaries: Supplement Eq. S4, with
  `G_top(air)=1 W/(m² K)` and
  `G_bottom(thermally-grown SiO2)=7.37e6 W/(m² K)`.
- Local PTE source and continuity: Supplement Eq. S5.
- Shockley–Ramo collection: Supplement Eq. S6.
- Weighting potential: Supplement Eq. S7; full-width top electrode is 1,
  bottom electrode is 0, and lateral sample-air edges are electrically
  insulating.
- `kappa_a=14.4`, `kappa_b=3.8 W/(m K)`;
  `sigma_a=4.91e5`, `sigma_b=1.10e5 S/m`;
  `S_a=-6`, `S_b=27 µV/K`; `T_bath=300 K`.

The lateral edges are thermal zero-flux exactly as stated for the paper's 2-D
IR edge calculation. Top and bottom are **not** adiabatic: both use the
explicit paper Robin conductances.

## Main sanity result

At 45°, the analytic lab-frame PTE coupling is

- `|(sigma S)_yy| = {abs(coupling['alpha45_M_yy_A_mK']):.6g} A/(m K)`
- `|(sigma S)_yx| = {abs(coupling['alpha45_M_yx_A_mK']):.6g} A/(m K)`
- transverse/electrode coupling ratio =
  `{coupling['alpha45_transverse_to_electrode_ratio']:.6g}`.

Thus the electrode-direction term is nearly cancelled by the paper's p×n
Seebeck/conductivity values, while the transverse term remains. The simulated
50-nm-grid maps show:

| case | side/electrode peak | expected odd-symmetry residual | peak response |
|---|---:|---:|---:|
| a-axis aligned | {aligned['side_to_electrode_peak_ratio']:.6g} | {aligned['expected_odd_symmetry_relative_residual']:.3e} | {aligned['map_peak_abs_A_per_Wabsorbed']:.6e} A/W_abs |
| a-axis at 45° | {transverse['side_to_electrode_peak_ratio']:.6g} | {transverse['expected_odd_symmetry_relative_residual']:.3e} | {transverse['map_peak_abs_A_per_Wabsorbed']:.6e} A/W_abs |

The boundary-pair sign tests pass: the aligned case has opposite signs at
top/bottom electrode midpoints and the 45° case has opposite signs at
left/right edge midpoints. The odd-reflection residual is retained only as a
diagnostic; it is not a gate at 45° because rotating the anisotropic thermal
tensor introduces a nonzero `kappa_xy`, so the thermal problem is not
invariant under an isolated x reflection.

This reproduces the paper's central Fig. 1 sanity claim: aligned axes give the
longitudinal, opposite-sign electrode response, whereas the 45° p×n geometry
suppresses electrode response and leaves opposite-sign side-edge response.

## Numerical checks

| case | map NRMSE, 100→50 nm | peak difference | edge-ratio difference |
|---|---:|---:|---:|
| α=0° | {convergence['alpha_0deg']['map_nrmse_range_normalized']:.3%} | {convergence['alpha_0deg']['map_peak_relative_difference']:.3%} | {convergence['alpha_0deg']['side_to_electrode_ratio_relative_difference']:.3%} |
| α=45° | {convergence['alpha_45deg']['map_nrmse_range_normalized']:.3%} | {convergence['alpha_45deg']['map_peak_relative_difference']:.3%} | {convergence['alpha_45deg']['side_to_electrode_ratio_relative_difference']:.3%} |

The direct forward and thermal-adjoint currents agree to
`{transverse['forward_adjoint_relative_difference']:.3e}` relative error in
the 45° check. Its linear residual is
`{transverse['linear_residual_relative']:.3e}` and energy-balance error is
`{transverse['energy_balance_relative']:.3e}`.

## What is and is not reproduced

Reproduced:

- the paper's published material tensor values and explicit thermal-interface
  laws;
- Gaussian local heating, insulating crystal edges, solved weighting
  potential, and Shockley–Ramo collection;
- the longitudinal-electrode versus transverse-edge sign/symmetry change;
- numerical mesh convergence and conservation checks.

Not claimed:

- exact Fig. 2H/5H magnitude or pixelwise agreement, because device CAD,
  exact electrode masks, local 635-nm absorption, objective transmission, and
  exact beam-radius input are not supplied as numerical data;
- 3-D COMSOL reproduction, transient response, optical FDTD/RCWA,
  inverse design, or optimization;
- an experimental current prediction. The reported scale is A per absorbed W.

## Artifacts

- `PAPER_SPCM_SANITY_MAPS.png`
- `PAPER_SPCM_NUMERICAL_CONVERGENCE.png`
- `PAPER_SPCM_BEAM_RADIUS_SENSITIVITY.png`
- `paper_spcm_sanity_summary.json`
- `paper_spcm_sanity_cases.csv`
- `{manifest_name}`
"""
    (output / "PAPER_SPCM_SANITY_CHECK_REPORT.md").write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.raw_dir.mkdir(parents=True, exist_ok=True)

    cases: dict[tuple[float, float], dict[str, Any]] = {}
    for step_nm in (100.0, 50.0):
        for angle in (0.0, 45.0):
            cases[(step_nm, angle)] = run_case(
                step_nm * 1e-9, angle, BEAM_RADIUS_M
            )
    # Beam-radius sensitivity on the fine grid.  Keys use radius in µm to
    # distinguish them from the mesh-step keys above.
    sensitivity: dict[tuple[float, float], dict[str, Any]] = {}
    for radius_um in (0.40, 0.50, 0.75):
        sensitivity[(radius_um, 45.0)] = run_case(
            50.0e-9, 45.0, radius_um * 1e-6
        )

    comparisons = {
        "alpha_0deg": compare_grids(cases[(100.0, 0.0)], cases[(50.0, 0.0)]),
        "alpha_45deg": compare_grids(
            cases[(100.0, 45.0)], cases[(50.0, 45.0)]
        ),
    }
    m45 = rotated_tensor(
        SIGMA_A_S_M * SEEBECK_A_V_K,
        SIGMA_B_S_M * SEEBECK_B_V_K,
        45.0,
    )
    analytic = {
        "sigma_a_times_Sa_A_mK": SIGMA_A_S_M * SEEBECK_A_V_K,
        "sigma_b_times_Sb_A_mK": SIGMA_B_S_M * SEEBECK_B_V_K,
        "alpha45_M_yy_A_mK": float(m45[1, 1]),
        "alpha45_M_yx_A_mK": float(m45[1, 0]),
        "alpha45_transverse_to_electrode_ratio": float(
            abs(m45[1, 0]) / abs(m45[1, 1])
        ),
    }
    fine_metrics = {
        "alpha_0deg": cases[(50.0, 0.0)]["metrics"],
        "alpha_45deg": cases[(50.0, 45.0)]["metrics"],
    }
    gates = {
        "alpha0_electrode_dominates": (
            fine_metrics["alpha_0deg"]["side_to_electrode_peak_ratio"] < 0.25
        ),
        "alpha45_side_edge_dominates": (
            fine_metrics["alpha_45deg"]["side_to_electrode_peak_ratio"] > 3.0
        ),
        "dominance_switch_above_20x": (
            fine_metrics["alpha_45deg"]["side_to_electrode_peak_ratio"]
            / fine_metrics["alpha_0deg"]["side_to_electrode_peak_ratio"]
            > 20.0
        ),
        "alpha0_top_bottom_have_opposite_sign": fine_metrics["alpha_0deg"][
            "bottom_top_opposite_sign"
        ],
        "alpha45_left_right_have_opposite_sign": fine_metrics["alpha_45deg"][
            "left_right_opposite_sign"
        ],
        "alpha45_analytic_cancellation_above_100x": (
            analytic["alpha45_transverse_to_electrode_ratio"] > 100.0
        ),
        "linear_residual_below_1e-8": all(
            item["linear_residual_relative"] < 1e-8
            for item in fine_metrics.values()
        ),
        "energy_balance_below_1pct": all(
            item["energy_balance_relative"] < 0.01
            for item in fine_metrics.values()
        ),
        "forward_adjoint_match_below_1e-10": all(
            item["forward_adjoint_relative_difference"] < 1e-10
            for item in fine_metrics.values()
        ),
        "grid_map_nrmse_below_1pct": all(
            item["map_nrmse_range_normalized"] < 0.01
            for item in comparisons.values()
        ),
    }
    status = (
        "VALIDATED_PAPER_SPCM_MECHANISM_SANITY_CHECK"
        if all(gates.values())
        else "FAILED_PAPER_SPCM_MECHANISM_SANITY_CHECK"
    )
    summary: dict[str, Any] = {
        "status": status,
        "scope": "minimal 2-D mechanism sanity check; not exact device reproduction",
        "paper_sources": {
            "main": {"path": str(PAPER), "sha256": sha256(PAPER)},
            "supplement": {
                "path": str(SUPPLEMENT),
                "sha256": sha256(SUPPLEMENT),
            },
        },
        "published_parameters": {
            "kappa_a_W_mK": KAPPA_A_W_MK,
            "kappa_b_W_mK": KAPPA_B_W_MK,
            "sigma_a_S_m": SIGMA_A_S_M,
            "sigma_b_S_m": SIGMA_B_S_M,
            "Seebeck_a_V_K": SEEBECK_A_V_K,
            "Seebeck_b_V_K": SEEBECK_B_V_K,
            "G_top_air_W_m2K": G_TOP_AIR_W_M2K,
            "G_bottom_thermal_SiO2_W_m2K": G_BOTTOM_THERMAL_SIO2_W_M2K,
            "T_bath_K": T_BATH_K,
        },
        "explicit_assumptions_not_numerically_published": {
            "rectangle_width_um": WIDTH_M * 1e6,
            "rectangle_height_um": HEIGHT_M * 1e6,
            "flake_thickness_nm": THICKNESS_M * 1e9,
            "baseline_beam_radius_um": BEAM_RADIUS_M * 1e6,
            "source_normalization": "unit absorbed power on unbounded plane",
            "absolute_current_claim": False,
        },
        "analytic_coupling": analytic,
        "fine_cases": fine_metrics,
        "grid_convergence": comparisons,
        "beam_radius_sensitivity": {
            f"w0_{radius:.2f}um": case["metrics"]
            for (radius, _), case in sensitivity.items()
        },
        "gates": gates,
    }

    raw_path = args.raw_dir / "paper_spcm_sanity_raw.npz"
    np.savez_compressed(
        raw_path,
        x_50nm_m=cases[(50.0, 0.0)]["mesh"].x_m,
        y_50nm_m=cases[(50.0, 0.0)]["mesh"].y_m,
        aligned_current_A_per_W=cases[(50.0, 0.0)]["current_map"],
        transverse_current_A_per_W=cases[(50.0, 45.0)]["current_map"],
        aligned_temperature_K_per_W=cases[(50.0, 0.0)]["temperature_rise"],
        transverse_temperature_K_per_W=cases[(50.0, 45.0)]["temperature_rise"],
        weighting_potential=cases[(50.0, 45.0)]["psi"],
    )
    raw_sha = sha256(raw_path)
    summary["raw_artifact"] = {
        "path": str(raw_path),
        "byte_size": raw_path.stat().st_size,
        "sha256": raw_sha,
    }

    plot_maps(
        cases[(50.0, 0.0)],
        cases[(50.0, 45.0)],
        args.output_dir / "PAPER_SPCM_SANITY_MAPS.png",
    )
    plot_convergence(
        cases,
        comparisons,
        args.output_dir / "PAPER_SPCM_NUMERICAL_CONVERGENCE.png",
    )
    plot_beam_sensitivity(
        sensitivity,
        args.output_dir / "PAPER_SPCM_BEAM_RADIUS_SENSITIVITY.png",
    )

    summary_path = args.output_dir / "paper_spcm_sanity_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    csv_path = args.output_dir / "paper_spcm_sanity_cases.csv"
    rows: list[dict[str, Any]] = []
    for (step_nm, angle), case in sorted(cases.items()):
        rows.append({"kind": "grid", **case["metrics"]})
    for (radius_um, angle), case in sorted(sensitivity.items()):
        rows.append({"kind": "beam_sensitivity", **case["metrics"]})
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    manifest = {
        "status": status,
        "generation_command": (
            "/home/eidl/miniconda3/envs/EIDL-Lumapi/bin/python "
            "photothermal_pte/validation/paper_spcm_sanity/"
            "validate_paper_spcm_sanity.py "
            f"--output-dir {args.output_dir} --raw-dir {args.raw_dir}"
        ),
        "raw_artifact_not_committed": summary["raw_artifact"],
        "published_files": [],
    }
    for path in sorted(args.output_dir.iterdir()):
        if path.name == "RAW_ARTIFACT_MANIFEST.json":
            continue
        manifest["published_files"].append(
            {
                "path": str(path),
                "byte_size": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    manifest_path = args.output_dir / "RAW_ARTIFACT_MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    write_report(args.output_dir, summary, manifest_path.name)
    # Refresh hashes after the report is written.
    manifest["published_files"] = []
    for path in sorted(args.output_dir.iterdir()):
        if path.name == manifest_path.name:
            continue
        manifest["published_files"].append(
            {
                "path": str(path),
                "byte_size": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"status": status, "gates": gates}, indent=2))


if __name__ == "__main__":
    main()
