#!/usr/bin/env python3
"""Certify 81x81 nodal physical-density optical/thermal coupling."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import traceback

import numpy as np

from .contract import (
    DESIGN_DX_M,
    DESIGN_DY_M,
    DESIGN_NX,
    DESIGN_NY,
    design_extrusion_nodes_m,
    design_nodal_coordinates_m,
)
from .nodal_physical_coupling import NodalPhysicalCoupling


STATUS_PASS = "VALIDATED_81X81_NODAL_OPTICAL_THERMAL_MAPPING_JVP_VJP"
STATUS_FAIL = "FAILED_81X81_NODAL_OPTICAL_THERMAL_MAPPING_JVP_VJP"
TRANSPOSE_LIMIT = 1.0e-12
JVP_FD_LIMIT = 1.0e-9
CONSTANT_LIMIT = 2.0e-13


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fd-step", type=float, default=1.0e-5)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_norm(value: np.ndarray, reference: np.ndarray) -> float:
    return float(
        np.linalg.norm(value - reference)
        / max(
            np.linalg.norm(value),
            np.linalg.norm(reference),
            np.finfo(float).tiny,
        )
    )


def relative_scalar(left: float, right: float) -> float:
    return abs(left - right) / max(
        abs(left), abs(right), np.finfo(float).tiny
    )


def model(cell_nm: float) -> NodalPhysicalCoupling:
    x, y = design_nodal_coordinates_m()
    cells = int(round(2000.0 / cell_nm))
    edges = np.linspace(-1.0e-6, 1.0e-6, cells + 1)
    return NodalPhysicalCoupling(
        x_nodes_m=x,
        y_nodes_m=y,
        optical_z_nodes_m=design_extrusion_nodes_m(),
        thermal_x_edges_m=edges,
        thermal_y_edges_m=edges,
    )


def main() -> int:
    args = parse_args()
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "nodal_physical_coupling_summary.json"
    result: dict[str, object] = {
        "status": "BLOCKED_81X81_NODAL_COUPLING_NOT_RUN",
        "passed": False,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "solver-free 81x81 nodal physical-density coupling to optical "
            "import nodes and thermal cell averages"
        ),
        "filter_run": False,
        "projection_run": False,
        "maxwell_run": False,
        "thermal_solve_run": False,
        "optimization_run": False,
    }
    try:
        if args.fd_step <= 0.0:
            raise ValueError("FD step must be positive")
        x, y = design_nodal_coordinates_m()
        z = design_extrusion_nodes_m()
        xx = x[:, None] / 1.0e-6
        yy = y[None, :] / 1.0e-6
        rho = (
            0.5
            + 0.08 * np.sin(0.7 * np.pi * xx)
            * np.cos(0.6 * np.pi * yy)
            + 0.03 * xx * yy
        )
        rng = np.random.default_rng(2026072708)
        raw_directions = {
            "smooth": (
                np.cos(0.4 * np.pi * (xx + 0.2))
                * np.sin(0.8 * np.pi * (yy - 0.1))
            ),
            "seeded_random": rng.normal(size=rho.shape),
        }
        directions = {
            name: value / np.max(np.abs(value))
            for name, value in raw_directions.items()
        }
        if (
            np.min(rho) - args.fd_step < 0.0
            or np.max(rho) + args.fd_step > 1.0
        ):
            raise RuntimeError("FD endpoints leave [0,1]")

        coupling_100 = model(100.0)
        optical_covector = rng.normal(size=coupling_100.optical_shape)
        optical_covector /= np.linalg.norm(optical_covector)
        optical_records = []
        raw_arrays: dict[str, np.ndarray] = {
            "physical_rho": rho,
            "x_nodes_m": x,
            "y_nodes_m": y,
            "optical_z_nodes_m": z,
            "optical_covector": optical_covector,
        }
        for name, direction in directions.items():
            jvp = coupling_100.optical_jvp(direction)
            fd = (
                coupling_100.optical(rho + args.fd_step * direction)
                - coupling_100.optical(rho - args.fd_step * direction)
            ) / (2.0 * args.fd_step)
            vjp = coupling_100.optical_vjp(optical_covector)
            left = float(np.sum(jvp * optical_covector))
            right = float(np.sum(direction * vjp))
            optical_records.append(
                {
                    "direction": name,
                    "JVP_centered_FD_relative_error": relative_norm(jvp, fd),
                    "JVP_VJP_dot_relative_error": relative_scalar(
                        left, right
                    ),
                    "dot_left": left,
                    "dot_right": right,
                }
            )
            raw_arrays[f"direction_{name}"] = direction
            raw_arrays[f"optical_jvp_{name}"] = jvp
            raw_arrays[f"optical_vjp_{name}"] = vjp

        optical_zero = coupling_100.optical(np.zeros_like(rho))
        optical_one = coupling_100.optical(np.ones_like(rho))
        optical_rho = coupling_100.optical(rho)
        optical_contract = {
            "shape": list(coupling_100.optical_shape),
            "x_bounds_m": [float(x[0]), float(x[-1])],
            "y_bounds_m": [float(y[0]), float(y[-1])],
            "z_bounds_m": [float(z[0]), float(z[-1])],
            "x_spacing_m": float(np.diff(x)[0]),
            "y_spacing_m": float(np.diff(y)[0]),
            "z_spacing_m": float(np.diff(z)[0]),
            "rho0_max_abs_error": float(np.max(np.abs(optical_zero))),
            "rho1_max_abs_error": float(
                np.max(np.abs(optical_one - 1.0))
            ),
            "z_extrusion_max_abs_error": float(
                np.max(np.abs(optical_rho - optical_rho[:, :, :1]))
            ),
            "mapping": (
                "identity on exact x-y physical nodes, exact repeat on "
                "13 z nodes; no interpolation, fencepost merge, or wrap"
            ),
            "directions": optical_records,
        }
        raw_arrays["optical_rho"] = optical_rho

        nodal_weight_x = np.empty(x.size)
        nodal_weight_x[0] = 0.5 * (x[1] - x[0])
        nodal_weight_x[-1] = 0.5 * (x[-1] - x[-2])
        nodal_weight_x[1:-1] = 0.5 * (x[2:] - x[:-2])
        nodal_weight_y = nodal_weight_x.copy()
        nodal_integral = float(
            np.sum(
                rho
                * nodal_weight_x[:, None]
                * nodal_weight_y[None, :]
            )
        )
        thermal_records = []
        for cell_nm in (100.0, 50.0):
            current = model(cell_nm)
            covector = rng.normal(size=current.thermal_shape)
            covector /= np.linalg.norm(covector)
            direction_records = []
            for name, direction in directions.items():
                jvp = current.thermal_jvp(direction)
                fd = (
                    current.thermal(rho + args.fd_step * direction)
                    - current.thermal(rho - args.fd_step * direction)
                ) / (2.0 * args.fd_step)
                vjp = current.thermal_vjp(covector)
                left = float(np.sum(jvp * covector))
                right = float(np.sum(direction * vjp))
                direction_records.append(
                    {
                        "direction": name,
                        "JVP_centered_FD_relative_error": relative_norm(
                            jvp, fd
                        ),
                        "JVP_VJP_dot_relative_error": relative_scalar(
                            left, right
                        ),
                        "dot_left": left,
                        "dot_right": right,
                    }
                )
                key = f"thermal_{cell_nm:g}nm_{name}"
                raw_arrays[f"{key}_jvp"] = jvp
                raw_arrays[f"{key}_vjp"] = vjp
            thermal_rho = current.thermal(rho)
            zero = current.thermal(np.zeros_like(rho))
            one = current.thermal(np.ones_like(rho))
            dx = np.diff(current.thermal_x_edges_m)
            dy = np.diff(current.thermal_y_edges_m)
            thermal_integral = float(
                np.sum(thermal_rho * dx[:, None] * dy[None, :])
            )
            affine = 0.5 + 0.1 * xx - 0.07 * yy
            xc = 0.5 * (
                current.thermal_x_edges_m[:-1]
                + current.thermal_x_edges_m[1:]
            )
            yc = 0.5 * (
                current.thermal_y_edges_m[:-1]
                + current.thermal_y_edges_m[1:]
            )
            affine_expected = (
                0.5
                + 0.1 * xc[:, None] / 1.0e-6
                - 0.07 * yc[None, :] / 1.0e-6
            )
            corner = np.zeros_like(rho)
            corner[0, 0] = 1.0
            corner_mapped = current.thermal(corner)
            record = {
                "core_xy_cell_size_nm": cell_nm,
                "shape": list(current.thermal_shape),
                "bounds_m": {
                    "x": [
                        float(current.thermal_x_edges_m[0]),
                        float(current.thermal_x_edges_m[-1]),
                    ],
                    "y": [
                        float(current.thermal_y_edges_m[0]),
                        float(current.thermal_y_edges_m[-1]),
                    ],
                },
                "mapping": (
                    "exact area average of the nonperiodic piecewise-bilinear "
                    "nodal interpolant"
                ),
                "rho0_max_abs_error": float(np.max(np.abs(zero))),
                "rho1_max_abs_error": float(
                    np.max(np.abs(one - 1.0))
                ),
                "affine_cell_average_max_abs_error": float(
                    np.max(np.abs(current.thermal(affine) - affine_expected))
                ),
                "area_integral_relative_error": relative_scalar(
                    thermal_integral, nodal_integral
                ),
                "corner_to_opposite_corner_value": float(
                    corner_mapped[-1, -1]
                ),
                "corner_to_opposite_x_edge_max_abs": float(
                    np.max(np.abs(corner_mapped[-1, :]))
                ),
                "corner_to_opposite_y_edge_max_abs": float(
                    np.max(np.abs(corner_mapped[:, -1]))
                ),
                "directions": direction_records,
            }
            thermal_records.append(record)
            raw_arrays[f"thermal_{cell_nm:g}nm_rho"] = thermal_rho
            raw_arrays[f"thermal_{cell_nm:g}nm_covector"] = covector

        all_direction_records = optical_records + [
            direction
            for record in thermal_records
            for direction in record["directions"]
        ]
        worst_jvp_fd = max(
            record["JVP_centered_FD_relative_error"]
            for record in all_direction_records
        )
        worst_transpose = max(
            record["JVP_VJP_dot_relative_error"]
            for record in all_direction_records
        )
        worst_constant = max(
            optical_contract["rho0_max_abs_error"],
            optical_contract["rho1_max_abs_error"],
            *[
                value
                for record in thermal_records
                for value in (
                    record["rho0_max_abs_error"],
                    record["rho1_max_abs_error"],
                )
            ],
        )
        worst_conservation = max(
            record["area_integral_relative_error"]
            for record in thermal_records
        )
        worst_no_wrap = max(
            value
            for record in thermal_records
            for value in (
                record["corner_to_opposite_corner_value"],
                record["corner_to_opposite_x_edge_max_abs"],
                record["corner_to_opposite_y_edge_max_abs"],
            )
        )
        passed = bool(
            x.size == DESIGN_NX
            and y.size == DESIGN_NY
            and np.allclose(np.diff(x), DESIGN_DX_M, atol=2e-18, rtol=0)
            and np.allclose(np.diff(y), DESIGN_DY_M, atol=2e-18, rtol=0)
            and worst_jvp_fd < JVP_FD_LIMIT
            and worst_transpose < TRANSPOSE_LIMIT
            and worst_constant < CONSTANT_LIMIT
            and worst_conservation < TRANSPOSE_LIMIT
            and worst_no_wrap == 0.0
            and optical_contract["z_extrusion_max_abs_error"] == 0.0
        )
        raw_npz = output / "nodal_physical_coupling_certificate.npz"
        np.savez_compressed(raw_npz, **raw_arrays)
        result.update(
            {
                "status": STATUS_PASS if passed else STATUS_FAIL,
                "passed": passed,
                "physical_density_contract": {
                    "shape": [DESIGN_NX, DESIGN_NY],
                    "kind": (
                        "nodal physical density on exact finite support; "
                        "not 81 finite-width pixels"
                    ),
                    "x_bounds_m": [float(x[0]), float(x[-1])],
                    "y_bounds_m": [float(y[0]), float(y[-1])],
                    "dx_m": DESIGN_DX_M,
                    "dy_m": DESIGN_DY_M,
                    "periodic": False,
                },
                "optical_mapping": optical_contract,
                "thermal_mappings": thermal_records,
                "gates": {
                    "JVP_centered_FD_limit": JVP_FD_LIMIT,
                    "worst_JVP_centered_FD_relative_error": worst_jvp_fd,
                    "transpose_limit": TRANSPOSE_LIMIT,
                    "worst_JVP_VJP_dot_relative_error": worst_transpose,
                    "constant_limit": CONSTANT_LIMIT,
                    "worst_endpoint_constant_error": worst_constant,
                    "worst_area_integral_relative_error": (
                        worst_conservation
                    ),
                    "worst_opposite_boundary_leakage": worst_no_wrap,
                    "optical_z_extrusion_error": optical_contract[
                        "z_extrusion_max_abs_error"
                    ],
                },
                "raw_artifact": {
                    "path": str(raw_npz),
                    "byte_size": raw_npz.stat().st_size,
                    "sha256": sha256(raw_npz),
                },
                "next_gate": "IMPORTED_PERMITTIVITY_ENDPOINT_EQUIVALENCE",
            }
        )
    except Exception as exc:
        result["status"] = "BLOCKED_81X81_NODAL_COUPLING_EXECUTION"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
    summary_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result.get("passed") else 2


if __name__ == "__main__":
    raise SystemExit(main())
