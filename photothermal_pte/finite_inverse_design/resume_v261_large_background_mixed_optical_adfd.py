#!/usr/bin/env python3
"""Resume mixed optical AD--FD from completed CPU/GPU adjoint FSPs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import traceback

import numpy as np

from .native_yee_q import extract_native_yee_q
from .large_background_contract import baseline_contract
from .probe_v261_cpu_tfsf_device import (
    FREQUENCY_HZ,
    PABS_FIELD,
    PABS_INDEX,
    SOURCE_NAME,
)
from .probe_v261_gpu_plane_wave_roi import (
    APPROVED_API,
    APPROVED_ROOT,
    json_default,
    load_lumapi,
    scalar,
)
from .run_v261_large_background_mixed_optical_adfd import (
    DESIGN_FIELD,
    DESIGN_INDEX,
    FIELD_REGION,
    SIO2_EPSILON,
    WAVELENGTH_M,
    absorption_objective_and_source,
    fieldregion_profile,
    gradient_from_adjoint,
    monitor_electric,
    monitor_epsilon,
    run_forward,
    sha256,
    strip_arrays,
)


FLUX_SIGNS = {
    "device_flux_x_min": -1.0,
    "device_flux_x_max": 1.0,
    "device_flux_y_min": -1.0,
    "device_flux_y_max": 1.0,
    "device_flux_z_min": -1.0,
    "device_flux_z_max": 1.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument(
        "--fd-checkpoint-dir",
        help="Reuse already completed centered-FD FSPs from this directory.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rho", type=float, default=0.5)
    parser.add_argument(
        "--fd-step",
        type=float,
        action="append",
        dest="fd_steps",
        default=None,
    )
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args()
    args.fd_steps = args.fd_steps or [0.02, 0.01]
    if not 0.0 < args.rho < 1.0:
        parser.error("rho must lie strictly inside (0,1)")
    if any(
        step <= 0.0
        or args.rho - step <= 0.0
        or args.rho + step >= 1.0
        for step in args.fd_steps
    ):
        parser.error("invalid centered FD step")
    return args


def log_contract(path: Path, engine: str) -> dict[str, object]:
    text = path.read_text(errors="replace")
    return {
        "path": str(path),
        "byte_size": path.stat().st_size,
        "sha256": sha256(path),
        "engine_marker_present": (
            "-gpu" in text if engine == "GPU" else "fdtd-engine -t 16" in text
        ),
        "completed_successfully": "Simulation completed successfully" in text,
        "divergence_marker_absent": (
            "electromagnetic fields are diverging" not in text
        ),
        "autoshutoff_satisfied": (
            "autoshutoff criteria are satisfied" in text
        ),
    }


def load_base(fdtd: object, project: Path) -> dict[str, object]:
    fdtd.switchtolayout()
    fdtd.load(str(project))
    electric, grid = monitor_electric(fdtd, PABS_FIELD)
    epsilon, index_grid = monitor_epsilon(fdtd, PABS_INDEX)
    for axis in "xyzf":
        if not np.array_equal(grid[axis], index_grid[axis]):
            raise RuntimeError(f"saved base E/index {axis} grids differ")
    objective, q_source, components = absorption_objective_and_source(
        electric, epsilon, grid
    )
    native = extract_native_yee_q(
        fdtd,
        field_monitor=PABS_FIELD,
        index_monitor=PABS_INDEX,
        wavelength_m=WAVELENGTH_M,
    )
    source_power = scalar(
        fdtd.sourcepower(FREQUENCY_HZ, 2, SOURCE_NAME), "source power"
    )
    net_outward = 0.0
    faces = {}
    for name, sign in FLUX_SIGNS.items():
        signed = scalar(fdtd.transmission(name), name) * source_power
        outward = sign * signed
        net_outward += outward
        faces[name] = {
            "signed_axis_power_W": signed,
            "outward_power_W": outward,
        }
    p_six = -net_outward
    return {
        "objective_W": objective,
        "component_power_W": components,
        "q_source": q_source,
        "grid": grid,
        "pabs_P_Q_W": native["P_Q_W"],
        "objective_vs_native_pabs_relative_error": abs(
            objective - native["P_Q_W"]
        )
        / max(abs(native["P_Q_W"]), np.finfo(float).tiny),
        "P_six_W": p_six,
        "six_face_closure_relative_error": abs(objective - p_six)
        / max(abs(p_six), np.finfo(float).tiny),
        "six_face_power": faces,
        "source_intensity_W_m2": scalar(
            fdtd.sourceintensity(FREQUENCY_HZ, 2, SOURCE_NAME),
            "source intensity",
        ),
        "project": {
            "path": str(project),
            "byte_size": project.stat().st_size,
            "sha256": sha256(project),
        },
    }


def load_adjoint(
    fdtd: object, project: Path, log: Path, engine: str
) -> dict[str, object]:
    fdtd.switchtolayout()
    fdtd.load(str(project))
    if int(fdtd.getnamednumber(SOURCE_NAME)) != 0:
        raise RuntimeError(f"{engine} adjoint contains TFSF")
    electric, grid = monitor_electric(fdtd, DESIGN_FIELD)
    return {
        "electric": electric,
        "grid": grid,
        "project": {
            "path": str(project),
            "byte_size": project.stat().st_size,
            "sha256": sha256(project),
        },
        "log": log_contract(log, engine),
    }


def main() -> int:
    args = parse_args()
    checkpoint = Path(args.checkpoint_dir).expanduser().resolve()
    fd_checkpoint = (
        Path(args.fd_checkpoint_dir).expanduser().resolve()
        if args.fd_checkpoint_dir
        else None
    )
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "mixed_optical_adfd_resumed_result.json"
    result: dict[str, object] = {
        "status": "BLOCKED_RESUMED_MIXED_OPTICAL_ADFD_NOT_RUN",
        "passed": False,
        "checkpoint_dir": str(checkpoint),
        "fd_checkpoint_dir": (
            str(fd_checkpoint) if fd_checkpoint is not None else None
        ),
        "rho": args.rho,
        "fd_steps": args.fd_steps,
        "solver_root": str(APPROVED_ROOT),
        "lumapi_path": str(APPROVED_API),
        "forward_engine": "CPU TFSF",
        "production_adjoint_engine": "GPU FieldRegion",
        "equivalence_adjoint_engine": "CPU FieldRegion",
        "periodic_or_bloch": False,
        "thermal_run": False,
        "pte_run": False,
        "optimization_run": False,
    }
    fdtd = None
    started = time.monotonic()
    try:
        base_project = checkpoint / "mixed_base_forward_cpu_tfsf.fsp"
        template = checkpoint / "mixed_adjoint_source_template.fsp"
        cpu_project = checkpoint / "mixed_adjoint_cpu.fsp"
        gpu_project = checkpoint / "mixed_adjoint_gpu.fsp"
        required = (
            base_project,
            template,
            cpu_project,
            gpu_project,
            checkpoint / "mixed_adjoint_cpu_p0.log",
            checkpoint / "mixed_adjoint_gpu_p0.log",
        )
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"missing checkpoint artifacts: {missing}")

        lumapi = load_lumapi()
        fdtd = lumapi.FDTD(hide=True, serverArgs={"platform": "offscreen"})
        base = load_base(fdtd, base_project)
        result["base_forward"] = strip_arrays(base)
        profile, profile_scale = fieldregion_profile(base["q_source"])

        fdtd.switchtolayout()
        fdtd.load(str(template))
        template_tfsf_count = int(fdtd.getnamednumber(SOURCE_NAME))
        base_amplitude = float(
            fdtd.getnamed(FIELD_REGION, "base amplitude")
        )
        imported = np.asarray(
            fdtd.getresult(FIELD_REGION, "source profile")["E"],
            np.complex128,
        )
        if imported.ndim == 4:
            imported = imported[..., None, :]
        profile_reload_error = float(np.max(np.abs(imported - profile)))
        result["adjoint_source"] = {
            "definition": (
                "profile=conj(dP_Q/dE*)/max|dP_Q/dE*| on the exact "
                "certified pabs native Yee grid"
            ),
            "profile_scale": profile_scale,
            "fieldregion_base_amplitude": base_amplitude,
            "saved_profile_reload_max_abs_error": profile_reload_error,
            "tfsf_count": template_tfsf_count,
            "template": {
                "path": str(template),
                "byte_size": template.stat().st_size,
                "sha256": sha256(template),
            },
        }

        gpu = load_adjoint(
            fdtd,
            gpu_project,
            checkpoint / "mixed_adjoint_gpu_p0.log",
            "GPU",
        )
        cpu = load_adjoint(
            fdtd,
            cpu_project,
            checkpoint / "mixed_adjoint_cpu_p0.log",
            "CPU",
        )
        grid_audit = {}
        for axis in "xyzf":
            same_shape = cpu["grid"][axis].shape == gpu["grid"][axis].shape
            maximum = (
                float(
                    np.max(
                        np.abs(cpu["grid"][axis] - gpu["grid"][axis])
                    )
                )
                if same_shape
                else None
            )
            grid_audit[axis] = {
                "same_shape": same_shape,
                "maximum_absolute_difference": maximum,
                "allclose_rtol0_atol1e-15": bool(
                    same_shape
                    and np.allclose(
                        cpu["grid"][axis],
                        gpu["grid"][axis],
                        rtol=0.0,
                        atol=1.0e-15,
                    )
                ),
            }
        if not all(
            audit["allclose_rtol0_atol1e-15"]
            for audit in grid_audit.values()
        ):
            raise RuntimeError("CPU/GPU adjoint grids differ physically")
        field_nrmse = float(
            np.linalg.norm(gpu["electric"] - cpu["electric"])
            / max(
                np.linalg.norm(cpu["electric"]), np.finfo(float).tiny
            )
        )
        result["cpu_gpu_adjoint_equivalence"] = {
            "grid_audit": grid_audit,
            "complex_field_NRMSE": field_nrmse,
            "limit": 5.0e-3,
            "CPU": strip_arrays(cpu),
            "GPU": strip_arrays(gpu),
        }

        fdtd.switchtolayout()
        fdtd.load(str(base_project))
        forward_design_e, design_grid = monitor_electric(
            fdtd, DESIGN_FIELD
        )
        forward_adjoint_grid_audit = {}
        for engine, adjoint in (("CPU", cpu), ("GPU", gpu)):
            forward_adjoint_grid_audit[engine] = {}
            for axis in "xyzf":
                same_shape = (
                    design_grid[axis].shape
                    == adjoint["grid"][axis].shape
                )
                forward_adjoint_grid_audit[engine][axis] = {
                    "same_shape": same_shape,
                    "allclose_rtol0_atol1e-15": bool(
                        same_shape
                        and np.allclose(
                            design_grid[axis],
                            adjoint["grid"][axis],
                            rtol=0.0,
                            atol=1.0e-15,
                        )
                    ),
                }
        if not all(
            audit["allclose_rtol0_atol1e-15"]
            for engine in forward_adjoint_grid_audit.values()
            for audit in engine.values()
        ):
            raise RuntimeError("forward/adjoint design grids differ")
        result["forward_adjoint_design_grid_audit"] = (
            forward_adjoint_grid_audit
        )
        fd_rows = []
        for step in args.fd_steps:
            pair = []
            pair_meta = []
            for sign in (1.0, -1.0):
                rho = args.rho + sign * step
                label = f"fd_rho_{rho:.8f}".replace(".", "p")
                if fd_checkpoint is not None:
                    completed = (
                        fd_checkpoint / f"{label}_cpu_tfsf.fsp"
                    )
                    if not completed.is_file():
                        raise FileNotFoundError(completed)
                    forward = load_base(fdtd, completed)
                    forward["rho"] = rho
                    forward["reused_completed_forward"] = True
                else:
                    fdtd.switchtolayout()
                    fdtd.load(str(base_project))
                    forward = run_forward(
                        fdtd,
                        rho=rho,
                        project=output / f"{label}_cpu_tfsf.fsp",
                        threads=args.threads,
                        flux_signs=FLUX_SIGNS,
                    )
                pair.append(forward["objective_W"])
                pair_meta.append(strip_arrays(forward))
            centered = (pair[0] - pair[1]) / (2.0 * step)
            fd_rows.append(
                {
                    "step": step,
                    "objective_plus_W": pair[0],
                    "objective_minus_W": pair[1],
                    "centered_FD_dP_Q_d_rho_W": centered,
                    "forward_cases": pair_meta,
                }
            )

        # Ordinary index-monitor d-cards are unavailable in layout mode for
        # this simple rectangle.  Measure the actual conformal
        # d(epsilon_Yee)/d(rho) from the completed +/- FSP pair at the smallest
        # FD step.  These are the same forward solves used below; no extra EM
        # solve and no analytic epsilon assumption is introduced.
        derivative_row = min(fd_rows, key=lambda row: row["step"])
        epsilon_pair = []
        epsilon_grids = []
        for case in derivative_row["forward_cases"]:
            fdtd.switchtolayout()
            fdtd.load(case["project"]["path"])
            epsilon, epsilon_grid = monitor_epsilon(fdtd, DESIGN_INDEX)
            epsilon_pair.append(epsilon)
            epsilon_grids.append(epsilon_grid)
        derivative = (
            epsilon_pair[0] - epsilon_pair[1]
        ) / (2.0 * derivative_row["step"])
        expected_derivative = SIO2_EPSILON - 1.0
        derivative_uniform_error = float(
            np.max(np.abs(derivative - expected_derivative))
        )
        if derivative_uniform_error > 1.0e-12:
            raise RuntimeError(
                "uniform scalar conformal epsilon derivative is not spatially "
                f"constant: max error {derivative_uniform_error:.3e}"
            )
        index_field_grid_audit = []
        for grid in epsilon_grids:
            axes = {}
            for axis in "xy":
                axes[axis] = bool(
                    grid[axis].shape == design_grid[axis].shape
                    and np.allclose(
                        grid[axis],
                        design_grid[axis],
                        rtol=0.0,
                        atol=1.0e-15,
                    )
                )
            axes["f"] = bool(
                grid["f"].shape == design_grid["f"].shape
                and np.allclose(
                    grid["f"],
                    design_grid["f"],
                    rtol=1.0e-12,
                    atol=0.0,
                )
            )
            z_same_shape = grid["z"].shape == design_grid["z"].shape
            z_prefix_equal = bool(
                z_same_shape
                and np.allclose(
                    grid["z"][:-1],
                    design_grid["z"][:-1],
                    rtol=0.0,
                    atol=1.0e-15,
                )
            )
            endpoint_only = bool(
                z_same_shape
                and z_prefix_equal
                and not np.isclose(
                    grid["z"][-1],
                    design_grid["z"][-1],
                    rtol=0.0,
                    atol=1.0e-15,
                )
            )
            if not (
                all(axes.values())
                and z_prefix_equal
                and (endpoint_only or np.isclose(
                    grid["z"][-1],
                    design_grid["z"][-1],
                    rtol=0.0,
                    atol=1.0e-15,
                ))
            ):
                raise RuntimeError(
                    "FD index/design-field grids differ beyond the one "
                    "documented z-max endpoint"
                )
            index_field_grid_audit.append(
                {
                    "xyf_allclose_rtol0_atol1e-15": all(axes.values()),
                    "z_same_shape": z_same_shape,
                    "z_prefix_allclose_rtol0_atol1e-15": z_prefix_equal,
                    "z_endpoint_only_difference": endpoint_only,
                    "index_z_max_m": float(grid["z"][-1]),
                    "field_z_max_m": float(design_grid["z"][-1]),
                    "mapping": (
                        "no interpolation; same array ownership is valid only "
                        "because measured d-epsilon/d-rho is exactly constant "
                        "for this uniform scalar certificate"
                    ),
                }
            )
        derivative_meta = {
            "step": derivative_row["step"],
            "method": (
                "centered conformal epsilon_Yee derivative extracted from "
                "the completed smallest-step FD forward FSP pair"
            ),
            "additional_electromagnetic_solves": 0,
            "max_abs_imaginary_derivative": float(
                np.max(np.abs(np.imag(derivative)))
            ),
            "expected_uniform_derivative": expected_derivative,
            "maximum_absolute_nonuniformity": derivative_uniform_error,
            "index_to_field_grid_audit": index_field_grid_audit,
            "plus_project": derivative_row["forward_cases"][0]["project"],
            "minus_project": derivative_row["forward_cases"][1]["project"],
        }
        cpu_gradient, cpu_components = gradient_from_adjoint(
            forward_electric=forward_design_e,
            adjoint_electric=cpu["electric"],
            design_grid=design_grid,
            d_epsilon_d_rho=derivative,
            profile_scale=profile_scale,
            base_amplitude=base_amplitude,
            design_bounds_m={
                axis: getattr(baseline_contract().design, axis)
                for axis in "xyz"
            },
        )
        gpu_gradient, gpu_components = gradient_from_adjoint(
            forward_electric=forward_design_e,
            adjoint_electric=gpu["electric"],
            design_grid=design_grid,
            d_epsilon_d_rho=derivative,
            profile_scale=profile_scale,
            base_amplitude=base_amplitude,
            design_bounds_m={
                axis: getattr(baseline_contract().design, axis)
                for axis in "xyz"
            },
        )
        gradient_difference = abs(gpu_gradient - cpu_gradient) / max(
            abs(cpu_gradient), np.finfo(float).tiny
        )
        result["adjoint_gradient_dP_Q_d_rho_W"] = {
            "CPU": cpu_gradient,
            "GPU": gpu_gradient,
            "CPU_components_xyz": cpu_components,
            "GPU_components_xyz": gpu_components,
            "CPU_GPU_relative_difference": gradient_difference,
            "conformal_epsilon_derivative": derivative_meta,
            "explicit_design_loss_term": (
                "zero: lossless air/SiO2 design and fixed TaIrTe4"
            ),
        }
        for row in fd_rows:
            centered = row["centered_FD_dP_Q_d_rho_W"]
            row["GPU_adjoint_dP_Q_d_rho_W"] = gpu_gradient
            row["relative_error"] = abs(gpu_gradient - centered) / max(
                abs(centered), np.finfo(float).tiny
            )
        result["centered_FD_step_sweep"] = fd_rows
        best_error = min(row["relative_error"] for row in fd_rows)
        step_change = abs(
            fd_rows[-1]["centered_FD_dP_Q_d_rho_W"]
            - fd_rows[-2]["centered_FD_dP_Q_d_rho_W"]
        ) / max(
            abs(fd_rows[-1]["centered_FD_dP_Q_d_rho_W"]),
            np.finfo(float).tiny,
        )
        result["best_AD_FD_relative_error"] = best_error
        result["FD_step_change_relative"] = step_change
        result["gates"] = {
            "base_native_objective_exact": (
                base["objective_vs_native_pabs_relative_error"] < 1.0e-12
            ),
            "base_Q_six_face_closure_lt_0p5pct": (
                base["six_face_closure_relative_error"] < 5.0e-3
            ),
            "source_intensity_error_lt_0p5pct": (
                abs(base["source_intensity_W_m2"] - 1.0) < 5.0e-3
            ),
            "tfsf_absent_from_all_adjoint_projects": (
                template_tfsf_count == 0
            ),
            "saved_source_profile_error_lt_1e-14": (
                profile_reload_error < 1.0e-14
            ),
            "cpu_gpu_grids_same_with_1e-15m_tolerance": all(
                audit["allclose_rtol0_atol1e-15"]
                for audit in grid_audit.values()
            ),
            "cpu_gpu_field_NRMSE_lt_0p5pct": field_nrmse < 5.0e-3,
            "cpu_gpu_gradient_difference_lt_1pct": (
                gradient_difference < 1.0e-2
            ),
            "forward_adjoint_design_grids_same": all(
                audit["allclose_rtol0_atol1e-15"]
                for engine in forward_adjoint_grid_audit.values()
                for audit in engine.values()
            ),
            "best_AD_FD_error_lt_1pct": best_error < 1.0e-2,
            "FD_step_change_lt_1pct": step_change < 1.0e-2,
            "all_FD_Q_six_face_closures_lt_0p5pct": all(
                case["six_face_closure_relative_error"] < 5.0e-3
                for row in fd_rows
                for case in row["forward_cases"]
            ),
            "completed_engine_logs_clean": all(
                data["log"]["engine_marker_present"]
                and data["log"]["completed_successfully"]
                and data["log"]["divergence_marker_absent"]
                and data["log"]["autoshutoff_satisfied"]
                for data in (cpu, gpu)
            ),
        }
        result["passed"] = bool(all(result["gates"].values()))
        result["status"] = (
            "VALIDATED_MIXED_CPU_TFSF_GPU_FIELDREGION_OPTICAL_ADFD"
            if result["passed"]
            else "FAILED_MIXED_CPU_TFSF_GPU_FIELDREGION_OPTICAL_ADFD"
        )
    except Exception as exc:
        result["status"] = "BLOCKED_RESUMED_MIXED_OPTICAL_ADFD_EXECUTION"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
        result["passed"] = False
    finally:
        result["total_wall_s"] = time.monotonic() - started
        if fdtd is not None:
            try:
                fdtd.close()
            except Exception:
                pass
        result_path.write_text(
            json.dumps(result, indent=2, default=json_default) + "\n"
        )
    print(json.dumps(result, indent=2, default=json_default), flush=True)
    return 0 if result.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
