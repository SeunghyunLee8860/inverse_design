#!/usr/bin/env python3
"""Combined 81x81 physical-density Maxwell/thermal/PTE AD--FD gate."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np

from .audit_v261_large_background_tfsf_geometry import (
    add_large_background_geometry,
)
from .contract import (
    DESIGN_BOUNDS_M,
    design_extrusion_nodes_m,
    design_nodal_coordinates_m,
)
from .explicit_thermal import (
    build_explicit_geometry,
    evaluate_explicit_thermal,
    solve_explicit_forward,
)
from .finite_q_mapping import (
    build_conservative_embedding_remap,
    exact_nonzero_box,
    nodal_control_volume_edges,
    project_remap_to_material_support_along_axis,
)
from .large_background_contract import baseline_contract
from .native_yee_q import EPS0, extract_native_yee_q
from .nodal_physical_coupling import NodalPhysicalCoupling
from .probe_v261_cpu_tfsf_device import (
    FREQUENCY_HZ,
    PABS_FIELD,
    PABS_INDEX,
    SOURCE_NAME,
)
from .probe_v261_cpu_tfsf_roi import run_on_cpu
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
    add_adjoint_monitors,
    configure_cpu,
    configure_pml_profile,
    fieldregion_profile,
    gradient_from_adjoint,
    monitor_electric,
    monitor_epsilon,
    prepare_adjoint_layout,
    run_adjoint,
)
from .run_v261_large_background_tfsf_forward import (
    add_monitors,
    configure_design,
    sha256,
)


STATUS_PASS = "VALIDATED_COMBINED_PHYSICAL_RHO_PTE_ADFD"
STATUS_FAIL = "FAILED_COMBINED_PHYSICAL_RHO_PTE_ADFD"
IMPORTED_OBJECT = "finite_design_gray_imported_permittivity"
SIO2_EPSILON = 1.38**2
ADFD_LIMIT = 5.0e-3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", default="0.01,0.005")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--gpu-device", default="GPU 1")
    parser.add_argument("--simulation-time-ps", type=float, default=4.0)
    parser.add_argument(
        "--resume-base-fsp",
        help=(
            "Completed identical baseline FSP to re-read fail-closed; "
            "only the baseline solve is skipped"
        ),
    )
    parser.add_argument("--expected-resume-base-sha256")
    parser.add_argument(
        "--quiet-json",
        action="store_true",
        help="write the full result JSON to disk but print only final status",
    )
    parser.add_argument(
        "--resume-project",
        action="append",
        default=[],
        metavar="ROLE=PATH@SHA256",
        help=(
            "SHA-verified completed forward/adjoint FSP to re-read. "
            "May be repeated; an unlisted solve is executed normally."
        ),
    )
    args = parser.parse_args()
    if bool(args.resume_base_fsp) != bool(
        args.expected_resume_base_sha256
    ):
        parser.error(
            "resume base FSP and expected SHA-256 must be supplied together"
        )
    return args


def parse_resume_projects(values: list[str]) -> dict[str, Path]:
    projects: dict[str, Path] = {}
    allowed = {
        "fd_plus_h0.005",
        "fd_minus_h0.005",
        "fd_plus_h0.01",
        "fd_minus_h0.01",
        "flake_4um_adjoint_gpu",
        "flake_4um_adjoint_cpu",
        "flake_6um_adjoint_gpu",
        "flake_6um_adjoint_cpu",
    }
    for value in values:
        try:
            role, path_and_sha = value.split("=", 1)
            path_text, expected_sha = path_and_sha.rsplit("@", 1)
        except ValueError as exc:
            raise ValueError(
                f"invalid --resume-project {value!r}; "
                "expected ROLE=PATH@SHA256"
            ) from exc
        if role not in allowed:
            raise ValueError(f"unsupported resume-project role {role!r}")
        if role in projects:
            raise ValueError(f"duplicate resume-project role {role!r}")
        path = Path(path_text).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        actual_sha = sha256(path)
        if actual_sha != expected_sha:
            raise RuntimeError(
                f"resume-project SHA-256 mismatch for {role}: "
                f"{actual_sha} != {expected_sha}"
            )
        projects[role] = path
    return projects


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(json.dumps(array.shape).encode())
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def physical_state() -> tuple[np.ndarray, np.ndarray]:
    x, y = design_nodal_coordinates_m()
    xn = x[:, None] / 1.0e-6
    yn = y[None, :] / 1.0e-6
    rho = (
        0.5
        + 0.07
        * np.sin(0.65 * np.pi * (xn + 0.11))
        * np.cos(0.55 * np.pi * (yn - 0.17))
        + 0.025 * xn
        - 0.015 * yn
    )
    direction = (
        np.cos(0.45 * np.pi * (xn - 0.19))
        * np.sin(0.70 * np.pi * (yn + 0.13))
        + 0.18 * xn * yn
        - 0.11 * xn
    )
    direction /= np.max(np.abs(direction))
    if np.min(rho) <= 0.1 or np.max(rho) >= 0.9:
        raise RuntimeError("baseline physical rho lacks FD margin")
    return rho, direction


def selected_design_edges(geometry) -> tuple[np.ndarray, np.ndarray]:
    result = []
    for edges in (geometry.x_edges_m, geometry.y_edges_m):
        centers = 0.5 * (edges[:-1] + edges[1:])
        selected = np.flatnonzero(
            (centers > -1.0e-6) & (centers < 1.0e-6)
        )
        if selected.size == 0 or np.any(np.diff(selected) != 1):
            raise RuntimeError("thermal design cells are not contiguous")
        target = edges[selected[0] : selected[-1] + 2]
        if not np.allclose(
            target[[0, -1]], [-1.0e-6, 1.0e-6], rtol=0.0, atol=2e-18
        ):
            raise RuntimeError("thermal design bounds changed")
        result.append(target)
    return result[0], result[1]


def coupling_for_geometry(geometry) -> NodalPhysicalCoupling:
    x, y = design_nodal_coordinates_m()
    tx, ty = selected_design_edges(geometry)
    return NodalPhysicalCoupling(
        x_nodes_m=x,
        y_nodes_m=y,
        optical_z_nodes_m=design_extrusion_nodes_m(),
        thermal_x_edges_m=tx,
        thermal_y_edges_m=ty,
    )


def imported_index(rho: np.ndarray) -> np.ndarray:
    density = np.asarray(rho, float)
    if density.shape != (81, 81):
        raise ValueError(f"physical rho shape {density.shape} != (81,81)")
    if np.any(density < 0.0) or np.any(density > 1.0):
        raise ValueError("physical rho leaves [0,1]")
    epsilon = 1.0 + density * (SIO2_EPSILON - 1.0)
    return np.repeat(
        np.sqrt(epsilon)[:, :, None],
        design_extrusion_nodes_m().size,
        axis=2,
    )


def set_imported_density(fdtd: object, rho: np.ndarray) -> None:
    if int(fdtd.getnamednumber(IMPORTED_OBJECT)) != 1:
        raise RuntimeError("expected exactly one imported design object")
    fdtd.select(IMPORTED_OBJECT)
    x, y = design_nodal_coordinates_m()
    fdtd.importnk2(imported_index(rho), x, y, design_extrusion_nodes_m())


def run_forward_density(
    fdtd: object,
    *,
    rho: np.ndarray,
    project: Path,
    threads: int,
    flux_signs: dict[str, float],
    reuse_completed: bool = False,
) -> dict[str, object]:
    if reuse_completed:
        if not project.is_file():
            raise FileNotFoundError(project)
        fdtd.load(str(project))
        resources = configure_cpu(fdtd, threads)
        resource_name = "REUSED_COMPLETED_FSP"
        run_wall = 0.0
    else:
        fdtd.switchtolayout()
        set_imported_density(fdtd, rho)
        fdtd.setnamed(FIELD_REGION, "source mode", False)
        fdtd.setnamed(SOURCE_NAME, "enabled", True)
        resources = configure_cpu(fdtd, threads)
        fdtd.save(str(project))
        started = time.monotonic()
        resource_name = run_on_cpu(fdtd)
        run_wall = time.monotonic() - started
        fdtd.save(str(project))
    electric, grid = monitor_electric(fdtd, PABS_FIELD)
    epsilon, index_grid = monitor_epsilon(fdtd, PABS_INDEX)
    for axis in "xyzf":
        if not np.array_equal(grid[axis], index_grid[axis]):
            raise RuntimeError(f"Q E/index {axis} grids differ")
    design_electric, design_grid = monitor_electric(fdtd, DESIGN_FIELD)
    design_epsilon, design_index_grid = monitor_epsilon(
        fdtd, DESIGN_INDEX
    )
    if design_electric.shape != design_epsilon.shape:
        raise RuntimeError(
            "design E/index component arrays have different shapes"
        )
    design_coordinate_readback = {}
    for axis in "xyzf":
        if design_grid[axis].shape != design_index_grid[axis].shape:
            raise RuntimeError(
                f"design E/index {axis} coordinate counts differ"
            )
        design_coordinate_readback[axis] = {
            "field_bounds": [
                float(design_grid[axis][0]),
                float(design_grid[axis][-1]),
            ],
            "index_bounds": [
                float(design_index_grid[axis][0]),
                float(design_index_grid[axis][-1]),
            ],
            "maximum_same_array_index_difference": float(
                np.max(
                    np.abs(
                        design_grid[axis] - design_index_grid[axis]
                    )
                )
            ),
        }
    native = extract_native_yee_q(
        fdtd,
        field_monitor=PABS_FIELD,
        index_monitor=PABS_INDEX,
        wavelength_m=4.0e-6,
    )
    source_power = scalar(
        fdtd.sourcepower(FREQUENCY_HZ, 2, SOURCE_NAME), "source power"
    )
    net_outward = 0.0
    face_power = {}
    for name, sign in flux_signs.items():
        signed = scalar(fdtd.transmission(name), name) * source_power
        outward = sign * signed
        face_power[name] = {
            "signed_axis_power_W": signed,
            "outward_power_W": outward,
        }
        net_outward += outward
    p_q = float(native["P_Q_W"])
    p_six = -net_outward
    closure = abs(p_q - p_six) / max(abs(p_six), np.finfo(float).tiny)
    return {
        "rho_sha256": array_sha256(rho),
        "native": native,
        "electric": electric,
        "epsilon": epsilon,
        "grid": grid,
        "design_electric": design_electric,
        "design_epsilon": design_epsilon,
        "design_grid": design_grid,
        "design_index_grid": design_index_grid,
        "design_coordinate_readback": design_coordinate_readback,
        "P_Q_W": p_q,
        "P_six_W": p_six,
        "six_face_closure_relative_error": closure,
        "component_power_W": native["component_power_W"],
        "source_intensity_W_m2": scalar(
            fdtd.sourceintensity(FREQUENCY_HZ, 2, SOURCE_NAME),
            "source intensity",
        ),
        "face_power": face_power,
        "run_wall_s": run_wall,
        "reused_completed_project": reuse_completed,
        "resource_name": resource_name,
        "resources": resources,
        "project": {
            "path": str(project),
            "byte_size": project.stat().st_size,
            "sha256": sha256(project),
        },
    }


def build_native_thermal_mapping(native: dict, geometry) -> dict:
    target_edges = (
        geometry.x_edges_m,
        geometry.y_edges_m,
        geometry.z_edges_m,
    )
    source = np.zeros(geometry.material_id.shape, float)
    records = {}
    for component in "xyz":
        density = np.asarray(native["Q_components"][component], float)
        if not np.any(density):
            records[component] = {
                "box": None,
                "remap": None,
                "native_power_W": float(
                    native["component_power_W"][component]
                ),
            }
            continue
        box, outside = exact_nonzero_box(density)
        if outside != 0:
            raise RuntimeError("native Q nonzero-box audit failed")
        source_edges = tuple(
            nodal_control_volume_edges(
                native["native_coordinates"][component][axis]
            )[section.start : section.stop + 1]
            for axis, section in zip("xyz", box)
        )
        remap = build_conservative_embedding_remap(
            source_edges_m=source_edges,
            target_edges_m=target_edges,
        )
        remap = project_remap_to_material_support_along_axis(
            remap,
            target_edges_m=target_edges,
            target_support_mask=geometry.flake_mask,
            axis=2,
        )
        mapped = remap.apply(density[box])
        source += mapped
        records[component] = {
            "box": box,
            "remap": remap,
            "native_power_W": float(
                native["component_power_W"][component]
            ),
        }
    volume = (
        np.diff(target_edges[0])[:, None, None]
        * np.diff(target_edges[1])[None, :, None]
        * np.diff(target_edges[2])[None, None, :]
    )
    mapped_power = float(np.sum(volume * source))
    native_power = float(native["P_Q_W"])
    outside_count = int(np.count_nonzero(source[~geometry.flake_mask]))
    return {
        "source_W_m3": source,
        "records": records,
        "mapped_power_W": mapped_power,
        "native_power_W": native_power,
        "relative_power_error": abs(mapped_power - native_power)
        / max(abs(native_power), np.finfo(float).tiny),
        "outside_flake_nonzero_count": outside_count,
    }


def apply_existing_mapping(native: dict, mapping: dict, geometry) -> dict:
    source = np.zeros(geometry.material_id.shape, float)
    for component in "xyz":
        density = np.asarray(native["Q_components"][component], float)
        record = mapping["records"][component]
        if record["remap"] is None:
            if np.any(density):
                raise RuntimeError("previously zero Q component became nonzero")
            continue
        box = record["box"]
        outside = np.array(density, copy=True)
        outside[box] = 0.0
        if np.any(outside):
            raise RuntimeError("perturbed Q escaped fixed native support")
        source += record["remap"].apply(density[box])
    volume = (
        np.diff(geometry.x_edges_m)[:, None, None]
        * np.diff(geometry.y_edges_m)[None, :, None]
        * np.diff(geometry.z_edges_m)[None, None, :]
    )
    mapped_power = float(np.sum(volume * source))
    native_power = float(native["P_Q_W"])
    return {
        "source_W_m3": source,
        "mapped_power_W": mapped_power,
        "native_power_W": native_power,
        "relative_power_error": abs(mapped_power - native_power)
        / max(abs(native_power), np.finfo(float).tiny),
        "outside_flake_nonzero_count": int(
            np.count_nonzero(source[~geometry.flake_mask])
        ),
    }


def native_weight_and_source(
    *,
    evaluation,
    native: dict,
    mapping: dict,
    electric: np.ndarray,
    epsilon: np.ndarray,
    frequency_Hz: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    sensitivity = evaluation.system.full_field(
        evaluation.gradient_Q_active_A_m3_W
    )
    coefficient = np.zeros((*electric.shape[:3], 3), float)
    transpose_errors = {}
    rng = np.random.default_rng(2026072710)
    for index, component in enumerate("xyz"):
        record = mapping["records"][component]
        if record["remap"] is None:
            continue
        box = record["box"]
        pulled = record["remap"].transpose(sensitivity)
        coefficient[..., index][box] = pulled
        probe = rng.normal(size=pulled.shape)
        left = float(
            np.sum(
                sensitivity * record["remap"].apply(probe)
            )
        )
        right = float(np.sum(pulled * probe))
        transpose_errors[component] = abs(left - right) / max(
            abs(left), abs(right), np.finfo(float).tiny
        )
    q = np.stack(
        [native["Q_components"][component] for component in "xyz"],
        axis=-1,
    )
    weighted_value = float(np.sum(coefficient * q))
    objective_identity = abs(weighted_value - evaluation.objective_A) / max(
        abs(weighted_value),
        abs(evaluation.objective_A),
        np.finfo(float).tiny,
    )
    omega = 2.0 * np.pi * frequency_Hz
    source = np.zeros_like(electric)
    for component in range(3):
        source[..., 0, component] = (
            0.5
            * EPS0
            * omega
            * np.imag(epsilon[..., 0, component])
            * coefficient[..., component]
            * electric[..., 0, component]
        )
    return coefficient, source, {
        "worst_transpose_relative_error": max(
            transpose_errors.values(), default=0.0
        ),
        "objective_from_weighted_native_Q_A": weighted_value,
        "thermal_forward_objective_A": float(evaluation.objective_A),
        "objective_identity_relative_error": objective_identity,
    }


def compact_forward(value: dict) -> dict:
    return {
        key: item
        for key, item in value.items()
        if key
        not in {
            "native",
            "electric",
            "epsilon",
            "grid",
            "design_electric",
            "design_epsilon",
            "design_grid",
            "design_index_grid",
        }
    }


def read_completed_adjoint(
    fdtd: object,
    *,
    project: Path,
    engine: str,
) -> dict[str, object]:
    fdtd.switchtolayout()
    fdtd.load(str(project))
    if int(fdtd.getnamednumber(SOURCE_NAME)) != 0:
        raise RuntimeError("completed adjoint unexpectedly contains TFSF")
    electric, grid = monitor_electric(fdtd, DESIGN_FIELD)
    return {
        "engine": engine,
        "electric": electric,
        "grid": grid,
        "run_wall_s": 0.0,
        "resource_name": "REUSED_COMPLETED_FSP",
        "resources": {"reuse": True},
        "project": {
            "path": str(project),
            "byte_size": project.stat().st_size,
            "sha256": sha256(project),
        },
    }


def main() -> int:
    args = parse_args()
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "combined_physical_rho_pte_adfd_result.json"
    result: dict[str, object] = {
        "status": "BLOCKED_COMBINED_PHYSICAL_RHO_PTE_ADFD_NOT_RUN",
        "passed": False,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "81x81 nodal physical-rho combined Maxwell-Q and explicit "
            "thermal-material PTE directional AD-FD"
        ),
        "optimization_run": False,
        "transient_run": False,
        "filter_projection_run": False,
    }
    fdtd = None
    started = time.monotonic()
    try:
        steps = [float(value) for value in args.steps.split(",")]
        if not steps or any(value <= 0.0 for value in steps):
            raise ValueError("positive FD steps required")
        rho, direction = physical_state()
        if any(
            np.min(rho - step * direction) < 0.0
            or np.max(rho + step * direction) > 1.0
            for step in steps
        ):
            raise ValueError("FD state leaves [0,1]")
        resume_projects = parse_resume_projects(args.resume_project)
        result["resume_projects"] = {
            role: {
                "path": str(path),
                "byte_size": path.stat().st_size,
                "sha256": sha256(path),
            }
            for role, path in resume_projects.items()
        }
        contract = baseline_contract(
            lateral_domain_um=7.2,
            z_min_um=-3.6,
            z_max_um=3.4,
            pml_layers=32,
            flake_dz_nm=2.5,
        )
        lumapi = load_lumapi()
        fdtd = lumapi.FDTD(hide=True, serverArgs={"platform": "offscreen"})
        add_large_background_geometry(fdtd, contract)
        pml_readback = configure_pml_profile(fdtd, "stabilized-xy")
        design_meta = configure_design(
            fdtd, "gray", 0.5, "imported-permittivity"
        )
        set_imported_density(fdtd, rho)
        fdtd.setnamed(
            "FDTD", "simulation time", args.simulation_time_ps * 1.0e-12
        )
        flux_signs = add_monitors(fdtd, contract)
        add_adjoint_monitors(fdtd, contract)
        base_project = (
            Path(args.resume_base_fsp).expanduser().resolve()
            if args.resume_base_fsp
            else output / "combined_base_forward_cpu_tfsf.fsp"
        )
        if args.resume_base_fsp:
            actual_base_sha = sha256(base_project)
            if actual_base_sha != args.expected_resume_base_sha256:
                raise RuntimeError("resume baseline FSP SHA-256 mismatch")
        base = run_forward_density(
            fdtd,
            rho=rho,
            project=base_project,
            threads=args.threads,
            flux_signs=flux_signs,
            reuse_completed=bool(args.resume_base_fsp),
        )
        result["optical_contract"] = {
            "forward_engine": "CPU TFSF",
            "adjoint_engine": "GPU FieldRegion",
            "periodic_or_bloch": False,
            "pml_layers": 32,
            "pml_profile_readback": pml_readback,
            "flake_dz_m": 2.5e-9,
            "design_representation": design_meta,
            "base_forward": compact_forward(base),
        }
        scenario_bases = {}
        for flake_um in (4.0, 6.0):
            initial_geometry = build_explicit_geometry(
                np.full((20, 20), 0.5),
                lateral_domain_m=32.0e-6,
                si_depth_m=20.0e-6,
                flake_span_m=flake_um * 1.0e-6,
                core_xy_cell_size_m=100.0e-9,
                flake_dz_m=25.0e-9,
                design_dz_m=100.0e-9,
            )
            coupling = coupling_for_geometry(initial_geometry)
            thermal_rho = coupling.thermal(rho)
            geometry = build_explicit_geometry(
                thermal_rho,
                lateral_domain_m=32.0e-6,
                si_depth_m=20.0e-6,
                flake_span_m=flake_um * 1.0e-6,
                core_xy_cell_size_m=100.0e-9,
                flake_dz_m=25.0e-9,
                design_dz_m=100.0e-9,
            )
            mapping = build_native_thermal_mapping(
                base["native"], geometry
            )
            evaluation = evaluate_explicit_thermal(
                rho=thermal_rho,
                source_W_m3=mapping["source_W_m3"],
                lateral_domain_m=32.0e-6,
                si_depth_m=20.0e-6,
                flake_span_m=flake_um * 1.0e-6,
                core_xy_cell_size_m=100.0e-9,
                flake_dz_m=25.0e-9,
                design_dz_m=100.0e-9,
            )
            coefficient, weighted_source, pullback = (
                native_weight_and_source(
                    evaluation=evaluation,
                    native=base["native"],
                    mapping=mapping,
                    electric=base["electric"],
                    epsilon=base["epsilon"],
                    frequency_Hz=float(base["grid"]["f"][0]),
                )
            )
            thermal_material_nodal = coupling.thermal_vjp(
                evaluation.gradient_rho_A
            )
            profile, profile_scale = fieldregion_profile(weighted_source)
            fdtd.switchtolayout()
            fdtd.load(str(base_project))
            template = output / f"flake_{flake_um:g}um_adjoint_template.fsp"
            source_meta = prepare_adjoint_layout(
                fdtd,
                grid=base["grid"],
                profile=profile,
                template=template,
            )
            scenario_bases[flake_um] = {
                "coupling": coupling,
                "thermal_rho": thermal_rho,
                "geometry": geometry,
                "mapping": mapping,
                "evaluation": evaluation,
                "coefficient": coefficient,
                "thermal_material_nodal": thermal_material_nodal,
                "profile_scale": profile_scale,
                "template": template,
                "source_meta": source_meta,
                "pullback": pullback,
            }

        # Imported-material updates invalidate layout-mode index d-cards.
        # Reuse the smallest actual centered-FD forward pair to obtain the
        # exact solver-realized directional epsilon_Yee derivative.  These
        # are not extra solves and the projects are reused below.
        derivative_step = min(steps)
        cached_forwards = {}
        derivative_pair = {}
        for sign, label in ((1.0, "plus"), (-1.0, "minus")):
            fdtd.switchtolayout()
            fdtd.load(str(base_project))
            role = f"fd_{label}_h{derivative_step:g}"
            resume_path = resume_projects.get(role)
            forward = run_forward_density(
                fdtd,
                rho=rho + sign * derivative_step * direction,
                project=resume_path
                or output
                / f"{role}_cpu_tfsf.fsp",
                threads=args.threads,
                flux_signs=flux_signs,
                reuse_completed=resume_path is not None,
            )
            cached_forwards[(derivative_step, label)] = forward
            derivative_pair[label] = forward
        derivative = (
            derivative_pair["plus"]["design_epsilon"]
            - derivative_pair["minus"]["design_epsilon"]
        ) / (2.0 * derivative_step)
        design_grid = base["design_grid"]
        forward_design_e = base["design_electric"]
        if derivative.shape != forward_design_e.shape:
            raise RuntimeError(
                "directional epsilon and forward E shapes differ"
            )
        for axis in "xyzf":
            if not np.array_equal(
                derivative_pair["plus"]["design_index_grid"][axis],
                derivative_pair["minus"]["design_index_grid"][axis],
            ):
                raise RuntimeError(
                    f"plus/minus design index {axis} grids differ"
                )
        derivative_meta = {
            "method": (
                "centered completed-CPU-TFSF imported-permittivity "
                "directional epsilon_Yee derivative"
            ),
            "pairing": (
                "same-shape Lumerical E/index component-array pairing; "
                "no coordinate interpolation"
            ),
            "step": derivative_step,
            "electromagnetic_solves": 2,
            "solves_shared_with_objective_centered_FD": True,
            "base_design_coordinate_readback": base[
                "design_coordinate_readback"
            ],
        }

        scenarios = {}
        for flake_um, data in scenario_bases.items():
            gpu_role = f"flake_{flake_um:g}um_adjoint_gpu"
            if gpu_role in resume_projects:
                gpu = read_completed_adjoint(
                    fdtd,
                    project=resume_projects[gpu_role],
                    engine="GPU",
                )
            else:
                gpu = run_adjoint(
                    fdtd,
                    template=data["template"],
                    project=output / f"{gpu_role}.fsp",
                    engine="GPU",
                    threads=args.threads,
                    gpu_device=args.gpu_device,
                )
            cpu_equivalence = None
            if flake_um == 4.0:
                cpu_role = "flake_4um_adjoint_cpu"
                if cpu_role in resume_projects:
                    cpu = read_completed_adjoint(
                        fdtd,
                        project=resume_projects[cpu_role],
                        engine="CPU",
                    )
                else:
                    cpu = run_adjoint(
                        fdtd,
                        template=data["template"],
                        project=output / f"{cpu_role}.fsp",
                        engine="CPU",
                        threads=args.threads,
                        gpu_device=args.gpu_device,
                    )
                cpu_equivalence = float(
                    np.linalg.norm(gpu["electric"] - cpu["electric"])
                    / max(
                        np.linalg.norm(cpu["electric"]),
                        np.finfo(float).tiny,
                    )
                )
            optical_directional, optical_components = gradient_from_adjoint(
                forward_electric=forward_design_e,
                adjoint_electric=gpu["electric"],
                design_grid=design_grid,
                d_epsilon_d_rho=derivative,
                profile_scale=data["profile_scale"],
                base_amplitude=data["source_meta"][
                    "fieldregion_base_amplitude"
                ],
                design_bounds_m=DESIGN_BOUNDS_M,
            )
            thermal_directional = float(
                np.sum(data["thermal_material_nodal"] * direction)
            )
            combined = optical_directional + thermal_directional
            scenarios[flake_um] = {
                "base_objective_A": float(
                    data["evaluation"].objective_A
                ),
                "thermal_rho_shape": list(data["thermal_rho"].shape),
                "thermal_material_directional_A": thermal_directional,
                "optical_Q_directional_A": optical_directional,
                "optical_components_xyz_A": optical_components,
                "combined_adjoint_directional_A": combined,
                "gradient_component_sum_relative_error": abs(
                    combined - optical_directional - thermal_directional
                )
                / max(abs(combined), np.finfo(float).tiny),
                "mapping_relative_power_error": data["mapping"][
                    "relative_power_error"
                ],
                "mapping_outside_flake_nonzero_count": data["mapping"][
                    "outside_flake_nonzero_count"
                ],
                "pullback": data["pullback"],
                "forward_energy_balance_relative_error": float(
                    data["evaluation"].solved.energy_balance_relative_error
                ),
                "forward_linear_residual_relative": float(
                    data["evaluation"].solved.linear_residual_relative
                ),
                "adjoint_linear_residual_relative": float(
                    data["evaluation"].adjoint_linear_residual_relative
                ),
                "source_profile_roundtrip_max_abs_error": data[
                    "source_meta"
                ]["source_profile_roundtrip_max_abs_error"],
                "cpu_gpu_adjoint_field_NRMSE": cpu_equivalence,
                "gpu_project": gpu["project"],
                "fd_rows": [],
            }

        for step in steps:
            pair = {}
            for sign, label in ((1.0, "plus"), (-1.0, "minus")):
                perturbed = rho + sign * step * direction
                forward = cached_forwards.get((step, label))
                if forward is None:
                    fdtd.switchtolayout()
                    fdtd.load(str(base_project))
                    role = f"fd_{label}_h{step:g}"
                    resume_path = resume_projects.get(role)
                    forward = run_forward_density(
                        fdtd,
                        rho=perturbed,
                        project=resume_path
                        or output
                        / f"{role}_cpu_tfsf.fsp",
                        threads=args.threads,
                        flux_signs=flux_signs,
                        reuse_completed=resume_path is not None,
                    )
                pair[label] = {
                    "forward": compact_forward(forward),
                    "objectives": {},
                }
                for flake_um, data in scenario_bases.items():
                    thermal_rho = data["coupling"].thermal(perturbed)
                    mapped = apply_existing_mapping(
                        forward["native"],
                        data["mapping"],
                        data["geometry"],
                    )
                    solved = solve_explicit_forward(
                        rho=thermal_rho,
                        source_W_m3=mapped["source_W_m3"],
                        lateral_domain_m=32.0e-6,
                        si_depth_m=20.0e-6,
                        flake_span_m=flake_um * 1.0e-6,
                        core_xy_cell_size_m=100.0e-9,
                        flake_dz_m=25.0e-9,
                        design_dz_m=100.0e-9,
                    )
                    pair[label]["objectives"][flake_um] = {
                        "objective_A": float(solved.objective_A),
                        "mapping_relative_power_error": mapped[
                            "relative_power_error"
                        ],
                        "mapping_outside_flake_nonzero_count": mapped[
                            "outside_flake_nonzero_count"
                        ],
                        "energy_balance_relative_error": float(
                            solved.solved.energy_balance_relative_error
                        ),
                        "linear_residual_relative": float(
                            solved.solved.linear_residual_relative
                        ),
                    }
            for flake_um in (4.0, 6.0):
                plus = pair["plus"]["objectives"][flake_um]["objective_A"]
                minus = pair["minus"]["objectives"][flake_um][
                    "objective_A"
                ]
                finite_difference = (plus - minus) / (2.0 * step)
                adjoint = scenarios[flake_um][
                    "combined_adjoint_directional_A"
                ]
                error = abs(adjoint - finite_difference) / max(
                    abs(adjoint),
                    abs(finite_difference),
                    np.finfo(float).tiny,
                )
                scenarios[flake_um]["fd_rows"].append(
                    {
                        "step": step,
                        "objective_plus_A": plus,
                        "objective_minus_A": minus,
                        "finite_difference_directional_A": finite_difference,
                        "combined_adjoint_directional_A": adjoint,
                        "relative_error": error,
                        "plus_forward": pair["plus"]["forward"],
                        "minus_forward": pair["minus"]["forward"],
                        "plus_thermal": pair["plus"]["objectives"][
                            flake_um
                        ],
                        "minus_thermal": pair["minus"]["objectives"][
                            flake_um
                        ],
                    }
                )
                print(
                    "COMBINED_PHYSICAL_RHO_ADFD "
                    f"flake={flake_um:g}um h={step:g} error={error:.6e}",
                    flush=True,
                )

        selected_step = min(steps)
        selected_errors = []
        all_energy = []
        all_residual = []
        all_mapping = []
        all_closure = [base["six_face_closure_relative_error"]]
        for flake_um, scenario in scenarios.items():
            selected = next(
                row
                for row in scenario["fd_rows"]
                if row["step"] == selected_step
            )
            selected_errors.append(selected["relative_error"])
            all_energy.append(
                scenario["forward_energy_balance_relative_error"]
            )
            all_residual.extend(
                [
                    scenario["forward_linear_residual_relative"],
                    scenario["adjoint_linear_residual_relative"],
                ]
            )
            all_mapping.append(scenario["mapping_relative_power_error"])
            for row in scenario["fd_rows"]:
                all_closure.extend(
                    [
                        row["plus_forward"][
                            "six_face_closure_relative_error"
                        ],
                        row["minus_forward"][
                            "six_face_closure_relative_error"
                        ],
                    ]
                )
                for key in ("plus_thermal", "minus_thermal"):
                    all_energy.append(
                        row[key]["energy_balance_relative_error"]
                    )
                    all_residual.append(
                        row[key]["linear_residual_relative"]
                    )
                    all_mapping.append(
                        row[key]["mapping_relative_power_error"]
                    )
        gates = {
            "worst_selected_combined_AD_FD_relative_error": max(
                selected_errors
            ),
            "combined_AD_FD_limit": ADFD_LIMIT,
            "worst_Q_mapping_relative_error": max(all_mapping),
            "Q_mapping_limit": 5.0e-3,
            "worst_six_face_closure_relative_error": max(all_closure),
            "six_face_closure_limit": 5.0e-3,
            "worst_thermal_energy_balance_relative_error": max(all_energy),
            "thermal_energy_balance_limit": 1.0e-2,
            "worst_forward_or_adjoint_linear_residual_relative": max(
                all_residual
            ),
            "linear_residual_limit": 1.0e-8,
            "worst_pullback_transpose_relative_error": max(
                data["pullback"]["worst_transpose_relative_error"]
                for data in scenario_bases.values()
            ),
            "worst_pullback_objective_identity_relative_error": max(
                data["pullback"]["objective_identity_relative_error"]
                for data in scenario_bases.values()
            ),
            "four_um_CPU_GPU_adjoint_field_NRMSE": scenarios[4.0][
                "cpu_gpu_adjoint_field_NRMSE"
            ],
        }
        passed = bool(
            gates["worst_selected_combined_AD_FD_relative_error"]
            < ADFD_LIMIT
            and gates["worst_Q_mapping_relative_error"] < 5.0e-3
            and gates["worst_six_face_closure_relative_error"] < 5.0e-3
            and gates["worst_thermal_energy_balance_relative_error"]
            < 1.0e-2
            and gates[
                "worst_forward_or_adjoint_linear_residual_relative"
            ]
            < 1.0e-8
            and gates["worst_pullback_transpose_relative_error"] < 1.0e-12
            and gates[
                "worst_pullback_objective_identity_relative_error"
            ]
            < 1.0e-10
            and gates["four_um_CPU_GPU_adjoint_field_NRMSE"] < 5.0e-3
            and all(
                scenario["source_profile_roundtrip_max_abs_error"] == 0.0
                and scenario["mapping_outside_flake_nonzero_count"] == 0
                and scenario["gradient_component_sum_relative_error"]
                < 1.0e-12
                for scenario in scenarios.values()
            )
        )
        raw_path = output / "combined_physical_rho_pte_adfd.npz"
        np.savez_compressed(
            raw_path,
            rho_physical=rho,
            direction_physical=direction,
            gradient_thermal_material_4um=scenario_bases[4.0][
                "thermal_material_nodal"
            ],
            gradient_thermal_material_6um=scenario_bases[6.0][
                "thermal_material_nodal"
            ],
            native_Q_coefficient_4um=scenario_bases[4.0][
                "coefficient"
            ],
            native_Q_coefficient_6um=scenario_bases[6.0][
                "coefficient"
            ],
        )
        result.update(
            {
                "status": STATUS_PASS if passed else STATUS_FAIL,
                "passed": passed,
                "physical_density": {
                    "shape": [81, 81],
                    "bounds_m": {
                        "x": [-1.0e-6, 1.0e-6],
                        "y": [-1.0e-6, 1.0e-6],
                    },
                    "spacing_m": 25.0e-9,
                    "baseline_min": float(np.min(rho)),
                    "baseline_max": float(np.max(rho)),
                    "baseline_sha256": array_sha256(rho),
                    "direction_sha256": array_sha256(direction),
                },
                "thermal_contract": {
                    "named_flake_scenarios_um": [4.0, 6.0],
                    "fabrication_truth_promoted": False,
                    "lateral_domain_um": 32.0,
                    "Si_depth_um": 20.0,
                    "core_xy_cell_size_nm": 100.0,
                    "flake_dz_nm": 25.0,
                    "design_dz_nm": 100.0,
                },
                "adjoint_source_definition": (
                    "native_weight_c=R_Q^T(dI/dQ_thermal); "
                    "dI/dE_c*=c_c*(omega*epsilon0/2)*"
                    "Im(epsilon_c)*E_c; vector FieldRegion source"
                ),
                "scalar_P_Q_source_reused": False,
                "layout_directional_epsilon": {
                    key: value
                    for key, value in derivative_meta.items()
                    if key != "grid"
                },
                "scenarios": {
                    f"{key:g}um": value
                    for key, value in scenarios.items()
                },
                "gates": gates,
                "raw_artifact": {
                    "path": str(raw_path),
                    "byte_size": raw_path.stat().st_size,
                    "sha256": sha256(raw_path),
                },
                "next_gate": (
                    "GRAY_LAW_SENSITIVITY"
                    if passed
                    else "STOP_FAIL_CLOSED"
                ),
            }
        )
    except Exception as exc:
        result["status"] = "BLOCKED_COMBINED_PHYSICAL_RHO_PTE_ADFD"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
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
        if args.quiet_json:
            print(
                "COMBINED_PHYSICAL_RHO_ADFD_FINAL "
                f"status={result.get('status')} "
                f"passed={result.get('passed')} "
                f"result={result_path}",
                flush=True,
            )
        else:
            print(
                json.dumps(result, indent=2, default=json_default),
                flush=True,
            )
    return 0 if result.get("passed") else 2


if __name__ == "__main__":
    raise SystemExit(main())
