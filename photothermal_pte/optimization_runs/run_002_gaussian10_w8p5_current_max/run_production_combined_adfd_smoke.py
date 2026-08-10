#!/usr/bin/env python3
"""GPU-only production combined physical-density PTE AD--FD smoke.

This gate deliberately uses the exact nonuniform density stored with the
production component-Yee Jacobian.  It does not combine the uniform-rho
thermal pullback control with a different optical material state.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np
from scipy import sparse


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from photothermal_pte.finite_inverse_design.finite_q_mapping import (  # noqa: E402
    apply_material_intersection_density_separable,
    nodal_control_volume_edges,
    transpose_material_intersection_density_separable,
)
from photothermal_pte.finite_inverse_design.native_yee_q import (  # noqa: E402
    EPS0,
    extract_native_yee_q,
)
from photothermal_pte.finite_inverse_design.probe_v261_cpu_tfsf_device import (  # noqa: E402
    PABS_FIELD,
    PABS_GROUP,
    PABS_INDEX,
)
from photothermal_pte.finite_inverse_design.run_v261_large_background_mixed_optical_adfd import (  # noqa: E402
    component_volumes,
    fieldregion_profile,
    import_named_fieldregion_profile,
    invert_fieldregion_linear_collocation,
    monitor_electric,
)
from photothermal_pte.finite_inverse_design.yee_material_jacobian import (  # noqa: E402
    SparseYeeMaterialJacobian,
)
from photothermal_pte.optimization_runs.cuda_thermal_adjoint import (  # noqa: E402
    PersistentCudaCSR,
    solve_forward_adjoint_cuda,
)
from photothermal_pte.optimization_runs.axis_contract import (  # noqa: E402
    LEGACY_X_A_Y_B,
    AxisContract,
)

import audit_production_candidate_geometry as geometry  # noqa: E402
from build_nonuniform_complex_yee_jacobian import (  # noqa: E402
    component_coordinates,
    index_detail,
    set_density,
)
from map_production_q_to_thermal_grid import (  # noqa: E402
    MATERIALS,
    material_masks,
    thermal_edges,
)
import run_complex_material_control as material_control  # noqa: E402
from run_production_candidate_forward import add_fieldregion  # noqa: E402
from validate_production_thermal_material_adfd import (  # noqa: E402
    boundary_energy,
    build_state,
    nodal_to_cell,
    nodal_to_cell_transpose,
    thermal_cell_gradient,
)
from selected_thermal_density_mapping import (  # noqa: E402
    selected_nodal_to_thermal_cell,
    selected_nodal_to_thermal_cell_transpose,
)


C0 = 299792458.0
WAVELENGTH_M = 10.0e-6
FREQUENCY_HZ = C0 / WAVELENGTH_M
FIELD_REGION = "run002_component_yee_adjoint_region"
SCENARIO = "grown_grown"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(first: float, second: float) -> float:
    return abs(first - second) / max(
        abs(first), abs(second), np.finfo(float).tiny
    )


def checked(path: Path, expected: str, label: str) -> Path:
    value = path.expanduser().resolve()
    if not value.is_file():
        raise FileNotFoundError(f"missing {label}: {value}")
    actual = sha256(value)
    if actual != expected:
        raise RuntimeError(
            f"{label} SHA mismatch: expected {expected}, got {actual}"
        )
    return value


def contract_configuration(name: str) -> dict[str, object]:
    if name == "selected_production":
        return {
            "operator_status": "VALIDATED_SELECTED_PRODUCTION_COMPLEX_COMPONENT_YEE_JACOBIAN",
            "imported_object": geometry.SELECTED_DESIGN_OBJECT,
            "nodes": geometry.design_nodes(
                geometry.SELECTED_DESIGN_BOUNDS,
                geometry.SELECTED_DESIGN_SHAPE,
            ),
            "design_half_span_m": 9.3e-6,
            "density_forward": selected_nodal_to_thermal_cell,
            "density_transpose": selected_nodal_to_thermal_cell_transpose,
            "nodal_shape": (373, 373),
            "thermal_cell_shape": (186, 186),
        }
    if name == "coarse_production":
        return {
            "operator_status": "VALIDATED_PRODUCTION_COMPLEX_COMPONENT_YEE_JACOBIAN",
            "imported_object": geometry.DESIGN_OBJECT,
            "nodes": geometry.design_nodes(),
            "design_half_span_m": 10.0e-6,
            "density_forward": nodal_to_cell,
            "density_transpose": nodal_to_cell_transpose,
            "nodal_shape": (201, 201),
            "thermal_cell_shape": (200, 200),
        }
    raise ValueError(f"unknown production contract {name!r}")


def load_operator(
    directory: Path, expected_status: str
) -> tuple[SparseYeeMaterialJacobian, np.ndarray, dict]:
    root = directory.expanduser().resolve()
    result_path = root / "component_yee_jacobian_result.json"
    result = json.loads(result_path.read_text())
    if result.get("status") != expected_status:
        raise RuntimeError(
            "component-Yee Jacobian status mismatch: "
            f"expected {expected_status}, got {result.get('status')}"
        )
    coordinate_path = Path(result["artifacts"]["coordinates_and_density"]["path"])
    checked(
        coordinate_path,
        result["artifacts"]["coordinates_and_density"]["sha256"],
        "Jacobian coordinate artifact",
    )
    layout = np.load(coordinate_path)
    rho = np.asarray(layout["rho"], float)
    matrices = {}
    shapes = {}
    for component in "xyz":
        artifact = result["artifacts"]["component_J"][component]
        path = checked(Path(artifact["path"]), artifact["sha256"], f"J_{component}")
        matrices[component] = sparse.load_npz(path)
        shapes[component] = tuple(
            result["coordinate_audit"]["components"][component]["shape"]
        )
    operator = SparseYeeMaterialJacobian(
        density_shape=tuple(rho.shape),
        component_shapes=shapes,
        matrices=matrices,
    )
    rng = np.random.default_rng(2026080607)
    direction = rng.normal(size=rho.shape)
    cotangent = {
        component: rng.normal(size=shapes[component])
        + 1j * rng.normal(size=shapes[component])
        for component in "xyz"
    }
    tangent = operator.jvp(direction)
    left = float(
        np.real(sum(np.sum(cotangent[c] * tangent[c]) for c in "xyz"))
    )
    right = float(np.vdot(direction, operator.vjp(cotangent)))
    dot_error = relative(left, right)
    if dot_error >= 1.0e-12:
        raise RuntimeError(f"fresh component-J transpose error {dot_error:.3e}")
    return operator, rho, {
        "result": {"path": str(result_path), "sha256": sha256(result_path)},
        "coordinates": {
            "path": str(coordinate_path),
            "sha256": sha256(coordinate_path),
        },
        "fresh_transpose_dot_error": dot_error,
        "maximum_coordinate_mismatch_m": result["maximum_coordinate_mismatch_m"],
        "density_sha256": result["density_sha256"],
        "density_range": result["density_range"],
    }


def open_fdtd(gpu_device: str):
    wrapper = material_control.load_source_wrapper()
    audit = wrapper.source_audit
    os.environ["VC_LUMERICAL_ROOT"] = str(audit.APPROVED_ROOT)
    os.environ["LUMERICAL_ROOT"] = str(audit.APPROVED_ROOT)
    os.environ["LUMERICAL_PYTHONPATH"] = str(audit.APPROVED_API)
    os.environ["LUMERICAL_SESSION_GPU_DEVICE"] = gpu_device
    os.environ["CL_GPU_DEVICE"] = gpu_device
    os.environ["FDTD_THREADS"] = "8"
    os.environ["PATH"] = (
        f"{audit.APPROVED_ROOT / 'bin'}:{os.environ.get('PATH', '')}"
    )
    for path in (audit.STAGE1, REPOSITORY / "photothermal_pte"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    helper = audit.load_module(audit.API_HELPER, "run002_combined_adfd_api")
    installation = type(
        "Installation",
        (),
        {
            "version_key": "v261",
            "root": audit.APPROVED_ROOT,
            "lumapi_path": audit.APPROVED_API / "lumapi.py",
            "device_executable": audit.APPROVED_ROOT / "bin" / "device",
        },
    )()
    lumapi = helper.load_lumapi(installation)
    import eqc_lib as runtime

    fdtd = lumapi.FDTD(hide=True, serverArgs={"platform": "offscreen"})
    return fdtd, audit, runtime


def native_arrays(q: dict) -> dict[str, np.ndarray]:
    arrays = {}
    for component in "xyz":
        arrays[f"Q{component}_W_m3"] = np.asarray(
            q["Q_components"][component], float
        )
        for axis in "xyz":
            arrays[f"Q{component}_{axis}_m"] = np.asarray(
                q["native_coordinates"][component][axis], float
            )
    return arrays


def run_forward(
    fdtd,
    audit,
    runtime,
    *,
    base_fsp: Path,
    rho: np.ndarray,
    role: str,
    output: Path,
    imported_object: str,
    nodes: tuple[np.ndarray, np.ndarray, np.ndarray],
    completed_project: Path | None = None,
    polarization_angle_deg: float | None = None,
) -> dict:
    project = (
        completed_project.resolve()
        if completed_project is not None
        else output / f"{role}.fsp"
    )
    q_path = output / f"{role}_native_q.npz"
    if completed_project is None:
        fdtd.load(str(base_fsp))
        fdtd.switchtolayout()
        if polarization_angle_deg is not None:
            fdtd.setnamed(
                audit.SOURCE_NAME,
                "polarization angle",
                float(polarization_angle_deg),
            )
        set_density(
            fdtd,
            rho,
            imported_object=imported_object,
            nodes=nodes,
        )
        add_fieldregion(fdtd)
        fdtd.setnamed(FIELD_REGION, "source mode", False)
        fdtd.setnamed(audit.SOURCE_NAME, "enabled", True)
        resources = runtime.configure_session_resources(fdtd)
        fdtd.save(str(project))
        started = time.monotonic()
        resource_used = audit.strict_gpu_run(fdtd, f"run002_combined_{role}")
        wall = time.monotonic() - started
    else:
        fdtd.load(str(project))
        resources = {"reuse_completed_FSP": True}
        resource_used = "REUSED_COMPLETED_GPU_FORWARD"
        wall = 0.0
    actual_polarization_angle_deg = float(
        fdtd.getnamed(audit.SOURCE_NAME, "polarization angle")
    )
    if (
        polarization_angle_deg is not None
        and not np.isclose(
            actual_polarization_angle_deg,
            float(polarization_angle_deg),
            rtol=0.0,
            atol=1.0e-12,
        )
    ):
        raise RuntimeError(
            "source polarization readback mismatch: "
            f"requested {polarization_angle_deg}, got "
            f"{actual_polarization_angle_deg}"
        )
    fdtd.runanalysis(PABS_GROUP)
    if completed_project is None:
        fdtd.save(str(project))

    source_power = audit.scalar(
        fdtd.sourcepower(FREQUENCY_HZ, 2, audit.SOURCE_NAME), "sourcepower"
    )
    net_outward = 0.0
    faces = {}
    for axis in "xyz":
        for side, sign in (("min", -1.0), ("max", 1.0)):
            name = f"run002_production_flux_{axis}_{side}"
            signed = audit.scalar(fdtd.transmission(name), name) * source_power
            faces[name] = {
                "signed_axis_power_W": signed,
                "outward_power_W": sign * signed,
            }
            net_outward += sign * signed
    p_six = -net_outward
    q = extract_native_yee_q(
        fdtd,
        field_monitor=geometry.PABS_FIELD,
        index_monitor=geometry.PABS_INDEX,
        wavelength_m=WAVELENGTH_M,
    )
    electric, grid = monitor_electric(fdtd, PABS_FIELD)
    detail = index_detail(fdtd)
    epsilon = np.stack(
        [detail[f"epsilon_{component}"] for component in "xyz"], axis=-1
    )[..., None, :]
    if epsilon.shape != electric.shape:
        raise RuntimeError(
            f"forward E/index component shape mismatch: "
            f"{electric.shape} versus {epsilon.shape}"
        )
    coordinate_mismatch = 0.0
    for component_index, component in enumerate("xyz"):
        if electric.shape[:3] != q["Q_components"][component].shape:
            raise RuntimeError(f"E/Q{component} shape mismatch")
        index_coordinates = component_coordinates(detail, component)
        for axis_index, (axis, index_coordinate) in enumerate(
            zip("xyz", index_coordinates)
        ):
            expected = np.asarray(grid[axis], float)
            if component_index == axis_index:
                expected = expected + np.asarray(grid[f"delta_{component}"], float)
            actual = np.asarray(q["native_coordinates"][component][axis], float)
            coordinate_mismatch = max(
                coordinate_mismatch,
                float(np.max(np.abs(expected - np.asarray(index_coordinate, float)))),
                float(np.max(np.abs(actual - np.asarray(index_coordinate, float)))),
            )
    arrays = native_arrays(q)
    np.savez_compressed(q_path, **arrays)
    p_q = float(q["P_Q_W"])
    closure = relative(p_q, p_six)
    q_min = min(float(np.min(arrays[f"Q{c}_W_m3"])) for c in "xyz")
    log_audit = audit.log_audit(project.parent)
    auto_shutoff = log_audit.get("final_auto_shutoff")
    if auto_shutoff is None or float(auto_shutoff) >= 1.0e-5:
        raise RuntimeError(
            f"forward auto-shutoff gate failed for {role}: {auto_shutoff}"
        )
    if not all(np.all(np.isfinite(arrays[f"Q{c}_W_m3"])) for c in "xyz"):
        raise RuntimeError("native Q contains NaN or Inf")
    if q_min < 0.0:
        raise RuntimeError(f"native Q contains negative density {q_min}")
    return {
        "rho": np.array(rho, copy=True),
        "q": q,
        "electric": electric,
        "epsilon": epsilon,
        "grid": grid,
        "P_Q_W": p_q,
        "P_six_W": p_six,
        "closure": closure,
        "source_power_W": source_power,
        "source_polarization_angle_deg": actual_polarization_angle_deg,
        "faces": faces,
        "coordinate_mismatch_m": coordinate_mismatch,
        "resources": resources,
        "resource_used": resource_used,
        "solver_mode": "GPU",
        "log_audit": log_audit,
        "wall_s": wall,
        "reused_completed": completed_project is not None,
        "project": {
            "path": str(project),
            "size_bytes": project.stat().st_size,
            "sha256": sha256(project),
        },
        "native_Q": {
            "path": str(q_path),
            "size_bytes": q_path.stat().st_size,
            "sha256": sha256(q_path),
        },
    }


def map_q(
    q: dict, *, design_half_span_m: float
) -> tuple[dict[str, np.ndarray], dict]:
    edges = thermal_edges()
    masks = material_masks(edges, design_half_span_m=design_half_span_m)
    shape = tuple(axis.size - 1 for axis in edges)
    sources = {name: np.zeros(shape, float) for name in MATERIALS}
    attributed = 0.0
    mapped = 0.0
    records = {}
    for component in "xyz":
        source_density = np.asarray(q["Q_components"][component], float)
        source_edges = tuple(
            nodal_control_volume_edges(
                np.asarray(q["native_coordinates"][component][axis], float)
            )
            for axis in "xyz"
        )
        records[component] = {}
        for material in MATERIALS:
            density, _, audit = apply_material_intersection_density_separable(
                source_density=source_density,
                source_edges_m=source_edges,
                target_edges_m=edges,
                target_material_support_mask=masks[material],
            )
            sources[material] += density
            attributed += audit["material_attributed_source_power_W"]
            mapped += audit["target_integrated_power_W"]
            records[component][material] = audit
    total = sum(sources.values())
    data = {
        "x_edges_m": edges[0],
        "y_edges_m": edges[1],
        "z_edges_m": edges[2],
        "Q_total_W_m3": total,
        **{f"Q_{name}_W_m3": value for name, value in sources.items()},
        **{f"mask_{name}": value for name, value in masks.items()},
    }
    return data, {
        "material_attributed_input_power_W": attributed,
        "mapped_power_W": mapped,
        "internal_relative_power_error": relative(attributed, mapped),
        "physical_fraction_of_native_P_Q": mapped
        / max(float(q["P_Q_W"]), np.finfo(float).tiny),
        "records": records,
    }


def solve_base_thermal(
    data: dict,
    rho: np.ndarray,
    cuda_device: int,
    density_forward,
    density_transpose,
    axis_contract: AxisContract = LEGACY_X_A_Y_B,
):
    state = build_state(
        data,
        SCENARIO,
        density_forward(rho),
        axis_contract=axis_contract,
    )
    pair = solve_forward_adjoint_cuda(
        state.system.matrix_W_K,
        state.source_power_W,
        state.c_A_K,
        cuda_device=cuda_device,
        relative_tolerance=1.0e-10,
        max_iterations=30000,
    )
    objective = float(np.dot(state.c_A_K, pair.forward.solution))
    cell_parts = thermal_cell_gradient(
        state, pair.forward.solution, pair.adjoint.solution
    )
    nodal_parts = {
        name: density_transpose(value) for name, value in cell_parts.items()
    }
    target_active = np.asarray(
        state.system.source_volume_operator_m3.T @ pair.adjoint.solution
    ).reshape(-1)
    target_sensitivity = np.zeros(state.active.shape, float)
    target_sensitivity[state.active] = target_active
    return state, pair, objective, nodal_parts, target_sensitivity


def pullback_q(
    q: dict,
    data: dict,
    target_sensitivity: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict]:
    target_edges = tuple(np.asarray(data[f"{axis}_edges_m"], float) for axis in "xyz")
    masks = {name: np.asarray(data[f"mask_{name}"], bool) for name in MATERIALS}
    pulled = {}
    records = {}
    rng = np.random.default_rng(2026080608)
    for component in "xyz":
        source_edges = tuple(
            nodal_control_volume_edges(
                np.asarray(q["native_coordinates"][component][axis], float)
            )
            for axis in "xyz"
        )
        total = np.zeros_like(q["Q_components"][component], float)
        probe = rng.normal(size=total.shape)
        left = 0.0
        for material in MATERIALS:
            total += transpose_material_intersection_density_separable(
                target_density_sensitivity=target_sensitivity,
                source_edges_m=source_edges,
                target_edges_m=target_edges,
                target_material_support_mask=masks[material],
            )
            mapped_probe, _, _ = apply_material_intersection_density_separable(
                source_density=probe,
                source_edges_m=source_edges,
                target_edges_m=target_edges,
                target_material_support_mask=masks[material],
            )
            left += float(np.sum(target_sensitivity * mapped_probe))
        right = float(np.sum(total * probe))
        pulled[component] = total
        records[component] = {"transpose_dot_error": relative(left, right)}
    return pulled, records


def prepare_solver_aligned_source(
    fdtd,
    audit,
    *,
    base_project: Path,
    grid: dict[str, np.ndarray],
    native_source: np.ndarray,
    template: Path,
) -> tuple[float, float, dict]:
    """Import the weighted vector source on the solver FieldRegion grid.

    Lumerical's FieldRegion source requires the rectilinear coordinates of
    the recorded FieldRegion dataset.  Its vector components are then placed
    on their Yee locations by the solver.  Supplying three independently
    shifted source-object grids is rejected by v261 as not mesh-aligned; that
    failed diagnostic is preserved outside this retry.
    """
    fdtd.load(str(base_project))
    solver_mesh = {}
    positive_extension = {}
    for axis in "xyz":
        raw_mesh = fdtd.getresult("FDTD", axis)
        if isinstance(raw_mesh, dict):
            raw_mesh = raw_mesh[axis]
        mesh = np.asarray(raw_mesh, float).reshape(-1)
        if mesh.size < 3 or np.any(np.diff(mesh) <= 0.0):
            raise RuntimeError(f"invalid completed-forward {axis} mesh")
        base_axis = np.asarray(grid[axis], float)
        after = mesh[mesh > base_axis[-1] + 2.0e-18]
        if after.size == 0:
            raise RuntimeError(f"no solver mesh plane after FieldRegion {axis}")
        solver_mesh[axis] = mesh
        positive_extension[axis] = float(after[0])
    native_profile, scale = fieldregion_profile(native_source)
    profile, source_grid, collocation = invert_fieldregion_linear_collocation(
        grid,
        native_profile,
        positive_extension_coordinate_m=positive_extension,
    )
    fdtd.switchtolayout()
    if int(fdtd.getnamednumber(audit.SOURCE_NAME)) != 1:
        raise RuntimeError("expected exactly one forward Gaussian source")
    # Keep the forward source enabled as an immutable mesh anchor.  A
    # disabled source is omitted from v261's auto-nonuniform meshing, so its
    # amplitude is set to exact zero instead.  It then contributes no field
    # while retaining the identical source geometry in runsetup.
    original_forward_amplitude = float(
        fdtd.getnamed(audit.SOURCE_NAME, "amplitude")
    )
    fdtd.setnamed(audit.SOURCE_NAME, "amplitude", 0.0)
    if not bool(fdtd.getnamed(audit.SOURCE_NAME, "enabled")):
        raise RuntimeError("forward Gaussian mesh-anchor source is disabled")
    if float(fdtd.getnamed(audit.SOURCE_NAME, "amplitude")) != 0.0:
        raise RuntimeError("forward Gaussian mesh-anchor amplitude is nonzero")
    if int(fdtd.getnamednumber(FIELD_REGION)) != 1:
        raise RuntimeError("production FieldRegion monitor is missing")
    original_bounds = {
        axis: [
            float(fdtd.getnamed(FIELD_REGION, f"{axis} min")),
            float(fdtd.getnamed(FIELD_REGION, f"{axis} max")),
        ]
        for axis in "xyz"
    }
    for axis in "xyz":
        fdtd.setnamed(FIELD_REGION, f"{axis} min", float(source_grid[axis][0]))
        fdtd.setnamed(FIELD_REGION, f"{axis} max", float(source_grid[axis][-1]))
    fdtd.setnamed(FIELD_REGION, "source mode", True)
    try:
        fdtd.setnamed(FIELD_REGION, "nuttall window pulse", False)
    except Exception:
        pass
    roundtrip = import_named_fieldregion_profile(
        fdtd, FIELD_REGION, source_grid, profile
    )
    amplitude = float(fdtd.getnamed(FIELD_REGION, "base amplitude"))
    fdtd.save(str(template))
    return scale, amplitude, {
        "method": (
            "one common-grid FieldRegion vector source obtained by exact "
            "backward inversion of v261's component-wise linear placement "
            "onto the native Ex/Ey/Ez Yee coordinates"
        ),
        "component_collocation": collocation,
        "solver_mesh_positive_extension_m": positive_extension,
        "source_profile_roundtrip_max_abs_error": roundtrip,
        "coordinate_bounds_m": {
            axis: [float(source_grid[axis][0]), float(source_grid[axis][-1])]
            for axis in "xyz"
        },
        "solver_source_layout_adjustment": {
            "original_bounds_m": original_bounds,
            "source_bounds_m": {
                axis: [float(source_grid[axis][0]), float(source_grid[axis][-1])]
                for axis in "xyz"
            },
            "reason": (
                "one positive-axis common-grid cell is added so the exact "
                "component-wise common-to-native interpolation inverse keeps "
                "the last native target sample; the exterior common sample "
                "is zero"
            ),
            "nonzero_source_deleted": False,
            "forward_Gaussian_source_object_preserved_as_mesh_anchor": True,
            "forward_Gaussian_source_enabled_in_adjoint": True,
            "forward_Gaussian_source_original_amplitude": original_forward_amplitude,
            "forward_Gaussian_source_adjoint_amplitude": 0.0,
            "source_samples_cropped": 0,
        },
        "profile_scale": scale,
        "template": {
            "path": str(template),
            "size_bytes": template.stat().st_size,
            "sha256": sha256(template),
        },
        "empirical_normalization": False,
        "gradient_rescaling": False,
        "component_values_built_from_component_specific_E_epsilon_and_Q_pullback": True,
        "direct_shifted_component_source_objects_used": False,
    }


def prepare_common_grid_source(
    fdtd,
    audit,
    *,
    base_project: Path,
    grid: dict[str, np.ndarray],
    native_source: np.ndarray,
    template: Path,
) -> tuple[float, float, dict]:
    """Import the official common-grid FieldRegion profile.

    Component staggering is handled by FieldRegion itself.  The forward
    Gaussian stays enabled with zero amplitude because disabling it changes
    the auto-nonuniform mesh in v261.
    """
    profile, scale = fieldregion_profile(native_source)
    fdtd.load(str(base_project))
    fdtd.switchtolayout()
    original_amplitude = float(fdtd.getnamed(audit.SOURCE_NAME, "amplitude"))
    fdtd.setnamed(audit.SOURCE_NAME, "amplitude", 0.0)
    fdtd.setnamed(audit.SOURCE_NAME, "enabled", True)
    for axis in "xyz":
        fdtd.setnamed(FIELD_REGION, f"{axis} min", float(grid[axis][0]))
        fdtd.setnamed(FIELD_REGION, f"{axis} max", float(grid[axis][-1]))
    fdtd.setnamed(FIELD_REGION, "source mode", True)
    try:
        fdtd.setnamed(FIELD_REGION, "nuttall window pulse", False)
    except Exception:
        pass
    roundtrip = import_named_fieldregion_profile(
        fdtd, FIELD_REGION, grid, profile
    )
    base_amplitude = float(fdtd.getnamed(FIELD_REGION, "base amplitude"))
    fdtd.save(str(template))
    return scale, base_amplitude, {
        "method": (
            "official common-grid FieldRegion vector profile; component "
            "staggering is handled by the solver"
        ),
        "component_collocation": {
            "method": "official common-grid FieldRegion placement",
            "components": {},
        },
        "source_profile_roundtrip_max_abs_error": roundtrip,
        "coordinate_bounds_m": {
            axis: [float(grid[axis][0]), float(grid[axis][-1])]
            for axis in "xyz"
        },
        "profile_scale": scale,
        "fieldregion_base_amplitude": base_amplitude,
        "template": {
            "path": str(template),
            "size_bytes": template.stat().st_size,
            "sha256": sha256(template),
        },
        "forward_Gaussian_source_object_preserved_as_mesh_anchor": True,
        "forward_Gaussian_source_original_amplitude": original_amplitude,
        "forward_Gaussian_source_enabled_in_adjoint": True,
        "forward_Gaussian_source_adjoint_amplitude": 0.0,
        "empirical_normalization": False,
        "gradient_rescaling": False,
    }


def reconstruct_fieldregion_only_cw(
    electric_first: np.ndarray,
    electric_average: np.ndarray,
) -> tuple[np.ndarray, dict[str, float | bool | str]]:
    """Remove the mesh-anchor Gaussian spectrum from a two-source CW field."""
    first_over_average = np.vdot(electric_average, electric_first) / np.vdot(
        electric_average, electric_average
    )
    residual = float(
        np.linalg.norm(
            electric_first - first_over_average * electric_average
        )
        / max(float(np.linalg.norm(electric_first)), np.finfo(float).tiny)
    )
    fieldregion_over_first = 2.0 * first_over_average - 1.0
    if abs(fieldregion_over_first) <= np.finfo(float).tiny:
        raise RuntimeError("invalid FieldRegion-only CW source-spectrum ratio")
    electric = electric_first / fieldregion_over_first
    return electric, {
        "method": (
            "FieldRegion-only CW field reconstructed from official cwnorm(1) "
            "and cwnorm(2) states; the zero-amplitude forward Gaussian remains "
            "active only to preserve the forward mesh"
        ),
        "first_over_average_real": float(np.real(first_over_average)),
        "first_over_average_imag": float(np.imag(first_over_average)),
        "fieldregion_over_first_source_spectrum_real": float(
            np.real(fieldregion_over_first)
        ),
        "fieldregion_over_first_source_spectrum_imag": float(
            np.imag(fieldregion_over_first)
        ),
        "fieldregion_only_field_multiplier_real": float(
            np.real(1.0 / fieldregion_over_first)
        ),
        "fieldregion_only_field_multiplier_imag": float(
            np.imag(1.0 / fieldregion_over_first)
        ),
        "two_normalization_state_spatial_residual": residual,
        "uses_finite_difference_fit": False,
        "empirical_gradient_rescaling": False,
    }


def run_adjoint(
    fdtd,
    audit,
    runtime,
    *,
    template: Path,
    project: Path,
    completed_project: Path | None = None,
) -> dict:
    fdtd.load(
        str(
            completed_project.resolve()
            if completed_project is not None
            else template
        )
    )
    if int(fdtd.getnamednumber(audit.SOURCE_NAME)) != 1:
        raise RuntimeError("adjoint template lost forward mesh-anchor source")
    forward_source_enabled = bool(fdtd.getnamed(audit.SOURCE_NAME, "enabled"))
    if (
        forward_source_enabled
        and float(fdtd.getnamed(audit.SOURCE_NAME, "amplitude")) != 0.0
    ):
        raise RuntimeError("forward mesh-anchor source has nonzero amplitude")
    if completed_project is None:
        resources = runtime.configure_session_resources(fdtd)
        fdtd.save(str(project))
        started = time.monotonic()
        resource_used = audit.strict_gpu_run(fdtd, "run002_combined_adjoint")
        wall = time.monotonic() - started
        fdtd.save(str(project))
    else:
        project = completed_project.resolve()
        resources = {"reuse_completed_FSP": True}
        resource_used = "REUSED_COMPLETED_GPU_ADJOINT"
        wall = 0.0
    log_audit = audit.log_audit(project.parent)
    auto_shutoff = log_audit.get("final_auto_shutoff")
    if auto_shutoff is None or float(auto_shutoff) >= 1.0e-5:
        raise RuntimeError(f"adjoint auto-shutoff gate failed: {auto_shutoff}")
    fdtd.cwnorm(1)
    electric_first, grid = monitor_electric(fdtd, PABS_FIELD)
    fdtd.cwnorm(2)
    electric_average, average_grid = monitor_electric(fdtd, PABS_FIELD)
    grid_mismatch = max(
        float(
            np.max(
                np.abs(
                    np.asarray(grid[key]) - np.asarray(average_grid[key])
                )
            )
        )
        for key in ("x", "y", "z", "delta_x", "delta_y", "delta_z")
    )
    electric, normalization = reconstruct_fieldregion_only_cw(
        electric_first, electric_average
    )
    normalization_residual = float(
        normalization["two_normalization_state_spatial_residual"]
    )
    if grid_mismatch >= 2.0e-18 or normalization_residual >= 1.0e-12:
        raise RuntimeError(
            "FieldRegion-only CW reconstruction failed: "
            f"grid={grid_mismatch:.3e}, residual={normalization_residual:.3e}"
        )
    normalization["grid_mismatch_m"] = grid_mismatch
    return {
        "electric": electric,
        "grid": grid,
        "resources": resources,
        "resource_used": resource_used,
        "solver_mode": "GPU",
        "forward_source_enabled_in_adjoint": forward_source_enabled,
        "named_source_normalization": normalization,
        "log_audit": log_audit,
        "wall_s": wall,
        "project": {
            "path": str(project),
            "size_bytes": project.stat().st_size,
            "sha256": sha256(project),
        },
    }


def optical_gradient(
    operator: SparseYeeMaterialJacobian,
    *,
    forward: dict,
    adjoint: dict,
    pulled: dict[str, np.ndarray],
    profile_scale: float,
    base_amplitude: float,
) -> tuple[np.ndarray, dict]:
    mismatch = 0.0
    for key in ("x", "y", "z", "delta_x", "delta_y", "delta_z"):
        mismatch = max(
            mismatch,
            float(
                np.max(
                    np.abs(
                        np.asarray(forward["grid"][key])
                        - np.asarray(adjoint["grid"][key])
                    )
                )
            ),
        )
    volumes = component_volumes(forward["grid"])
    indirect = {}
    direct = {}
    for index, component in enumerate("xyz"):
        field = forward["electric"][..., 0, index]
        adjoint_field = adjoint["electric"][..., 0, index]
        if operator.component_shapes[component] != field.shape:
            raise RuntimeError(f"J_{component}/field shape mismatch")
        indirect[component] = (
            (2.0 * EPS0 / base_amplitude)
            * volumes[index]
            * field
            * (adjoint_field * profile_scale)
        )
        direct[component] = (
            -1j
            * 0.5
            * EPS0
            * (2.0 * np.pi * FREQUENCY_HZ)
            * pulled[component]
            * np.abs(field) ** 2
        )
    indirect_gradient = operator.vjp(indirect)
    direct_gradient = operator.vjp(direct)
    return indirect_gradient + direct_gradient, {
        "forward_adjoint_coordinate_mismatch_m": mismatch,
        "indirect_gradient_L2_A": float(np.linalg.norm(indirect_gradient)),
        "direct_gradient_L2_A": float(np.linalg.norm(direct_gradient)),
    }


def compact_forward(value: dict) -> dict:
    return {
        key: item
        for key, item in value.items()
        if key not in {"rho", "q", "electric", "epsilon", "grid"}
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-fsp", required=True, type=Path)
    parser.add_argument("--base-sha256", required=True)
    parser.add_argument("--jacobian-dir", required=True, type=Path)
    parser.add_argument(
        "--contract",
        choices=("coarse_production", "selected_production"),
        default="coarse_production",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--gpu-device", default="GPU 4")
    parser.add_argument("--cuda-device", type=int, default=4)
    parser.add_argument("--step", type=float, default=0.005)
    parser.add_argument(
        "--incident-power-W", type=float, default=1.3822261103022194e-13
    )
    parser.add_argument("--artifact-audit", action="store_true")
    parser.add_argument("--resume-base-fsp", type=Path)
    parser.add_argument("--resume-base-sha256")
    parser.add_argument("--prepare-adjoint-template-only", action="store_true")
    args = parser.parse_args()
    config = contract_configuration(args.contract)
    base_fsp = checked(args.base_fsp, args.base_sha256, "base production FSP")
    operator, rho, operator_meta = load_operator(
        args.jacobian_dir, str(config["operator_status"])
    )
    if tuple(rho.shape) != tuple(config["nodal_shape"]):
        raise RuntimeError(
            f"operator density shape {rho.shape} != contract {config['nodal_shape']}"
        )
    resume_base = None
    if args.resume_base_fsp is not None:
        if not args.resume_base_sha256:
            raise ValueError("--resume-base-sha256 is required with --resume-base-fsp")
        resume_base = checked(
            args.resume_base_fsp,
            args.resume_base_sha256,
            "completed nonuniform base FSP",
        )
    if args.artifact_audit:
        print(
            json.dumps(
                {
                    "status": "PASSED_PRODUCTION_COMBINED_ADFD_ARTIFACT_AUDIT",
                    "base_FSP": {"path": str(base_fsp), "sha256": sha256(base_fsp)},
                    "operator": operator_meta,
                    "contract": args.contract,
                    "rho_shape": list(rho.shape),
                    "rho_range": [float(np.min(rho)), float(np.max(rho))],
                    "Maxwell_solves": 0,
                    "thermal_solves": 0,
                    "optimization_iterations": 0,
                },
                indent=2,
            )
        )
        return 0
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "production_combined_adfd_smoke_result.json"
    result: dict[str, object] = {
        "status": "BLOCKED_PRODUCTION_COMBINED_ADFD_SMOKE",
        "passed": False,
        "optimization_iterations": 0,
        "empirical_normalization": False,
        "gradient_rescaling": False,
        "CPU_FDTD_fallback": False,
        "CPU_thermal_linear_solve_fallback": False,
    }
    fdtd = None
    started = time.monotonic()
    try:
        if args.step <= 0.0 or args.step >= min(np.min(rho), np.min(1.0 - rho)):
            raise ValueError("FD step leaves no-clipping density margin")
        fdtd, audit, runtime = open_fdtd(args.gpu_device)
        base = run_forward(
            fdtd,
            audit,
            runtime,
            base_fsp=base_fsp,
            rho=rho,
            role="base_nonuniform",
            output=output,
            imported_object=str(config["imported_object"]),
            nodes=config["nodes"],
            completed_project=resume_base,
        )
        base_data, base_mapping = map_q(
            base["q"], design_half_span_m=float(config["design_half_span_m"])
        )
        state, pair, objective, thermal_parts, target_sensitivity = (
            solve_base_thermal(
                base_data,
                rho,
                args.cuda_device,
                config["density_forward"],
                config["density_transpose"],
            )
        )
        pulled, pullback_records = pullback_q(
            base["q"], base_data, target_sensitivity
        )
        native_source = np.zeros_like(base["electric"], complex)
        for index, component in enumerate("xyz"):
            native_source[..., 0, index] = (
                0.5
                * EPS0
                * (2.0 * np.pi * FREQUENCY_HZ)
                * np.imag(base["epsilon"][..., 0, index])
                * pulled[component]
                * base["electric"][..., 0, index]
            )
        template = output / "production_combined_adjoint_template.fsp"
        profile_scale, base_amplitude, source_meta = prepare_common_grid_source(
            fdtd,
            audit,
            base_project=Path(base["project"]["path"]),
            grid=base["grid"],
            native_source=native_source,
            template=template,
        )
        if args.prepare_adjoint_template_only:
            result = {
                "status": "COMPLETED_PRODUCTION_COMBINED_ADJOINT_TEMPLATE_PREPARATION",
                "passed": True,
                "base": compact_forward(base),
                "base_mapping": base_mapping,
                "objective_A": objective,
                "thermal": {
                    "forward_residual": pair.forward.explicit_relative_residual,
                    "adjoint_residual": pair.adjoint.explicit_relative_residual,
                    "energy_balance_relative": boundary_energy(
                        state, pair.forward.solution
                    ),
                },
                "pullback": pullback_records,
                "source": source_meta,
                "profile_scale": profile_scale,
                "base_amplitude": base_amplitude,
                "Maxwell_timestepping_solves_this_invocation": (
                    0 if base["reused_completed"] else 1
                ),
                "thermal_forward_solves": 1,
                "thermal_adjoint_solves": 1,
                "optimization_iterations": 0,
                "empirical_normalization": False,
                "gradient_rescaling": False,
                "CPU_FDTD_fallback": False,
                "CPU_thermal_linear_solve_fallback": False,
            }
            result_path.write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(result, indent=2))
            return 0
        adjoint = run_adjoint(
            fdtd,
            audit,
            runtime,
            template=template,
            project=output / "production_combined_adjoint_gpu.fsp",
        )
        optical, optical_meta = optical_gradient(
            operator,
            forward=base,
            adjoint=adjoint,
            pulled=pulled,
            profile_scale=profile_scale,
            base_amplitude=base_amplitude,
        )
        thermal = thermal_parts["total"]
        total = optical + thermal
        direction_scale = float(np.max(np.abs(total)))
        if not np.isfinite(direction_scale) or direction_scale == 0.0:
            raise RuntimeError("combined gradient is zero or nonfinite")
        direction = total / direction_scale
        pair_objectives = {}
        pair_records = {}
        for sign, label in ((1.0, "plus"), (-1.0, "minus")):
            local_rho = rho + sign * args.step * direction
            if np.min(local_rho) <= 0.0 or np.max(local_rho) >= 1.0:
                raise RuntimeError("FD perturbation requires clipping")
            forward = run_forward(
                fdtd,
                audit,
                runtime,
                base_fsp=base_fsp,
                rho=local_rho,
                role=f"adjoint_aligned_h{args.step:g}_{label}",
                output=output,
                imported_object=str(config["imported_object"]),
                nodes=config["nodes"],
            )
            local_data, mapping = map_q(
                forward["q"],
                design_half_span_m=float(config["design_half_span_m"]),
            )
            local_state = build_state(
                local_data,
                SCENARIO,
                config["density_forward"](local_rho),
            )
            linear = PersistentCudaCSR(
                local_state.system.matrix_W_K, cuda_device=args.cuda_device
            ).solve(
                local_state.source_power_W,
                relative_tolerance=1.0e-10,
                max_iterations=30000,
            )
            local_objective = float(np.dot(local_state.c_A_K, linear.solution))
            pair_objectives[label] = local_objective
            pair_records[label] = {
                "objective_A": local_objective,
                "objective_A_per_incident_W": local_objective
                / args.incident_power_W,
                "forward": compact_forward(forward),
                "mapping": mapping,
                "thermal_residual": linear.explicit_relative_residual,
                "thermal_energy_balance": boundary_energy(
                    local_state, linear.solution
                ),
                "thermal_solve_seconds": linear.solve_seconds,
            }
        finite_difference = (
            pair_objectives["plus"] - pair_objectives["minus"]
        ) / (2.0 * args.step)
        adjoint_directional = float(np.sum(total * direction))
        adfd_error = relative(adjoint_directional, finite_difference)
        worst_closure = max(
            base["closure"],
            pair_records["plus"]["forward"]["closure"],
            pair_records["minus"]["forward"]["closure"],
        )
        worst_mapping = max(
            base_mapping["internal_relative_power_error"],
            pair_records["plus"]["mapping"]["internal_relative_power_error"],
            pair_records["minus"]["mapping"]["internal_relative_power_error"],
        )
        worst_residual = max(
            pair.forward.explicit_relative_residual,
            pair.adjoint.explicit_relative_residual,
            pair_records["plus"]["thermal_residual"],
            pair_records["minus"]["thermal_residual"],
        )
        worst_energy = max(
            boundary_energy(state, pair.forward.solution),
            pair_records["plus"]["thermal_energy_balance"],
            pair_records["minus"]["thermal_energy_balance"],
        )
        worst_pullback = max(
            row["transpose_dot_error"] for row in pullback_records.values()
        )
        worst_auto_shutoff = max(
            float(base["log_audit"]["final_auto_shutoff"]),
            float(adjoint["log_audit"]["final_auto_shutoff"]),
            float(pair_records["plus"]["forward"]["log_audit"]["final_auto_shutoff"]),
            float(pair_records["minus"]["forward"]["log_audit"]["final_auto_shutoff"]),
        )
        passed = bool(
            adfd_error < 0.01
            and operator_meta["fresh_transpose_dot_error"] < 1.0e-12
            and worst_pullback < 1.0e-12
            and worst_closure < 0.005
            and worst_mapping < 0.005
            and worst_residual < 1.0e-8
            and worst_energy < 0.01
            and worst_auto_shutoff < 1.0e-5
            and optical_meta["forward_adjoint_coordinate_mismatch_m"] < 2.0e-18
        )
        raw = output / "production_combined_adfd_smoke.npz"
        np.savez_compressed(
            raw,
            rho=rho,
            direction=direction,
            gradient_total_A=total,
            gradient_optical_A=optical,
            gradient_thermal_A=thermal,
            gradient_thermal_bulk_A=thermal_parts["bulk_k"],
            gradient_thermal_interface_A=thermal_parts["interface_G"],
            target_Q_density_sensitivity_A_m3_W=target_sensitivity,
            **{
                f"native_Q{component}_density_sensitivity_A_m3_W": value
                for component, value in pulled.items()
            },
        )
        result = {
            "status": (
                "VALIDATED_PRODUCTION_COMBINED_PHYSICAL_RHO_ADFD_SMOKE"
                if passed
                else "FAILED_PRODUCTION_COMBINED_PHYSICAL_RHO_ADFD_SMOKE"
            ),
            "passed": passed,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "scope": (
                f"{args.contract}; one nonuniform production physical-rho baseline; one "
                "adjoint-aligned centered-FD direction at h=0.005"
            ),
            "contract": args.contract,
            "base_FSP": {"path": str(base_fsp), "sha256": sha256(base_fsp)},
            "operator": operator_meta,
            "scenario": SCENARIO,
            "density_range": [float(np.min(rho)), float(np.max(rho))],
            "step": args.step,
            "incident_power_W": args.incident_power_W,
            "base_objective_A": objective,
            "base_objective_A_per_incident_W": objective / args.incident_power_W,
            "adjoint_directional_A": adjoint_directional,
            "finite_difference_directional_A": finite_difference,
            "adjoint_directional_A_per_incident_W": adjoint_directional
            / args.incident_power_W,
            "finite_difference_directional_A_per_incident_W": finite_difference
            / args.incident_power_W,
            "combined_AD_FD_relative_error": adfd_error,
            "gradient_norms_A": {
                "total": float(np.linalg.norm(total)),
                "optical": float(np.linalg.norm(optical)),
                "thermal_material": float(np.linalg.norm(thermal)),
            },
            "base_forward": compact_forward(base),
            "base_mapping": base_mapping,
            "base_thermal": {
                "forward_residual": pair.forward.explicit_relative_residual,
                "adjoint_residual": pair.adjoint.explicit_relative_residual,
                "energy_balance": boundary_energy(state, pair.forward.solution),
            },
            "pullback": pullback_records,
            "adjoint_source": source_meta,
            "adjoint": {
                key: value for key, value in adjoint.items() if key not in {"electric", "grid"}
            },
            "optical_gradient": optical_meta,
            "FD_pair": pair_records,
            "gates": {
                "combined_AD_FD_relative_error": adfd_error,
                "combined_AD_FD_limit": 0.01,
                "worst_component_J_transpose_error": operator_meta[
                    "fresh_transpose_dot_error"
                ],
                "worst_Q_pullback_transpose_error": worst_pullback,
                "worst_optical_closure": worst_closure,
                "worst_Q_mapping_error": worst_mapping,
                "worst_thermal_residual": worst_residual,
                "worst_thermal_energy_balance": worst_energy,
                "worst_forward_auto_shutoff": worst_auto_shutoff,
                "forward_adjoint_coordinate_mismatch_m": optical_meta[
                    "forward_adjoint_coordinate_mismatch_m"
                ],
            },
            "raw_artifact": {
                "path": str(raw),
                "size_bytes": raw.stat().st_size,
                "sha256": sha256(raw),
            },
            "Maxwell_solves": {"forward": 3, "adjoint": 1},
            "thermal_solves": {"forward": 3, "adjoint": 1},
            "optimization_iterations": 0,
            "empirical_normalization": False,
            "gradient_rescaling": False,
            "Q_clipping_smoothing_gain_or_rescaling": False,
            "CPU_FDTD_fallback": False,
            "CPU_thermal_linear_solve_fallback": False,
            "wall_s": time.monotonic() - started,
        }
    except Exception as exc:
        result.update(
            {
                "status": "FAILED_PRODUCTION_COMBINED_PHYSICAL_RHO_ADFD_SMOKE",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "wall_s": time.monotonic() - started,
            }
        )
    finally:
        if fdtd is not None:
            try:
                fdtd.close()
            except Exception:
                pass
        result_path.write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(json.dumps(result, indent=2, default=str))
    return 0 if result.get("passed", False) else 1


if __name__ == "__main__":
    raise SystemExit(main())
