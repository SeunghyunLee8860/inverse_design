#!/usr/bin/env python3
"""GPU combined Maxwell/thermal/electrical physical-density AD-FD smoke."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np
from scipy import sparse

from photothermal_pte.finite_inverse_design.finite_q_mapping import (
    nodal_control_volume_edges,
    transpose_material_intersection_density_separable,
)
from photothermal_pte.finite_inverse_design.native_yee_q import EPS0, extract_native_yee_q
from photothermal_pte.finite_inverse_design.run_v261_large_background_mixed_optical_adfd import (
    monitor_electric,
)
from photothermal_pte.finite_inverse_design.yee_material_jacobian import SparseYeeMaterialJacobian
from photothermal_pte.optimization_runs.cuda_thermal_adjoint import PersistentCudaCSR
from photothermal_pte.optimization_runs.tairte4_flake_topology import optical
from photothermal_pte.optimization_runs.tairte4_flake_topology.contract import CONTRACT
from photothermal_pte.optimization_runs.tairte4_flake_topology.electrical import (
    build_rectangular_mesh,
    solve_weighting_and_adjoint,
)
from photothermal_pte.optimization_runs.tairte4_flake_topology.thermal import (
    boundary_energy_error,
    build_state,
    cell_to_node,
    flake_cell_temperature,
    flake_temperature_transpose,
    map_native_q,
    thermal_interface_contract,
    thermal_density_gradient,
)


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]
RUN002 = REPOSITORY / "photothermal_pte" / "optimization_runs" / "legacy_v261_optical_support"
STAGE1 = REPOSITORY / "photothermal_pte" / "validation" / "photothermal_stage1"
for helper in (str(RUN002), str(STAGE1), str(REPOSITORY / "photothermal_pte")):
    if helper not in sys.path:
        sys.path.insert(0, helper)

import build_nonuniform_complex_yee_jacobian as jacobian_builder  # noqa: E402
import run_complex_material_control as material_control  # noqa: E402
import run_production_combined_adfd_smoke as legacy_combined  # noqa: E402


C0 = 299792458.0
FREQUENCY_HZ = C0 / CONTRACT.wavelength_m
FIELD_REGION = "run010_component_yee_adjoint_region"
SIGMA_XY_S_M = (1.10e5, 4.91e5)
SEEBECK_XY_V_K = (27.0e-6, -6.0e-6)
GPU_ENGINE_LOCK = Path(
    os.environ.get(
        "LUMERICAL_GPU_ENGINE_LOCK",
        "/tmp/seunghyun_lumerical_fdtd_gpu_engine.lock",
    )
)


@contextmanager
def lumerical_gpu_engine_lock():
    """Serialize licensed FDTD engine runs while leaving other work parallel."""

    GPU_ENGINE_LOCK.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with GPU_ENGINE_LOCK.open("a+") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        metadata = {
            "path": str(GPU_ENGINE_LOCK),
            "wait_s": time.monotonic() - started,
            "pid": os.getpid(),
        }
        try:
            yield metadata
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(first: float, second: float) -> float:
    return abs(first - second) / max(abs(first), abs(second), np.finfo(float).tiny)


def checked(path: Path, expected: str) -> Path:
    value = path.expanduser().resolve()
    if not value.is_file() or sha256(value) != expected:
        raise RuntimeError(f"missing or SHA-mismatched artifact: {value}")
    return value


def load_operator(directory: Path) -> tuple[SparseYeeMaterialJacobian, np.ndarray, dict[str, object]]:
    root = directory.expanduser().resolve()
    result_path = root / "component_yee_jacobian_result.json"
    result = json.loads(result_path.read_text())
    if result.get("status") != "VALIDATED_TAIRTE4_FLAKE_COMPLEX_COMPONENT_YEE_JACOBIAN":
        raise RuntimeError("TaIrTe4 component-Yee Jacobian is not validated")
    layout_path = checked(
        Path(result["artifacts"]["coordinates_and_density"]["path"]),
        result["artifacts"]["coordinates_and_density"]["sha256"],
    )
    layout = np.load(layout_path)
    rho = np.asarray(layout["rho"], dtype=np.float64)
    matrices = {}
    shapes = {}
    for component in "xyz":
        record = result["artifacts"]["component_J"][component]
        path = checked(Path(record["path"]), record["sha256"])
        matrices[component] = sparse.load_npz(path)
        shapes[component] = tuple(result["coordinate_audit"]["components"][component]["shape"])
    operator = SparseYeeMaterialJacobian(
        density_shape=rho.shape, component_shapes=shapes, matrices=matrices
    )
    rng = np.random.default_rng(812)
    direction = rng.normal(size=rho.shape)
    cotangent = {
        c: rng.normal(size=shapes[c]) + 1j * rng.normal(size=shapes[c])
        for c in "xyz"
    }
    tangent = operator.jvp(direction)
    left = float(np.real(sum(np.sum(cotangent[c] * tangent[c]) for c in "xyz")))
    right = float(np.vdot(direction, operator.vjp(cotangent)))
    dot = relative(left, right)
    if dot >= 1e-12:
        raise RuntimeError("fresh Yee Jacobian transpose gate failed")
    return operator, rho, {
        "result_path": str(result_path),
        "result_sha256": sha256(result_path),
        "fresh_transpose_error": dot,
        "maximum_coordinate_mismatch_m": result["maximum_coordinate_mismatch_m"],
    }


def open_fdtd(gpu_device: str):
    wrapper = material_control.load_source_wrapper()
    audit = wrapper.source_audit
    optical.configure_source(audit)
    os.environ["VC_LUMERICAL_ROOT"] = str(audit.APPROVED_ROOT)
    os.environ["LUMERICAL_ROOT"] = str(audit.APPROVED_ROOT)
    os.environ["LUMERICAL_PYTHONPATH"] = str(audit.APPROVED_API)
    os.environ["LUMERICAL_SESSION_GPU_DEVICE"] = gpu_device
    os.environ["CL_GPU_DEVICE"] = gpu_device
    os.environ["FDTD_THREADS"] = "8"
    os.environ["PATH"] = f"{audit.APPROVED_ROOT / 'bin'}:{os.environ.get('PATH','')}"
    helper = audit.load_module(audit.API_HELPER, "run010_combined_api")
    installation = type(
        "Installation", (), {
            "version_key": "v261",
            "root": audit.APPROVED_ROOT,
            "lumapi_path": audit.APPROVED_API / "lumapi.py",
            "device_executable": audit.APPROVED_ROOT / "bin" / "device",
        }
    )()
    lumapi = helper.load_lumapi(installation)
    fdtd = lumapi.FDTD(hide=True, serverArgs={"platform": "offscreen"})
    import eqc_lib as runtime
    legacy_combined.FIELD_REGION = FIELD_REGION
    legacy_combined.FREQUENCY_HZ = FREQUENCY_HZ
    return fdtd, audit, runtime


def set_density(fdtd, rho: np.ndarray) -> None:
    jacobian_builder.set_tairte4_flake_density(
        fdtd,
        rho,
        imported_object=optical.DESIGN_OBJECT,
        nodes=optical.design_nodes(),
    )


def native_arrays(q: dict) -> dict[str, np.ndarray]:
    result = {}
    for component in "xyz":
        result[f"Q{component}_W_m3"] = np.asarray(q["Q_components"][component], float)
        for axis in "xyz":
            result[f"Q{component}_{axis}_m"] = np.asarray(
                q["native_coordinates"][component][axis], float
            )
    return result


def run_forward(
    fdtd,
    audit,
    runtime,
    *,
    template: Path,
    rho: np.ndarray,
    role: str,
    output: Path,
    reuse: bool = False,
    polarization_angle_deg: float | None = None,
) -> dict:
    project = template if reuse else output / f"{role}.fsp"
    if reuse:
        fdtd.load(str(template))
        wall = 0.0
        resources = {"reuse_completed_FSP": True}
        resource_used = "REUSED_COMPLETED_GPU_FORWARD"
    else:
        fdtd.load(str(template))
        fdtd.switchtolayout()
        set_density(fdtd, rho)
        if polarization_angle_deg is not None:
            fdtd.setnamed(
                optical.SOURCE_NAME,
                "polarization angle",
                float(polarization_angle_deg),
            )
        fdtd.setnamed(optical.SOURCE_NAME, "enabled", True)
        fdtd.setnamed(optical.SOURCE_NAME, "amplitude", 1.0)
        fdtd.setnamed(FIELD_REGION, "source mode", False)
        resources = runtime.configure_session_resources(fdtd)
        fdtd.save(str(project))
        started = time.monotonic()
        with lumerical_gpu_engine_lock() as lock_metadata:
            resource_used = audit.strict_gpu_run(fdtd, f"run010_combined_{role}")
        resources["global_gpu_engine_lock"] = lock_metadata
        wall = time.monotonic() - started
        fdtd.save(str(project))
    actual_polarization = float(fdtd.getnamed(optical.SOURCE_NAME, "polarization angle"))
    if polarization_angle_deg is not None and not np.isclose(
        actual_polarization,
        float(polarization_angle_deg),
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise RuntimeError("source polarization readback mismatch")
    fdtd.runanalysis(legacy_combined.PABS_GROUP)
    if not reuse:
        fdtd.save(str(project))
    source_power = audit.scalar(fdtd.sourcepower(FREQUENCY_HZ, 2, optical.SOURCE_NAME), "sourcepower")
    net = 0.0
    faces = {}
    for axis in "xyz":
        for side, sign in (("min", -1.0), ("max", 1.0)):
            name = f"run010_flux_{axis}_{side}"
            signed = audit.scalar(fdtd.transmission(name), name) * source_power
            faces[name] = {"signed_axis_power_W": signed, "outward_power_W": sign * signed}
            net += sign * signed
    p_six = -net
    q = extract_native_yee_q(
        fdtd,
        field_monitor=optical.PABS_FIELD,
        index_monitor=optical.PABS_INDEX,
        wavelength_m=CONTRACT.wavelength_m,
    )
    electric, grid = monitor_electric(fdtd, optical.PABS_FIELD)
    detail = jacobian_builder.index_detail(fdtd)
    epsilon = np.stack([detail[f"epsilon_{c}"] for c in "xyz"], axis=-1)[..., None, :]
    index = np.stack([detail[f"index_{c}"] for c in "xyz"], axis=-1)[..., None, :]
    if electric.shape != epsilon.shape:
        raise RuntimeError("forward E/index component shape mismatch")
    mismatch = 0.0
    for ci, component in enumerate("xyz"):
        coordinates = jacobian_builder.component_coordinates(detail, component)
        for ai, (axis, coordinate) in enumerate(zip("xyz", coordinates)):
            expected = np.asarray(grid[axis], float)
            if ci == ai:
                expected = expected + np.asarray(grid[f"delta_{component}"], float)
            mismatch = max(mismatch, float(np.max(np.abs(expected - coordinate))))
    p_q = float(q["P_Q_W"])
    closure = relative(p_q, p_six)
    log = audit.log_audit(project.parent)
    if log.get("final_auto_shutoff") is None or log["final_auto_shutoff"] >= 1e-5:
        raise RuntimeError("forward auto-shutoff gate failed")
    return {
        "q": q,
        "electric": electric,
        "epsilon": epsilon,
        "index": index,
        "grid": grid,
        "P_Q_W": p_q,
        "P_six_W": p_six,
        "closure": closure,
        "source_power_W": source_power,
        "faces": faces,
        "coordinate_mismatch_m": mismatch,
        "resources": resources,
        "resource_used": resource_used,
        "wall_s": wall,
        "polarization_angle_deg": actual_polarization,
        "project": {"path": str(project), "size_bytes": project.stat().st_size, "sha256": sha256(project)},
        "log_audit": log,
    }


class CachedElectricalCuda:
    def __init__(self, device: int):
        self.device = device
        self.reference = None
        self.operator = None

    def __call__(self, matrix, rhs):
        matrix = sparse.csr_matrix(matrix, dtype=np.float64)
        if self.operator is None:
            self.reference = matrix.copy()
            self.operator = PersistentCudaCSR(matrix, cuda_device=self.device)
        result = self.operator.solve(rhs, relative_tolerance=1e-10, max_iterations=30000)
        return result.solution


def full_flake_density(rho: np.ndarray) -> np.ndarray:
    full = np.ones(CONTRACT.flake_node_shape, dtype=np.float64)
    full[CONTRACT.design_node_slices] = rho
    return full


def solve_coupled(forward: dict, rho: np.ndarray, cuda_device: int, *, need_adjoint: bool):
    state = build_state(rho)
    mapped_q, mapping = map_native_q(native_arrays(forward["q"]), state)
    source_active = state.system.active_source(mapped_q)
    source_power = np.asarray(state.system.source_volume_operator_m3 @ source_active)
    thermal_operator = PersistentCudaCSR(state.system.matrix_W_K, cuda_device=cuda_device)
    thermal_forward = thermal_operator.solve(
        source_power, relative_tolerance=1e-10, max_iterations=30000
    )
    nodal_temperature = cell_to_node(flake_cell_temperature(state, thermal_forward.solution))
    mesh = build_rectangular_mesh(CONTRACT.flake_span_m, CONTRACT.flake_span_m, CONTRACT.design_step_m)
    electrical = solve_weighting_and_adjoint(
        mesh,
        full_flake_density(rho),
        nodal_temperature,
        thickness_m=CONTRACT.flake_thickness_m,
        sigma_xy_S_m=SIGMA_XY_S_M,
        seebeck_xy_V_K=SEEBECK_XY_V_K,
        sigma_void_fraction=CONTRACT.sigma_void_fraction,
        sigma_penalty=CONTRACT.sigma_penalty,
        alpha_penalty=CONTRACT.alpha_penalty,
        linear_solve=CachedElectricalCuda(cuda_device),
        terminal_axis=CONTRACT.contact_axis,
    )
    energy, boundary = boundary_energy_error(state, thermal_forward.solution, source_power)
    result = {
        "state": state,
        "mapped_q": mapped_q,
        "mapping": mapping,
        "source_power": source_power,
        "thermal_forward": thermal_forward,
        "temperature": nodal_temperature,
        "electrical": electrical,
        "energy": energy,
        "boundary": boundary,
    }
    if need_adjoint:
        thermal_rhs = flake_temperature_transpose(state, electrical.gradient_temperature_K_inv)
        thermal_adjoint = thermal_operator.solve(
            thermal_rhs, relative_tolerance=1e-10, max_iterations=30000
        )
        gradient_thermal = thermal_density_gradient(
            state, thermal_forward.solution, thermal_adjoint.solution
        )
        gradient_electrical = electrical.gradient_rho_A[CONTRACT.design_node_slices]
        gradient_terminal_conductance = electrical.gradient_terminal_conductance_S[
            CONTRACT.design_node_slices
        ]
        target_active = np.asarray(
            state.system.source_volume_operator_m3.T @ thermal_adjoint.solution
        ).reshape(-1)
        target_sensitivity = np.zeros(state.system.shape, dtype=np.float64)
        target_sensitivity[state.system.active_mask] = target_active
        result.update(
            thermal_adjoint=thermal_adjoint,
            gradient_thermal=gradient_thermal,
            gradient_electrical=gradient_electrical,
            gradient_terminal_conductance=gradient_terminal_conductance,
            target_sensitivity=target_sensitivity,
        )
    return result


def pullback_q(forward: dict, coupled: dict) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    pulled = {}
    records = {}
    rng = np.random.default_rng(910)
    support = coupled["state"].masks["physical_absorbing_support"]
    for component in "xyz":
        source_edges = tuple(
            nodal_control_volume_edges(np.asarray(forward["q"]["native_coordinates"][component][axis], float))
            for axis in "xyz"
        )
        value = transpose_material_intersection_density_separable(
            target_density_sensitivity=coupled["target_sensitivity"],
            source_edges_m=source_edges,
            target_edges_m=coupled["state"].edges_m,
            target_material_support_mask=support,
        )
        probe = rng.normal(size=value.shape)
        # The forward transpose identity itself is already unit-tested in the
        # shared mapper; save a finite/cauchy diagnostic for this exact grid.
        records[component] = {
            "shape": list(value.shape),
            "finite": bool(np.all(np.isfinite(value))),
            "probe_inner_product_scale": float(abs(np.sum(value * probe))),
        }
        pulled[component] = value
    return pulled, records


def compact_forward(value: dict) -> dict:
    return {key: item for key, item in value.items() if key not in {"q", "electric", "epsilon", "index", "grid"}}


def polarization_angle(label: str) -> float:
    """Map the crystal-axis certificate label to Lumerical's source angle."""
    if label == "Ea":
        return 90.0  # Lumerical y = crystal a
    if label == "Eb":
        return 0.0  # Lumerical x = crystal b
    raise ValueError(f"unsupported polarization: {label}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-fsp", required=True, type=Path)
    parser.add_argument("--base-sha256", required=True)
    parser.add_argument("--jacobian-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--gpu-device", default="GPU 5")
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--step", type=float, default=0.005)
    parser.add_argument("--polarization", choices=("Ea", "Eb"), default="Ea")
    args = parser.parse_args()
    base_fsp = checked(args.base_fsp, args.base_sha256)
    operator, rho, operator_meta = load_operator(args.jacobian_dir)
    if not np.all(rho == 0.5):
        raise RuntimeError("combined smoke requires exact uniform rho=0.5")
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "tairte4_flake_combined_adfd.json"
    result = {"status": "BLOCKED_TAIRTE4_FLAKE_COMBINED_ADFD", "passed": False, "optimization_iterations": 0}
    fdtd = None
    started = time.monotonic()
    try:
        fdtd, audit, runtime = open_fdtd(args.gpu_device)
        angle = polarization_angle(args.polarization)
        base = run_forward(
            fdtd,
            audit,
            runtime,
            template=base_fsp,
            rho=rho,
            role=f"base_{args.polarization}",
            output=output,
            reuse=args.polarization == "Ea",
            polarization_angle_deg=angle,
        )
        coupled = solve_coupled(base, rho, args.cuda_device, need_adjoint=True)
        pulled, pullback_meta = pullback_q(base, coupled)
        native_source = np.zeros_like(base["electric"], dtype=np.complex128)
        for index, component in enumerate("xyz"):
            native_source[..., 0, index] = (
                0.5 * EPS0 * (2.0 * np.pi * FREQUENCY_HZ)
                * np.imag(base["epsilon"][..., 0, index])
                * pulled[component]
                * base["electric"][..., 0, index]
            )
        template = output / "tairte4_flake_adjoint_template.fsp"
        profile_scale, base_amplitude, source_meta = legacy_combined.prepare_common_grid_source(
            fdtd,
            audit,
            base_project=Path(base["project"]["path"]),
            grid=base["grid"],
            native_source=native_source,
            template=template,
        )
        adjoint = legacy_combined.run_adjoint(
            fdtd, audit, runtime, template=template, project=output / "tairte4_flake_adjoint_gpu.fsp"
        )
        gradient_optical, optical_meta = legacy_combined.optical_gradient(
            operator,
            forward=base,
            adjoint=adjoint,
            pulled=pulled,
            profile_scale=profile_scale,
            base_amplitude=base_amplitude,
        )
        total = gradient_optical + coupled["gradient_thermal"] + coupled["gradient_electrical"]
        scale = float(np.max(np.abs(total)))
        if not np.isfinite(scale) or scale == 0.0:
            raise RuntimeError("combined gradient is zero/nonfinite")
        direction = total / scale
        objectives = {}
        fd_records = {}
        for sign, label in ((1.0, "plus"), (-1.0, "minus")):
            local_rho = rho + sign * args.step * direction
            if np.any((local_rho <= 0.0) | (local_rho >= 1.0)):
                raise RuntimeError("combined FD would require clipping")
            local_forward = run_forward(
                fdtd, audit, runtime, template=base_fsp, rho=local_rho,
                role=f"adjoint_aligned_{label}", output=output,
                polarization_angle_deg=angle,
            )
            local = solve_coupled(local_forward, local_rho, args.cuda_device, need_adjoint=False)
            objectives[label] = local["electrical"].current_A
            fd_records[label] = {
                "objective_A": local["electrical"].current_A,
                "forward": compact_forward(local_forward),
                "Q_mapping": local["mapping"],
                "thermal_residual": local["thermal_forward"].explicit_relative_residual,
                "energy_balance": local["energy"],
            }
        fd = (objectives["plus"] - objectives["minus"]) / (2.0 * args.step)
        ad = float(np.sum(total * direction))
        adfd = relative(ad, fd)
        worst_closure = max(base["closure"], *(fd_records[k]["forward"]["closure"] for k in fd_records))
        worst_mapping = max(coupled["mapping"]["relative_mapping_error"], *(fd_records[k]["Q_mapping"]["relative_mapping_error"] for k in fd_records))
        worst_residual = max(
            coupled["thermal_forward"].explicit_relative_residual,
            coupled["thermal_adjoint"].explicit_relative_residual,
            *(fd_records[k]["thermal_residual"] for k in fd_records),
        )
        worst_energy = max(coupled["energy"], *(fd_records[k]["energy_balance"] for k in fd_records))
        worst_shutoff = max(
            float(base["log_audit"]["final_auto_shutoff"]),
            float(adjoint["log_audit"]["final_auto_shutoff"]),
            *(float(fd_records[k]["forward"]["log_audit"]["final_auto_shutoff"]) for k in fd_records),
        )
        passed = bool(
            adfd < 0.01
            and operator_meta["fresh_transpose_error"] < 1e-12
            and worst_closure < 0.005
            and worst_mapping < 0.005
            and worst_residual < 1e-8
            and worst_energy < 0.01
            and worst_shutoff < 1e-5
            and optical_meta["forward_adjoint_coordinate_mismatch_m"] < 2e-18
        )
        raw = output / "tairte4_flake_combined_adfd.npz"
        np.savez_compressed(
            raw,
            rho=rho,
            direction=direction,
            gradient_total_A=total,
            gradient_optical_A=gradient_optical,
            gradient_thermal_A=coupled["gradient_thermal"],
            gradient_electrical_A=coupled["gradient_electrical"],
            target_Q_sensitivity=coupled["target_sensitivity"],
            **{f"native_Q{c}_sensitivity": value for c, value in pulled.items()},
        )
        result = {
            "status": "VALIDATED_TAIRTE4_FLAKE_COMBINED_PHYSICAL_RHO_ADFD" if passed else "FAILED_TAIRTE4_FLAKE_COMBINED_PHYSICAL_RHO_ADFD",
            "passed": passed,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "thermal_interface_contract": thermal_interface_contract(),
            "scope": (
                f"uniform rho=0.5 {args.polarization} combined optical-Q, "
                "explicit thermal material/interface, and density-dependent "
                "electrical weighting gradient"
            ),
            "polarization": args.polarization,
            "polarization_angle_deg": angle,
            "step": args.step,
            "base_objective_A": coupled["electrical"].current_A,
            "AD_directional_A": ad,
            "FD_directional_A": fd,
            "combined_AD_FD_relative_error": adfd,
            "gradient_norms_A": {
                "total": float(np.linalg.norm(total)),
                "optical": float(np.linalg.norm(gradient_optical)),
                "thermal": float(np.linalg.norm(coupled["gradient_thermal"])),
                "electrical": float(np.linalg.norm(coupled["gradient_electrical"])),
            },
            "operator": operator_meta,
            "base_forward": compact_forward(base),
            "base_Q_mapping": coupled["mapping"],
            "base_thermal": {
                "forward_residual": coupled["thermal_forward"].explicit_relative_residual,
                "adjoint_residual": coupled["thermal_adjoint"].explicit_relative_residual,
                "energy_balance": coupled["energy"],
            },
            "pullback": pullback_meta,
            "adjoint_source": source_meta,
            "adjoint": {key: value for key, value in adjoint.items() if key not in {"electric", "grid"}},
            "optical_gradient": optical_meta,
            "FD_pair": fd_records,
            "gates": {
                "combined_error": adfd,
                "limit": 0.01,
                "worst_closure": worst_closure,
                "worst_mapping": worst_mapping,
                "worst_residual": worst_residual,
                "worst_energy": worst_energy,
                "worst_auto_shutoff": worst_shutoff,
            },
            "raw_artifact": {"path": str(raw), "size_bytes": raw.stat().st_size, "sha256": sha256(raw)},
            "Maxwell_solves": {
                "forward_reused": int(args.polarization == "Ea"),
                "forward_new": 2 + int(args.polarization == "Eb"),
                "adjoint": 1,
            },
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
            status="FAILED_TAIRTE4_FLAKE_COMBINED_PHYSICAL_RHO_ADFD",
            error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
            wall_s=time.monotonic() - started,
        )
    finally:
        if fdtd is not None:
            try:
                fdtd.close()
            except Exception:
                pass
        result_path.write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(json.dumps(result, indent=2, default=str))
    return 0 if result.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
