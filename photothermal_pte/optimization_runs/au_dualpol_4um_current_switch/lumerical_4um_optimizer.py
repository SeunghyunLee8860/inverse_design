"""Fail-closed orchestration for the 81x81 Lumerical Au optimizer carrier.

This module deliberately contains no alternative Maxwell solver.  One
physical evaluation is assembled from the already validated executable
pieces in this directory:

* one imported-density Lumerical forward for Ea and Eb,
* one layout-only component-Yee material Jacobian,
* the custom CUDA thermal/electrical forward and adjoint for each polarization,
* one frozen-grid distributed-source Lumerical adjoint for each polarization.

The optimizer coordinate is the canonical 81x81 latent nodal density.  The
single projected nodal state produced by ``OPTIMIZER_250NM_MAPPING`` is passed to
every optical/PDE derivative path.  Lumerical HEAT and CHARGE are never used.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Callable

import numpy as np

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.contract import (
    CONTRACT,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.lumerical_4um_density import (
    density_state_audit,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.lumerical_4um_design_mapping import (
    OPTIMIZER_250NM_MAPPING,
    calibrated_lumerical_250nm_dfm_caps,
    smooth_lumerical_250nm_constraints,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.lumerical_4um_signed_objective import (
    signed_dual_objective_point,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.lumerical_only_boundary import (
    require_lumerical_only_source_boundary,
)


SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY = Path(__file__).resolve().parents[3]
MESH_LABEL = "fine_z2p5_bulk50_xy100_cv0_pml8_span20_z6_t1ps"
FINAL_XY50_MESH_LABEL = "fine_z2p5_bulk50_xy50_cv0_pml8_span20_z6_t1ps"
SOURCE_OBJECT_W0_UM = 3.9561433030461415
SMOKE_MAXEVAL = 2
CURRENT_SCALE_A = 1.0e-9
DFM_CONSTRAINT_SCALE = 0.01


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    value = path.expanduser().resolve()
    return {
        "path": str(value),
        "size_bytes": value.stat().st_size,
        "sha256": sha256(value),
    }


def initial_latent_density() -> np.ndarray:
    """Return the field-independent beta-4 state used by passed AD--FD gates."""

    x = np.linspace(-1.0, 1.0, CONTRACT.design_node_shape[0])[:, None]
    y = np.linspace(-1.0, 1.0, CONTRACT.design_node_shape[1])[None, :]
    return np.ascontiguousarray(
        0.5 + 0.16 * np.sin(0.8 * np.pi * x) * np.cos(0.6 * np.pi * y)
    )


def uniform_initial_latent_density() -> np.ndarray:
    """Return the exact uniform rho=0.5 production-continuation start."""

    return np.full(CONTRACT.design_node_shape, 0.5, dtype=np.float64)


def _required_path(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"required environment variable is unset: {name}")
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{name}: {path}")
    return path


def _required_output_root() -> Path:
    value = os.environ.get("AU_LUMERICAL_OPT_OUTPUT_ROOT")
    if not value:
        raise RuntimeError("AU_LUMERICAL_OPT_OUTPUT_ROOT is required")
    output = Path(value).expanduser().resolve()
    try:
        output.relative_to(REPOSITORY.resolve())
    except ValueError:
        pass
    else:
        raise RuntimeError("optimizer raw output must be outside the Git worktree")
    return output


def _physical_gpu_index() -> int:
    value = os.environ.get("CUDA_VISIBLE_DEVICES")
    if value is None or not value.strip().isdigit():
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES must contain exactly one physical GPU index"
        )
    return int(value)


def _threads() -> int:
    value = int(os.environ.get("FDTD_THREADS", "8"))
    if value < 1:
        raise RuntimeError("FDTD_THREADS must be positive")
    return value


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _matching_raw_artifact(
    result: dict[str, Any], suffix: str
) -> tuple[Path, dict[str, Any]]:
    records = [
        item
        for item in result.get("raw_artifacts", [])
        if str(item.get("path", "")).endswith(suffix)
    ]
    if len(records) != 1:
        raise RuntimeError(f"expected exactly one forward {suffix} artifact")
    record = records[0]
    path = Path(record["path"]).resolve()
    if not path.is_file() or sha256(path) != record["sha256"]:
        raise RuntimeError(f"forward {suffix} artifact path/SHA failed")
    return path, record


def _tail(path: Path, lines: int = 80) -> str:
    try:
        values = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(values[-lines:])


@dataclass(frozen=True)
class OptimizerRuntime:
    output_root: Path
    source_calibration: dict[str, Path]
    gpu_index: int
    threads: int
    accelerator_policy: str = "development"
    beta: float = 4.0
    final_xy50_source_calibration: dict[str, Path] | None = None

    @classmethod
    def from_environment(
        cls,
        *,
        require_smoke_beta: bool = True,
        require_final_xy50_source_calibration: bool = False,
    ) -> "OptimizerRuntime":
        policy = os.environ.get(
            "AU_LUMERICAL_ACCELERATOR_POLICY", "development"
        )
        if policy not in ("development", "b200"):
            raise RuntimeError(f"unsupported accelerator policy: {policy}")
        beta = float(os.environ.get("AU_LUMERICAL_OPT_BETA", "4"))
        if require_smoke_beta and beta != 4.0:
            raise RuntimeError(
                "the first smoke run is restricted to the beta-4 AD-FD state"
            )
        fine_source = None
        if require_final_xy50_source_calibration:
            fine_source = {
                "Ea": _required_path(
                    "AU_LUMERICAL_EA_FINAL_XY50_SOURCE_CALIBRATION"
                ),
                "Eb": _required_path(
                    "AU_LUMERICAL_EB_FINAL_XY50_SOURCE_CALIBRATION"
                ),
            }
        return cls(
            output_root=_required_output_root(),
            source_calibration={
                "Ea": _required_path("AU_LUMERICAL_EA_SOURCE_CALIBRATION"),
                "Eb": _required_path("AU_LUMERICAL_EB_SOURCE_CALIBRATION"),
            },
            gpu_index=_physical_gpu_index(),
            threads=_threads(),
            accelerator_policy=policy,
            beta=beta,
            final_xy50_source_calibration=fine_source,
        )

    def audit(self) -> dict[str, Any]:
        solver_boundary = require_lumerical_only_source_boundary()
        groups: list[
            tuple[str, dict[str, Path], str, float]
        ] = [
            ("optimizer_xy100", self.source_calibration, MESH_LABEL, 100.0e-9)
        ]
        if self.final_xy50_source_calibration is not None:
            groups.append(
                (
                    "final_xy50",
                    self.final_xy50_source_calibration,
                    FINAL_XY50_MESH_LABEL,
                    50.0e-9,
                )
            )
        audited: dict[str, dict[str, Any]] = {}
        expected_uuid: str | None = None
        expected_solver: str | None = None
        for group_name, paths, mesh_label, flake_dxy_m in groups:
            if set(paths) != {"Ea", "Eb"}:
                raise RuntimeError(
                    f"{group_name} source calibrations must contain exactly Ea/Eb"
                )
            audited[group_name] = {}
            for polarization, path in paths.items():
                record = _load_json(path)
                mesh = record.get("mesh_spec", {})
                gpu_uuid = record.get("GPU_log_evidence", {}).get(
                    "requested_gpu_uuid"
                )
                gates = {
                    "source_only_case": record.get("case") == "source_only",
                    "polarization_matches": record.get("polarization")
                    == polarization,
                    "source_status_passed": str(record.get("status", "")).startswith(
                        "PASSED_EXACT_AU_4UM_SOURCE_ONLY"
                    ),
                    "all_source_gates_passed": record.get("all_gates_passed")
                    is True,
                    "mesh_label_matches": mesh.get("label") == mesh_label,
                    "flake_dxy_matches": bool(
                        np.isclose(
                            float(mesh.get("flake_dxy_m", np.nan)),
                            flake_dxy_m,
                            rtol=0.0,
                            atol=1.0e-18,
                        )
                    ),
                    "outer_dxy_is_200nm": bool(
                        np.isclose(
                            float(mesh.get("outer_dxy_m", np.nan)),
                            200.0e-9,
                            rtol=0.0,
                            atol=1.0e-18,
                        )
                    ),
                    "stack_dz_is_2p5nm": bool(
                        np.isclose(
                            float(mesh.get("stack_dz_m", np.nan)),
                            2.5e-9,
                            rtol=0.0,
                            atol=1.0e-18,
                        )
                    ),
                    "bulk_dz_is_50nm": bool(
                        np.isclose(
                            float(mesh.get("bulk_dz_m", np.nan)),
                            50.0e-9,
                            rtol=0.0,
                            atol=1.0e-18,
                        )
                    ),
                    "mesh_accuracy_is_3": mesh.get("mesh_accuracy") == 3,
                    "PML_is_8_layers": mesh.get("pml_layers") == 8,
                    "lateral_span_is_20um": bool(
                        np.isclose(
                            float(mesh.get("lateral_span_m", np.nan)),
                            20.0e-6,
                            rtol=0.0,
                            atol=1.0e-15,
                        )
                    ),
                    "z_span_is_minus3_to_plus3um": bool(
                        np.isclose(
                            float(mesh.get("z_min_m", np.nan)),
                            -3.0e-6,
                            rtol=0.0,
                            atol=1.0e-15,
                        )
                        and np.isclose(
                            float(mesh.get("z_max_m", np.nan)),
                            3.0e-6,
                            rtol=0.0,
                            atol=1.0e-15,
                        )
                    ),
                    "simulation_time_is_1ps": bool(
                        np.isclose(
                            float(mesh.get("simulation_time_s", np.nan)),
                            1.0e-12,
                            rtol=0.0,
                            atol=1.0e-21,
                        )
                    ),
                    "auto_shutoff_min_is_1e_minus7": bool(
                        np.isclose(
                            float(mesh.get("auto_shutoff_min", np.nan)),
                            1.0e-7,
                            rtol=0.0,
                            atol=1.0e-16,
                        )
                    ),
                    "CV0": mesh.get("conformal_mesh") == "conformal variant 0",
                    "accelerator_policy_matches": record.get("accelerator_policy")
                    == self.accelerator_policy,
                    "B200_promotion_state_matches": bool(
                        record.get("B200_promotion_certified")
                    )
                    == (self.accelerator_policy == "b200"),
                    "GPU_UUID_present": isinstance(gpu_uuid, str)
                    and bool(gpu_uuid),
                    "solver_version_present": isinstance(
                        record.get("solver_version"), str
                    ),
                }
                if not all(gates.values()):
                    raise RuntimeError(
                        f"{group_name}/{polarization} source calibration "
                        f"contract failed: {gates}"
                    )
                if expected_uuid is None:
                    expected_uuid = str(gpu_uuid)
                    expected_solver = str(record["solver_version"])
                elif (
                    gpu_uuid != expected_uuid
                    or record["solver_version"] != expected_solver
                ):
                    raise RuntimeError(
                        "100/50-nm Ea/Eb source GPU UUID or solver version differs"
                    )
                audited[group_name][polarization] = {
                    "artifact": artifact(path),
                    "status": record["status"],
                    "solver_version": record["solver_version"],
                    "requested_gpu_uuid": gpu_uuid,
                    "mesh_spec": mesh,
                    "gates": gates,
                }
        return {
            "output_root": str(self.output_root),
            "GPU_physical_index": self.gpu_index,
            "accelerator_policy": self.accelerator_policy,
            "threads": self.threads,
            "beta": self.beta,
            "mesh_label": MESH_LABEL,
            "source_calibrations": audited["optimizer_xy100"],
            "final_xy50_source_calibrations": audited.get("final_xy50"),
            "final_xy50_source_calibrations_required": (
                self.final_xy50_source_calibration is not None
            ),
            "Maxwell_solver": "Lumerical FDTD 2026 R1.2 build 4522",
            "thermal_electrical_solver": "repository custom CUDA PDE",
            "Lumerical_HEAT_or_CHARGE_used": False,
            "FDTDX_used": False,
            "Lumerical_only_source_boundary": solver_boundary,
        }


class LumericalEvaluationDriver:
    """Evaluate and cache complete signed-current gradients by latent state."""

    def __init__(
        self,
        runtime: OptimizerRuntime,
        *,
        prune_heavy_intermediates: bool = False,
    ):
        self.runtime = runtime
        self.prune_heavy_intermediates = bool(prune_heavy_intermediates)
        self.evaluations_root = runtime.output_root / "evaluations"
        self.evaluations_root.mkdir(parents=True, exist_ok=True)
        self.history: list[dict[str, Any]] = []
        self._cache: dict[str, dict[str, Any]] = {}

    @staticmethod
    def _is_prunable_heavy_artifact(path: Path) -> bool:
        return bool(
            path.suffix.lower() in {".fsp", ".h5"}
            or path.name.endswith("_raw.npz")
            or path.name == "gray_q_cuda_pde_pullback.npz"
        )

    def _prune_completed_evaluation(
        self, evaluation_dir: Path
    ) -> dict[str, Any]:
        """Remove production-only transients after gradients are persisted.

        JSON, logs, latent/projected density, final gradients, and small
        Jacobian matrices remain.  The fresh final exact-binary evaluation is
        run by a different driver and is never pruned by this method.
        """

        candidates = [
            path
            for path in evaluation_dir.rglob("*")
            if path.is_file() and self._is_prunable_heavy_artifact(path)
        ]
        records = [
            {
                "path": str(path.resolve()),
                "size_bytes_before_prune": path.stat().st_size,
                "reason": "production optimizer transient; gradients and provenance retained",
            }
            for path in candidates
        ]
        for path in candidates:
            path.unlink()
        retention = {
            "policy": "prune_heavy_continuous_optimizer_intermediates_v1",
            "pruned_at_utc": datetime.now(timezone.utc).isoformat(),
            "pruned_file_count": len(records),
            "pruned_size_bytes": int(
                sum(row["size_bytes_before_prune"] for row in records)
            ),
            "retained": [
                "evaluation_result.json",
                "signed_projected_gradients.npz",
                "latent_density.npy",
                "projected_density.npz",
                "JSON and command provenance",
                "solver logs",
                "small component-Yee Jacobian matrices",
                "per-polarization final adjoint gradients",
            ],
            "pruned": records,
            "final_exact_binary_evaluation_exempt": True,
        }
        _write_json(evaluation_dir / "ARTIFACT_RETENTION.json", retention)
        return retention

    def _command(self, script: str, *arguments: str, log_path: Path) -> None:
        command = [sys.executable, str(SCRIPT_DIR / script), *map(str, arguments)]
        log_path.parent.mkdir(parents=True, exist_ok=True)
        record_path = log_path.with_suffix(".command.json")
        _write_json(
            record_path,
            {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "command": command,
                "cwd": str(REPOSITORY),
            },
        )
        with log_path.open("w", encoding="utf-8", errors="replace") as stream:
            completed = subprocess.run(
                command,
                cwd=REPOSITORY,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if completed.returncode != 0:
            raise RuntimeError(
                f"{script} failed with exit {completed.returncode}\n{_tail(log_path)}"
            )

    def _forward(
        self, polarization: str, density_path: Path, output: Path
    ) -> dict[str, Any]:
        self._command(
            "25_run_lumerical_4um_exact_au_control.py",
            "--case",
            "import_density",
            "--rho-file",
            str(density_path),
            "--rho-key",
            "projected_density_nodal",
            "--polarization",
            polarization,
            "--gpu-index",
            str(self.runtime.gpu_index),
            "--accelerator-policy",
            self.runtime.accelerator_policy,
            "--output-dir",
            str(output),
            "--source-calibration-json",
            str(self.runtime.source_calibration[polarization]),
            "--source-object-w0-um",
            str(SOURCE_OBJECT_W0_UM),
            "--mesh-label",
            MESH_LABEL,
            "--flake-dxy-nm",
            "100",
            "--stack-dz-nm",
            "2.5",
            "--bulk-dz-nm",
            "50",
            "--outer-dxy-nm",
            "200",
            "--mesh-accuracy",
            "3",
            "--au-max-coefficients",
            "6",
            "--au-fit-tolerance",
            "0",
            "--mesh-refinement",
            "conformal variant 0",
            "--pml-layers",
            "8",
            "--lateral-span-um",
            "20",
            "--z-min-um",
            "-3",
            "--z-max-um",
            "3",
            "--simulation-time-ps",
            "1",
            "--auto-shutoff-min",
            "1e-7",
            "--threads",
            str(self.runtime.threads),
            "--include-adjoint-field-region",
            log_path=output.parent / f"forward_{polarization}.log",
        )
        json_paths = sorted(output.glob("*.json"))
        if len(json_paths) != 1:
            raise RuntimeError(
                f"expected one {polarization} forward JSON, found {json_paths}"
            )
        result = _load_json(json_paths[0])
        if (
            result.get("all_gates_passed") is not True
            or result.get("case") != "import_density"
            or result.get("polarization") != polarization
        ):
            raise RuntimeError(f"{polarization} forward gates did not pass")
        fsp, _ = _matching_raw_artifact(result, ".fsp")
        raw, _ = _matching_raw_artifact(result, "_raw.npz")
        return {
            "result_path": json_paths[0],
            "result": result,
            "fsp": fsp,
            "raw": raw,
        }

    def _pde(
        self,
        polarization: str,
        forward: dict[str, Any],
        density_path: Path,
        output: Path,
    ) -> dict[str, Any]:
        self._command(
            "33_validate_lumerical_4um_gray_q_cuda_pde.py",
            "--forward-result",
            str(forward["result_path"]),
            "--raw-npz",
            str(forward["raw"]),
            "--density-file",
            str(density_path),
            "--density-key",
            "projected_density_nodal",
            "--output-dir",
            str(output),
            "--cuda-device",
            "0",
            log_path=output.parent / f"cuda_pde_{polarization}.log",
        )
        result_path = output / "gray_q_cuda_pde_result.json"
        pullback = output / "gray_q_cuda_pde_pullback.npz"
        result = _load_json(result_path)
        if (
            result.get("passed") is not True
            or result.get("polarization") != polarization
        ):
            raise RuntimeError(f"{polarization} custom-CUDA PDE gates failed")
        return {"result_path": result_path, "result": result, "pullback": pullback}

    def _jacobian(
        self, forward: dict[str, Any], density_path: Path, output: Path
    ) -> dict[str, Any]:
        self._command(
            "26_build_lumerical_4um_yee_jacobian.py",
            "--forward-project",
            str(forward["fsp"]),
            "--forward-project-sha256",
            sha256(forward["fsp"]),
            "--forward-result-json",
            str(forward["result_path"]),
            "--forward-result-sha256",
            sha256(forward["result_path"]),
            "--density-file",
            str(density_path),
            "--density-key",
            "projected_density_nodal",
            "--output-dir",
            str(output),
            log_path=output.parent / "component_yee_jacobian.log",
        )
        result_path = output / "component_yee_jacobian_result.json"
        result = _load_json(result_path)
        if result.get("passed") is not True:
            raise RuntimeError("component-Yee material Jacobian gates failed")
        return {"result_path": result_path, "result": result}

    def _adjoint(
        self,
        polarization: str,
        forward: dict[str, Any],
        pde: dict[str, Any],
        jacobian: dict[str, Any],
        density_path: Path,
        output: Path,
    ) -> dict[str, Any]:
        self._command(
            "34_run_lumerical_4um_gray_maxwell_adjoint.py",
            "--forward-result",
            str(forward["result_path"]),
            "--forward-fsp",
            str(forward["fsp"]),
            "--forward-raw-npz",
            str(forward["raw"]),
            "--density-file",
            str(density_path),
            "--density-key",
            "projected_density_nodal",
            "--jacobian-dir",
            str(Path(jacobian["result_path"]).parent),
            "--pde-result",
            str(pde["result_path"]),
            "--pde-pullback-npz",
            str(pde["pullback"]),
            "--output-dir",
            str(output),
            "--gpu-index",
            str(self.runtime.gpu_index),
            "--accelerator-policy",
            self.runtime.accelerator_policy,
            "--threads",
            str(self.runtime.threads),
            log_path=output.parent / f"adjoint_{polarization}.log",
        )
        result_path = output / "gray_maxwell_adjoint_result.json"
        gradient_path = output / "gray_maxwell_adjoint_gradient.npz"
        result = _load_json(result_path)
        if (
            result.get("passed") is not True
            or result.get("polarization") != polarization
            or result.get("AD_FD_claimed") is not False
        ):
            raise RuntimeError(f"{polarization} Lumerical adjoint gates failed")
        with np.load(gradient_path, allow_pickle=False) as arrays:
            gradient = np.asarray(arrays["gradient_total_A"], np.float64)
            rho = np.asarray(arrays["rho_nodal"], np.float64)
        if gradient.shape != CONTRACT.design_node_shape:
            raise RuntimeError(f"{polarization} gradient has wrong shape")
        return {
            "result_path": result_path,
            "result": result,
            "gradient_path": gradient_path,
            "gradient": gradient,
            "rho": rho,
        }

    def _load_completed(self, evaluation_dir: Path) -> dict[str, Any]:
        result_path = evaluation_dir / "evaluation_result.json"
        result = _load_json(result_path)
        if result.get("passed") is not True:
            raise RuntimeError("cached evaluation is not passed")
        gradient_path = evaluation_dir / "signed_projected_gradients.npz"
        if sha256(gradient_path) != result["artifacts"]["gradients"]["sha256"]:
            raise RuntimeError("cached evaluation gradient SHA changed")
        with np.load(gradient_path, allow_pickle=False) as arrays:
            return {
                **result,
                "gradient_Ea_projected_A": np.asarray(
                    arrays["gradient_Ea_projected_A"], np.float64
                ),
                "gradient_Eb_projected_A": np.asarray(
                    arrays["gradient_Eb_projected_A"], np.float64
                ),
            }

    def evaluate(self, latent: np.ndarray) -> dict[str, Any]:
        latent_value = np.asarray(latent, np.float64)
        if latent_value.shape != CONTRACT.design_node_shape:
            raise ValueError("latent density has the wrong shape")
        if (
            not np.all(np.isfinite(latent_value))
            or np.min(latent_value) < 0.0
            or np.max(latent_value) > 1.0
        ):
            raise ValueError("latent density must be finite inside [0,1]")
        projected = OPTIMIZER_250NM_MAPPING.physical(
            latent_value, self.runtime.beta
        )
        state = density_state_audit(projected)
        state_hash = str(state["density_state_sha256"])
        if state_hash in self._cache:
            return self._cache[state_hash]
        evaluation_dir = self.evaluations_root / (
            f"eval_{len(self.history):04d}_{state_hash[:12]}"
        )
        if evaluation_dir.exists():
            result_path = evaluation_dir / "evaluation_result.json"
            if not result_path.is_file():
                raise RuntimeError(
                    f"refusing incomplete cached evaluation: {evaluation_dir}"
                )
            loaded = self._load_completed(evaluation_dir)
            self._cache[state_hash] = loaded
            return loaded
        evaluation_dir.mkdir(parents=True)
        latent_path = evaluation_dir / "latent_density.npy"
        density_path = evaluation_dir / "projected_density.npz"
        np.save(latent_path, latent_value, allow_pickle=False)
        np.savez_compressed(
            density_path, projected_density_nodal=projected
        )
        started = time.monotonic()
        forward = {
            polarization: self._forward(
                polarization,
                density_path,
                evaluation_dir / f"forward_{polarization}",
            )
            for polarization in ("Ea", "Eb")
        }
        pde = {
            polarization: self._pde(
                polarization,
                forward[polarization],
                density_path,
                evaluation_dir / f"cuda_pde_{polarization}",
            )
            for polarization in ("Ea", "Eb")
        }
        jacobian = self._jacobian(
            forward["Ea"], density_path, evaluation_dir / "yee_jacobian"
        )
        adjoint = {
            polarization: self._adjoint(
                polarization,
                forward[polarization],
                pde[polarization],
                jacobian,
                density_path,
                evaluation_dir / f"adjoint_{polarization}",
            )
            for polarization in ("Ea", "Eb")
        }
        if not np.array_equal(adjoint["Ea"]["rho"], projected) or not np.array_equal(
            adjoint["Eb"]["rho"], projected
        ):
            raise RuntimeError("Ea/Eb adjoints do not share the projected state")
        current_a = float(adjoint["Ea"]["result"]["current_A"])
        current_b = float(adjoint["Eb"]["result"]["current_A"])
        if current_a != float(pde["Ea"]["result"]["current_A"]) or current_b != float(
            pde["Eb"]["result"]["current_A"]
        ):
            raise RuntimeError("adjoint and custom-PDE currents differ")
        gradient_path = evaluation_dir / "signed_projected_gradients.npz"
        np.savez_compressed(
            gradient_path,
            projected_density=projected,
            gradient_Ea_projected_A=adjoint["Ea"]["gradient"],
            gradient_Eb_projected_A=adjoint["Eb"]["gradient"],
        )
        record: dict[str, Any] = {
            "status": "PASSED_LUMERICAL_4UM_DUALPOL_OPTIMIZER_EVALUATION",
            "passed": True,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "evaluation_index": len(self.history),
            "density_state": state,
            "currents_A": {"Ea": current_a, "Eb": current_b},
            "utilities_A": {"Ea": current_a, "Eb": -current_b},
            "balanced_utility_A": min(current_a, -current_b),
            "opposite_current_switching_achieved": current_a > 0.0
            and current_b < 0.0,
            "solver_counts": {
                "Lumerical_forward": 2,
                "Lumerical_adjoint": 2,
                "Lumerical_layout_only_Jacobian_sessions": 1,
                "custom_CUDA_thermal_forward_adjoint_pairs": 2,
                "custom_CUDA_electrical_forward_adjoint_pairs": 2,
                "Lumerical_HEAT_or_CHARGE": 0,
                "FDTDX_Maxwell": 0,
            },
            "inputs": {
                "latent": artifact(latent_path),
                "projected_density": artifact(density_path),
            },
            "polarizations": {
                polarization: {
                    "forward_result": artifact(forward[polarization]["result_path"]),
                    "PDE_result": artifact(pde[polarization]["result_path"]),
                    "adjoint_result": artifact(adjoint[polarization]["result_path"]),
                    "adjoint_gradient": artifact(adjoint[polarization]["gradient_path"]),
                }
                for polarization in ("Ea", "Eb")
            },
            "Jacobian_result": artifact(jacobian["result_path"]),
            "artifacts": {"gradients": artifact(gradient_path)},
            "wall_s": time.monotonic() - started,
            "gradient_Ea_projected_A": adjoint["Ea"]["gradient"],
            "gradient_Eb_projected_A": adjoint["Eb"]["gradient"],
        }
        persisted = {
            key: value
            for key, value in record.items()
            if key
            not in {"gradient_Ea_projected_A", "gradient_Eb_projected_A"}
        }
        _write_json(evaluation_dir / "evaluation_result.json", persisted)
        if self.prune_heavy_intermediates:
            retention = self._prune_completed_evaluation(evaluation_dir)
            record["artifact_retention"] = retention
            persisted["artifact_retention"] = retention
            _write_json(evaluation_dir / "evaluation_result.json", persisted)
        self.history.append(persisted)
        _write_json(self.runtime.output_root / "evaluation_history.json", self.history)
        self._cache[state_hash] = record
        return record


class SmokeEpigraphProblem:
    """NLopt callback surface with exact same-design physics caching."""

    def __init__(
        self,
        evaluate: Callable[[np.ndarray], dict[str, Any]],
        *,
        beta: float,
        dfm_caps: np.ndarray,
    ):
        self.evaluate_physics = evaluate
        self.beta = float(beta)
        self.dfm_caps = np.asarray(dfm_caps, np.float64)
        if self.dfm_caps.shape != (2,) or not np.all(np.isfinite(self.dfm_caps)):
            raise ValueError("DFM caps must contain two finite values")
        self.callback_history: list[dict[str, Any]] = []
        self._last_latent: np.ndarray | None = None
        self._last_point: dict[str, Any] | None = None

    @property
    def variable_count(self) -> int:
        return int(np.prod(CONTRACT.design_node_shape)) + 1

    def point(self, vector: np.ndarray) -> dict[str, Any]:
        value = np.asarray(vector, np.float64)
        if value.shape != (self.variable_count,):
            raise ValueError("optimizer vector has the wrong shape")
        latent = value[:-1].reshape(CONTRACT.design_node_shape)
        if self._last_latent is not None and np.array_equal(latent, self._last_latent):
            assert self._last_point is not None
            point = {**self._last_point, "epigraph_nA": float(value[-1])}
            return point
        evaluated = self.evaluate_physics(latent)
        signed = signed_dual_objective_point(
            latent=latent,
            beta=self.beta,
            current_a_A=float(evaluated["currents_A"]["Ea"]),
            current_b_A=float(evaluated["currents_A"]["Eb"]),
            gradient_a_projected_A=evaluated["gradient_Ea_projected_A"],
            gradient_b_projected_A=evaluated["gradient_Eb_projected_A"],
            epigraph_A=float(value[-1]) * CURRENT_SCALE_A,
            mapping=OPTIMIZER_250NM_MAPPING,
        )
        dfm_values, dfm_gradients, _ = smooth_lumerical_250nm_constraints(
            latent, self.beta
        )
        point = {
            **evaluated,
            **signed,
            "latent": latent,
            "epigraph_nA": float(value[-1]),
            "DFM_values": dfm_values,
            "DFM_gradients": dfm_gradients,
        }
        self._last_latent = latent.copy()
        self._last_point = point
        self.callback_history.append(
            {
                "callback_index": len(self.callback_history),
                "current_Ea_nA": 1.0e9 * float(point["current_a_A"]),
                "current_Eb_nA": 1.0e9 * float(point["current_b_A"]),
                "balanced_utility_nA": 1.0e9
                * float(point["balanced_utility_A"]),
                "DFM_values": dfm_values.tolist(),
            }
        )
        return point

    def objective(self, vector: np.ndarray, gradient: np.ndarray) -> float:
        if gradient.size:
            gradient[:] = 0.0
            gradient[-1] = 1.0
        return float(vector[-1])

    def constraints(
        self, result: np.ndarray, vector: np.ndarray, gradient: np.ndarray
    ) -> None:
        point = self.point(vector)
        current_constraints_nA = (
            np.asarray(point["epigraph_constraints_A"], np.float64)
            / CURRENT_SCALE_A
        )
        dfm_constraints = (
            np.asarray(point["DFM_values"], np.float64) - self.dfm_caps
        ) / DFM_CONSTRAINT_SCALE
        result[:] = np.concatenate((current_constraints_nA, dfm_constraints))
        if gradient.size:
            gradient[:] = 0.0
            gradient[:2, :-1] = (
                np.asarray(point["constraint_gradients_latent_A"], np.float64)
                / CURRENT_SCALE_A
            ).reshape(2, -1)
            gradient[:2, -1] = 1.0
            gradient[2:, :-1] = (
                np.asarray(point["DFM_gradients"], np.float64)
                / DFM_CONSTRAINT_SCALE
            ).reshape(2, -1)


def smoke_preflight(runtime: OptimizerRuntime) -> dict[str, Any]:
    latent = initial_latent_density()
    projected = OPTIMIZER_250NM_MAPPING.physical(latent, runtime.beta)
    dfm_values, dfm_gradients, _ = smooth_lumerical_250nm_constraints(
        latent, runtime.beta
    )
    dfm_caps, dfm_calibration = calibrated_lumerical_250nm_dfm_caps()
    gates = {
        "latent_shape_81x81": latent.shape == (81, 81),
        "latent_inside_bounds": bool(np.min(latent) >= 0.0 and np.max(latent) <= 1.0),
        "projected_shape_81x81": projected.shape == (81, 81),
        "projected_finite": bool(np.all(np.isfinite(projected))),
        "two_DFM_constraints": dfm_values.shape == (2,)
        and dfm_gradients.shape == (2, 81, 81),
        "calibrated_250nm_DFM_caps": dfm_caps.shape == (2,)
        and bool(np.all(np.isfinite(dfm_caps)))
        and bool(np.all(dfm_caps > 0.0)),
        "maxeval_is_two": SMOKE_MAXEVAL == 2,
        "raw_output_outside_Git": not str(runtime.output_root).startswith(
            str(REPOSITORY.resolve()) + os.sep
        ),
    }
    return {
        "status": "PASSED_LUMERICAL_4UM_OPTIMIZER_SMOKE_PREFLIGHT"
        if all(gates.values())
        else "FAILED_LUMERICAL_4UM_OPTIMIZER_SMOKE_PREFLIGHT",
        "passed": all(gates.values()),
        "runtime": runtime.audit(),
        "initial_density_state": density_state_audit(projected),
        "initial_DFM_values": dfm_values.tolist(),
        "DFM_caps_for_smoke": dfm_caps.tolist(),
        "initial_DFM_constraints_satisfied": bool(np.all(dfm_values <= dfm_caps)),
        "DFM_calibration": dfm_calibration,
        "gates": gates,
    }
