#!/usr/bin/env python3
"""Build and validate the Au density component-Yee material Jacobian.

This command consumes a completed nonuniform ``import_density`` FSP.  It
never calls ``fdtd.run``: all colored perturbations are layout-only queries
of the frozen Lumerical ``index_detail`` map.  Raw sparse matrices and
coordinates must be written outside Git.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
from typing import Any

import numpy as np
from scipy import sparse


REPOSITORY = Path(__file__).resolve().parents[3]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from photothermal_pte.finite_inverse_design.probe_v261_cpu_tfsf_device import (  # noqa: E402
    PABS_FIELD,
    PABS_INDEX,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.lumerical_4um_density import (  # noqa: E402
    density_nodes,
    density_state_audit,
    load_projected_density_file,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.lumerical_4um_yee_jacobian import (  # noqa: E402
    COMPONENTS,
    build_colored_material_jacobian,
    component_coordinates,
    read_lumerical_index_detail,
    set_lumerical_projected_density,
    validate_completed_density_record,
    validate_material_jacobian,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.lumerical_maxwell_contract import (  # noqa: E402
    LUMAPI_PATH,
    LUMERICAL_ROOT,
)


COORDINATE_MISMATCH_LIMIT_M = 2.0e-18


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "UNKNOWN"


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--forward-project", required=True, type=Path)
    parser.add_argument("--forward-project-sha256", required=True)
    parser.add_argument("--forward-result-json", required=True, type=Path)
    parser.add_argument("--forward-result-sha256", required=True)
    parser.add_argument("--density-file", required=True, type=Path)
    parser.add_argument("--density-key", default="projected_density_nodal")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--audit-only", action="store_true")
    return parser.parse_args()


def _maximum_difference(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, float).reshape(-1)
    b = np.asarray(right, float).reshape(-1)
    if a.shape != b.shape:
        return float("inf")
    return float(np.max(np.abs(a - b)))


def _field_component_coordinates(fdtd: Any) -> dict[str, tuple[np.ndarray, ...]]:
    base = {
        axis: np.asarray(fdtd.getdata(PABS_FIELD, axis, 1), float).reshape(-1)
        for axis in COMPONENTS
    }
    delta = {
        axis: np.asarray(
            fdtd.getdata(PABS_FIELD, f"delta_{axis}", 1), float
        ).reshape(-1)
        for axis in COMPONENTS
    }
    for axis in COMPONENTS:
        if base[axis].shape != delta[axis].shape:
            raise RuntimeError(f"PABS field delta_{axis} shape mismatch")
    return {
        component: tuple(
            base[axis] + delta[axis] if axis == component else base[axis]
            for axis in COMPONENTS
        )
        for component in COMPONENTS
    }


def _coordinate_audit(
    fdtd: Any, detail: dict[str, np.ndarray]
) -> dict[str, Any]:
    field_coordinates = _field_component_coordinates(fdtd)
    records: dict[str, Any] = {}
    maximum = 0.0
    for component in COMPONENTS:
        index_coordinates = component_coordinates(detail, component)
        mismatch = max(
            _maximum_difference(left, right)
            for left, right in zip(
                field_coordinates[component], index_coordinates, strict=True
            )
        )
        maximum = max(maximum, mismatch)
        records[component] = {
            "maximum_field_index_coordinate_mismatch_m": mismatch,
            "coordinate_bounds_m": {
                axis: [float(value[0]), float(value[-1])]
                for axis, value in zip(
                    COMPONENTS, index_coordinates, strict=True
                )
            },
        }
    return {
        "components": records,
        "maximum_mismatch_m": maximum,
        "limit_m": COORDINATE_MISMATCH_LIMIT_M,
        "passed": maximum < COORDINATE_MISMATCH_LIMIT_M,
    }


def _epsilon_difference(
    left: dict[str, np.ndarray], right: dict[str, np.ndarray]
) -> float:
    return max(
        float(
            np.max(
                np.abs(
                    np.asarray(left[f"epsilon_{component}"])
                    - np.asarray(right[f"epsilon_{component}"])
                )
            )
        )
        for component in COMPONENTS
    )


def _configure_lumapi() -> None:
    os.environ["VC_LUMERICAL_ROOT"] = str(LUMERICAL_ROOT)
    os.environ["LUMERICAL_ROOT"] = str(LUMERICAL_ROOT)
    os.environ["LUMERICAL_PYTHONPATH"] = str(LUMAPI_PATH.parent)
    os.environ["PATH"] = f"{LUMERICAL_ROOT / 'bin'}:{os.environ.get('PATH', '')}"
    if str(LUMAPI_PATH.parent) not in sys.path:
        sys.path.insert(0, str(LUMAPI_PATH.parent))


def main() -> int:
    args = _parse_args()
    forward_project = args.forward_project.expanduser().resolve()
    forward_result_path = args.forward_result_json.expanduser().resolve()
    density_path = args.density_file.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    for path, expected, label in (
        (forward_project, args.forward_project_sha256, "forward FSP"),
        (forward_result_path, args.forward_result_sha256, "forward result JSON"),
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = _sha256(path)
        if actual != expected:
            raise RuntimeError(f"{label} SHA mismatch: {actual} != {expected}")
    rho = load_projected_density_file(density_path, key=args.density_key)
    forward_record = json.loads(forward_result_path.read_text(encoding="utf-8"))
    forward_validation = validate_completed_density_record(
        forward_record,
        rho,
        forward_fsp_sha256=args.forward_project_sha256,
    )
    if not forward_validation["passed"]:
        raise RuntimeError(
            f"completed import-density forward gate failed: {forward_validation}"
        )
    audit = {
        "status": "AUDITED_LUMERICAL_4UM_YEE_JACOBIAN_INPUTS_NOT_RUN",
        "forward_validation": forward_validation,
        "density_state": density_state_audit(rho),
        "inputs": {
            "forward_project": _artifact(forward_project),
            "forward_result_json": _artifact(forward_result_path),
            "density_file": _artifact(density_path),
            "density_key": args.density_key,
        },
        "Maxwell_solves": 0,
    }
    if args.audit_only:
        print(json.dumps(audit, indent=2))
        return 0

    try:
        output.relative_to(REPOSITORY.resolve())
    except ValueError:
        pass
    else:
        raise RuntimeError("Jacobian raw outputs must be outside the Git worktree")
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing nonempty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "component_yee_jacobian_result.json"
    result: dict[str, Any] = {
        **audit,
        "status": "BLOCKED_LUMERICAL_4UM_COMPONENT_YEE_JACOBIAN",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "forward_accelerator_policy": forward_record.get("accelerator_policy"),
        "B200_promotion_certified": bool(
            forward_record.get("B200_promotion_certified")
        ),
        "scope": (
            "layout-only material-map certificate; no Maxwell, thermal, "
            "electrical, adjoint, or optimization solve"
        ),
    }
    fdtd = None
    try:
        _configure_lumapi()
        import lumapi

        fdtd = lumapi.FDTD(hide=True, serverArgs={"platform": "offscreen"})
        solver_version = str(fdtd.version())
        result["solver_version"] = solver_version
        if solver_version != str(forward_record["solver_version"]):
            raise RuntimeError(
                "forward/Jacobian solver-version mismatch: "
                f"{forward_record['solver_version']} != {solver_version}"
            )
        fdtd.load(str(forward_project))
        completed_detail = read_lumerical_index_detail(
            fdtd, monitor_name=PABS_INDEX
        )
        coordinate_audit = _coordinate_audit(fdtd, completed_detail)
        if not coordinate_audit["passed"]:
            raise RuntimeError(
                f"field/index native-Yee coordinate mismatch: {coordinate_audit}"
            )
        fdtd.switchtolayout()

        def evaluate(value: np.ndarray) -> dict[str, np.ndarray]:
            set_lumerical_projected_density(fdtd, value)
            return read_lumerical_index_detail(fdtd, monitor_name=PABS_INDEX)

        operator, construction, layout_baseline = build_colored_material_jacobian(
            evaluate, rho
        )
        completed_layout_error = _epsilon_difference(
            completed_detail, layout_baseline
        )
        validation = validate_material_jacobian(evaluate, rho, operator)
        gates = {
            "completed_vs_layout_baseline_epsilon_exact": (
                completed_layout_error == 0.0
            ),
            "field_index_coordinate_match": coordinate_audit["passed"],
            "construction_roundtrip_exact": construction[
                "baseline_roundtrip_epsilon_max_abs_error"
            ]
            == 0.0,
            "mapping_FD_and_transpose_passed": validation["passed"],
            "zero_Maxwell_solves": construction["Maxwell_solves"] == 0
            and validation["Maxwell_solves"] == 0,
            "forward_density_FSP_hash_link_passed": forward_validation["passed"],
        }
        matrix_artifacts: dict[str, Any] = {}
        for component, matrix in operator.matrices.items():
            path = output / f"J_{component}.npz"
            sparse.save_npz(path, matrix)
            matrix_artifacts[component] = _artifact(path)
        x, y, _ = density_nodes()
        layout_path = output / "component_yee_coordinates_and_density.npz"
        arrays: dict[str, np.ndarray] = {
            "projected_density_nodal": rho,
            "projected_density_x_m": x,
            "projected_density_y_m": y,
        }
        for component in COMPONENTS:
            for axis, value in zip(
                COMPONENTS,
                component_coordinates(completed_detail, component),
                strict=True,
            ):
                arrays[f"{component}_{axis}_m"] = value
        np.savez_compressed(layout_path, **arrays)
        passed = all(gates.values())
        result.update(
            {
                "status": (
                    "PASSED_LUMERICAL_4UM_COMPONENT_YEE_JACOBIAN"
                    if passed
                    else "FAILED_LUMERICAL_4UM_COMPONENT_YEE_JACOBIAN"
                ),
                "passed": passed,
                "completed_vs_layout_baseline_epsilon_max_abs_error": (
                    completed_layout_error
                ),
                "coordinate_audit": coordinate_audit,
                "construction": construction,
                "validation": validation,
                "gates": gates,
                "artifacts": {
                    "component_J": matrix_artifacts,
                    "coordinates_and_density": _artifact(layout_path),
                },
            }
        )
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
    finally:
        if fdtd is not None:
            try:
                fdtd.close()
            except Exception:
                pass
        _write_json(result_path, result)
    print(json.dumps(result, indent=2))
    return 0 if result.get("passed") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
