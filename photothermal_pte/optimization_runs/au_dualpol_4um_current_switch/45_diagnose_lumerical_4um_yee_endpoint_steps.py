#!/usr/bin/env python3
"""Diagnose component-Yee material derivatives near rho=0 and rho=1.

This is a layout-only diagnostic.  It reloads a completed nonuniform FSP,
changes only the authorized ``importnk2`` density object, and reads
``index_detail``.  It never calls ``fdtd.run`` and therefore performs zero
Maxwell solves.  Fixed density subsets are swept over several finite-
difference steps so exact endpoints are not conflated with near-endpoints.
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
import time
import traceback
from typing import Any, Mapping

import numpy as np


REPOSITORY = Path(__file__).resolve().parents[3]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from photothermal_pte.finite_inverse_design.probe_v261_cpu_tfsf_device import (  # noqa: E402
    PABS_INDEX,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.lumerical_4um_density import (  # noqa: E402
    density_state_audit,
    load_projected_density_file,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.lumerical_4um_yee_jacobian import (  # noqa: E402
    COMPONENTS,
    read_lumerical_index_detail,
    set_lumerical_projected_density,
    validate_index_detail,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.lumerical_maxwell_contract import (  # noqa: E402
    LUMAPI_PATH,
    LUMERICAL_ROOT,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.lumerical_only_boundary import (  # noqa: E402
    require_lumerical_only_source_boundary,
)


DEFAULT_STEPS = (1.0e-3, 3.0e-4, 1.0e-4, 3.0e-5, 1.0e-5, 3.0e-6)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--forward-project", required=True, type=Path)
    parser.add_argument("--density-file", required=True, type=Path)
    parser.add_argument("--density-key", default="projected_density_nodal")
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument(
        "--steps",
        type=float,
        nargs="+",
        default=list(DEFAULT_STEPS),
    )
    return parser.parse_args()


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
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _configure_lumapi() -> None:
    os.environ["VC_LUMERICAL_ROOT"] = str(LUMERICAL_ROOT)
    os.environ["LUMERICAL_ROOT"] = str(LUMERICAL_ROOT)
    os.environ["LUMERICAL_PYTHONPATH"] = str(LUMAPI_PATH.parent)
    os.environ["PATH"] = f"{LUMERICAL_ROOT / 'bin'}:{os.environ.get('PATH', '')}"
    if str(LUMAPI_PATH.parent) not in sys.path:
        sys.path.insert(0, str(LUMAPI_PATH.parent))


def _epsilon(detail: Mapping[str, np.ndarray], component: str) -> np.ndarray:
    return np.asarray(detail[f"epsilon_{component}"], np.complex128)


def _derivative_summary(
    derivative: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    component_records: dict[str, Any] = {}
    squared_norm = 0.0
    maximum = 0.0
    for component in COMPONENTS:
        value = np.asarray(derivative[component], np.complex128)
        norm = float(np.linalg.norm(value))
        max_abs = float(np.max(np.abs(value)))
        squared_norm += norm * norm
        maximum = max(maximum, max_abs)
        component_records[component] = {
            "l2_norm": norm,
            "maximum_abs": max_abs,
        }
    return {
        "components": component_records,
        "combined_l2_norm": float(np.sqrt(squared_norm)),
        "combined_maximum_abs": maximum,
    }


def _difference_summary(
    current: Mapping[str, np.ndarray],
    previous: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    component_records: dict[str, Any] = {}
    squared_difference = 0.0
    squared_current = 0.0
    squared_previous = 0.0
    maximum = 0.0
    tiny = np.finfo(float).tiny
    for component in COMPONENTS:
        left = np.asarray(current[component], np.complex128)
        right = np.asarray(previous[component], np.complex128)
        difference = left - right
        difference_norm = float(np.linalg.norm(difference))
        left_norm = float(np.linalg.norm(left))
        right_norm = float(np.linalg.norm(right))
        max_abs = float(np.max(np.abs(difference)))
        squared_difference += difference_norm * difference_norm
        squared_current += left_norm * left_norm
        squared_previous += right_norm * right_norm
        maximum = max(maximum, max_abs)
        component_records[component] = {
            "absolute_l2_difference": difference_norm,
            "maximum_abs_difference": max_abs,
            "relative_l2_difference": difference_norm
            / max(left_norm, right_norm, tiny),
        }
    combined_difference = float(np.sqrt(squared_difference))
    combined_scale = max(
        float(np.sqrt(squared_current)),
        float(np.sqrt(squared_previous)),
        tiny,
    )
    return {
        "components": component_records,
        "combined_absolute_l2_difference": combined_difference,
        "combined_maximum_abs_difference": maximum,
        "combined_relative_l2_difference": combined_difference / combined_scale,
    }


def _directions(rho: np.ndarray) -> dict[str, tuple[np.ndarray, str]]:
    exact_zero = rho == 0.0
    near_zero = (rho > 0.0) & (rho < 1.0e-4)
    lower_combined = rho < 1.0e-4
    interior = (rho >= 1.0e-3) & (rho <= 1.0 - 1.0e-3)
    exact_one = rho == 1.0
    definitions = {
        "exact_zero": (exact_zero.astype(float), "forward"),
        "near_zero_positive": (near_zero.astype(float), "forward"),
        "lower_combined": (lower_combined.astype(float), "forward"),
        "fixed_interior": (interior.astype(float), "centered"),
        "exact_one": (-exact_one.astype(float), "forward"),
    }
    return {
        name: (direction, scheme)
        for name, (direction, scheme) in definitions.items()
        if np.any(direction)
    }


def main() -> int:
    require_lumerical_only_source_boundary()
    args = _parse_args()
    forward_project = args.forward_project.expanduser().resolve()
    density_file = args.density_file.expanduser().resolve()
    output_json = args.output_json.expanduser().resolve()
    if not forward_project.is_file():
        raise FileNotFoundError(forward_project)
    if not density_file.is_file():
        raise FileNotFoundError(density_file)
    try:
        output_json.relative_to(REPOSITORY.resolve())
    except ValueError:
        pass
    else:
        raise RuntimeError("diagnostic output must be outside the Git worktree")
    steps = tuple(float(value) for value in args.steps)
    if (
        not steps
        or any(not np.isfinite(value) or not 0.0 < value < 0.5 for value in steps)
        or any(left <= right for left, right in zip(steps, steps[1:]))
    ):
        raise ValueError("steps must be finite, positive, and strictly descending")
    rho = load_projected_density_file(density_file, key=args.density_key)
    directions = _directions(rho)
    result: dict[str, Any] = {
        "status": "RUNNING_LUMERICAL_4UM_YEE_ENDPOINT_STEP_SWEEP",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "scope": "layout-only importnk2/index_detail diagnostic",
        "Maxwell_solves": 0,
        "inputs": {
            "forward_project": _artifact(forward_project),
            "density_file": _artifact(density_file),
            "density_key": args.density_key,
        },
        "density_state": density_state_audit(rho),
        "steps": list(steps),
        "subsets": {
            name: {
                "scheme": scheme,
                "node_count": int(np.count_nonzero(direction)),
                "minimum_rho": float(np.min(rho[direction != 0.0])),
                "maximum_rho": float(np.max(rho[direction != 0.0])),
            }
            for name, (direction, scheme) in directions.items()
        },
        "directions": {},
        "layout_index_detail_evaluations": 0,
    }
    _write_json(output_json, result)
    fdtd = None
    started = time.perf_counter()
    try:
        _configure_lumapi()
        import lumapi

        fdtd = lumapi.FDTD(hide=True, serverArgs={"platform": "offscreen"})
        result["solver_version"] = str(fdtd.version())
        fdtd.load(str(forward_project))
        fdtd.switchtolayout()

        def evaluate(value: np.ndarray) -> dict[str, np.ndarray]:
            set_lumerical_projected_density(fdtd, value)
            detail = read_lumerical_index_detail(fdtd, monitor_name=PABS_INDEX)
            result["layout_index_detail_evaluations"] += 1
            return detail

        baseline = evaluate(rho)
        baseline_audit = validate_index_detail(baseline)
        result["baseline"] = baseline_audit
        _write_json(output_json, result)
        for name, (direction, scheme) in directions.items():
            direction_record: dict[str, Any] = {
                "scheme": scheme,
                "node_count": int(np.count_nonzero(direction)),
                "steps": [],
            }
            result["directions"][name] = direction_record
            previous: dict[str, np.ndarray] | None = None
            previous_step: float | None = None
            for step in steps:
                positive = evaluate(rho + step * direction)
                if scheme == "centered":
                    negative = evaluate(rho - step * direction)
                    denominator = 2.0 * step
                else:
                    negative = baseline
                    denominator = step
                derivative = {
                    component: (
                        _epsilon(positive, component)
                        - _epsilon(negative, component)
                    )
                    / denominator
                    for component in COMPONENTS
                }
                record: dict[str, Any] = {
                    "step": step,
                    "derivative": _derivative_summary(derivative),
                }
                if previous is not None:
                    record["difference_from_next_larger_step"] = {
                        "next_larger_step": previous_step,
                        **_difference_summary(derivative, previous),
                    }
                direction_record["steps"].append(record)
                previous = derivative
                previous_step = step
                _write_json(output_json, result)
        roundtrip = evaluate(rho)
        result["roundtrip"] = validate_index_detail(roundtrip)
        result["baseline_roundtrip_epsilon_max_abs_error"] = max(
            float(
                np.max(
                    np.abs(
                        _epsilon(roundtrip, component)
                        - _epsilon(baseline, component)
                    )
                )
            )
            for component in COMPONENTS
        )
        result["status"] = "COMPLETED_LUMERICAL_4UM_YEE_ENDPOINT_STEP_SWEEP"
        result["passed"] = result["baseline_roundtrip_epsilon_max_abs_error"] == 0.0
    except Exception as exc:
        result["status"] = "FAILED_LUMERICAL_4UM_YEE_ENDPOINT_STEP_SWEEP"
        result["passed"] = False
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
    finally:
        result["wall_time_s"] = time.perf_counter() - started
        if fdtd is not None:
            try:
                fdtd.close()
            except Exception:
                pass
        _write_json(output_json, result)
    print(json.dumps(result, indent=2))
    return 0 if result.get("passed") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
