#!/usr/bin/env python3
"""Read back opt-in two-pole coefficients from a placed FDTDX solver state.

No FDTD time step or optimizer is called.  The preflight binds exact case and
material-law files, builds the physical grid/objects through pinned FDTDX,
places a fixed 500 nm-DFM binary reference mask, and copies only material
regions back to the host for exact coefficient and permittivity checks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import traceback
from typing import Any

import numpy as np

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_exact_binary_convergence import (
    mesh_audit,
    reference_mask,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_exact_binary_material import (
    arrays_for_exact_binary,
    coefficient_endpoint_matrix,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_case_contract import (
    file_sha256,
    load_case_contract,
    realized_time_contract,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_mesh import (
    build_model,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_two_pole_material_contract import (
    load_material_law_contract,
)


VERSION = "fdtdx-fresh-two-pole-placed-solver-array-preflight-v1"
REFERENCE_NAME = "l_shape_4um_with_500nm_arms"
HERE = Path(__file__).resolve().parent
IMPLEMENTATION_FILES = (
    HERE / "fdtdx_4um_model.py",
    HERE / "fdtdx_fresh_candidate_model_material.py",
    HERE / "fdtdx_exact_binary_material.py",
    HERE / "fdtdx_fresh_exact_binary_pilot.py",
    Path(__file__).resolve(),
)


def _shape(value: Any) -> list[int] | None:
    return None if value is None else [int(item) for item in value.shape]


def placed_solver_array_audit(
    model: dict[str, Any],
    arrays: Any,
    law: dict[str, Any],
    spec: Any,
    material_stack: dict[str, Any],
) -> dict[str, Any]:
    """Prove the placed state retains the exact candidate pole endpoints."""

    requested_dt = float(
        law["case_binding"]["realized_float32_cfl"]["time_step_s"]
    )
    expected_contract_sha = law["material_law_contract_sha256"]
    array_shapes = {
        name: _shape(getattr(arrays, name))
        for name in (
            "inv_permittivities",
            "dispersive_c1",
            "dispersive_c2",
            "dispersive_c3",
            "dispersive_c4",
        )
    }
    endpoint_readback = {
        name: coefficient_endpoint_matrix(model, name).astype(float).tolist()
        for name in ("au", "a", "b", "c")
    }
    expected_endpoints = {
        name: np.asarray(
            [
                [pole["c1"], pole["c2"], pole["c3"]]
                for pole in law["material_axes"][name]["candidate"]["poles"]
            ],
            dtype=np.float32,
        )
        for name in ("au", "a", "b", "c")
    }
    fixed_shapes = {
        name: _shape(model[name]) for name in ("fixed_c1", "fixed_c2", "fixed_c3")
    }
    checks = {
        "material_stack_readback_ready": material_stack["ready"],
        "candidate_model_mode_exact": (
            model.get("material_law_mode") == "candidate-two-pole-contract"
        ),
        "material_law_contract_hash_exact": (
            model.get("material_law_contract_sha256") == expected_contract_sha
        ),
        "two_poles_declared": model.get("num_dispersive_poles") == 2,
        "all_model_endpoints_equal_contract_float32": all(
            np.array_equal(
                coefficient_endpoint_matrix(model, name), expected_endpoints[name]
            )
            for name in expected_endpoints
        ),
        "placed_arrays_have_two_poles_and_three_components": all(
            array_shapes[name] is not None and array_shapes[name][:2] == [2, 3]
            for name in ("dispersive_c1", "dispersive_c2", "dispersive_c3")
        ),
        "fixed_arrays_match_placed_array_shapes": all(
            fixed_shapes[f"fixed_c{index}"]
            == array_shapes[f"dispersive_c{index}"]
            for index in (1, 2, 3)
        ),
        "lorentz_drude_c4_array_absent": array_shapes["dispersive_c4"] is None,
        "realized_time_step_exact": (
            float(model["config"].time_step_duration) == requested_dt
        ),
        "fresh_mesh_contract_exact": model.get("fresh_mesh_audit")
        == mesh_audit(spec),
        "candidate_only_remains_true": law["promotion"]["candidate_only"] is True,
        "optimizer_remains_forbidden": (
            law["promotion"]["optimizer_start_allowed"] is False
        ),
        "adjoint_source_absent": (
            "distributed_adjoint_source" not in model["slices"]
        ),
        "full_material_model_not_air_calibration": (
            model.get("air_only_source_calibration") is False
        ),
    }
    return {
        "ready": all(checks.values()),
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "array_shapes": array_shapes,
        "fixed_array_shapes": fixed_shapes,
        "coefficient_endpoints": endpoint_readback,
        "devices": [str(device) for device in model["jax"].devices()],
        "scope": "placed material arrays only; zero FDTD time steps",
    }


def _write(output: Path, payload: dict[str, Any]) -> None:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    payload["preflight_payload_sha256"] = hashlib.sha256(encoded).hexdigest()
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-contract", type=Path, required=True)
    parser.add_argument("--case-contract-sha256", required=True)
    parser.add_argument("--material-law", type=Path, required=True)
    parser.add_argument("--material-law-sha256", required=True)
    parser.add_argument("--fdtdx-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    output = args.output.expanduser()
    if not output.is_absolute():
        parser.error("--output must be absolute")
    output = output.resolve()
    if not output.parent.is_dir() or output.exists():
        parser.error("output parent must exist and output must not exist")

    fdtdx_source = args.fdtdx_source.expanduser().resolve()
    spec, case_payload, case_audit = load_case_contract(
        args.case_contract, args.case_contract_sha256
    )
    law, law_audit = load_material_law_contract(
        args.material_law,
        args.material_law_sha256,
        spec,
        case_payload,
        case_audit["actual_sha256"],
        fdtdx_source,
    )
    common = {
        "version": VERSION,
        "case_file_audit": case_audit,
        "material_law_file_audit": law_audit,
        "reference_name": REFERENCE_NAME,
        "implementation_sha256": {
            str(path.relative_to(HERE)): file_sha256(path)
            for path in IMPLEMENTATION_FILES
        },
        "promotion": {
            "candidate_only": True,
            "is_solver_array_certificate": False,
            "is_material_certificate": False,
            "is_mesh_certificate": False,
            "optimizer_start_allowed": False,
        },
    }

    try:
        os.environ["FDTDX_SOURCE_DIR"] = str(fdtdx_source)
        from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch import (
            fdtdx_4um_model as optical_model,
        )
        from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_exact_binary_pilot import (
            material_stack_audit,
        )

        optical_model.FDTDX_SOURCE = fdtdx_source
        model = build_model(
            spec.mesh,
            "Ea",
            total_periods=spec.time.total_periods,
            window_periods=spec.time.window_periods,
            courant_factor=spec.time.courant_factor,
            alpha_scale=spec.pml_alpha_scale,
            target_reflection=spec.pml_target_reflection,
            include_adjoint_source=False,
            air_only_source_calibration=False,
            material_law_contract=law,
        )
        mask = reference_mask(REFERENCE_NAME)
        arrays = arrays_for_exact_binary(model, mask, spec.mesh)
        material_stack = material_stack_audit(model, arrays, mask, spec.mesh)
        array_audit = placed_solver_array_audit(
            model, arrays, law, spec.mesh, material_stack
        )
        case_time = realized_time_contract(spec, model)
        extra_checks = {
            "requested_time_contract_exact": case_time
            == {
                **dict(case_payload["time_spec"]),
                "time_step_s": float(model["config"].time_step_duration),
                "time_steps_total": int(model["config"].time_steps_total),
            },
            "pml_contract_exact": model["pml_face_parameters"]
            == case_payload["resolved_pml_face_parameters"],
        }
        ready = array_audit["ready"] and all(extra_checks.values())
        payload = {
            **common,
            "status": (
                "VALIDATED_CANDIDATE_TWO_POLE_PLACED_SOLVER_ARRAY_READBACK"
                if ready
                else "BLOCKED_CANDIDATE_TWO_POLE_PLACED_SOLVER_ARRAY_READBACK"
            ),
            "ready": ready,
            "mesh": mesh_audit(spec.mesh),
            "realized_time_contract": case_time,
            "material_stack": material_stack,
            "placed_solver_array_audit": array_audit,
            "checks": extra_checks,
            "failed_checks": [
                name for name, passed in extra_checks.items() if not passed
            ]
            + array_audit["failed_checks"],
        }
    except Exception as error:
        payload = {
            **common,
            "status": "BLOCKED_CANDIDATE_TWO_POLE_PLACED_SOLVER_ARRAY_READBACK",
            "ready": False,
            "failure": {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            },
        }

    _write(output, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "ready": payload["ready"],
                "output": str(output),
                "file_sha256": file_sha256(output),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if payload["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
