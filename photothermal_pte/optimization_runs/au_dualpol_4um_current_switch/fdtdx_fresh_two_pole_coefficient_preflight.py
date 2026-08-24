#!/usr/bin/env python3
"""Read back candidate two-pole coefficients from the pinned FDTDX library.

This is deliberately narrower than a model build or field solve.  It proves
that the physical pole parameters frozen in a candidate material-law contract
produce the exact expected float32 c1/c2/c3 values and zero c4 through FDTDX's
own coefficient generator.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_case_contract import (
    file_sha256,
    load_case_contract,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_two_pole_material_contract import (
    load_material_law_contract,
)


VERSION = "fdtdx-fresh-two-pole-coefficient-preflight-v1"


def _import_pinned_fdtdx(source: Path):
    root = source.expanduser().resolve()
    expected = root / "src"
    if not expected.is_dir():
        raise RuntimeError("pinned FDTDX source has no src directory")
    if str(expected) not in sys.path:
        sys.path.insert(0, str(expected))
    fdtdx = importlib.import_module("fdtdx")
    imported = Path(fdtdx.__file__).resolve()
    if expected not in imported.parents:
        raise RuntimeError(f"unpinned FDTDX import: {imported}")
    dispersion = importlib.import_module("fdtdx.dispersion")
    return fdtdx, dispersion, imported


def contract_matrix(axis: dict[str, Any]) -> np.ndarray:
    poles = axis["candidate"]["poles"]
    if len(poles) != 2:
        raise RuntimeError("candidate material axis must contain exactly two poles")
    result = np.asarray(
        [[pole["c1"], pole["c2"], pole["c3"], 0.0] for pole in poles],
        dtype=np.float32,
    )
    if result.shape != (2, 4) or not np.all(np.isfinite(result)):
        raise RuntimeError("invalid contract coefficient matrix")
    return result


def coefficient_readback(
    expected: np.ndarray,
    observed: np.ndarray,
) -> dict[str, Any]:
    expected = np.asarray(expected, dtype=np.float32)
    observed = np.asarray(observed, dtype=np.float32)
    exact = expected.shape == observed.shape and np.array_equal(expected, observed)
    return {
        "expected_shape": list(expected.shape),
        "observed_shape": list(observed.shape),
        "expected": expected.astype(np.float64).tolist(),
        "observed": observed.astype(np.float64).tolist(),
        "maximum_absolute_error": (
            float(np.max(np.abs(observed - expected)))
            if expected.shape == observed.shape
            else None
        ),
        "exact": exact,
    }


def _fdtdx_poles(fdtdx: Any, axis: dict[str, Any]) -> tuple[Any, ...]:
    output = []
    for item in axis["candidate"]["poles"]:
        if item["kind"] == "Drude":
            pole = fdtdx.DrudePole(
                plasma_frequency=float(item["omega_p_rad_s"]),
                damping=float(item["gamma_rad_s"]),
            )
        elif item["kind"] == "Lorentz":
            pole = fdtdx.LorentzPole(
                resonance_frequency=float(item["omega_0_rad_s"]),
                damping=float(item["gamma_rad_s"]),
                delta_epsilon=float(item["delta_epsilon"]),
            )
        else:
            raise RuntimeError(f"unsupported pole kind {item['kind']!r}")
        output.append(pole)
    return tuple(output)


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

    spec, case_payload, case_audit = load_case_contract(
        args.case_contract, args.case_contract_sha256
    )
    law, law_audit = load_material_law_contract(
        args.material_law,
        args.material_law_sha256,
        spec,
        case_payload,
        case_audit["actual_sha256"],
        args.fdtdx_source,
    )
    fdtdx, dispersion, imported = _import_pinned_fdtdx(args.fdtdx_source)
    dt_s = float(law["case_binding"]["realized_float32_cfl"]["time_step_s"])
    axes: dict[str, Any] = {}
    for name, axis in law["material_axes"].items():
        poles = _fdtdx_poles(fdtdx, axis)
        c1, c2, c3, c4 = dispersion.compute_pole_coefficients_per_axis(
            poles, dt_s
        )
        observed = np.stack(
            (
                np.asarray(c1[:, 0], dtype=np.float32),
                np.asarray(c2[:, 0], dtype=np.float32),
                np.asarray(c3[:, 0], dtype=np.float32),
                np.asarray(c4[:, 0], dtype=np.float32),
            ),
            axis=1,
        )
        readback = coefficient_readback(contract_matrix(axis), observed)
        readback["three_FDTDX_axis_columns_identical"] = all(
            np.array_equal(np.asarray(item, dtype=np.float32), observed[:, index])
            for index, item in enumerate((c1[:, 1], c2[:, 1], c3[:, 1], c4[:, 1]))
        ) and all(
            np.array_equal(np.asarray(item, dtype=np.float32), observed[:, index])
            for index, item in enumerate((c1[:, 2], c2[:, 2], c3[:, 2], c4[:, 2]))
        )
        readback["c4_exactly_zero"] = bool(np.all(observed[:, 3] == 0.0))
        readback["ready"] = bool(
            readback["exact"]
            and readback["three_FDTDX_axis_columns_identical"]
            and readback["c4_exactly_zero"]
        )
        axes[name] = readback
    checks = {
        "all_material_axis_coefficients_exact": all(
            item["ready"] for item in axes.values()
        ),
        "law_remains_candidate_only": law["promotion"]["candidate_only"] is True,
        "optimizer_remains_forbidden": (
            law["promotion"]["optimizer_start_allowed"] is False
        ),
    }
    ready = all(checks.values())
    payload = {
        "version": VERSION,
        "status": (
            "VALIDATED_CANDIDATE_TWO_POLE_FDTDX_COEFFICIENT_READBACK"
            if ready
            else "BLOCKED_CANDIDATE_TWO_POLE_FDTDX_COEFFICIENT_READBACK"
        ),
        "ready": ready,
        "scope": "pinned FDTDX coefficient generation only; no grid/model/field solve",
        "case_file_audit": case_audit,
        "material_law_file_audit": law_audit,
        "fdtdx": {
            "imported_module": str(imported),
            "update_sha256": law["implementation_binding"]["pinned_fdtdx"][
                "update_sha256"
            ],
            "dispersion_sha256": law["implementation_binding"]["pinned_fdtdx"][
                "dispersion_sha256"
            ],
        },
        "time_step_s": dt_s,
        "material_axis_readback": axes,
        "checks": checks,
        "promotion": {
            "candidate_only": True,
            "is_solver_array_certificate": False,
            "is_material_certificate": False,
            "is_mesh_certificate": False,
            "optimizer_start_allowed": False,
        },
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    payload["preflight_payload_sha256"] = hashlib.sha256(encoded).hexdigest()
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "ready": ready,
                "output": str(output),
                "file_sha256": file_sha256(output),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
