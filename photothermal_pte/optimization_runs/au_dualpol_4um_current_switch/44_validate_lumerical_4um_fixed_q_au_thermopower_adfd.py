#!/usr/bin/env python3
"""Bounded custom-CUDA AD--FD gate using a completed Lumerical Q artifact.

The optical heat source is intentionally frozen.  This validates the complete
downstream density derivative, including Au thermal transport, Au/Ta thermal
and electrical contact, explicit 3-D top-contact weighting, bulk Au
thermopower, and both custom-PDE adjoints.  It does not claim to validate the
Maxwell density term.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import traceback
from typing import Any

import numpy as np

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.contract import (
    CONTRACT,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.lumerical_4um_density import (
    load_projected_density_file,
    nodal_to_cell_average,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.lumerical_4um_gray_q_coupling import (
    GrayYeeQCoupling,
    component_coordinates_from_raw,
    component_q_from_raw,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.lumerical_4um_provenance import (
    sha256,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.lumerical_4um_yee_jacobian import (
    validate_completed_density_record,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.lumerical_only_boundary import (
    require_lumerical_only_source_boundary,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.multiphysics_4um import thermal_edges
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.volumetric_electrical_4um import (
    evaluate_fixed_source_volumetric,
)


STATUS = "VALIDATED_LUMERICAL_Q_FIXED_CUSTOM_CUDA_AU_THERMOPOWER_ADFD"
STEPS = (2.0e-3, 1.0e-3)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--forward-result", required=True, type=Path)
    parser.add_argument("--raw-npz", required=True, type=Path)
    parser.add_argument("--density-file", required=True, type=Path)
    parser.add_argument("--density-key", default="projected_density_nodal")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--cuda-device", type=int, default=0)
    return parser.parse_args()


def _artifact(path: Path) -> dict[str, Any]:
    value = path.expanduser().resolve()
    return {
        "path": str(value),
        "size_bytes": value.stat().st_size,
        "sha256": sha256(value),
    }


def _matching_artifact(result: dict[str, Any], suffix: str) -> dict[str, Any]:
    matches = [
        item
        for item in result.get("raw_artifacts", [])
        if str(item.get("path", "")).endswith(suffix)
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one {suffix} forward artifact")
    return matches[0]


def main() -> int:
    solver_boundary = require_lumerical_only_source_boundary()
    args = _parse_args()
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "fixed_q_au_thermopower_adfd.json"
    result: dict[str, Any] = {
        "status": "FAILED_LUMERICAL_Q_FIXED_CUSTOM_CUDA_AU_THERMOPOWER_ADFD",
        "passed": False,
    }
    started = time.monotonic()
    try:
        forward_path = args.forward_result.expanduser().resolve()
        raw_path = args.raw_npz.expanduser().resolve()
        density_path = args.density_file.expanduser().resolve()
        forward = json.loads(forward_path.read_text(encoding="utf-8"))
        rho_nodal = load_projected_density_file(density_path, key=args.density_key)
        rho = nodal_to_cell_average(rho_nodal)

        fsp_record = _matching_artifact(forward, ".fsp")
        raw_record = _matching_artifact(forward, "_raw.npz")
        fsp_path = Path(fsp_record["path"]).resolve()
        if sha256(fsp_path) != fsp_record["sha256"]:
            raise RuntimeError("completed Lumerical FSP SHA256 changed")
        if raw_path != Path(raw_record["path"]).resolve():
            raise RuntimeError("raw NPZ differs from completed Lumerical record")
        if sha256(raw_path) != raw_record["sha256"]:
            raise RuntimeError("completed Lumerical raw NPZ SHA256 changed")
        binding = validate_completed_density_record(
            forward,
            rho_nodal,
            forward_fsp_sha256=str(fsp_record["sha256"]),
        )
        if not binding["passed"]:
            raise RuntimeError("Lumerical forward/density binding failed")

        with np.load(raw_path, allow_pickle=False) as raw:
            coupling = GrayYeeQCoupling.from_component_coordinates(
                component_coordinates_from_raw(raw), thermal_edges()
            )
            source_raw, mapping = coupling.map_power(component_q_from_raw(raw))
        source_scale = float(
            forward["reporting_normalization"]["scalar_reporting_factor"]
        )
        source = source_raw * source_scale

        base = evaluate_fixed_source_volumetric(
            rho, source, args.cuda_device, need_gradient=True
        )
        gradient = np.asarray(base["gradient_direct_A"], dtype=np.float64)
        norm = float(np.max(np.abs(gradient)))
        if not np.isfinite(norm) or norm == 0.0:
            raise RuntimeError("downstream adjoint gradient is invalid")
        direction = gradient / norm

        rows = []
        for step in STEPS:
            plus_rho = rho + step * direction
            minus_rho = rho - step * direction
            if np.min(minus_rho) <= 0.0 or np.max(plus_rho) >= 1.0:
                raise RuntimeError("AD-FD perturbation would clip rho")
            plus = float(
                evaluate_fixed_source_volumetric(
                    plus_rho, source, args.cuda_device, need_gradient=False
                )["objective_A"]
            )
            minus = float(
                evaluate_fixed_source_volumetric(
                    minus_rho, source, args.cuda_device, need_gradient=False
                )["objective_A"]
            )
            ad = float(np.vdot(gradient, direction))
            fd = (plus - minus) / (2.0 * step)
            rows.append(
                {
                    "step": step,
                    "AD_A": ad,
                    "FD_A": fd,
                    "absolute_error_A": abs(ad - fd),
                    "relative_error": abs(ad - fd)
                    / max(abs(ad), abs(fd), np.finfo(float).tiny),
                    "plus_current_A": plus,
                    "minus_current_A": minus,
                }
            )

        thermal_adjoint = base["thermal_adjoint_audit"]
        electrical_adjoint = base["electrical_adjoint_audit"]
        electrical = base["electrical_audit"]
        gates = {
            "Lumerical_forward_density_binding": bool(binding["passed"]),
            "fixed_Q_mapping_conservation_lt_1e_12": bool(
                float(mapping["relative_power_error"]) < 1.0e-12
            ),
            "two_step_directional_ADFD_lt_1pct": bool(
                max(float(row["relative_error"]) for row in rows) < 1.0e-2
            ),
            "thermal_adjoint_residual_lt_1e_8": bool(
                float(thermal_adjoint["relative_residual"]) < 1.0e-8
            ),
            "electrical_adjoint_residual_lt_1e_8": bool(
                float(electrical_adjoint["relative_residual"]) < 1.0e-8
            ),
            "Au_bulk_thermopower_active": bool(
                float(electrical["S_Au_V_K"]) == CONTRACT.au_bulk_seebeck_V_K
                and electrical["Au_thermopower_model"]
                == CONTRACT.au_thermopower_discretization
            ),
            "Au_Ta_interface_thermopower_explicitly_zero": bool(
                float(electrical["S_Au_Ta_contact_V_K"])
                == CONTRACT.au_tairte4_interfacial_seebeck_V_K
                == 0.0
            ),
            "explicit_3d_top_contact_volumetric_model": bool(
                electrical["electrical_model"] == CONTRACT.electrical_model
            ),
            "volumetric_integral_matches_terminal_lt_1e_12": bool(
                float(electrical["volumetric_integral_relative_error"])
                < 1.0e-12
            ),
            "finite": bool(
                np.all(np.isfinite(gradient))
                and np.all(np.isfinite(base["temperature"]))
                and np.all(np.isfinite(base["weighting"]))
            ),
        }
        passed = all(gates.values())
        raw_output = output / "fixed_q_au_thermopower_adfd.npz"
        np.savez_compressed(
            raw_output,
            rho_cell=rho,
            source_power_W=source,
            direction=direction,
            gradient_total_A=gradient,
            gradient_thermal_A=np.asarray(base["gradient_thermal_A"]),
            gradient_electrical_A=np.asarray(base["gradient_electrical_A"]),
            temperature_K=np.asarray(base["temperature"]),
            TaIrTe4_temperature_K=np.asarray(base["ta_temperature"]),
            Au_temperature_K=np.asarray(base["au_temperature"]),
            weighting=np.asarray(base["weighting"]),
        )
        result = {
            "status": STATUS if passed else result["status"],
            "passed": passed,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "scope": (
                "completed Lumerical forward Q held fixed; custom CUDA thermal/"
                "electrical forward and adjoint density derivative including S_Au"
            ),
            "explicitly_not_claimed": (
                "Maxwell density derivative; no Maxwell solve is performed here"
            ),
            "polarization": forward.get("polarization"),
            "objective_A": float(base["objective_A"]),
            "objective_nA": 1.0e9 * float(base["objective_A"]),
            "thermoelectric_current_components_A": {
                "TaIrTe4": float(electrical["tairte4_thermoelectric_current_A"]),
                "Au": float(electrical["au_thermoelectric_current_A"]),
            },
            "gradient_norms_A": {
                "total": float(np.linalg.norm(gradient)),
                "thermal": float(np.linalg.norm(base["gradient_thermal_A"])),
                "electrical": float(np.linalg.norm(base["gradient_electrical_A"])),
            },
            "steps": list(STEPS),
            "rows": rows,
            "gates": gates,
            "solver_boundary": solver_boundary,
            "inputs": {
                "forward_result": _artifact(forward_path),
                "forward_FSP": _artifact(fsp_path),
                "raw_NPZ": _artifact(raw_path),
                "density_file": _artifact(density_path),
                "forward_binding": binding,
            },
            "raw_output": _artifact(raw_output),
            "Maxwell_solves": 0,
            "reused_completed_Maxwell_artifact": "Lumerical FDTD",
            "custom_CUDA_thermal_solves": {"forward": 5, "adjoint": 1},
            "custom_CUDA_electrical_solves": {"forward": 5, "adjoint": 1},
            "Lumerical_HEAT_or_CHARGE_solves": 0,
            "wall_s": time.monotonic() - started,
        }
    except Exception as error:
        result.update(
            error=f"{type(error).__name__}: {error}",
            traceback=traceback.format_exc(),
            wall_s=time.monotonic() - started,
        )
    result_path.write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(json.dumps(result, indent=2, default=str), flush=True)
    return 0 if result.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
