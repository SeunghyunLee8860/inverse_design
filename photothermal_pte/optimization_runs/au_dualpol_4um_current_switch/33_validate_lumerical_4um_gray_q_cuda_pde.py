#!/usr/bin/env python3
"""Validate one completed gray-density Lumerical forward through custom PDEs.

This script performs no Maxwell solve.  It verifies the completed FSP/raw
artifact hashes, conservatively maps all native component-Yee absorption to
thermal-cell power, solves the repository CUDA thermal/electrical forward and
adjoint systems, and saves the exact native-Q pullback needed by the upcoming
Lumerical Maxwell adjoint.
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
    density_state_audit,
    load_projected_density_file,
    nodal_to_cell_average,
    nodal_to_cell_vjp,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.lumerical_4um_gray_q_coupling import (
    GrayYeeQCoupling,
    adjoint_bilinear_dot_audit,
    component_coordinates_from_raw,
    component_q_from_raw,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.lumerical_4um_yee_jacobian import (
    validate_completed_density_record,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.multiphysics_4um import thermal_edges
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.volumetric_electrical_4um import (
    evaluate_fixed_source_volumetric,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.lumerical_4um_provenance import (
    sha256,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.lumerical_only_boundary import (
    require_lumerical_only_source_boundary,
)


STATUS = "VALIDATED_LUMERICAL_4UM_GRAY_Q_CUSTOM_CUDA_PDE"


def _artifact(path: Path) -> dict[str, Any]:
    value = path.expanduser().resolve()
    return {
        "path": str(value),
        "size_bytes": value.stat().st_size,
        "sha256": sha256(value),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--forward-result", required=True, type=Path)
    parser.add_argument("--raw-npz", required=True, type=Path)
    parser.add_argument("--density-file", required=True, type=Path)
    parser.add_argument("--density-key", default="rho")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--cuda-device", type=int, default=5)
    return parser.parse_args()


def _matching_artifact(result: dict[str, Any], suffix: str) -> dict[str, Any]:
    matches = [
        item
        for item in result.get("raw_artifacts", [])
        if str(item.get("path", "")).endswith(suffix)
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one {suffix} artifact in forward result")
    return matches[0]


def main() -> int:
    require_lumerical_only_source_boundary()
    args = _parse_args()
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "gray_q_cuda_pde_result.json"
    result: dict[str, Any] = {
        "status": "FAILED_LUMERICAL_4UM_GRAY_Q_CUSTOM_CUDA_PDE",
        "passed": False,
        "Maxwell_solves": 0,
        "Lumerical_HEAT_or_CHARGE_solves": 0,
        "optimizer_iterations": 0,
    }
    started = time.monotonic()
    try:
        forward_path = args.forward_result.expanduser().resolve()
        raw_path = args.raw_npz.expanduser().resolve()
        density_path = args.density_file.expanduser().resolve()
        forward = json.loads(forward_path.read_text(encoding="utf-8"))
        rho_nodal = load_projected_density_file(density_path, key=args.density_key)
        rho_cell = nodal_to_cell_average(rho_nodal)

        fsp_record = _matching_artifact(forward, ".fsp")
        raw_record = _matching_artifact(forward, "_raw.npz")
        fsp_path = Path(fsp_record["path"]).resolve()
        if sha256(fsp_path) != fsp_record["sha256"]:
            raise RuntimeError("completed forward FSP SHA256 changed")
        if raw_path != Path(raw_record["path"]).resolve():
            raise RuntimeError("raw NPZ path differs from the forward record")
        if sha256(raw_path) != raw_record["sha256"]:
            raise RuntimeError("completed forward raw NPZ SHA256 changed")
        forward_binding = validate_completed_density_record(
            forward,
            rho_nodal,
            forward_fsp_sha256=str(fsp_record["sha256"]),
        )
        if not forward_binding["passed"]:
            raise RuntimeError("completed Lumerical density-forward binding failed")

        with np.load(raw_path, allow_pickle=False) as raw:
            coupling = GrayYeeQCoupling.from_component_coordinates(
                component_coordinates_from_raw(raw), thermal_edges()
            )
            q = component_q_from_raw(raw)
            source_power_raw, mapping = coupling.map_power(q)
            transpose = coupling.transpose_dot_audit()
            normalization = forward["reporting_normalization"]
            scale = float(normalization["scalar_reporting_factor"])
            source_power = source_power_raw * scale
            evaluated = evaluate_fixed_source_volumetric(
                rho_cell,
                source_power,
                args.cuda_device,
                need_gradient=True,
            )
            native_q_pullback_reporting = coupling.pullback_cell_power(
                np.asarray(evaluated["thermal_adjoint"], dtype=np.float64)
            )

        gradient_direct_cell = np.asarray(evaluated["gradient_direct_A"], float)
        gradient_direct_nodal = nodal_to_cell_vjp(gradient_direct_cell)
        contraction_audit = adjoint_bilinear_dot_audit(
            source_components=q,
            source_pullback=native_q_pullback_reporting,
            mapped_output=source_power,
            output_cotangent=np.asarray(evaluated["thermal_adjoint"]),
            source_scale=scale,
        )
        expected_raw_power = float(forward["P_Q_native_W_raw"])
        raw_power_error = abs(float(np.sum(source_power_raw)) - expected_raw_power) / max(
            abs(expected_raw_power), np.finfo(float).tiny
        )
        thermal_audit = evaluated["thermal_audit"]
        electrical_audit = evaluated["electrical_audit"]
        thermal_adjoint_audit = evaluated["thermal_adjoint_audit"]
        electrical_adjoint_audit = evaluated["electrical_adjoint_audit"]
        gates = {
            "completed_forward_binding": bool(forward_binding["passed"]),
            "native_Q_json_power_match_lt_1e_12": raw_power_error < 1.0e-12,
            "Q_mapping_conservation_lt_1e_12": float(mapping["relative_power_error"]) < 1.0e-12,
            "Q_mapping_transpose_lt_1e_12": float(transpose["relative_error"]) < 1.0e-12,
            "Q_pullback_contraction_normwise_lt_1e_12": (
                contraction_audit["normwise_relative_error"] < 1.0e-12
            ),
            "thermal_forward_residual_lt_1e_8": float(thermal_audit["relative_residual"]) < 1.0e-8,
            "thermal_adjoint_residual_lt_1e_8": float(thermal_adjoint_audit["relative_residual"]) < 1.0e-8,
            "thermal_energy_balance_lt_1pct": float(thermal_audit["energy_balance_relative"]) < 1.0e-2,
            "electrical_forward_residual_lt_1e_8": float(electrical_audit["relative_residual"]) < 1.0e-8,
            "electrical_adjoint_residual_lt_1e_8": float(electrical_adjoint_audit["relative_residual"]) < 1.0e-8,
            "electrical_terminal_balance_lt_1pct": float(electrical_audit["terminal_balance_relative"]) < 1.0e-2,
            "explicit_3d_top_contact_volumetric_model": bool(
                electrical_audit["electrical_model"] == CONTRACT.electrical_model
            ),
            "volumetric_integral_matches_terminal_lt_1e_12": bool(
                float(electrical_audit["volumetric_integral_relative_error"])
                < 1.0e-12
            ),
            "electrical_matrix_symmetric_lt_1e_13": bool(
                float(electrical_audit["matrix_symmetry_relative"]) < 1.0e-13
            ),
            "finite_nonzero_current_and_gradients": bool(
                np.isfinite(evaluated["objective_A"])
                and float(evaluated["objective_A"]) != 0.0
                and np.all(np.isfinite(gradient_direct_nodal))
                and np.linalg.norm(gradient_direct_nodal) > 0.0
                and all(
                    np.all(np.isfinite(native_q_pullback_reporting[component]))
                    for component in "xyz"
                )
            ),
            "Au_bulk_thermopower_is_active": bool(
                float(electrical_audit["S_Au_V_K"])
                == CONTRACT.au_bulk_seebeck_V_K
                and electrical_audit["Au_thermopower_model"]
                == CONTRACT.au_thermopower_discretization
            ),
            "unknown_Au_Ta_interface_thermopower_not_invented": bool(
                float(electrical_audit["S_Au_Ta_contact_V_K"])
                == CONTRACT.au_tairte4_interfacial_seebeck_V_K
                == 0.0
            ),
        }
        raw_output = output / "gray_q_cuda_pde_pullback.npz"
        np.savez_compressed(
            raw_output,
            rho_nodal=rho_nodal,
            rho_cell=rho_cell,
            source_power_W=source_power,
            temperature_K=np.asarray(evaluated["temperature"]),
            ta_temperature_K=np.asarray(evaluated["ta_temperature"]),
            au_temperature_K=np.asarray(evaluated["au_temperature"]),
            volumetric_current_density_A_m3=np.asarray(
                evaluated["volumetric_current_density_A_m3"]
            ),
            weighting_potential_all_electrical_nodes=np.asarray(
                evaluated["weighting"]
            ),
            gradient_direct_cell_A=gradient_direct_cell,
            gradient_direct_nodal_A=gradient_direct_nodal,
            **{
                f"native_Q{component}_sensitivity_A_m3_W_reporting": value
                for component, value in native_q_pullback_reporting.items()
            },
        )
        passed = all(gates.values())
        result = {
            "status": STATUS if passed else "FAILED_LUMERICAL_4UM_GRAY_Q_CUSTOM_CUDA_PDE",
            "passed": passed,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "scope": (
                "completed nonuniform Lumerical forward -> all native component-Yee Q "
                "exact-overlap remap -> custom CUDA thermal/electrical forward+adjoint"
            ),
            "density_state": density_state_audit(rho_nodal),
            "polarization": forward.get("polarization"),
            "source_scale_to_reporting_power": scale,
            "native_Q_power_W_raw": float(np.sum(source_power_raw)),
            "native_Q_json_power_relative_error": raw_power_error,
            "mapped_source_power_W_reporting": float(np.sum(source_power)),
            "current_A": float(evaluated["objective_A"]),
            "current_nA": 1.0e9 * float(evaluated["objective_A"]),
            "gradient_direct_cell_L2_A": float(np.linalg.norm(gradient_direct_cell)),
            "gradient_direct_nodal_L2_A": float(np.linalg.norm(gradient_direct_nodal)),
            "mapping": mapping,
            "mapping_transpose": transpose,
            "native_vs_thermal_adjoint_contraction": contraction_audit,
            "thermal": thermal_audit,
            "thermal_adjoint": thermal_adjoint_audit,
            "electrical": electrical_audit,
            "thermoelectric_current_components_A": {
                "TaIrTe4": float(
                    electrical_audit["tairte4_thermoelectric_current_A"]
                ),
                "Au": float(electrical_audit["au_thermoelectric_current_A"]),
                "sum": float(evaluated["objective_A"]),
            },
            "Au_transport_contract": {
                "sigma_S_m": CONTRACT.au_bulk_electrical_conductivity_S_m,
                "k_W_mK": CONTRACT.au_bulk_thermal_conductivity_W_mK,
                "S_V_K": CONTRACT.au_bulk_seebeck_V_K,
                "Au_Ta_interface_S_V_K": (
                    CONTRACT.au_tairte4_interfacial_seebeck_V_K
                ),
                "parameter_scope": CONTRACT.au_transport_parameter_scope,
                "thermopower_discretization": (
                    CONTRACT.au_thermopower_discretization
                ),
            },
            "electrical_adjoint": electrical_adjoint_audit,
            "gates": gates,
            "inputs": {
                "forward_result": _artifact(forward_path),
                "forward_FSP": _artifact(fsp_path),
                "raw_NPZ": _artifact(raw_path),
                "density_file": _artifact(density_path),
                "forward_binding": forward_binding,
            },
            "raw_output": _artifact(raw_output),
            "Maxwell_solves": 0,
            "custom_CUDA_thermal_solves": {"forward": 1, "adjoint": 1},
            "custom_CUDA_electrical_solves": {"forward": 1, "adjoint": 1},
            "Lumerical_HEAT_or_CHARGE_solves": 0,
            "optimizer_iterations": 0,
            "material_equality_filter": False,
            "gray_design_absorption_discarded": False,
            "wall_s": time.monotonic() - started,
        }
    except Exception as error:
        result.update(
            error=f"{type(error).__name__}: {error}",
            traceback=traceback.format_exc(),
            wall_s=time.monotonic() - started,
        )
    result_path.write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(json.dumps(result, indent=2, default=str))
    return 0 if result.get("passed", False) else 1


if __name__ == "__main__":
    raise SystemExit(main())
