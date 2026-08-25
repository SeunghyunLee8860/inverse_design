"""Shared official-Pabs provenance and custom CUDA downstream execution."""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.contract import (
    CONTRACT,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.lumerical_4um_multiphysics_comparison import (
    map_lumerical_component_yee_material_q_to_thermal,
    map_lumerical_official_pabs_to_thermal,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.multiphysics_4um import (
    N_TA,
    build_electrical_system,
    build_thermal_state,
    current_integrand,
    solve_electrical,
    solve_thermal,
    tairte4_temperature,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.validation_provenance import (
    sha256,
)


REQUIRED_PABS_KEYS = {
    "Pabs_W_m3",
    "Pabs_index_x",
    "Pabs_x_m",
    "Pabs_y_m",
    "Pabs_z_m",
}


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path} does not contain a JSON object")
    return payload


def validate_official_pabs_npz(
    pabs_path: Path,
    *,
    source_result_json: Path,
    source_raw_npz: Path,
) -> dict[str, str]:
    """Validate direct raw arrays or a script-29 companion and its hashes."""

    pabs = Path(pabs_path).resolve()
    result_json = Path(source_result_json).resolve()
    raw_npz = Path(source_raw_npz).resolve()
    if not pabs.is_file():
        raise FileNotFoundError(pabs)
    with np.load(pabs, allow_pickle=False) as probe:
        if not REQUIRED_PABS_KEYS.issubset(probe.files):
            raise RuntimeError(
                "Pabs NPZ lacks official arrays; run "
                "29_extract_lumerical_4um_official_pabs.py"
            )
    pabs_sha = sha256(pabs)
    if pabs == raw_npz:
        return {"path": str(pabs), "sha256": pabs_sha, "kind": "direct_raw_npz"}

    audit_path = pabs.with_suffix(".json")
    if not audit_path.is_file():
        raise FileNotFoundError(f"official Pabs companion audit is missing: {audit_path}")
    audit = _load_json(audit_path)
    if audit.get("status") != "EXTRACTED_LUMERICAL_OFFICIAL_PABS_INDEX_X":
        raise RuntimeError("official Pabs companion audit did not pass")
    output = audit.get("output_npz")
    source = audit.get("source_result_json")
    if not isinstance(output, dict) or not isinstance(source, dict):
        raise RuntimeError("official Pabs companion provenance is incomplete")
    if Path(output.get("path", "")).resolve() != pabs:
        raise RuntimeError("official Pabs companion path does not match its audit")
    if output.get("sha256") != pabs_sha:
        raise RuntimeError("official Pabs companion SHA256 does not match")
    if Path(source.get("path", "")).resolve() != result_json:
        raise RuntimeError("official Pabs companion source JSON path does not match")
    if source.get("sha256") != sha256(result_json):
        raise RuntimeError("official Pabs companion source JSON SHA256 does not match")
    return {
        "path": str(pabs),
        "sha256": pabs_sha,
        "kind": "script_29_companion_npz",
        "audit_json": str(audit_path),
        "audit_json_sha256": sha256(audit_path),
    }


def material_index_x_from_result(result: dict[str, Any]) -> dict[str, complex]:
    fit = result["material_fit_readback"]["materials"]
    return {
        material: complex(
            np.sqrt(
                complex(
                    float(
                        fit[material]["axes"]["x"]["fitted_epsilon_at_4um"][
                            "real"
                        ]
                    ),
                    float(
                        fit[material]["axes"]["x"]["fitted_epsilon_at_4um"][
                            "imag"
                        ]
                    ),
                )
            )
        )
        for material in ("Au", "TaIrTe4", "SiO2")
    }


def material_fitted_epsilon_from_result(
    result: dict[str, Any],
) -> dict[str, dict[str, complex]]:
    """Return the exact fitted epsilon saved by each component index monitor."""

    fit = result["material_fit_readback"]["materials"]
    targets: dict[str, dict[str, complex]] = {}
    for material in ("Au", "TaIrTe4", "SiO2"):
        if material not in fit:
            continue
        targets[material] = {}
        for component in "xyz":
            value = fit[material]["axes"][component]["fitted_epsilon_at_4um"]
            targets[material][component] = complex(
                float(value["real"]),
                float(value["imag"]),
            )
    return targets


def run_component_yee_downstream(
    result: dict[str, Any],
    raw_path: Path,
    rho: np.ndarray,
    cuda_device: int,
    case: str,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Map collocated component-Yee Q and solve the custom CUDA PDEs."""

    state = build_thermal_state(rho)
    normalization = result["reporting_normalization"]
    incident = float(normalization["source_only_incident_power_W_raw"])
    reporting = float(normalization["target_reporting_incident_power_W"])
    scale = reporting / incident
    with np.load(raw_path, allow_pickle=False) as raw:
        source_power, mapping = (
            map_lumerical_component_yee_material_q_to_thermal(
                raw,
                state.edges,
                scale,
                case=case,
                material_fitted_epsilon=material_fitted_epsilon_from_result(
                    result
                ),
            )
        )
    expected_power = float(result["P_Q_native_W_raw"]) * scale
    spatial_vs_json = abs(mapping["native_total_power_W"] - expected_power) / max(
        abs(expected_power), np.finfo(float).tiny
    )
    source_norm = max(float(np.linalg.norm(source_power)), np.finfo(float).tiny)
    source_mirror = {
        "x_relative": float(
            np.linalg.norm(source_power - source_power[::-1, :, :]) / source_norm
        ),
        "y_relative": float(
            np.linalg.norm(source_power - source_power[:, ::-1, :]) / source_norm
        ),
    }
    start = time.perf_counter()
    temperature, thermal_audit = solve_thermal(state, source_power, cuda_device)
    ta_temperature = tairte4_temperature(state, temperature)
    electrical = build_electrical_system(
        rho, ta_temperature, exact_binary_geometry=True
    )
    psi, current, electrical_audit = solve_electrical(electrical, cuda_device)
    runtime = time.perf_counter() - start
    integrand = current_integrand(ta_temperature, psi)
    integrand_current = float(np.sum(integrand) * CONTRACT.design_pitch_m**2)
    current_absolute_scale = float(
        np.sum(np.abs(integrand)) * CONTRACT.design_pitch_m**2
    )
    current_consistency = abs(integrand_current - current) / max(
        current_absolute_scale, np.finfo(float).tiny
    )
    current_cancellation = abs(current) / max(
        current_absolute_scale, np.finfo(float).tiny
    )
    gates = {
        "mapping_conservation_lt_1e-12": (
            mapping["relative_conservation_error"] < 1.0e-12
        ),
        "native_spatial_Q_matches_json_lt_1e-12": spatial_vs_json < 1.0e-12,
        "component_yee_filter_unassigned_absorption_lt_0p5pct": (
            mapping["unassigned_absorption_relative"] < 5.0e-3
        ),
        "component_yee_mapping_finite_nonnegative": bool(
            mapping["finite_nonnegative"]
        ),
        "thermal_residual_lt_1e-8": thermal_audit["relative_residual"] < 1.0e-8,
        "thermal_energy_balance_lt_1pct": (
            thermal_audit["energy_balance_relative"] < 1.0e-2
        ),
        "electrical_residual_lt_1e-8": (
            electrical_audit["relative_residual"] < 1.0e-8
        ),
        "electrical_terminal_balance_lt_1pct": (
            electrical_audit["terminal_balance_relative"] < 1.0e-2
        ),
        "exact_binary_void_Au_nodes_removed": bool(
            electrical_audit["exact_binary_geometry"]
            and electrical_audit["electrical_void_Au_nodes_removed"]
            and int(electrical_audit["inactive_void_Au_node_count"])
            == int(np.count_nonzero(np.asarray(rho) == 0))
        ),
        "current_integrand_consistency_lt_1e-12": current_consistency < 1.0e-12,
        "finite": bool(
            np.all(np.isfinite(source_power))
            and np.all(np.isfinite(temperature))
            and np.all(np.isfinite(ta_temperature))
            and np.all(np.isfinite(psi))
            and np.all(np.isfinite(integrand))
        ),
    }
    summary = {
        "runtime_s": runtime,
        "source_scale_to_reporting_power": scale,
        "source_power_W": float(np.sum(source_power)),
        "expected_native_Q_at_reporting_power_W": expected_power,
        "native_spatial_Q_vs_json_relative": spatial_vs_json,
        "native_yee_npz": {
            "path": str(raw_path),
            "sha256": sha256(raw_path),
        },
        "mapping": mapping,
        "source_power_mirror_error": source_mirror,
        "Tmax_K": float(np.max(temperature)),
        "TaIrTe4_Tmax_K": float(np.max(ta_temperature)),
        "current_A": current,
        "current_nA": current * 1.0e9,
        "current_from_integrand_A": integrand_current,
        "current_absolute_integrand_scale_A": current_absolute_scale,
        "current_integrand_consistency_relative": current_consistency,
        "current_cancellation_relative": current_cancellation,
        "thermal": thermal_audit,
        "electrical": electrical_audit,
        "gates": gates,
        "all_gates_passed": all(gates.values()),
    }
    arrays = {
        "source_power_W": source_power,
        "temperature_K": temperature,
        "TaIrTe4_temperature_K": ta_temperature,
        "weighting_potential_TaIrTe4": psi[: N_TA * N_TA].reshape(N_TA, N_TA),
        "current_integrand_A_m2": integrand,
    }
    return summary, arrays


def run_official_pabs_downstream(
    result: dict[str, Any],
    pabs_path: Path,
    rho: np.ndarray,
    cuda_device: int,
    case: str,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Map official material-filtered Pabs and solve custom thermal/electrical."""

    state = build_thermal_state(rho)
    normalization = result["reporting_normalization"]
    incident = float(normalization["source_only_incident_power_W_raw"])
    reporting = float(normalization["target_reporting_incident_power_W"])
    scale = reporting / incident
    with np.load(pabs_path, allow_pickle=False) as raw:
        source_power, mapping = map_lumerical_official_pabs_to_thermal(
            raw,
            state.edges,
            scale,
            case=case,
            material_index_x=material_index_x_from_result(result),
        )
    expected_power = float(result["P_Q_pabs_W_raw"]) * scale
    spatial_vs_json = abs(mapping["native_total_power_W"] - expected_power) / max(
        abs(expected_power), np.finfo(float).tiny
    )
    start = time.perf_counter()
    temperature, thermal_audit = solve_thermal(state, source_power, cuda_device)
    ta_temperature = tairte4_temperature(state, temperature)
    electrical = build_electrical_system(
        rho, ta_temperature, exact_binary_geometry=True
    )
    psi, current, electrical_audit = solve_electrical(electrical, cuda_device)
    runtime = time.perf_counter() - start
    integrand = current_integrand(ta_temperature, psi)
    integrand_current = float(np.sum(integrand) * CONTRACT.design_pitch_m**2)
    current_absolute_scale = float(
        np.sum(np.abs(integrand)) * CONTRACT.design_pitch_m**2
    )
    current_consistency = abs(integrand_current - current) / max(
        current_absolute_scale, np.finfo(float).tiny
    )
    current_cancellation = abs(current) / max(
        current_absolute_scale, np.finfo(float).tiny
    )
    gates = {
        "mapping_conservation_lt_1e-12": (
            mapping["relative_conservation_error"] < 1.0e-12
        ),
        "official_spatial_Pabs_matches_json_lt_1e-12": spatial_vs_json < 1.0e-12,
        "official_material_filter_unassigned_absorption_lt_0p5pct": (
            mapping["unassigned_absorption_relative"] < 5.0e-3
        ),
        "official_Pabs_negative_interpolation_artifact_lt_1e-12": (
            mapping["negative_absorption_relative"] < 1.0e-12
        ),
        "thermal_residual_lt_1e-8": thermal_audit["relative_residual"] < 1.0e-8,
        "thermal_energy_balance_lt_1pct": (
            thermal_audit["energy_balance_relative"] < 1.0e-2
        ),
        "electrical_residual_lt_1e-8": (
            electrical_audit["relative_residual"] < 1.0e-8
        ),
        "electrical_terminal_balance_lt_1pct": (
            electrical_audit["terminal_balance_relative"] < 1.0e-2
        ),
        "exact_binary_void_Au_nodes_removed": bool(
            electrical_audit["exact_binary_geometry"]
            and electrical_audit["electrical_void_Au_nodes_removed"]
            and int(electrical_audit["inactive_void_Au_node_count"])
            == int(np.count_nonzero(np.asarray(rho) == 0))
        ),
        "current_integrand_consistency_lt_1e-12": current_consistency < 1.0e-12,
        "finite": bool(
            np.all(np.isfinite(source_power))
            and np.all(np.isfinite(temperature))
            and np.all(np.isfinite(ta_temperature))
            and np.all(np.isfinite(psi))
            and np.all(np.isfinite(integrand))
        ),
    }
    summary = {
        "runtime_s": runtime,
        "source_scale_to_reporting_power": scale,
        "source_power_W": float(np.sum(source_power)),
        "expected_official_Pabs_at_reporting_power_W": expected_power,
        "official_spatial_Pabs_vs_json_relative": spatial_vs_json,
        "official_pabs_npz": {
            "path": str(pabs_path),
            "sha256": sha256(pabs_path),
        },
        "mapping": mapping,
        "Tmax_K": float(np.max(temperature)),
        "TaIrTe4_Tmax_K": float(np.max(ta_temperature)),
        "current_A": current,
        "current_nA": current * 1.0e9,
        "current_from_integrand_A": integrand_current,
        "current_absolute_integrand_scale_A": current_absolute_scale,
        "current_integrand_consistency_relative": current_consistency,
        "current_cancellation_relative": current_cancellation,
        "thermal": thermal_audit,
        "electrical": electrical_audit,
        "gates": gates,
        "all_gates_passed": all(gates.values()),
    }
    arrays = {
        "source_power_W": source_power,
        "temperature_K": temperature,
        "TaIrTe4_temperature_K": ta_temperature,
        "weighting_potential_TaIrTe4": psi[: N_TA * N_TA].reshape(N_TA, N_TA),
        "current_integrand_A_m2": integrand,
    }
    return summary, arrays
