"""Reusable discrete Lumerical Maxwell-adjoint pieces for the 4-um Au route."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy import sparse

from photothermal_pte.finite_inverse_design.native_yee_q import EPS0
from photothermal_pte.finite_inverse_design.run_v261_large_background_mixed_optical_adfd import (
    component_volumes,
)
from photothermal_pte.finite_inverse_design.yee_material_jacobian import (
    SparseYeeMaterialJacobian,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.contract import (
    CONTRACT,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.lumerical_4um_density import (
    canonical_density_nodes,
    density_state_sha256,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.lumerical_4um_yee_jacobian import (
    validate_index_detail,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.validation_provenance import (
    sha256,
)


COMPONENTS = "xyz"
FREQUENCY_HZ = 299_792_458.0 / CONTRACT.wavelength_m


def _artifact(path: Path, expected_sha256: str, label: str) -> Path:
    value = Path(path).expanduser().resolve()
    if not value.is_file():
        raise FileNotFoundError(f"missing {label}: {value}")
    actual = sha256(value)
    if actual != expected_sha256:
        raise RuntimeError(
            f"{label} SHA256 mismatch: expected {expected_sha256}, got {actual}"
        )
    return value


def load_component_yee_jacobian(
    directory: Path, projected_density: np.ndarray
) -> tuple[SparseYeeMaterialJacobian, dict[str, Any]]:
    """Load and re-audit one script-26 sparse material Jacobian."""

    root = Path(directory).expanduser().resolve()
    result_path = root / "component_yee_jacobian_result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("status") != "PASSED_LUMERICAL_4UM_COMPONENT_YEE_JACOBIAN":
        raise RuntimeError("component-Yee Jacobian certificate did not pass")
    if result.get("passed") is not True or not all(result.get("gates", {}).values()):
        raise RuntimeError("component-Yee Jacobian gates are incomplete")
    rho = canonical_density_nodes(projected_density)
    if (
        result.get("density_state", {}).get("density_state_sha256")
        != density_state_sha256(rho)
    ):
        raise RuntimeError("component-Yee Jacobian density state differs")
    coordinate_record = result["artifacts"]["coordinates_and_density"]
    coordinate_path = _artifact(
        Path(coordinate_record["path"]),
        str(coordinate_record["sha256"]),
        "component-Yee coordinate artifact",
    )
    with np.load(coordinate_path, allow_pickle=False) as coordinates:
        stored_rho = np.asarray(coordinates["projected_density_nodal"], float)
        if not np.array_equal(stored_rho, rho):
            raise RuntimeError("component-Yee coordinate artifact density differs")
        coordinate_arrays = {
            component: tuple(
                np.asarray(coordinates[f"{component}_{axis}_m"], float)
                for axis in COMPONENTS
            )
            for component in COMPONENTS
        }
    component_shape = {
        component: tuple(result["construction"]["baseline"]["component_shapes"][component])
        for component in COMPONENTS
    }
    matrices = {}
    matrix_artifacts = {}
    for component in COMPONENTS:
        record = result["artifacts"]["component_J"][component]
        path = _artifact(
            Path(record["path"]), str(record["sha256"]), f"J_{component}"
        )
        matrices[component] = sparse.load_npz(path)
        matrix_artifacts[component] = {
            "path": str(path),
            "sha256": sha256(path),
        }
    operator = SparseYeeMaterialJacobian(
        density_shape=rho.shape,
        component_shapes=component_shape,
        matrices=matrices,
    )
    # Re-run the real-design transpose contract after deserialization.
    rng = np.random.default_rng(4_034_608_24)
    direction = rng.normal(size=rho.shape)
    cotangent = {
        component: rng.normal(size=component_shape[component])
        + 1j * rng.normal(size=component_shape[component])
        for component in COMPONENTS
    }
    tangent = operator.jvp(direction)
    left = float(
        np.real(
            sum(
                np.sum(cotangent[component] * tangent[component])
                for component in COMPONENTS
            )
        )
    )
    right = float(np.sum(direction * operator.vjp(cotangent)))
    transpose_error = abs(left - right) / max(
        abs(left), abs(right), np.finfo(float).tiny
    )
    if transpose_error >= 1.0e-12:
        raise RuntimeError(f"reloaded component-J transpose error {transpose_error}")
    return operator, {
        "result": {"path": str(result_path), "sha256": sha256(result_path)},
        "coordinates": {
            "path": str(coordinate_path),
            "sha256": sha256(coordinate_path),
        },
        "component_J": matrix_artifacts,
        "component_coordinates_m": coordinate_arrays,
        "baseline_index_detail_audit": result["construction"]["baseline"],
        "fresh_transpose_relative_error": transpose_error,
        "source_forward_result": result["inputs"]["forward_result_json"],
        "source_forward_project": result["inputs"]["forward_project"],
    }


def index_detail_from_raw(raw: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Reconstruct script-26's index-detail audit from a script-25 NPZ."""

    detail: dict[str, np.ndarray] = {
        "x": np.asarray(raw["Qy_x_m"], float),
        "y": np.asarray(raw["Qx_y_m"], float),
        "z": np.asarray(raw["Qx_z_m"], float),
        "x_offset": np.asarray(raw["Qx_x_m"], float),
        "y_offset": np.asarray(raw["Qy_y_m"], float),
        "z_offset": np.asarray(raw["Qz_z_m"], float),
        "frequency_hz": np.asarray([FREQUENCY_HZ], float),
    }
    for component in COMPONENTS:
        detail[f"epsilon_{component}"] = np.asarray(
            raw[f"epsilon_{component}"], np.complex128
        )
    validate_index_detail(detail)
    return detail


def validate_raw_against_jacobian(
    raw: Mapping[str, np.ndarray], operator_meta: Mapping[str, Any]
) -> dict[str, Any]:
    """Prove a completed forward has the Jacobian's exact Yee state."""

    detail = index_detail_from_raw(raw)
    audit = validate_index_detail(detail)
    baseline = operator_meta["baseline_index_detail_audit"]
    coordinate_error = 0.0
    for component in COMPONENTS:
        expected = operator_meta["component_coordinates_m"][component]
        actual = tuple(np.asarray(raw[f"Q{component}_{axis}_m"], float) for axis in COMPONENTS)
        coordinate_error = max(
            coordinate_error,
            *(float(np.max(np.abs(left - right))) for left, right in zip(expected, actual, strict=True)),
        )
    gates = {
        "component_shapes_match": audit["component_shapes"] == baseline["component_shapes"],
        "epsilon_sha256_matches": audit["epsilon_sha256"] == baseline["epsilon_sha256"],
        "coordinate_arrays_match_lt_2e_18": coordinate_error < 2.0e-18,
        "frequency_matches_rel_lt_1e_12": abs(
            float(audit["frequency_hz"]) - float(baseline["frequency_hz"])
        )
        / max(abs(float(baseline["frequency_hz"])), 1.0)
        < 1.0e-12,
    }
    return {
        "passed": all(gates.values()),
        "gates": gates,
        # The monitor raw NPZ and layout-mode index_detail are separate
        # Lumerical API readbacks.  Their coordinate values can differ by a
        # few binary ulps even when they describe the same frozen Yee grid;
        # retain the byte hashes as diagnostics but gate on the complete six
        # coordinate arrays above at a sub-attometre tolerance.
        "coordinate_sha256_identical": (
            audit["coordinate_sha256"] == baseline["coordinate_sha256"]
        ),
        "coordinate_max_abs_error_m": coordinate_error,
        "forward_index_detail_audit": audit,
        "jacobian_baseline_index_detail_audit": baseline,
    }


def material_jacobian_reuse_audit(
    raw_binding: Mapping[str, Any],
    *,
    source_raw_sha256: str,
    target_raw_sha256: str,
    source_polarization: str,
    target_polarization: str,
) -> dict[str, Any]:
    """Authorize material-J reuse without requiring polarization-dependent E/Q."""

    polarizations_valid = source_polarization in ("Ea", "Eb") and (
        target_polarization in ("Ea", "Eb")
    )
    gates = {
        "polarizations_are_Ea_or_Eb": polarizations_valid,
        "target_epsilon_grid_frequency_match_jacobian": bool(
            raw_binding.get("passed")
        ),
    }
    return {
        "passed": all(gates.values()),
        "gates": gates,
        "basis": (
            "same projected-density state plus exact epsilon SHA, component-Yee "
            "shapes, sub-attometre coordinates, and frequency"
        ),
        "polarization_dependent_E_and_Q_required_to_match": False,
        "source_polarization": source_polarization,
        "target_polarization": target_polarization,
        "forward_raw_SHA_identical_to_jacobian_source": (
            source_raw_sha256 == target_raw_sha256
        ),
    }


def reconstruct_fieldregion_only_cw(
    electric_first: np.ndarray, electric_average: np.ndarray
) -> tuple[np.ndarray, dict[str, float | bool | str]]:
    """Remove the zero-amplitude mesh-anchor source from Lumerical CW norm."""

    first = np.asarray(electric_first, np.complex128)
    average = np.asarray(electric_average, np.complex128)
    first_over_average = np.vdot(average, first) / np.vdot(average, average)
    residual = float(
        np.linalg.norm(first - first_over_average * average)
        / max(float(np.linalg.norm(first)), np.finfo(float).tiny)
    )
    fieldregion_over_first = 2.0 * first_over_average - 1.0
    if abs(fieldregion_over_first) <= np.finfo(float).tiny:
        raise RuntimeError("invalid FieldRegion-only source-spectrum ratio")
    return first / fieldregion_over_first, {
        "method": (
            "FieldRegion-only CW field reconstructed from official cwnorm(1) "
            "and cwnorm(2) while the zero-amplitude Gaussian anchors the mesh"
        ),
        "first_over_average_real": float(np.real(first_over_average)),
        "first_over_average_imag": float(np.imag(first_over_average)),
        "fieldregion_over_first_real": float(np.real(fieldregion_over_first)),
        "fieldregion_over_first_imag": float(np.imag(fieldregion_over_first)),
        "two_normalization_state_spatial_residual": residual,
        "uses_finite_difference_fit": False,
        "empirical_gradient_rescaling": False,
    }


def native_adjoint_source(
    forward_electric: np.ndarray,
    epsilon: Mapping[str, np.ndarray],
    native_q_sensitivity_A_m3_W_raw: Mapping[str, np.ndarray],
) -> np.ndarray:
    """Return dI/dE* on the native component-Yee arrays."""

    field = np.asarray(forward_electric, np.complex128)
    source = np.zeros_like(field)
    omega = 2.0 * np.pi * FREQUENCY_HZ
    for index, component in enumerate(COMPONENTS):
        permittivity = np.asarray(epsilon[component], np.complex128)
        weight = np.asarray(native_q_sensitivity_A_m3_W_raw[component], float)
        if permittivity.shape != field.shape[:3] or weight.shape != field.shape[:3]:
            raise ValueError(f"{component} E/epsilon/Q-sensitivity shape mismatch")
        source[..., 0, index] = (
            0.5
            * EPS0
            * omega
            * np.imag(permittivity)
            * weight
            * field[..., 0, index]
        )
    if not np.all(np.isfinite(source)) or np.linalg.norm(source) == 0.0:
        raise RuntimeError("native Maxwell-adjoint source is zero or non-finite")
    return source


def optical_density_gradient(
    operator: SparseYeeMaterialJacobian,
    *,
    forward_electric: np.ndarray,
    adjoint_electric: np.ndarray,
    epsilon: Mapping[str, np.ndarray],
    native_q_sensitivity_A_m3_W_raw: Mapping[str, np.ndarray],
    grid: Mapping[str, np.ndarray],
    profile_scale: float,
    fieldregion_base_amplitude: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return field-mediated plus explicit-loss dI/d projected density."""

    forward = np.asarray(forward_electric, np.complex128)
    adjoint = np.asarray(adjoint_electric, np.complex128)
    if forward.shape != adjoint.shape:
        raise ValueError("forward and adjoint electric arrays differ")
    if not np.isfinite(profile_scale) or profile_scale <= 0.0:
        raise ValueError("invalid FieldRegion profile scale")
    if not np.isfinite(fieldregion_base_amplitude) or fieldregion_base_amplitude == 0.0:
        raise ValueError("invalid FieldRegion base amplitude")
    volumes = component_volumes(dict(grid))
    indirect_cotangent = {}
    direct_cotangent = {}
    omega = 2.0 * np.pi * FREQUENCY_HZ
    component_records = {}
    for index, component in enumerate(COMPONENTS):
        field = forward[..., 0, index]
        adjoint_field = adjoint[..., 0, index]
        weight = np.asarray(native_q_sensitivity_A_m3_W_raw[component], float)
        indirect_cotangent[component] = (
            (2.0 * EPS0 / fieldregion_base_amplitude)
            * volumes[index]
            * field
            * (adjoint_field * profile_scale)
        )
        direct_cotangent[component] = (
            -1j
            * 0.5
            * EPS0
            * omega
            * weight
            * np.abs(field) ** 2
        )
        component_records[component] = {
            "indirect_cotangent_L2": float(np.linalg.norm(indirect_cotangent[component])),
            "direct_cotangent_L2": float(np.linalg.norm(direct_cotangent[component])),
            "epsilon_imaginary_range": [
                float(np.min(np.imag(epsilon[component]))),
                float(np.max(np.imag(epsilon[component]))),
            ],
        }
    indirect = operator.vjp(indirect_cotangent)
    direct = operator.vjp(direct_cotangent)
    total = indirect + direct
    return total, {
        "indirect_gradient": indirect,
        "direct_loss_gradient": direct,
        "component": component_records,
        "gradient_norms_A": {
            "indirect": float(np.linalg.norm(indirect)),
            "direct_loss": float(np.linalg.norm(direct)),
            "total_optical": float(np.linalg.norm(total)),
        },
        "empirical_gradient_rescaling": False,
    }
