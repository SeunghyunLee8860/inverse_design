"""Fail-closed certificate chain required by every production entry point."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.material_fraction import (
    audit as material_fraction_audit,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.validation_provenance import (
    load_current_source_calibration,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_4um_model import (
    ABSORPTION_LOSS_BASIS,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.objective import (
    PTE_CURRENT_SIGN_CONVENTION,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.mesh_variants import (
    FULL_DOMAIN_Z,
    edges_sha256,
    variant_edges,
)


HERE = Path(__file__).resolve().parent
LUMERICAL_DENSITY_ROUTE_STATUS = (
    "RTX_EA_EB_FOUR_LATENT_DIRECTIONS_EACH_ADFD_PASSED; BLOCKED_PENDING_B200_"
    "ENDPOINT_BANDWIDTH_RESONANCE_OPTIMIZER_AND_MESH_GATES"
)
DEVICE_CERTIFICATE = HERE / "physical_device_contract.json"
SOURCE_CALIBRATION = (
    HERE
    / "results_fdtdx_4um_source_calibration"
    / "fdtdx_4um_source_calibration.json"
)
MESH_CERTIFICATE = (
    HERE
    / "results_4um_shared_linear_mesh_convergence"
    / "MESH_CONVERGENCE_SUMMARY.json"
)
GRADIENT_CERTIFICATE = (
    HERE
    / "results_4um_shared_linear_combined_adfd"
    / "COMBINED_ADFD_SUMMARY.json"
)
MESH_STATUS = "VALIDATED_SHARED_LINEAR_FULL_MESH_CONVERGENCE"
GRADIENT_STATUS = "VALIDATED_SHARED_LINEAR_COMBINED_ADFD"
DEVICE_STATUS = "VALIDATED_AU_TAIRTE4_PHYSICAL_DEVICE_CONTRACT"
REQUIRED_DEVICE_CONFIRMATIONS = (
    "flake_geometry_and_thickness_from_target_device",
    "crystal_axis_angle_and_solver_mapping_confirmed",
    "terminal_shapes_and_locations_confirmed",
    "floating_au_role_and_direct_contact_confirmed",
    "sio2_and_si_stack_confirmed",
    "beam_wavelength_power_waist_center_and_incidence_confirmed",
    "au_tairte4_thermal_contact_scenario_accepted",
    "au_tairte4_electrical_contact_scenario_accepted",
    "electrical_void_floor_sensitivity_passed",
    "signed_output_terminal_and_current_convention_confirmed",
)
REQUIRED_MESH_COVERAGE = (
    "optical_z_full_domain",
    "optical_xy",
    "time_window_stationarity",
    "q_closed_flux_closure",
    "thermal_mesh",
    "electrical_mesh",
)
CURRENT_IMPLEMENTATIONS = {
    "fdtdx_4um_model.py": HERE / "fdtdx_4um_model.py",
    "multiphysics_4um.py": HERE / "multiphysics_4um.py",
    "combined_4um.py": HERE / "combined_4um.py",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path: Path) -> tuple[dict[str, object] | None, str | None]:
    if not path.is_file():
        return None, f"missing certificate: {path}"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return None, f"unreadable certificate {path}: {error}"
    if not isinstance(value, dict):
        return None, f"certificate root is not an object: {path}"
    return value, None


def readiness_audit(
    mesh_path: Path = MESH_CERTIFICATE,
    gradient_path: Path = GRADIENT_CERTIFICATE,
    device_path: Path = DEVICE_CERTIFICATE,
    calibration_path: Path = SOURCE_CALIBRATION,
) -> dict[str, object]:
    mesh_path = Path(mesh_path)
    gradient_path = Path(gradient_path)
    device_path = Path(device_path)
    calibration_path = Path(calibration_path)
    device, device_error = _read(device_path)
    mesh, mesh_error = _read(mesh_path)
    gradient, gradient_error = _read(gradient_path)
    try:
        load_current_source_calibration(calibration_path)
        calibration_error = None
    except (OSError, ValueError, KeyError, json.JSONDecodeError, RuntimeError) as error:
        calibration_error = str(error)
    material = material_fraction_audit()

    checks: dict[str, bool] = {
        # The certificates below describe the historical shared-gray/FDTDX
        # implementation.  They cannot promote the new Lumerical dispersive
        # density route even if every historical certificate is complete.
        "lumerical_dispersive_density_route_validated": False,
        "device_certificate_readable": device_error is None,
        "source_calibration_current": calibration_error is None,
        "mesh_certificate_readable": mesh_error is None,
        "gradient_certificate_readable": gradient_error is None,
    }
    errors = [
        error
        for error in (device_error, calibration_error, mesh_error, gradient_error)
        if error is not None
    ]
    device_sha256 = None
    if device is not None:
        confirmations = device.get("confirmations", {})
        device_sha256 = sha256(device_path)
        checks.update(
            device_status=device.get("status") == DEVICE_STATUS,
            device_confirmations=bool(
                isinstance(confirmations, dict)
                and all(
                    confirmations.get(name) is True
                    for name in REQUIRED_DEVICE_CONFIRMATIONS
                )
            ),
        )
    else:
        checks.update(device_status=False, device_confirmations=False)

    calibration_sha256 = (
        sha256(calibration_path) if calibration_error is None else None
    )
    mesh_sha256 = None
    selected_numerical_contract = None
    selected_source_calibration = None
    if mesh is not None:
        coverage = mesh.get("coverage", {})
        implementation_sha256 = mesh.get("implementation_sha256", {})
        selected = mesh.get("selected_numerical_contract")
        optical = selected.get("optical", {}) if isinstance(selected, dict) else {}
        selected_calibration = (
            selected.get("source_calibration", {})
            if isinstance(selected, dict)
            else {}
        )
        factor = optical.get("mesh_factor") if isinstance(optical, dict) else None
        try:
            factor_valid = bool(int(factor) == factor and int(factor) >= 1)
        except (TypeError, ValueError):
            factor_valid = False
        expected_grid_sha = (
            edges_sha256(variant_edges(int(factor), FULL_DOMAIN_Z))
            if factor_valid
            else None
        )
        total_periods = optical.get("total_periods") if isinstance(optical, dict) else None
        window_periods = optical.get("window_periods") if isinstance(optical, dict) else None
        courant_factor = optical.get("courant_factor") if isinstance(optical, dict) else None
        try:
            time_valid = bool(
                int(total_periods) == total_periods
                and int(window_periods) == window_periods
                and int(total_periods) > 2 * int(window_periods) > 0
                and 0.0 < float(courant_factor) <= 1.0
            )
        except (TypeError, ValueError):
            time_valid = False
        calibration_cases = (
            selected_calibration.get("cases", [])
            if isinstance(selected_calibration, dict)
            else []
        )
        calibration_by_pol = {
            str(case.get("polarization")): case
            for case in calibration_cases
            if isinstance(case, dict)
        }
        try:
            selected_powers = [
                float(calibration_by_pol[pol]["incident_power_W"])
                for pol in ("Ea", "Eb")
            ]
            selected_common = float(
                selected_calibration["common_reference_incident_power_W"]
            )
            selected_mismatch = abs(selected_powers[0] - selected_powers[1]) / max(
                selected_powers
            )
            selected_calibration_valid = bool(
                len(calibration_cases) == 2
                and set(calibration_by_pol) == {"Ea", "Eb"}
                and all(power > 0.0 for power in selected_powers)
                and selected_common > 0.0
                and selected_mismatch < 5.0e-3
                and selected_calibration.get("grid_edges_sha256")
                == optical.get("grid_edges_sha256")
                and selected_calibration.get("courant_factor") == courant_factor
                and selected_calibration.get("total_periods") == total_periods
                and selected_calibration.get("window_periods") == window_periods
            )
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            selected_calibration_valid = False
        mesh_sha256 = sha256(mesh_path)
        checks.update(
            mesh_status=mesh.get("status") == MESH_STATUS,
            mesh_material_fraction=mesh.get("au_material_fraction") == material,
            mesh_absorption_loss_basis=(
                mesh.get("absorption_loss_basis") == ABSORPTION_LOSS_BASIS
            ),
            mesh_current_sign_convention=(
                mesh.get("pte_current_sign_convention")
                == PTE_CURRENT_SIGN_CONVENTION
            ),
            mesh_coverage=bool(
                isinstance(coverage, dict)
                and all(coverage.get(name) is True for name in REQUIRED_MESH_COVERAGE)
            ),
            mesh_uses_device_certificate=(
                device_sha256 is not None
                and mesh.get("device_certificate_sha256") == device_sha256
            ),
            mesh_uses_source_calibration=(
                calibration_sha256 is not None
                and mesh.get("source_calibration_sha256") == calibration_sha256
            ),
            mesh_uses_current_implementations=bool(
                isinstance(implementation_sha256, dict)
                and all(
                    implementation_sha256.get(name) == sha256(path)
                    for name, path in CURRENT_IMPLEMENTATIONS.items()
                )
            ),
            mesh_selected_numerical_contract=isinstance(selected, dict),
            mesh_selected_optical_full_z=bool(
                factor_valid
                and optical.get("mesh_mode") == FULL_DOMAIN_Z
                and optical.get("grid_edges_sha256") == expected_grid_sha
            ),
            mesh_selected_time_contract=time_valid,
            mesh_selected_source_calibration=selected_calibration_valid,
        )
        if all(
            checks[name]
            for name in (
                "mesh_selected_numerical_contract",
                "mesh_selected_optical_full_z",
                "mesh_selected_time_contract",
                "mesh_selected_source_calibration",
            )
        ):
            selected_numerical_contract = selected
            selected_source_calibration = selected_calibration
    else:
        checks.update(
            mesh_status=False,
            mesh_material_fraction=False,
            mesh_absorption_loss_basis=False,
            mesh_current_sign_convention=False,
            mesh_coverage=False,
            mesh_uses_device_certificate=False,
            mesh_uses_source_calibration=False,
            mesh_uses_current_implementations=False,
            mesh_selected_numerical_contract=False,
            mesh_selected_optical_full_z=False,
            mesh_selected_time_contract=False,
            mesh_selected_source_calibration=False,
        )

    if gradient is not None:
        checks.update(
            gradient_status=gradient.get("status") == GRADIENT_STATUS,
            gradient_material_fraction=(
                gradient.get("au_material_fraction") == material
            ),
            gradient_absorption_loss_basis=(
                gradient.get("absorption_loss_basis") == ABSORPTION_LOSS_BASIS
            ),
            gradient_current_sign_convention=(
                gradient.get("pte_current_sign_convention")
                == PTE_CURRENT_SIGN_CONVENTION
            ),
            gradient_uses_mesh_certificate=(
                mesh_sha256 is not None
                and gradient.get("mesh_certificate_sha256") == mesh_sha256
            ),
            gradient_multidirection_gate=bool(
                int(gradient.get("direction_count", 0)) >= 4
                and float(gradient.get("maximum_normalized_error", float("inf")))
                < 0.01
            ),
        )
    else:
        checks.update(
            gradient_status=False,
            gradient_material_fraction=False,
            gradient_absorption_loss_basis=False,
            gradient_current_sign_convention=False,
            gradient_uses_mesh_certificate=False,
            gradient_multidirection_gate=False,
        )

    failed = [name for name, passed in checks.items() if not passed]
    return {
        "ready": not failed,
        "checks": checks,
        "failed_checks": failed,
        "errors": errors,
        "device_certificate": str(device_path),
        "device_certificate_sha256": device_sha256,
        "source_calibration": str(calibration_path),
        "source_calibration_sha256": calibration_sha256,
        "mesh_certificate": str(mesh_path),
        "mesh_certificate_sha256": mesh_sha256,
        "gradient_certificate": str(gradient_path),
        "required_mesh_coverage": list(REQUIRED_MESH_COVERAGE),
        "required_device_confirmations": list(REQUIRED_DEVICE_CONFIRMATIONS),
        "lumerical_density_route_status": LUMERICAL_DENSITY_ROUTE_STATUS,
        "au_material_fraction": material,
        "absorption_loss_basis": ABSORPTION_LOSS_BASIS,
        "pte_current_sign_convention": PTE_CURRENT_SIGN_CONVENTION,
        "selected_numerical_contract": selected_numerical_contract,
        "selected_source_calibration": selected_source_calibration,
    }


def require_production_readiness() -> dict[str, object]:
    result = readiness_audit()
    if not result["ready"]:
        raise RuntimeError(
            "production inverse design is blocked by certificate readiness:\n"
            + json.dumps(result, indent=2)
        )
    return result


def calibrated_source_scales(
    readiness: dict[str, object], target_incident_power_W: float
) -> dict[str, float]:
    """Return polarization-specific power scales for the selected optical grid."""
    if readiness.get("ready") is not True:
        raise RuntimeError("source scales require a passing production-readiness audit")
    if not float(target_incident_power_W) > 0.0:
        raise ValueError("target incident power must be positive")
    calibration = readiness.get("selected_source_calibration")
    if not isinstance(calibration, dict):
        raise RuntimeError("readiness result has no selected source calibration")
    cases = calibration.get("cases")
    if not isinstance(cases, list):
        raise RuntimeError("selected source calibration has no case list")
    incident_by_pol = {
        str(case.get("polarization")): float(case["incident_power_W"])
        for case in cases
        if isinstance(case, dict)
    }
    if set(incident_by_pol) != {"Ea", "Eb"}:
        raise RuntimeError("selected source calibration must contain only Ea and Eb")
    return {
        pol: float(target_incident_power_W) / incident_by_pol[pol]
        for pol in ("Ea", "Eb")
    }
