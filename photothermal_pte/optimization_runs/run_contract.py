"""Fail-closed validation for an inverse-design run directory.

This module intentionally has no Lumerical or SciPy dependency.  It verifies
that a run is self-describing before a licensed solver process is launched.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any


RUN_NAME = re.compile(r"^run_[0-9]{3}_[a-z0-9][a-z0-9_]*$")
REQUIRED_DIRECTORIES = ("checkpoints", "manifests", "plots", "results")
REQUIRED_FILES = (
    "README.md",
    "run_config.json",
    "run_optimization.py",
    "STATUS.json",
)
ALLOWED_STATUS = {
    "PLANNED",
    "PREFLIGHT_PASSED",
    "SOURCE_ONLY_GPU_VALIDATED",
    "UNIFORM_COMPLEX_MATERIAL_EQUIVALENCE_VALIDATED",
    "PRODUCTION_CANDIDATE_FORWARD_VALIDATED",
    "PRODUCTION_COMPONENT_YEE_MAPPING_VALIDATED",
    "PRODUCTION_MATERIAL_Q_ATTRIBUTION_VALIDATED",
    "PRODUCTION_3D_THERMAL_Q_DEPOSITION_VALIDATED",
    "PRODUCTION_CUDA_THERMAL_PTE_VALIDATED",
    "PRODUCTION_FIXED_Q_THERMAL_MATERIAL_ADFD_VALIDATED",
    "PRODUCTION_THERMAL_TO_NATIVE_YEE_PULLBACK_VALIDATED",
    "PRODUCTION_COMBINED_PHYSICAL_RHO_ADFD_SMOKE_VALIDATED",
    "PRODUCTION_DESIGN_WINDOW_SELECTED",
    "PRODUCTION_FINITE_FILTER_PROJECTION_VALIDATED",
    "SELECTED_PRODUCTION_OPTICAL_CHAIN_VALIDATED",
    "SELECTED_PRODUCTION_THERMAL_MAPPING_VALIDATED",
    "SELECTED_THERMAL_GRAY_LAW_ADFD_VALIDATED",
    "RUNNING",
    "COMPLETED",
    "FAILED",
    "BLOCKED",
}


class ValidationError(RuntimeError):
    """Raised when a run violates the repository contract."""


@dataclass(frozen=True)
class ValidationResult:
    run_directory: str
    run_id: str
    status: str
    source_commit: str
    repository_artifacts_checked: int
    external_artifacts_checked: int
    external_artifacts_missing: int
    valid: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_directory": self.run_directory,
            "run_id": self.run_id,
            "status": self.status,
            "source_commit": self.source_commit,
            "repository_artifacts_checked": self.repository_artifacts_checked,
            "external_artifacts_checked": self.external_artifacts_checked,
            "external_artifacts_missing": self.external_artifacts_missing,
            "valid": self.valid,
        }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValidationError(f"JSON root must be an object: {path}")
    return value


def _require(mapping: dict[str, Any], key: str, kind: type) -> Any:
    if key not in mapping or not isinstance(mapping[key], kind):
        raise ValidationError(f"missing or invalid {key!r}")
    return mapping[key]


def _positive(value: Any, label: str) -> float:
    if not isinstance(value, (int, float)) or value <= 0:
        raise ValidationError(f"{label} must be positive")
    return float(value)


def _validate_prohibitions(config: dict[str, Any]) -> None:
    prohibitions = _require(config, "prohibitions", dict)
    for key in (
        "q_clipping",
        "q_smoothing",
        "q_gain",
        "q_rescaling",
        "periodic_tiling",
        "gradient_rescaling",
    ):
        if prohibitions.get(key) is not False:
            raise ValidationError(f"prohibition {key} must be explicitly false")


def _validate_physics_v1(config: dict[str, Any]) -> None:
    design = _require(config, "design", dict)
    if design.get("latent_shape") != [81, 81]:
        raise ValidationError("design.latent_shape must remain [81, 81]")
    if design.get("bounds_um") != {"x": [-1.0, 1.0], "y": [-1.0, 1.0]}:
        raise ValidationError("design bounds must remain the finite 2 um ROI")
    if design.get("node_spacing_nm") != 25.0:
        raise ValidationError("design node spacing must remain 25 nm")
    if design.get("filter_radius_nm") != 500.0:
        raise ValidationError("filter radius must remain 500 nm")
    if design.get("periodic_wrap") is not False:
        raise ValidationError("periodic wrapping is forbidden")
    if design.get("projection_eta") != 0.5:
        raise ValidationError("projection eta must remain 0.5")

    optical = _require(config, "optical", dict)
    if optical.get("boundary") != "six_pml" or optical.get("periodic") is not False:
        raise ValidationError("optical boundary must be nonperiodic six-PML")
    if optical.get("illumination") != "normal_incidence_cpu_tfsf":
        raise ValidationError("unreviewed optical illumination change")
    if optical.get("analysis_wavelength_um") != 4.0:
        raise ValidationError("analysis wavelength must remain 4 um")
    if optical.get("incident_intensity_W_m2") != 1.0:
        raise ValidationError("incident intensity must remain 1 W/m2")

    thermal = _require(config, "thermal", dict)
    if thermal.get("tairte4_kappa_W_mK") != [14.4, 3.8, 1.0]:
        raise ValidationError("TaIrTe4 anisotropic kappa changed")
    if thermal.get("G_tairte4_air_W_m2K") != 1.0:
        raise ValidationError("TaIrTe4/air G changed")
    _positive(thermal.get("G_tairte4_design_sio2_W_m2K"), "design-interface G")
    if thermal.get("gray_law_exponent") not in (1.0, 2.0, 3.0):
        raise ValidationError("gray-law exponent must be a named p=1,2,3 scenario")

    objective = _require(config, "objective", dict)
    if objective.get("type") != "uniform_45deg_pte_current_A":
        raise ValidationError("objective type changed without a new contract")
    if objective.get("sense") not in {"maximize", "minimize"}:
        raise ValidationError("objective sense must be maximize or minimize")

    _validate_prohibitions(config)


def _validate_physics_v2(config: dict[str, Any]) -> None:
    design = _require(config, "design", dict)
    if design.get("periodic_wrap") is not False:
        raise ValidationError("periodic wrapping is forbidden")
    if design.get("production_node_spacing_nm") != 50.0:
        raise ValidationError("Gaussian-10 production nodes must use 50 nm spacing")
    if design.get("filter_radius_nm") != 500.0:
        raise ValidationError("finite production filter radius must remain 500 nm")
    expected_window = {
        "name": "centered_18p6um",
        "x_um": [-9.3, 9.3],
        "y_um": [-9.3, 9.3],
        "node_shape": [373, 373],
        "absolute_gradient_L1_fraction": 0.9088729683975348,
    }
    if design.get("production_window") != expected_window:
        raise ValidationError("reviewed 18.6 um production window changed")
    if design.get("minimum_solid_nm") != 500.0 or design.get("minimum_void_nm") != 500.0:
        raise ValidationError("minimum solid and void dimensions must remain 500 nm")
    if float(design.get("differentiable_steering_target_nm", 0.0)) < 500.0:
        raise ValidationError("differentiable length-scale steering cannot be below DRC")
    if design.get("design_height_um") != 1.0:
        raise ValidationError("reviewed Gaussian-10 baseline height must remain 1.0 um")

    optical = _require(config, "optical", dict)
    if optical.get("illumination") != "scalar_gaussian_waist_size_and_position":
        raise ValidationError("Gaussian-10 illumination contract changed")
    if optical.get("analysis_wavelength_um") != 10.0:
        raise ValidationError("Gaussian-10 analysis wavelength must remain 10 um")
    if optical.get("target_realized_waist_um") != 8.5:
        raise ValidationError("Gaussian-10 realized waist target must remain 8.5 um")
    if optical.get("boundary") != "six_pml" or optical.get("periodic") is not False:
        raise ValidationError("optical boundary must be nonperiodic six-PML")
    source_span = _positive(optical.get("source_span_um"), "source span")
    domain = _positive(optical.get("candidate_domain_um"), "FDTD domain")
    if source_span >= domain:
        raise ValidationError("Gaussian source must stay strictly inside the FDTD domain")
    if optical.get("sio2_optical_model") != "Kitamura_2007_complex_epsilon_at_10um":
        raise ValidationError("10-um SiO2 must not silently revert to lossless n=1.38")

    thermal = _require(config, "thermal", dict)
    if thermal.get("tairte4_kappa_W_mK") != [14.4, 3.8, 1.0]:
        raise ValidationError("TaIrTe4 anisotropic kappa changed")
    if thermal.get("G_tairte4_air_W_m2K") != 1.0:
        raise ValidationError("TaIrTe4/air G changed")
    scenarios = _require(thermal, "interface_scenarios", list)
    if len(scenarios) != 4:
        raise ValidationError("all four bottom/design SiO2 interface scenarios are required")
    pairs = {
        (value.get("bottom"), value.get("design"))
        for value in scenarios
        if isinstance(value, dict)
    }
    expected = {
        ("thermally_grown", "thermally_grown"),
        ("thermally_grown", "evaporated"),
        ("evaporated", "thermally_grown"),
        ("evaporated", "evaporated"),
    }
    if pairs != expected:
        raise ValidationError("bottom/design interface scenario matrix is incomplete")
    if thermal.get("linear_solver_device") != "cuda_only":
        raise ValidationError("production thermal forward/adjoint must be CUDA-only")
    boundaries = _require(thermal, "boundary_conditions", dict)
    if boundaries.get("x_min_x_max_y_min_y_max") != "Dirichlet_T_bath_at_far_64um_domain":
        raise ValidationError("far lateral thermal boundary contract changed")
    if boundaries.get("z_min") != "Dirichlet_T_bath_at_bottom_of_20um_Si":
        raise ValidationError("bottom thermal boundary contract changed")
    if boundaries.get("z_max") != "material_specific_Robin_to_T_bath":
        raise ValidationError("top thermal boundary contract changed")
    expected_exposed = {
        "TaIrTe4": "Robin_G_1_W_m2K_to_T_bath",
        "SiO2_and_gray_design": "Robin_h_10_W_m2K_to_T_bath",
    }
    if boundaries.get("solid_air_exposed_surfaces") != expected_exposed:
        raise ValidationError("material-specific exposed thermal boundaries changed")
    if boundaries.get("periodic_axes") != []:
        raise ValidationError("thermal periodic boundaries are forbidden")

    objective = _require(config, "objective", dict)
    if objective.get("type") != "signed_uniform_45deg_pte_current_per_incident_power":
        raise ValidationError("Gaussian-10 objective contract changed")
    if objective.get("sign_runs") != [1, -1]:
        raise ValidationError("absolute current must be handled as two signed runs")
    _validate_prohibitions(config)


def _validate_physics(config: dict[str, Any]) -> None:
    schema = config.get("schema_version")
    if schema == 1:
        _validate_physics_v1(config)
    elif schema == 2:
        _validate_physics_v2(config)
    else:
        raise ValidationError(f"unsupported run schema_version: {schema}")


def _validate_artifact(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise ValidationError(f"missing {label}: {path}")
    actual = sha256(path)
    if actual != expected:
        raise ValidationError(
            f"SHA mismatch for {label}: expected {expected}, got {actual}"
        )


def validate_run_directory(
    run_dir: Path,
    *,
    repository_root: Path,
    require_external: bool = False,
) -> ValidationResult:
    """Validate layout, immutable physics, and pinned input artifacts."""

    run_dir = run_dir.resolve()
    repository_root = repository_root.resolve()
    if not RUN_NAME.fullmatch(run_dir.name):
        raise ValidationError(f"invalid run directory name: {run_dir.name}")
    for directory in REQUIRED_DIRECTORIES:
        if not (run_dir / directory).is_dir():
            raise ValidationError(f"missing run subdirectory: {directory}")
    for filename in REQUIRED_FILES:
        if not (run_dir / filename).is_file():
            raise ValidationError(f"missing run file: {filename}")

    config = _load_json(run_dir / "run_config.json")
    status = _load_json(run_dir / "STATUS.json")
    run_id = _require(config, "run_id", str)
    if run_id != run_dir.name:
        raise ValidationError("run_config.run_id must equal directory name")
    if status.get("run_id") != run_id:
        raise ValidationError("STATUS.run_id does not match run_config")
    status_name = _require(status, "status", str)
    if status_name not in ALLOWED_STATUS:
        raise ValidationError(f"unknown run status: {status_name}")
    source = _require(config, "source", dict)
    source_commit = _require(source, "git_commit", str)
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise ValidationError("source.git_commit must be a full Git SHA")
    expected_pr = 7 if config.get("schema_version") == 1 else 8
    if source.get("base_pr") != expected_pr:
        raise ValidationError(
            f"baseline provenance must identify Draft PR #{expected_pr}"
        )
    _validate_physics(config)

    repository_checked = 0
    for artifact in _require(config, "repository_inputs", list):
        if not isinstance(artifact, dict):
            raise ValidationError("repository input entries must be objects")
        relative = _require(artifact, "path", str)
        expected = _require(artifact, "sha256", str)
        candidate = (repository_root / relative).resolve()
        try:
            candidate.relative_to(repository_root)
        except ValueError as exc:
            raise ValidationError(f"repository input escapes root: {relative}") from exc
        _validate_artifact(candidate, expected, "repository input")
        repository_checked += 1

    external_checked = 0
    external_missing = 0
    for artifact in _require(config, "external_inputs", list):
        if not isinstance(artifact, dict):
            raise ValidationError("external input entries must be objects")
        candidate = Path(_require(artifact, "path", str)).expanduser()
        expected = _require(artifact, "sha256", str)
        if not candidate.is_file():
            external_missing += 1
            if require_external:
                raise ValidationError(f"missing external input: {candidate}")
            continue
        _validate_artifact(candidate, expected, "external input")
        external_checked += 1

    return ValidationResult(
        run_directory=str(run_dir),
        run_id=run_id,
        status=status_name,
        source_commit=source_commit,
        repository_artifacts_checked=repository_checked,
        external_artifacts_checked=external_checked,
        external_artifacts_missing=external_missing,
        valid=True,
    )
