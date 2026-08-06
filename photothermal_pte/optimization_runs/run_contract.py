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


def _validate_physics(config: dict[str, Any]) -> None:
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
    if source.get("base_pr") != 7:
        raise ValidationError("baseline provenance must identify Draft PR #7")
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
