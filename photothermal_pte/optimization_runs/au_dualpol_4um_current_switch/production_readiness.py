"""Fail-closed certificate chain required by every production entry point."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.material_fraction import (
    audit as material_fraction_audit,
)


HERE = Path(__file__).resolve().parent
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
REQUIRED_MESH_COVERAGE = (
    "optical_z_full_domain",
    "optical_xy",
    "time_window_stationarity",
    "q_closed_flux_closure",
    "thermal_mesh",
    "electrical_mesh",
)


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
) -> dict[str, object]:
    mesh_path = Path(mesh_path)
    gradient_path = Path(gradient_path)
    mesh, mesh_error = _read(mesh_path)
    gradient, gradient_error = _read(gradient_path)
    material = material_fraction_audit()

    checks: dict[str, bool] = {
        "mesh_certificate_readable": mesh_error is None,
        "gradient_certificate_readable": gradient_error is None,
    }
    errors = [error for error in (mesh_error, gradient_error) if error is not None]
    mesh_sha256 = None
    if mesh is not None:
        coverage = mesh.get("coverage", {})
        mesh_sha256 = sha256(mesh_path)
        checks.update(
            mesh_status=mesh.get("status") == MESH_STATUS,
            mesh_material_fraction=mesh.get("au_material_fraction") == material,
            mesh_coverage=bool(
                isinstance(coverage, dict)
                and all(coverage.get(name) is True for name in REQUIRED_MESH_COVERAGE)
            ),
        )
    else:
        checks.update(
            mesh_status=False,
            mesh_material_fraction=False,
            mesh_coverage=False,
        )

    if gradient is not None:
        checks.update(
            gradient_status=gradient.get("status") == GRADIENT_STATUS,
            gradient_material_fraction=(
                gradient.get("au_material_fraction") == material
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
            gradient_uses_mesh_certificate=False,
            gradient_multidirection_gate=False,
        )

    failed = [name for name, passed in checks.items() if not passed]
    return {
        "ready": not failed,
        "checks": checks,
        "failed_checks": failed,
        "errors": errors,
        "mesh_certificate": str(mesh_path),
        "mesh_certificate_sha256": mesh_sha256,
        "gradient_certificate": str(gradient_path),
        "required_mesh_coverage": list(REQUIRED_MESH_COVERAGE),
        "au_material_fraction": material,
    }


def require_production_readiness() -> dict[str, object]:
    result = readiness_audit()
    if not result["ready"]:
        raise RuntimeError(
            "production inverse design is blocked by certificate readiness:\n"
            + json.dumps(result, indent=2)
        )
    return result
