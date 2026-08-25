#!/usr/bin/env python3
"""Final solver-free quarantine audit for historical FDTDX gray-Au paths.

This audit does not select a replacement FDTDX gray law.  It proves that the
historical O3/TE1 relaxation cannot reach an optimizer, that the completed
FDTDX mesh evidence remains blocked, and that exact-binary endpoint placement
rejects gray inputs.  The separately owned Lumerical route is only recorded as
an unvalidated future route; it is neither launched nor modified here.
"""

from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping

import numpy as np

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.au_density_relaxation import (
    audit as optical_relaxation_audit,
    d_epsilon_d_projected_density,
    epsilon_relaxation,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.contract import (
    CONTRACT,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_exact_binary_convergence import (
    MeshSpec,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_exact_binary_material import (
    mask_material_audit,
    normalize_exact_mask,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_user_balanced_pte_tail_certificate import (
    STATUS_BLOCKED as PTE_STATUS_BLOCKED,
    STATUS_PASS as PTE_STATUS_PASS,
    VERSION as PTE_CERTIFICATE_VERSION,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_user_balanced_z_certificate import (
    STATUS_BLOCKED as OPTICAL_STATUS_BLOCKED,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_user_balanced_z_tail_certificate import (
    TAIL_VERSION,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.lumerical_4um_density import (
    density_state_audit,
    nodal_to_cell_average,
    nodal_to_cell_jvp,
    nodal_to_cell_vjp,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.material_fraction import (
    audit as historical_fraction_audit,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.production_readiness import (
    readiness_audit,
)


VERSION = "fdtdx-gray-law-final-audit-v1"
STATUS_READY = "VALIDATED_FDTDX_GRAY_LAW_QUARANTINE_AUDIT"
STATUS_INVALID = "INVALID_FDTDX_GRAY_LAW_QUARANTINE_AUDIT"
REPORT_NAME = "FDTDX_GRAY_LAW_FINAL_AUDIT.json"
POLARIZATIONS = ("Ea", "Eb")
ACTIVE_FDTDX_RUNTIME_FILES = (
    "material_fraction.py",
    "combined_4um.py",
    "multiphysics_4um.py",
    "10_optimize_4um_dualpol_au_ld_mma.py",
    "12_optimize_exact_binary_au_topology.py",
    "13_optimize_robust_binary_au_ld_mma.py",
)
OPTIMIZER_ENTRYPOINTS = (
    "10_optimize_4um_dualpol_au_ld_mma.py",
    "12_optimize_exact_binary_au_topology.py",
    "13_optimize_robust_binary_au_ld_mma.py",
)
LOCAL_PAPERS = (
    "s41467-022-32309-w.pdf",
    "41467_2022_32309_MOESM1_ESM.pdf",
    "s41467-024-51599-w.pdf",
    "41467_2024_51599_MOESM1_ESM.pdf",
    "Adv Funct Materials - 2026 - Blevins - Large Transverse Thermoelectric Effect in Weyl Semimetal TaIrTe4 Engineered for.pdf",
)
PAPER_SCOPE = {
    "10.1038/s41467-022-32309-w": (
        "explicit 50-nm Z-shaped Au resonators/backplate; geometric global "
        "optimization and ordinary optical-to-thermal heat transfer"
    ),
    "10.1038/s41467-024-51599-w": (
        "explicit inverse-T/T Au resonators; dimension/orientation sweeps and "
        "electrode-defined directional-current projection"
    ),
    "10.1002/adfm.75986": (
        "TaIrTe4 Jloc=-sigma*S*grad(T) and device-dependent Shockley-Ramo "
        "weighting; electrode/crystal/thermal geometry controls collection"
    ),
    "10.1016/j.cma.2018.08.034": (
        "nonlinear metallic bi-material epsilon interpolation through n and k"
    ),
    "10.1021/acsphotonics.1c00260": (
        "plasmonic discrete-adjoint FDTD with nonlinear interpolation plus "
        "filter/projection; poor interpolation can cause amplification"
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ("git", "-C", str(repository), *arguments),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _literal_number(node: ast.AST) -> float | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    return None


def literal_power_three_locations(path: Path) -> list[dict[str, Any]]:
    """Return executable AST locations containing a literal power of three."""

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    result = []
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            if _literal_number(node.right) == 3.0:
                result.append(
                    {
                        "line": int(node.lineno),
                        "expression": ast.unparse(node),
                    }
                )
    return result


def optimizer_gate_order(path: Path) -> dict[str, Any]:
    """Audit call order inside the executable ``main`` function only."""

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    main = next(
        (
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "main"
        ),
        None,
    )
    readiness_lines: list[int] = []
    output_mutation_lines: list[int] = []
    compilation_lines: list[int] = []
    if main is not None:
        for node in ast.walk(main):
            if not isinstance(node, ast.Call):
                continue
            function = node.func
            if isinstance(function, ast.Name) and function.id == "require_production_readiness":
                readiness_lines.append(int(node.lineno))
            if isinstance(function, ast.Attribute) and function.attr == "mkdir":
                owner = function.value
                if isinstance(owner, ast.Name) and owner.id in {"OUT", "RAW"}:
                    output_mutation_lines.append(int(node.lineno))
            if (
                isinstance(function, ast.Attribute)
                and function.attr == "create"
                and isinstance(function.value, ast.Name)
                and function.value.id == "CompiledOpticalRunner"
            ):
                compilation_lines.append(int(node.lineno))
    readiness = min(readiness_lines, default=-1)
    output_mutation = min(output_mutation_lines, default=-1)
    compilation = min(compilation_lines, default=-1)
    checks = {
        "main_function_present": main is not None,
        "one_readiness_gate_present": len(readiness_lines) == 1,
        "output_mutation_present": output_mutation >= 0,
        "maxwell_compilation_present": compilation >= 0,
        "readiness_precedes_output_mutation": readiness >= 0
        and output_mutation >= 0
        and readiness < output_mutation,
        "readiness_precedes_maxwell_compilation": readiness >= 0
        and compilation >= 0
        and readiness < compilation,
    }
    return {
        "path": str(path),
        "lines": {
            "readiness": readiness,
            "first_output_mutation": output_mutation,
            "first_maxwell_compilation": compilation,
        },
        "checks": checks,
        "ready": all(checks.values()),
    }


def shared_state_numeric_audit() -> dict[str, Any]:
    rng = np.random.default_rng(20260825)
    nodes = 0.1 + 0.8 * rng.random(CONTRACT.design_node_shape)
    direction = rng.standard_normal(CONTRACT.design_node_shape)
    direction /= np.max(np.abs(direction))
    cotangent = rng.standard_normal(CONTRACT.design_shape)
    left = float(np.vdot(nodal_to_cell_jvp(direction), cotangent))
    right = float(np.vdot(direction, nodal_to_cell_vjp(cotangent)))
    transpose_error = abs(left - right) / max(abs(left), np.finfo(float).tiny)
    cells = nodal_to_cell_average(nodes)
    gradient = nodal_to_cell_vjp(2.0 * cells)
    analytic_cell = float(np.vdot(gradient, direction))
    cell_fd_rows = []
    optical_fd_rows = []
    optical_direction = direction.astype(np.float64)
    optical_analytic = d_epsilon_d_projected_density(nodes) * optical_direction
    for step in (1.0e-4, 5.0e-5, 2.5e-5, 1.25e-5):
        plus_cell = float(
            np.sum(nodal_to_cell_average(nodes + step * direction) ** 2)
        )
        minus_cell = float(
            np.sum(nodal_to_cell_average(nodes - step * direction) ** 2)
        )
        cell_fd = (plus_cell - minus_cell) / (2.0 * step)
        cell_fd_rows.append(
            {
                "step": step,
                "relative_error": abs(cell_fd - analytic_cell)
                / max(abs(analytic_cell), np.finfo(float).tiny),
            }
        )
        optical_fd = (
            epsilon_relaxation(nodes + step * optical_direction)
            - epsilon_relaxation(nodes - step * optical_direction)
        ) / (2.0 * step)
        optical_fd_rows.append(
            {
                "step": step,
                "complex_relative_error": float(
                    np.linalg.norm(optical_fd - optical_analytic)
                    / max(
                        np.linalg.norm(optical_analytic),
                        np.finfo(float).tiny,
                    )
                ),
            }
        )
    state = density_state_audit(nodes)
    checks = {
        "one_nodal_shape_is_81x81": state["nodal_shape_xy"] == [81, 81],
        "one_pde_cell_shape_is_80x80": state["pde_cell_shape_xy"] == [80, 80],
        "cell_map_transpose_lt_1e_minus_12": transpose_error < 1.0e-12,
        "cell_map_fd_lt_1e_minus_7": max(row["relative_error"] for row in cell_fd_rows)
        < 1.0e-7,
        "optical_nk_law_fd_lt_1e_minus_9": max(
            row["complex_relative_error"] for row in optical_fd_rows
        )
        < 1.0e-9,
        "optical_rho_power_is_none": state["optical_rho_power"] is None,
        "gray_not_claimed_as_fabricated_material": state[
            "gray_state_claimed_as_fabricated_material"
        ]
        is False,
    }
    return {
        "state": state,
        "cell_map_transpose_relative_error": transpose_error,
        "cell_map_finite_difference": cell_fd_rows,
        "optical_law_finite_difference": optical_fd_rows,
        "checks": checks,
        "ready": all(checks.values()),
    }


def exact_binary_audit() -> dict[str, Any]:
    mask = np.zeros(CONTRACT.design_shape, dtype=np.uint8)
    mask[35:45, 35:45] = 1
    accepted = normalize_exact_mask(mask)
    float_rejected = False
    gray_rejected = False
    try:
        normalize_exact_mask(mask.astype(np.float64))
    except ValueError:
        float_rejected = True
    gray = mask.astype(np.uint8)
    gray[0, 0] = 2
    try:
        normalize_exact_mask(gray)
    except ValueError:
        gray_rejected = True
    material = mask_material_audit(mask, MeshSpec())
    checks = {
        "integer_binary_mask_accepted": np.array_equal(accepted, mask),
        "float_mask_rejected_even_at_binary_values": float_rejected,
        "nonendpoint_integer_rejected": gray_rejected,
        "gray_density_forbidden": material["gray_density_allowed"] is False,
        "rho_power_absent": material["rho_power"] is None,
        "piecewise_constant_replication": material["mapping"]
        == "integer piecewise-constant replication",
    }
    return {"material": material, "checks": checks, "ready": all(checks.values())}


def certificate_audit(
    path: Path,
    expected_sha256: str,
    *,
    kind: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    supplied = path.expanduser()
    resolved = supplied.resolve()
    exists = resolved.is_file()
    actual = sha256(resolved) if exists else None
    payload = (
        json.loads(resolved.read_text(encoding="utf-8")) if exists else {}
    )
    if kind == "optical":
        semantic = {
            "version_exact": payload.get("version") == TAIL_VERSION,
            "artifact_certificate_valid": payload.get("certificate_valid")
            is True,
            "strict_convergence_blocked": payload.get("status")
            == OPTICAL_STATUS_BLOCKED
            and payload.get("convergence_pass") is False
            and payload.get("mesh_selected") is None,
            "optimizer_forbidden": payload.get("optimizer_start_allowed")
            is False,
        }
    elif kind == "pte":
        semantic = {
            "version_exact": payload.get("version") == PTE_CERTIFICATE_VERSION,
            "artifact_certificate_valid": payload.get("certificate_valid")
            is True,
            "recognized_valid_status": payload.get("status")
            in (PTE_STATUS_PASS, PTE_STATUS_BLOCKED),
            "strict_optical_mesh_unselected": payload.get(
                "strict_optical_mesh_selected"
            )
            is None,
            "optimizer_forbidden": payload.get("optimizer_start_allowed")
            is False,
        }
    else:  # pragma: no cover - internal misuse
        raise ValueError(kind)
    checks = {
        "path_is_absolute": supplied.is_absolute(),
        "file_exists": exists,
        "sha256_matches": actual == expected_sha256,
        **semantic,
    }
    audit = {
        "path": str(resolved),
        "expected_sha256": expected_sha256,
        "actual_sha256": actual,
        "checks": checks,
        "ready": all(checks.values()),
    }
    return payload, audit


def build_audit(
    optical_certificate_path: Path,
    expected_optical_certificate_sha256: str,
    pte_certificate_path: Path,
    expected_pte_certificate_sha256: str,
    papers_root: Path,
) -> dict[str, Any]:
    here = Path(__file__).resolve().parent
    repository = here.parents[2]
    dirty = _git(repository, "status", "--porcelain", "--untracked-files=all")
    optical_certificate, optical_certificate_audit = certificate_audit(
        optical_certificate_path,
        expected_optical_certificate_sha256,
        kind="optical",
    )
    pte_certificate, pte_certificate_audit = certificate_audit(
        pte_certificate_path,
        expected_pte_certificate_sha256,
        kind="pte",
    )
    active_scan = {
        name: {
            "path": str(here / name),
            "sha256": sha256(here / name),
            "literal_power_three_locations": literal_power_three_locations(
                here / name
            ),
        }
        for name in ACTIVE_FDTDX_RUNTIME_FILES
    }
    optimizer_gates = {
        name: optimizer_gate_order(here / name) for name in OPTIMIZER_ENTRYPOINTS
    }
    shared_state = shared_state_numeric_audit()
    binary = exact_binary_audit()
    historical = historical_fraction_audit()
    optical_relaxation = optical_relaxation_audit()
    readiness = readiness_audit()
    paper_root = papers_root.expanduser().resolve()
    paper_files = {
        name: {
            "path": str(paper_root / name),
            "exists": (paper_root / name).is_file(),
            "sha256": sha256(paper_root / name)
            if (paper_root / name).is_file()
            else None,
        }
        for name in LOCAL_PAPERS
    }
    checks = {
        "repository_clean_while_auditing": dirty == "",
        "optical_block_certificate_revalidates": optical_certificate_audit[
            "ready"
        ],
        "pte_tail_certificate_revalidates": pte_certificate_audit["ready"],
        "active_runtime_has_no_literal_rho_power_three": all(
            not record["literal_power_three_locations"]
            for record in active_scan.values()
        ),
        "all_optimizer_entrypoints_fail_closed_before_mutation_or_compile": all(
            record["ready"] for record in optimizer_gates.values()
        ),
        "exact_binary_path_rejects_gray_and_float": binary["ready"],
        "shared_nodal_state_and_discrete_map_numerically_valid": shared_state[
            "ready"
        ],
        "historical_fdtdx_fraction_is_linear_and_nonphysical": historical[
            "scope"
        ]
        == "historical_fdtdx_consistency_baseline"
        and historical["exponent"] == 1.0
        and historical["gray_density_is_physical_geometry"] is False,
        "future_optical_law_has_no_rho_power_and_is_not_promoted": optical_relaxation[
            "rho_cubed_used"
        ]
        is False
        and optical_relaxation["optical_rho_power"] is None
        and bool(optical_relaxation["remaining_gates"]),
        "production_readiness_is_fail_closed": readiness["ready"] is False
        and readiness["checks"]["lumerical_dispersive_density_route_validated"]
        is False,
        "all_local_papers_rebound_by_bytes": all(
            record["exists"] and record["sha256"] is not None
            for record in paper_files.values()
        ),
        "fdtdx_strict_mesh_unselected": optical_certificate.get("mesh_selected")
        is None,
        "pte_diagnostic_does_not_select_strict_optical_mesh": pte_certificate.get(
            "strict_optical_mesh_selected"
        )
        is None,
    }
    ready = all(checks.values())
    generator = Path(__file__).resolve()
    return {
        "version": VERSION,
        "status": STATUS_READY if ready else STATUS_INVALID,
        "ready": ready,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "solver-free forensic quarantine of historical FDTDX gray-Au and "
            "optimizer paths; no Lumerical/GPU solve and no replacement gray "
            "law promotion"
        ),
        "decision": {
            "historical_O3_TE1_allowed": False,
            "shared_linear_FDTDX_gray_allowed_for_production": False,
            "generic_FDTDX_Device_gray_allowed_for_production": False,
            "FDTDX_exact_binary_reference_allowed": True,
            "FDTDX_strict_mesh_selected": None,
            "FDTDX_optimizer_start_allowed": False,
            "future_Lumerical_nk_route_owned_by_this_audit": False,
            "future_Lumerical_nk_route_validated": False,
        },
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "active_runtime_scan": active_scan,
        "optimizer_gate_order": optimizer_gates,
        "exact_binary": binary,
        "historical_shared_linear_fraction": historical,
        "future_optical_relaxation_solver_free_audit": optical_relaxation,
        "shared_nodal_state_numeric_audit": shared_state,
        "production_readiness": {
            "ready": readiness["ready"],
            "failed_checks": readiness["failed_checks"],
            "errors": readiness["errors"],
            "lumerical_density_route_status": readiness[
                "lumerical_density_route_status"
            ],
        },
        "optical_certificate": optical_certificate_audit,
        "pte_certificate": pte_certificate_audit,
        "local_papers": {
            "root": str(paper_root),
            "files": paper_files,
            "scope_by_doi": PAPER_SCOPE,
            "local_detector_papers_justify_rho_power_three": False,
            "local_detector_papers_use_explicit_binary_Au_geometry": True,
        },
        "open_physical_blockers": [
            "actual flake/contact/electrode geometry and crystal-axis orientation",
            "TaIrTe4-SiO2 and Au-TaIrTe4 thermal conductance bounds",
            "Au-TaIrTe4 electrical contact and void-floor sensitivity",
            "thermal/electrical mesh convergence on actual geometry",
            "future Maxwell endpoint/bandwidth/resonance/Yee-Jacobian/AD-FD gates",
        ],
        "provenance": {
            "repository_commit": _git(repository, "rev-parse", "HEAD"),
            "repository_dirty_porcelain": dirty,
            "generator_path": str(generator),
            "generator_sha256": sha256(generator),
            "lumerical_used": False,
            "gpu_used": False,
        },
        "optimizer_start_allowed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--optical-tail-certificate", type=Path, required=True)
    parser.add_argument("--optical-tail-certificate-sha256", required=True)
    parser.add_argument("--pte-tail-certificate", type=Path, required=True)
    parser.add_argument("--pte-tail-certificate-sha256", required=True)
    parser.add_argument(
        "--papers-root", type=Path, default=Path("/home/seunghyun200/papers")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.expanduser()
    if not output.is_absolute() or not output.parent.is_dir() or output.exists():
        parser.error("--output must be a new absolute file under an existing directory")
    payload = build_audit(
        args.optical_tail_certificate,
        args.optical_tail_certificate_sha256,
        args.pte_tail_certificate,
        args.pte_tail_certificate_sha256,
        args.papers_root,
    )
    _atomic_json(output, payload)
    print(
        json.dumps(
            {
                "output": str(output.resolve()),
                "status": payload["status"],
                "ready": payload["ready"],
                "failed_checks": payload["failed_checks"],
                "optimizer_start_allowed": False,
            }
        )
    )
    return 0 if payload["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
