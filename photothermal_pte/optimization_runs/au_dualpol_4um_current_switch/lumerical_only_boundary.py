"""Fail-closed solver boundary for the production Lumerical/CUDA workflow."""

from __future__ import annotations

import ast
from pathlib import Path
import sys
from typing import Any


HERE = Path(__file__).resolve().parent
FORBIDDEN_MAXWELL_IMPORT_ROOTS = ("fdtdx",)
PRODUCTION_ENTRYPOINTS = (
    "25_run_lumerical_4um_exact_au_control.py",
    "26_build_lumerical_4um_yee_jacobian.py",
    "33_validate_lumerical_4um_gray_q_cuda_pde.py",
    "34_run_lumerical_4um_gray_maxwell_adjoint.py",
    "40_optimize_lumerical_4um_dualpol_smoke.py",
    "41_optimize_lumerical_4um_dualpol_continuation.py",
    "42_evaluate_lumerical_4um_exact_binary.py",
    "43_certify_lumerical_4um_exact_binary_lateral.py",
    "44_validate_lumerical_4um_fixed_q_au_thermopower_adfd.py",
)
PRODUCTION_SUPPORT = (
    "contract.py",
    "multiphysics_4um.py",
    "au_density_relaxation.py",
    "dfm.py",
    "robust_contract.py",
    "objective.py",
    "lumerical_only_boundary.py",
)


def _production_sources() -> tuple[Path, ...]:
    paths = {HERE / name for name in (*PRODUCTION_ENTRYPOINTS, *PRODUCTION_SUPPORT)}
    paths.update(HERE.glob("lumerical_4um_*.py"))
    paths.add(HERE / "lumerical_maxwell_contract.py")
    return tuple(sorted(paths))


def audit_production_imports() -> dict[str, Any]:
    """Statically prove that the production source set imports no other solver."""

    forbidden: list[dict[str, str]] = []
    imported: set[str] = set()
    sources = _production_sources()
    for path in sources:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                names.append(node.module)
            for name in names:
                imported.add(name)
                root = name.split(".", 1)[0].lower()
                if root in FORBIDDEN_MAXWELL_IMPORT_ROOTS:
                    forbidden.append(
                        {"path": str(path), "module": name, "line": str(node.lineno)}
                    )
    return {
        "passed": not forbidden,
        "source_count": len(sources),
        "sources": [str(path) for path in sources],
        "forbidden_imports": forbidden,
        "solver_contract": {
            "Maxwell_forward": "Lumerical FDTD",
            "Maxwell_adjoint": "Lumerical FDTD",
            "thermal_forward_adjoint": "custom CUDA PDE",
            "electrical_forward_adjoint": "custom CUDA PDE",
            "Lumerical_HEAT": False,
            "Lumerical_CHARGE": False,
            "alternative_Maxwell_solver": False,
        },
    }


def assert_lumerical_only_process() -> None:
    """Abort if a forbidden Maxwell package has entered this Python process."""

    loaded = sorted(
        name
        for name in sys.modules
        if name.lower().split(".", 1)[0] in FORBIDDEN_MAXWELL_IMPORT_ROOTS
    )
    if loaded:
        raise RuntimeError(
            "Lumerical-only process contains a forbidden Maxwell module: "
            + ", ".join(loaded)
        )


def require_lumerical_only_source_boundary() -> dict[str, Any]:
    """Run both source and live-process gates or fail closed."""

    assert_lumerical_only_process()
    audit = audit_production_imports()
    if not audit["passed"]:
        raise RuntimeError(
            f"Lumerical-only production import boundary failed: {audit['forbidden_imports']}"
        )
    return audit
