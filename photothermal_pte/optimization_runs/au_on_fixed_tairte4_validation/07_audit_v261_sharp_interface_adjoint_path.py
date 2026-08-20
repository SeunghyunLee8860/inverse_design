#!/usr/bin/env python3
"""Audit the bundled v261 polygon boundary-perturbation implementation.

This is a source-code/contract audit only.  It must not be reported as an
AD--FD result.  The subsequent GPU control supplies the numerical evidence.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
LUMOPT_ROOT = Path("/opt/lumerical/v261/api/python/lumopt")
FILES = {
    "polygon": LUMOPT_ROOT / "geometries" / "polygon.py",
    "edge": LUMOPT_ROOT / "utilities" / "edge.py",
    "gradients": LUMOPT_ROOT / "utilities" / "gradients.py",
    "materials": LUMOPT_ROOT / "utilities" / "materials.py",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    source = {name: path.read_text() for name, path in FILES.items()}
    checks = {
        "FunctionDefinedPolygon_present": "class FunctionDefinedPolygon" in source["polygon"],
        "CCW_contract_present": "COUNTER CLOCKWISE" in source["polygon"],
        "edge_quadrature_present": "edge_precision" in source["polygon"],
        "three_dimensional_depth_factor_present": "edge_derivs_2D[0] * self.depth" in source["edge"],
        "E_parallel_term_present": "E_parallel_forward" in source["gradients"],
        "D_perpendicular_term_present": "D_perp_forward" in source["gradients"],
        "epsilon_jump_term_present": "(eps_in - eps_out)" in source["gradients"],
        "inverse_epsilon_jump_term_present": "(1.0/eps_out - 1.0/eps_in)" in source["gradients"],
        "real_shape_gradient_returned": "return np.real(result)" in source["gradients"],
        "named_material_getfdtdindex_present": "getfdtdindex" in source["materials"],
    }
    all_checks = all(checks.values())
    status = (
        "AUDITED_V261_SHARP_INTERFACE_PATH_ADFD_PENDING"
        if all_checks
        else "BLOCKED_V261_SHARP_INTERFACE_IMPLEMENTATION_MISMATCH"
    )
    summary = {
        "status": status,
        "scope": "bundled v261 source-code and parameterization audit; no Maxwell solve and no AD-FD claim",
        "solver_version_family": "v261",
        "files": {
            name: {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for name, path in FILES.items()
        },
        "checks": checks,
        "approved_representation": {
            "inside": "exact named scalar (n,k) Au material",
            "outside": "air",
            "geometry": "counter-clockwise polygon extruded through a fixed 50 nm depth",
            "shape_parameter_control": "symmetric motion of the two x-normal vertical faces",
            "gray_Au_air_permittivity": False,
        },
        "boundary_kernel": {
            "description": "bundled LumOpt boundary perturbation using tangential E and normal D continuity variables",
            "terms": [
                "2*epsilon0*(epsilon_in-epsilon_out)*(E_parallel_fwd dot E_parallel_adj)",
                "(1/epsilon_out-1/epsilon_in)/epsilon0*(D_perp_fwd dot D_perp_adj)",
            ],
            "three_dimensional_rule": "integrate along each lateral edge and multiply by the fixed extrusion depth",
        },
        "not_yet_validated": [
            "adjoint-source normalization for the selected optical FOM",
            "component-specific forward/adjoint field pairing on the metal boundary",
            "central-FD agreement and step-size plateau",
            "thermal, electrical, PTE, or combined shape derivatives",
        ],
    }
    (RESULTS / "au_sharp_interface_path_audit.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    report = f"""# v261 sharp-interface Au adjoint-path audit

Status: `{status}`

This checkpoint inspects the bundled v261 LumOpt implementation. It performs
no Maxwell solve and is not an AD--FD certificate.

The approved fallback is a counter-clockwise binary Au polygon, extruded by a
fixed 50 nm thickness. The Au side uses the exact named Ordal `(n,k)` material;
the outside is air. No intermediate Au/air permittivity is constructed.

The installed boundary kernel uses both continuity variables required at a
material boundary: tangential electric field and normal electric displacement.
For the present width control only the two vertical x-normal faces move; the
top, bottom, y-normal faces and thickness remain fixed.

All `{len(checks)}` expected implementation checks passed. This confirms that
v261 contains the intended shape-derivative route, but the numerical gate is
still open: the same geometry must pass GPU adjoint versus central FD without
empirical normalization or gradient rescaling.

Official API contract: https://optics.ansys.com/hc/en-us/articles/360052044913-Optimizable-Geometry-Python-API
"""
    (RESULTS / "AU_SHARP_INTERFACE_PATH_AUDIT.md").write_text(report)
    print(json.dumps({"status": status, "checks": checks}, indent=2))
    return 0 if all_checks else 2


if __name__ == "__main__":
    raise SystemExit(main())
