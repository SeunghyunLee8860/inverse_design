#!/usr/bin/env python3
"""Audit the exact-Au 4-um Lumerical run matrix without running Maxwell."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys


REPOSITORY = Path(__file__).resolve().parents[3]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.contract import (  # noqa: E402
    CONTRACT as DEVICE_ASSUMPTIONS,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.lumerical_4um_exact_au import (  # noqa: E402
    control_geometry_audits,
    material_contract_audit,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.lumerical_4um_mesh_contract import (  # noqa: E402
    convergence_contract_audit,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.lumerical_maxwell_contract import (  # noqa: E402
    CONTRACT as SOLVER_CONTRACT,
    audit_environment,
)


HERE = Path(__file__).resolve().parent
PHYSICAL_DEVICE = HERE / "physical_device_contract.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-index", type=int)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    physical_device = json.loads(PHYSICAL_DEVICE.read_text(encoding="utf-8"))
    environment = audit_environment(requested_gpu_index=args.gpu_index)
    payload = {
        "status": "BLOCKED_EXACT_AU_LUMERICAL_4UM_RUNSETUP",
        "scope": "audit only; no FDTD/thermal/electrical/adjoint/optimization solve",
        "solver_contract": asdict(SOLVER_CONTRACT),
        "provisional_device_assumptions": DEVICE_ASSUMPTIONS.audit(),
        "physical_device_contract": {
            "path": str(PHYSICAL_DEVICE.relative_to(REPOSITORY)),
            "sha256": _sha256(PHYSICAL_DEVICE),
            "status": physical_device.get("status"),
        },
        "environment": environment,
        "dispersive_material_inputs": material_contract_audit(),
        "exact_geometry_controls": control_geometry_audits(),
        "mesh_convergence": convergence_contract_audit(),
    }
    blockers = []
    if environment["status"] != "READY_FOR_LUMERICAL_B200_MAXWELL_DEVELOPMENT":
        blockers.append("actual_B200_preflight")
    if physical_device.get("status") != "VALIDATED_AU_TAIRTE4_PHYSICAL_DEVICE_CONTRACT":
        blockers.append("physical_device_contract")
    blockers.extend(("Lumerical_material_fit_readback", "forward_runner_not_yet_executed"))
    payload["blockers"] = blockers
    rendered = json.dumps(payload, indent=2) + "\n"
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output_json.with_suffix(args.output_json.suffix + ".tmp")
        temporary.write_text(rendered, encoding="utf-8")
        temporary.replace(args.output_json)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
