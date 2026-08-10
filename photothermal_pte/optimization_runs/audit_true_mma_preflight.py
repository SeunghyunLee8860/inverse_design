#!/usr/bin/env python3
"""Hash the full Run016/017 dependency closure and enforce preflight gates."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

from photothermal_pte.optimization_runs.tairte4_flake_topology.contract import CONTRACT


REPOSITORY = Path(__file__).resolve().parents[2]
CODE_PATHS = (
    "photothermal_pte/optimization_runs/run_contract.py",
    "photothermal_pte/optimization_runs/tairte4_flake_topology/contract.py",
    "photothermal_pte/optimization_runs/tairte4_flake_topology/optical.py",
    "photothermal_pte/optimization_runs/tairte4_flake_topology/thermal.py",
    "photothermal_pte/optimization_runs/tairte4_flake_topology/electrical.py",
    "photothermal_pte/optimization_runs/tairte4_flake_topology/optimization_support.py",
    "photothermal_pte/optimization_runs/tairte4_flake_topology/mma.py",
    "photothermal_pte/optimization_runs/tairte4_flake_topology/audit_optical_runsetup.py",
    "photothermal_pte/optimization_runs/tairte4_flake_topology/run_forward_gpu.py",
    "photothermal_pte/optimization_runs/tairte4_flake_topology/compare_forward_meshes.py",
    "photothermal_pte/optimization_runs/tairte4_flake_topology/evaluate_objective_gradient.py",
    "photothermal_pte/optimization_runs/tairte4_flake_topology/evaluate_binary_objective.py",
    "photothermal_pte/optimization_runs/tairte4_flake_topology/validate_combined_adfd.py",
    "photothermal_pte/optimization_runs/tairte4_flake_topology/run_true_mma_optimization.py",
    "photothermal_pte/optimization_runs/run_true_mma_dual_supervisor.py",
    "photothermal_pte/optimization_runs/publish_true_mma_accepted_updates.py",
    "photothermal_pte/optimization_runs/cuda_thermal_adjoint.py",
    "photothermal_pte/optimization_runs/run_002_gaussian10_w8p5_current_max/production_density_mapping.py",
    "photothermal_pte/optimization_runs/run_002_gaussian10_w8p5_current_max/build_nonuniform_complex_yee_jacobian.py",
    "photothermal_pte/optimization_runs/run_002_gaussian10_w8p5_current_max/run_complex_material_control.py",
    "photothermal_pte/optimization_runs/run_002_gaussian10_w8p5_current_max/run_production_combined_adfd_smoke.py",
    "photothermal_pte/finite_inverse_design/finite_q_mapping.py",
    "photothermal_pte/finite_inverse_design/native_yee_q.py",
    "photothermal_pte/finite_inverse_design/yee_material_jacobian.py",
    "photothermal_pte/finite_inverse_design/probe_v261_cpu_tfsf_device.py",
    "photothermal_pte/finite_inverse_design/run_v261_large_background_mixed_optical_adfd.py",
    "photothermal_pte/validation/photothermal_stage1/anisotropic_heat_fvm.py",
    "photothermal_pte/optimization_runs/tairte4_flake_topology/tests/test_mma.py",
    "photothermal_pte/optimization_runs/tairte4_flake_topology/tests/test_true_mma_driver.py",
    "photothermal_pte/optimization_runs/tairte4_flake_topology/tests/test_thermal.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_gate(path: Path, expected_status: str) -> dict[str, object]:
    if not path.is_file():
        return {
            "path": str(path),
            "exists": False,
            "passed": False,
            "expected_status": expected_status,
        }
    payload = json.loads(path.read_text())
    return {
        "path": str(path),
        "exists": True,
        "sha256": sha256(path),
        "reported_status": payload.get("status"),
        "expected_status": expected_status,
        "passed": bool(payload.get("passed") and payload.get("status") == expected_status),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--preflight-root", required=True, type=Path)
    args = parser.parse_args()
    if CONTRACT.geometry_mode != "contact_anchored":
        raise RuntimeError("set TAIRTE4_TOPOLOGY_GEOMETRY=contact_anchored")
    CONTRACT.validate()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    preflight = args.preflight_root.expanduser().resolve()

    code = {}
    for relative in CODE_PATHS:
        path = REPOSITORY / relative
        if not path.is_file():
            raise RuntimeError(f"missing required tracked code: {relative}")
        code[relative] = {
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }

    historical_root = Path(
        "/data/seunghyun/tairte4/artifacts/tairte4_contact_anchored"
    )
    gates = {
        "runsetup": json_gate(
            historical_root / "run012_runsetup_20260810/tairte4_flake_optical_runsetup_audit.json",
            "VALIDATED_TAIRTE4_FLAKE_OPTICAL_RUNSETUP",
        ),
        "thermal_adfd": json_gate(
            historical_root / "run012_thermal_adfd_20260810/tairte4_flake_fixed_Q_thermal_adfd.json",
            "VALIDATED_TAIRTE4_FLAKE_FIXED_Q_THERMAL_ADFD",
        ),
        "electrical_adfd": json_gate(
            historical_root / "run012_electrical_adfd_20260810/tairte4_weighting_electrical_adfd.json",
            "VALIDATED_TAIRTE4_DENSITY_DEPENDENT_WEIGHTING_ADFD",
        ),
        "combined_Ea_adfd": json_gate(
            historical_root / "run012_combined_adfd_Ea_20260810/tairte4_flake_combined_adfd.json",
            "VALIDATED_TAIRTE4_FLAKE_COMBINED_PHYSICAL_RHO_ADFD",
        ),
        "contact_anchored_domain": json_gate(
            preflight / "domain_comparison/tairte4_flake_40um_48um_domain_comparison.json",
            "VALIDATED_TAIRTE4_FLAKE_40UM_OPTICAL_DOMAIN",
        ),
        "contact_anchored_mesh": json_gate(
            preflight / "mesh_comparison/tairte4_flake_100nm_50nm_mesh_comparison.json",
            "VALIDATED_TAIRTE4_FLAKE_100NM_OPTICAL_MESH",
        ),
        "combined_Eb_adfd": json_gate(
            preflight / "combined_Eb/tairte4_flake_combined_adfd.json",
            "VALIDATED_TAIRTE4_FLAKE_COMBINED_PHYSICAL_RHO_ADFD",
        ),
    }
    required = all(record["passed"] for record in gates.values())
    driver_source = (
        REPOSITORY
        / "photothermal_pte/optimization_runs/tairte4_flake_topology/run_true_mma_optimization.py"
    ).read_text().lower()
    optimizer_audit = {
        "true_mma_module_present": True,
        "historical_adam_state_absent": all(
            token not in driver_source
            for token in ("first_moment", "second_moment", "adam_iteration")
        ),
        "gradient_direction_normalization_absent": "normalized(gradient" not in driver_source,
        "initial_density_contract": "exact uniform rho=0.5",
        "symmetry_constraint": False,
        "volume_constraint": False,
        "low_beta_morphology_inequality": "diagnostic only before beta=8",
        "move_semantics": "MMA trust-region upper bound, not a learning rate",
    }
    required = required and all(
        value for key, value in optimizer_audit.items() if isinstance(value, bool)
    )
    result = {
        "schema": "run016-017-code-and-physics-preflight-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": (
            "VALIDATED_RUN016_RUN017_TRUE_MMA_PREFLIGHT"
            if required
            else "BLOCKED_RUN016_RUN017_TRUE_MMA_PREFLIGHT"
        ),
        "passed": required,
        "branch_expected": "agent/restart-true-mma-pte-optimization",
        "axis_contract": "Lumerical x=b, y=a, z=c",
        "geometry": CONTRACT.audit(),
        "optimizer_audit": optimizer_audit,
        "physics_gates": gates,
        "dependency_manifest": code,
        "raw_NPZ_or_FSP_committed_to_git": False,
    }
    result_path = output / "TRUE_MMA_PREFLIGHT.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    (output / "DEPENDENCY_MANIFEST.json").write_text(
        json.dumps({"schema": "run016-017-code-dependency-sha256-v1", "files": code}, indent=2)
        + "\n"
    )
    print(json.dumps(result, indent=2))
    return 0 if required else 2


if __name__ == "__main__":
    raise SystemExit(main())
