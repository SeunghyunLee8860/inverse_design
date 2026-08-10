#!/usr/bin/env python3
"""Prepare the exact selected-grid full-latent combined adjoint state."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]
for path in (HERE, REPOSITORY):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from photothermal_pte.finite_inverse_design.native_yee_q import EPS0  # noqa: E402
from photothermal_pte.optimization_runs.axis_contract import get_axis_contract  # noqa: E402
from run_production_combined_adfd_smoke import (  # noqa: E402
    FREQUENCY_HZ,
    boundary_energy,
    compact_forward,
    contract_configuration,
    load_operator,
    map_q,
    open_fdtd,
    optical_gradient,
    prepare_common_grid_source,
    pullback_q,
    run_adjoint,
    run_forward,
    solve_base_thermal,
)
from production_density_mapping import ProductionDensityMapping  # noqa: E402


STATUS = "COMPLETED_SELECTED_FULL_LATENT_ADJOINT_PREPARATION"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, object]:
    path = path.expanduser().resolve()
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-fsp", type=Path, required=True)
    parser.add_argument("--base-sha256", required=True)
    parser.add_argument("--jacobian-dir", type=Path, required=True)
    parser.add_argument("--latent-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu-device", default="GPU 0")
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--incident-power-W", type=float, default=1.3822950233084244e-13)
    parser.add_argument("--axis-contract", default="legacy_lumerical_x_a_y_b")
    parser.add_argument("--polarization-label", default=None)
    parser.add_argument("--polarization-angle-deg", type=float, default=None)
    parser.add_argument("--objective-sign", type=float, choices=(-1.0, 1.0), default=1.0)
    parser.add_argument(
        "--resume-completed-solves",
        action="store_true",
        help="Reuse completed forward/adjoint FSP files in a partial output directory.",
    )
    parser.add_argument(
        "--allow-closed-unit-interval-latent",
        action="store_true",
        help="Allow optimizer box-bound latent values exactly at 0 or 1; values outside [0,1] remain invalid.",
    )
    args = parser.parse_args()
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()) and not args.resume_completed_solves:
        raise RuntimeError(f"refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    completed_forward = output / "full_latent_base.fsp"
    completed_adjoint = output / "selected_full_latent_adjoint_gpu.fsp"
    if args.resume_completed_solves and not (
        completed_forward.is_file() and completed_adjoint.is_file()
    ):
        raise RuntimeError(
            "resume requested without both completed forward and adjoint FSP files"
        )
    result_path = output / "selected_full_latent_adjoint_preparation_result.json"
    result: dict[str, object] = {"status": "FAILED_SELECTED_FULL_LATENT_ADJOINT_PREPARATION", "passed": False}
    fdtd = None
    started = time.monotonic()
    try:
        base_fsp = args.base_fsp.expanduser().resolve()
        if sha256(base_fsp) != args.base_sha256:
            raise RuntimeError("base FSP SHA mismatch")
        latent_path = args.latent_npz.expanduser().resolve()
        latent_data = np.load(latent_path)
        latent = np.asarray(latent_data["latent"], float)
        mapping = ProductionDensityMapping()
        finite = bool(np.all(np.isfinite(latent)))
        if args.allow_closed_unit_interval_latent:
            domain_valid = finite and np.min(latent) >= 0.0 and np.max(latent) <= 1.0
            latent_domain_contract = "closed unit interval [0,1] for box-constrained optimization"
        else:
            domain_valid = finite and np.min(latent) > 0.0 and np.max(latent) < 1.0
            latent_domain_contract = "strict interior (0,1) for AD-FD certification"
        if latent.shape != mapping.shape or not domain_valid:
            raise RuntimeError(f"latent baseline violates {latent_domain_contract}")
        rho = mapping.physical(latent, args.beta)
        if "rho_reconstructed" in latent_data:
            reconstruction_error = float(np.max(np.abs(rho - latent_data["rho_reconstructed"])))
            if reconstruction_error >= 1.0e-14:
                raise RuntimeError("latent artifact reconstruction changed")
        operator, operator_rho, operator_meta = load_operator(
            args.jacobian_dir,
            "VALIDATED_SELECTED_PRODUCTION_COMPLEX_COMPONENT_YEE_JACOBIAN",
        )
        # The stored component operator is the derivative of the explicitly
        # linear epsilon(rho) law plus fixed solver interpolation. It is therefore
        # state independent; do not claim that its stored baseline equals rho.
        if operator_rho.shape != rho.shape:
            raise RuntimeError("component-J and latent density shapes differ")
        config = contract_configuration("selected_production")
        axis_contract = get_axis_contract(args.axis_contract)
        if args.polarization_label is None or args.polarization_angle_deg is None:
            if args.axis_contract != "legacy_lumerical_x_a_y_b":
                raise RuntimeError("corrected runs require explicit polarization label and angle")
        else:
            axis_contract.validate_polarization(
                args.polarization_label, args.polarization_angle_deg
            )
        fdtd, audit, runtime = open_fdtd(args.gpu_device)
        base = run_forward(
            fdtd,
            audit,
            runtime,
            base_fsp=base_fsp,
            rho=rho,
            role="full_latent_base",
            output=output,
            imported_object=str(config["imported_object"]),
            nodes=config["nodes"],
            polarization_angle_deg=args.polarization_angle_deg,
            completed_project=(completed_forward if args.resume_completed_solves else None),
        )
        base_data, base_mapping = map_q(
            base["q"], design_half_span_m=float(config["design_half_span_m"])
        )
        state, pair, signed_current, thermal_parts, target_sensitivity = solve_base_thermal(
            base_data,
            rho,
            args.cuda_device,
            config["density_forward"],
            config["density_transpose"],
            axis_contract=axis_contract,
        )
        objective = args.objective_sign * signed_current
        thermal_parts = {
            name: args.objective_sign * np.asarray(value, float)
            for name, value in thermal_parts.items()
        }
        target_sensitivity = args.objective_sign * target_sensitivity
        pulled, pullback_records = pullback_q(base["q"], base_data, target_sensitivity)
        native_source = np.zeros_like(base["electric"], complex)
        for index, component in enumerate("xyz"):
            native_source[..., 0, index] = (
                0.5
                * EPS0
                * (2.0 * np.pi * FREQUENCY_HZ)
                * np.imag(base["epsilon"][..., 0, index])
                * pulled[component]
                * base["electric"][..., 0, index]
            )
        template = output / "selected_full_latent_adjoint_template.fsp"
        profile_scale, base_amplitude, source_meta = prepare_common_grid_source(
            fdtd,
            audit,
            base_project=Path(base["project"]["path"]),
            grid=base["grid"],
            native_source=native_source,
            template=template,
        )
        adjoint = run_adjoint(
            fdtd,
            audit,
            runtime,
            template=template,
            project=output / "selected_full_latent_adjoint_gpu.fsp",
            completed_project=(completed_adjoint if args.resume_completed_solves else None),
        )
        gradient_optical, optical_meta = optical_gradient(
            operator,
            forward=base,
            adjoint=adjoint,
            pulled=pulled,
            profile_scale=profile_scale,
            base_amplitude=base_amplitude,
        )
        gradient_thermal = np.asarray(thermal_parts["total"], float)
        gradient_physical = gradient_optical + gradient_thermal
        gradient_latent = mapping.vjp(latent, gradient_physical, args.beta)
        if not np.all(np.isfinite(gradient_latent)) or np.linalg.norm(gradient_latent) == 0.0:
            raise RuntimeError("latent gradient is zero or nonfinite")
        worst_pullback = max(float(row["transpose_dot_error"]) for row in pullback_records.values())
        worst_residual = max(pair.forward.explicit_relative_residual, pair.adjoint.explicit_relative_residual)
        energy = boundary_energy(state, pair.forward.solution)
        passed = bool(
            base["closure"] < 0.005
            and base_mapping["internal_relative_power_error"] < 0.005
            and worst_pullback < 1.0e-12
            and worst_residual < 1.0e-8
            and energy < 0.01
            and float(base["log_audit"]["final_auto_shutoff"]) < 1.0e-5
            and float(adjoint["log_audit"]["final_auto_shutoff"]) < 1.0e-5
            and optical_meta["forward_adjoint_coordinate_mismatch_m"] < 2.0e-18
        )
        raw = output / "selected_full_latent_adjoint_preparation.npz"
        np.savez_compressed(
            raw,
            latent=latent,
            filtered=mapping.filtered(latent),
            rho=rho,
            gradient_physical_A=gradient_physical,
            gradient_latent_A=gradient_latent,
            gradient_optical_A=gradient_optical,
            gradient_thermal_A=gradient_thermal,
        )
        result = {
            "status": STATUS if passed else "FAILED_SELECTED_FULL_LATENT_ADJOINT_PREPARATION",
            "passed": passed,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "scope": "selected 373x373 latent -> finite conic filter -> beta=2 projection -> optical/thermal/PTE",
            "beta": args.beta,
            "axis_contract": {
                "name": axis_contract.name,
                "crystal_axis_by_solver_axis": axis_contract.crystal_axis_by_solver_axis,
                "epsilon_axis_by_solver_axis": axis_contract.epsilon_axis_by_solver_axis,
                "kappa_xyz_W_mK": axis_contract.kappa_xyz_W_mK,
                "sigma_xy_S_m": axis_contract.sigma_xy_S_m,
                "seebeck_xy_V_K": axis_contract.seebeck_xy_V_K,
            },
            "polarization": {
                "label": args.polarization_label,
                "requested_angle_deg": args.polarization_angle_deg,
                "readback_angle_deg": base["source_polarization_angle_deg"],
            },
            "latent_domain_contract": latent_domain_contract,
            "latent_range": [float(np.min(latent)), float(np.max(latent))],
            "physical_density_range": [float(np.min(rho)), float(np.max(rho))],
            "objective_A": objective,
            "signed_current_A": signed_current,
            "objective_sign": args.objective_sign,
            "objective_definition": "objective_sign * signed_current_A / incident power",
            "incident_power_W": args.incident_power_W,
            "gradient_norms_A": {
                "physical": float(np.linalg.norm(gradient_physical)),
                "latent": float(np.linalg.norm(gradient_latent)),
                "optical_physical": float(np.linalg.norm(gradient_optical)),
                "thermal_physical": float(np.linalg.norm(gradient_thermal)),
            },
            "component_J_state_independence": "epsilon(rho)=1+rho*(epsilon_SiO2-1), followed by fixed linear Yee interpolation",
            "operator": operator_meta,
            "base_forward": compact_forward(base),
            "base_mapping": base_mapping,
            "thermal": {
                "forward_residual": pair.forward.explicit_relative_residual,
                "adjoint_residual": pair.adjoint.explicit_relative_residual,
                "energy_balance": energy,
            },
            "pullback": pullback_records,
            "adjoint_source": source_meta,
            "optical_gradient": optical_meta,
            "gates": {
                "optical_closure": base["closure"],
                "Q_mapping_error": base_mapping["internal_relative_power_error"],
                "Q_pullback_transpose_error": worst_pullback,
                "thermal_residual": worst_residual,
                "thermal_energy_balance": energy,
                "base_auto_shutoff": base["log_audit"]["final_auto_shutoff"],
                "adjoint_auto_shutoff": adjoint["log_audit"]["final_auto_shutoff"],
                "coordinate_mismatch_m": optical_meta["forward_adjoint_coordinate_mismatch_m"],
            },
            "inputs": {
                "base_FSP": artifact(base_fsp),
                "latent_reconstruction": artifact(latent_path),
            },
            "raw_artifact": artifact(raw),
            "Maxwell_forward_solves": 1,
            "Maxwell_adjoint_solves": 1,
            "thermal_forward_solves": 1,
            "thermal_adjoint_solves": 1,
            "optimizer_started": False,
            "CPU_FDTD_fallback": False,
            "CPU_thermal_solve_fallback": False,
            "empirical_normalization": False,
            "gradient_rescaling": False,
            "wall_s": time.monotonic() - started,
        }
    except Exception as exc:
        result.update({"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(), "wall_s": time.monotonic() - started})
    finally:
        if fdtd is not None:
            try:
                fdtd.close()
            except Exception:
                pass
        result_path.write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(json.dumps(result, indent=2, default=str))
    return 0 if result.get("passed", False) else 1


if __name__ == "__main__":
    raise SystemExit(main())
