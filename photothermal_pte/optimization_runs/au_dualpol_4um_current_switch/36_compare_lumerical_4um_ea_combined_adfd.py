#!/usr/bin/env python3
"""Hash-validate one Ea/Eb projected-density or latent AD--FD pair."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import traceback
from typing import Any

import numpy as np

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.lumerical_4um_adfd import (
    array_sha256,
    centered_adfd_metrics,
    centered_pair_reconstruction_metrics,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.lumerical_4um_density import (
    density_state_sha256,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.lumerical_4um_design_mapping import (
    OPTIMIZER_250NM_MAPPING,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.validation_provenance import (
    sha256,
)


def _pair_contract(manifest: dict[str, Any]) -> dict[str, str | bool]:
    """Resolve polarization and design coordinate from a prepared manifest."""

    status = str(manifest.get("status", ""))
    for polarization in ("Ea", "Eb"):
        token = polarization.upper()
        projected = f"PREPARED_LUMERICAL_4UM_{token}_COMBINED_ADFD_PAIR"
        latent = f"PREPARED_LUMERICAL_4UM_{token}_LATENT_COMBINED_ADFD_PAIR"
        if status not in (projected, latent):
            continue
        recorded_polarization = manifest.get("polarization", polarization)
        if recorded_polarization != polarization:
            raise RuntimeError("pair manifest polarization/status mismatch")
        is_latent = status == latent
        coordinate = (
            "latent_81x81_before_filter_projection"
            if is_latent
            else "shared_projected_nodal_occupancy"
        )
        coordinate_token = "LATENT" if is_latent else "PROJECTED_DENSITY"
        return {
            "polarization": polarization,
            "is_latent": is_latent,
            "design_coordinate": coordinate,
            "validated_status": (
                f"VALIDATED_LUMERICAL_4UM_{token}_{coordinate_token}_COMBINED_ADFD"
            ),
            "failed_status": (
                f"FAILED_LUMERICAL_4UM_{token}_{coordinate_token}_COMBINED_ADFD"
            ),
        }
    raise RuntimeError("unsupported centered-pair manifest status")


def _artifact(path: Path) -> dict[str, Any]:
    value = path.expanduser().resolve()
    return {
        "path": str(value),
        "size_bytes": value.stat().st_size,
        "sha256": sha256(value),
    }


def _load_record(path: Path) -> tuple[Path, dict[str, Any]]:
    value = path.expanduser().resolve()
    return value, json.loads(value.read_text(encoding="utf-8"))


def _check_manifest_artifact(manifest: dict[str, Any], name: str) -> Path:
    record = manifest["artifacts"][name]
    path = Path(record["path"]).resolve()
    if sha256(path) != record["sha256"]:
        raise RuntimeError(f"prepared {name} artifact changed")
    return path


def _density_sha(record: dict[str, Any]) -> str:
    return str(record["projected_density_input"]["density_state"]["density_state_sha256"])


def _validate_case(
    *,
    polarization: str,
    sign: str,
    density_state: dict[str, Any],
    baseline_forward: dict[str, Any],
    forward_path: Path,
    forward: dict[str, Any],
    pde_path: Path,
    pde: dict[str, Any],
) -> dict[str, Any]:
    expected_density_sha = str(density_state["density_state_sha256"])
    if (
        forward.get("all_gates_passed") is not True
        or forward.get("case") != "import_density"
    ):
        raise RuntimeError(f"{sign} Lumerical forward did not pass")
    if (
        forward.get("polarization") != polarization
        or pde.get("polarization") != polarization
    ):
        raise RuntimeError(f"{sign} polarization is not {polarization}")
    if _density_sha(forward) != expected_density_sha:
        raise RuntimeError(f"{sign} forward density hash differs")
    if (
        pde.get("status") != "VALIDATED_LUMERICAL_4UM_GRAY_Q_CUSTOM_CUDA_PDE"
        or pde.get("passed") is not True
    ):
        raise RuntimeError(f"{sign} custom-CUDA PDE result did not pass")
    if pde.get("density_state", {}).get("density_state_sha256") != expected_density_sha:
        raise RuntimeError(f"{sign} PDE density hash differs")
    forward_input = pde.get("inputs", {}).get("forward_result", {})
    if (
        Path(forward_input.get("path", "")).resolve() != forward_path
        or sha256(forward_path) != forward_input.get("sha256")
    ):
        raise RuntimeError(f"{sign} PDE/forward JSON binding differs")
    for key in ("solver_version", "mesh_spec", "source_calibration_sha256", "accelerator_policy"):
        if forward.get(key) != baseline_forward.get(key):
            raise RuntimeError(f"{sign} forward changed baseline {key}")
    if (
        forward.get("GPU_log_evidence", {}).get("requested_gpu_uuid")
        != baseline_forward.get("GPU_log_evidence", {}).get("requested_gpu_uuid")
    ):
        raise RuntimeError(f"{sign} forward changed physical GPU UUID")
    return {
        "forward_result": _artifact(forward_path),
        "PDE_result": _artifact(pde_path),
        "density_state_sha256": expected_density_sha,
        "current_A": float(pde["current_A"]),
        "forward_solver_wall_time_s": float(forward["solver_wall_time_s"]),
        "PDE_wall_time_s": float(pde["wall_s"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-manifest", required=True, type=Path)
    parser.add_argument("--adjoint-result", required=True, type=Path)
    parser.add_argument("--plus-forward-result", required=True, type=Path)
    parser.add_argument("--plus-pde-result", required=True, type=Path)
    parser.add_argument("--minus-forward-result", required=True, type=Path)
    parser.add_argument("--minus-pde-result", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    try:
        _, manifest_preview = _load_record(args.pair_manifest)
        preview_contract = _pair_contract(manifest_preview)
        result_name = (
            f"{str(preview_contract['polarization']).lower()}_combined_adfd_result.json"
        )
    except Exception:
        result_name = "combined_adfd_result.json"
    result_path = output / result_name
    result: dict[str, Any] = {
        "status": "FAILED_LUMERICAL_4UM_COMBINED_ADFD_INPUT_AUDIT",
        "passed": False,
        "Maxwell_solves_this_invocation": 0,
        "custom_CUDA_solves_this_invocation": 0,
        "Lumerical_HEAT_or_CHARGE_solves": 0,
        "optimizer_iterations": 0,
    }
    started = time.monotonic()
    try:
        manifest_path, manifest = _load_record(args.pair_manifest)
        adjoint_path, adjoint = _load_record(args.adjoint_result)
        plus_forward_path, plus_forward = _load_record(args.plus_forward_result)
        plus_pde_path, plus_pde = _load_record(args.plus_pde_result)
        minus_forward_path, minus_forward = _load_record(args.minus_forward_result)
        minus_pde_path, minus_pde = _load_record(args.minus_pde_result)
        pair_contract = _pair_contract(manifest)
        polarization = str(pair_contract["polarization"])
        is_latent = bool(pair_contract["is_latent"])
        design_coordinate = str(pair_contract["design_coordinate"])
        validated_status = str(pair_contract["validated_status"])
        failed_status = str(pair_contract["failed_status"])
        if manifest.get("passed") is not True:
            raise RuntimeError("centered-pair manifest did not pass")
        if (
            adjoint.get("status")
            != "COMPLETED_LUMERICAL_4UM_GRAY_MAXWELL_ADJOINT_PREPARATION"
            or adjoint.get("passed") is not True
        ):
            raise RuntimeError("adjoint preparation did not pass")
        if (
            adjoint.get("polarization") != polarization
            or adjoint.get("AD_FD_claimed") is not False
        ):
            raise RuntimeError("adjoint record has invalid pre-AD-FD state")
        direction_path = _check_manifest_artifact(manifest, "direction")
        baseline_density_path = _check_manifest_artifact(manifest, "baseline_density")
        plus_density_path = _check_manifest_artifact(manifest, "plus_density")
        minus_density_path = _check_manifest_artifact(manifest, "minus_density")
        direction = np.load(direction_path, allow_pickle=False)
        baseline_density = np.load(baseline_density_path, allow_pickle=False)
        plus_density = np.load(plus_density_path, allow_pickle=False)
        minus_density = np.load(minus_density_path, allow_pickle=False)
        direction_label = (
            "adfd-latent-direction-v1"
            if is_latent
            else "adfd-direction-v1"
        )
        if array_sha256(direction, label=direction_label) != manifest["direction_sha256"]:
            raise RuntimeError("direction semantic SHA differs")
        loaded_density_shas = {
            label: density_state_sha256(value)
            for label, value in (
                ("baseline_density", baseline_density),
                ("plus_density", plus_density),
                ("minus_density", minus_density),
            )
        }
        for label, state_sha in loaded_density_shas.items():
            if state_sha != manifest[label]["density_state_sha256"]:
                raise RuntimeError(f"prepared {label} semantic SHA differs")
        baseline_density_sha = loaded_density_shas["baseline_density"]
        if baseline_density_sha != adjoint.get("density_state", {}).get(
            "density_state_sha256"
        ):
            raise RuntimeError("pair baseline density differs from adjoint density")
        baseline_forward_path = Path(
            adjoint["artifacts"]["forward_result"]["path"]
        ).resolve()
        if sha256(baseline_forward_path) != adjoint["artifacts"]["forward_result"]["sha256"]:
            raise RuntimeError("baseline forward record changed")
        baseline_forward = json.loads(baseline_forward_path.read_text(encoding="utf-8"))
        if baseline_forward.get("polarization") != polarization:
            raise RuntimeError("adjoint forward polarization differs from pair")
        gradient_path = Path(
            adjoint["artifacts"]["gradient_NPZ"]["path"]
        ).resolve()
        if sha256(gradient_path) != adjoint["artifacts"]["gradient_NPZ"]["sha256"]:
            raise RuntimeError("adjoint gradient artifact changed")
        with np.load(gradient_path, allow_pickle=False) as gradient_file:
            gradient = np.asarray(gradient_file["gradient_total_A"], float)
            gradient_density = np.asarray(gradient_file["rho_nodal"], float)
        if density_state_sha256(gradient_density) != baseline_density_sha:
            raise RuntimeError("gradient NPZ density differs from pair baseline density")
        plus = _validate_case(
            polarization=polarization,
            sign="plus",
            density_state=manifest["plus_density"],
            baseline_forward=baseline_forward,
            forward_path=plus_forward_path,
            forward=plus_forward,
            pde_path=plus_pde_path,
            pde=plus_pde,
        )
        minus = _validate_case(
            polarization=polarization,
            sign="minus",
            density_state=manifest["minus_density"],
            baseline_forward=baseline_forward,
            forward_path=minus_forward_path,
            forward=minus_forward,
            pde_path=minus_pde_path,
            pde=minus_pde,
        )
        step = float(manifest["step"])
        if is_latent:
            if manifest.get("mapping") != OPTIMIZER_250NM_MAPPING.audit():
                raise RuntimeError("latent pair mapping contract differs")
            if manifest.get("mapping_role") != "active_lumerical_optimizer_250nm":
                raise RuntimeError("latent pair is not bound to the active 250-nm mapping")
            latent_path = _check_manifest_artifact(manifest, "latent_baseline")
            latent_plus_path = _check_manifest_artifact(manifest, "latent_plus")
            latent_minus_path = _check_manifest_artifact(manifest, "latent_minus")
            latent = np.load(latent_path, allow_pickle=False)
            latent_plus = np.load(latent_plus_path, allow_pickle=False)
            latent_minus = np.load(latent_minus_path, allow_pickle=False)
            if (
                array_sha256(latent, label="adfd-latent-baseline-v1")
                != manifest["latent_baseline_sha256"]
            ):
                raise RuntimeError("latent baseline semantic SHA differs")
            beta = float(manifest["beta"])
            if not np.array_equal(
                OPTIMIZER_250NM_MAPPING.physical(latent, beta), baseline_density
            ):
                raise RuntimeError("latent baseline does not reproduce projected baseline")
            if not np.array_equal(
                OPTIMIZER_250NM_MAPPING.physical(latent_plus, beta), plus_density
            ):
                raise RuntimeError("latent plus state does not reproduce projected plus")
            if not np.array_equal(
                OPTIMIZER_250NM_MAPPING.physical(latent_minus, beta), minus_density
            ):
                raise RuntimeError("latent minus state does not reproduce projected minus")
            gradient_for_metrics = OPTIMIZER_250NM_MAPPING.vjp(
                latent, gradient, beta
            )
            projected_direction = OPTIMIZER_250NM_MAPPING.jvp(
                latent, direction, beta
            )
            projected_contraction = float(np.vdot(gradient, projected_direction))
            latent_contraction = float(np.vdot(gradient_for_metrics, direction))
            chain_scale = max(
                abs(projected_contraction),
                abs(latent_contraction),
                np.finfo(float).tiny,
            )
            gradient_chain = {
                "beta": beta,
                "projected_gradient_L2_A": float(np.linalg.norm(gradient)),
                "latent_gradient_L2_A": float(np.linalg.norm(gradient_for_metrics)),
                "projected_JVP_L2": float(np.linalg.norm(projected_direction)),
                "projected_gradient_dot_projected_JVP_A": projected_contraction,
                "latent_gradient_dot_latent_direction_A": latent_contraction,
                "chain_transpose_relative_error": abs(
                    projected_contraction - latent_contraction
                )
                / chain_scale,
            }
            pair_baseline = latent
            pair_plus = latent_plus
            pair_minus = latent_minus
            scope = (
                f"one independent smooth direction of {polarization} current with "
                "respect "
                "to 81x81 latent density before filter/projection"
            )
        else:
            gradient_for_metrics = gradient
            gradient_chain = {
                "projected_gradient_used_directly": True,
            }
            pair_baseline = baseline_density
            pair_plus = plus_density
            pair_minus = minus_density
            scope = (
                f"one independent smooth direction of {polarization} current with "
                "respect "
                "to shared projected nodal occupancy"
            )
        metrics = centered_adfd_metrics(
            gradient=gradient_for_metrics,
            direction=direction,
            step=step,
            baseline_current_A=float(adjoint["current_A"]),
            plus_current_A=plus["current_A"],
            minus_current_A=minus["current_A"],
        )
        pair_reconstruction = centered_pair_reconstruction_metrics(
            baseline=pair_baseline,
            direction=direction,
            plus=pair_plus,
            minus=pair_minus,
            step=step,
        )
        gates = {
            "hash_bound_inputs_passed": True,
            "centered_pair_reconstruction_within_float64_roundoff": pair_reconstruction[
                "within_float64_roundoff"
            ],
            "plus_minus_signal_relative_gt_1e_6": metrics[
                "plus_minus_signal_relative_to_current"
            ]
            > 1.0e-6,
            "AD_and_FD_have_same_nonzero_sign": metrics["same_nonzero_sign"],
            "one_direction_combined_AD_FD_relative_error_lt_1pct": metrics[
                "relative_error"
            ]
            < 1.0e-2,
        }
        passed = all(gates.values())
        result = {
            "status": validated_status if passed else failed_status,
            "passed": passed,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "scope": scope,
            "polarization": polarization,
            "design_coordinate": design_coordinate,
            "step": step,
            "metrics": metrics,
            "gradient_chain": gradient_chain,
            "centered_pair_reconstruction": pair_reconstruction,
            "gates": gates,
            "baseline": {
                "current_A": float(adjoint["current_A"]),
                "adjoint_result": _artifact(adjoint_path),
                "gradient_NPZ": _artifact(gradient_path),
            },
            "plus": plus,
            "minus": minus,
            "pair_manifest": _artifact(manifest_path),
            "direction_sha256": manifest["direction_sha256"],
            "density_state_binding": {
                "baseline_density_sha256": baseline_density_sha,
                "plus_density_sha256": loaded_density_shas["plus_density"],
                "minus_density_sha256": loaded_density_shas["minus_density"],
                "pair_baseline_equals_adjoint_density": True,
                "gradient_NPZ_density_equals_pair_baseline": True,
            },
            "relative_error_definition": "abs(AD-FD)/max(abs(AD),abs(FD))",
            "empirical_gradient_rescaling": False,
            "finite_difference_fit": False,
            "evidence_Maxwell_forward_solves": 2,
            "evidence_custom_CUDA_forward_evaluations": 2,
            "Maxwell_solves_this_invocation": 0,
            "custom_CUDA_solves_this_invocation": 0,
            "Lumerical_HEAT_or_CHARGE_solves": 0,
            "optimizer_iterations": 0,
            "wall_s": time.monotonic() - started,
        }
    except Exception as error:
        result.update(
            error=f"{type(error).__name__}: {error}",
            traceback=traceback.format_exc(),
            wall_s=time.monotonic() - started,
        )
    result_path.write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, default=str))
    return 0 if result.get("passed", False) else 1


if __name__ == "__main__":
    raise SystemExit(main())
