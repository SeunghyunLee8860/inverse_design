#!/usr/bin/env python3
"""Combine passed Ea/Eb latent certificates into one signed objective audit."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.lumerical_4um_adfd import (
    array_sha256,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.lumerical_4um_density import (
    density_state_audit,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.lumerical_4um_design_mapping import (
    NOMINAL_MAPPING,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.lumerical_4um_signed_objective import (
    signed_dual_objective_point,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.objective import (
    PTE_CURRENT_SIGN_CONVENTION,
    epigraph_constraints,
    useful_currents,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.validation_provenance import (
    sha256,
)


STATUS = "VALIDATED_LUMERICAL_4UM_SIGNED_DUAL_OBJECTIVE_ONE_DIRECTION"


def _artifact(path: Path) -> dict[str, object]:
    value = path.expanduser().resolve()
    return {
        "path": str(value),
        "size_bytes": value.stat().st_size,
        "sha256": sha256(value),
    }


def _relative_error(left: float, right: float) -> float:
    return abs(left - right) / max(abs(left), abs(right), np.finfo(float).tiny)


def _load_certificate(path: Path, polarization: str) -> dict[str, Any]:
    result_path = path.expanduser().resolve()
    result = json.loads(result_path.read_text(encoding="utf-8"))
    expected_status = f"VALIDATED_LUMERICAL_4UM_{polarization.upper()}_LATENT_COMBINED_ADFD"
    if result.get("status") != expected_status or result.get("passed") is not True:
        raise RuntimeError(f"{polarization} input is not a passed latent AD-FD certificate")
    if result.get("polarization") != polarization:
        raise RuntimeError(f"{polarization} certificate polarization mismatch")
    if result.get("design_coordinate") != "latent_81x81_before_filter_projection":
        raise RuntimeError(f"{polarization} certificate is not in latent coordinates")
    if not all(bool(value) for value in result.get("gates", {}).values()):
        raise RuntimeError(f"{polarization} latent AD-FD gate is incomplete")

    pair_record = result["pair_manifest"]
    pair_path = Path(pair_record["path"]).resolve()
    if sha256(pair_path) != pair_record["sha256"]:
        raise RuntimeError(f"{polarization} pair manifest SHA mismatch")
    pair = json.loads(pair_path.read_text(encoding="utf-8"))
    expected_pair_status = (
        f"PREPARED_LUMERICAL_4UM_{polarization.upper()}_LATENT_COMBINED_ADFD_PAIR"
    )
    recorded_pair_polarization = pair.get("polarization", polarization)
    if (
        pair.get("passed") is not True
        or pair.get("status") != expected_pair_status
        or recorded_pair_polarization != polarization
    ):
        raise RuntimeError(f"{polarization} pair manifest binding failed")

    latent_record = pair["artifacts"]["latent_baseline"]
    latent_path = Path(latent_record["path"]).resolve()
    if sha256(latent_path) != latent_record["sha256"]:
        raise RuntimeError(f"{polarization} latent baseline file SHA mismatch")
    latent = np.asarray(np.load(latent_path, allow_pickle=False), dtype=np.float64)
    if array_sha256(latent, label="adfd-latent-baseline-v1") != pair["latent_baseline_sha256"]:
        raise RuntimeError(f"{polarization} semantic latent baseline hash mismatch")

    direction_record = pair["artifacts"]["direction"]
    direction_path = Path(direction_record["path"]).resolve()
    if sha256(direction_path) != direction_record["sha256"]:
        raise RuntimeError(f"{polarization} direction file SHA mismatch")
    direction = np.asarray(np.load(direction_path, allow_pickle=False), dtype=np.float64)
    if array_sha256(direction, label="adfd-latent-direction-v1") != result["direction_sha256"]:
        raise RuntimeError(f"{polarization} semantic direction hash mismatch")

    gradient_record = result["baseline"]["gradient_NPZ"]
    gradient_path = Path(gradient_record["path"]).resolve()
    if sha256(gradient_path) != gradient_record["sha256"]:
        raise RuntimeError(f"{polarization} gradient NPZ SHA mismatch")
    with np.load(gradient_path, allow_pickle=False) as gradient_file:
        projected = np.asarray(gradient_file["rho_nodal"], dtype=np.float64)
        gradient = np.asarray(gradient_file["gradient_total_A"], dtype=np.float64)
    density_sha = density_state_audit(projected)["density_state_sha256"]
    if density_sha != result["density_state_binding"]["baseline_density_sha256"]:
        raise RuntimeError(f"{polarization} projected baseline density mismatch")
    beta = float(result["gradient_chain"]["beta"])
    mapped = NOMINAL_MAPPING.physical(latent, beta)
    mapping_error = float(np.max(np.abs(mapped - projected)))
    if mapping_error != 0.0:
        raise RuntimeError(f"{polarization} latent-to-projected state is not exact")
    return {
        "path": result_path,
        "result": result,
        "pair_path": pair_path,
        "latent_path": latent_path,
        "direction_path": direction_path,
        "gradient_path": gradient_path,
        "latent": latent,
        "direction": direction,
        "projected": projected,
        "gradient": gradient,
        "beta": beta,
        "mapping_error": mapping_error,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ea-certificate", required=True, type=Path)
    parser.add_argument("--eb-certificate", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    ea = _load_certificate(args.ea_certificate, "Ea")
    eb = _load_certificate(args.eb_certificate, "Eb")
    if ea["beta"] != eb["beta"]:
        raise RuntimeError("Ea/Eb projection beta differs")
    if not np.array_equal(ea["latent"], eb["latent"]):
        raise RuntimeError("Ea/Eb latent baseline differs")
    if not np.array_equal(ea["projected"], eb["projected"]):
        raise RuntimeError("Ea/Eb projected density differs")
    if not np.array_equal(ea["direction"], eb["direction"]):
        raise RuntimeError("Ea/Eb validation direction differs")
    if float(ea["result"]["step"]) != float(eb["result"]["step"]):
        raise RuntimeError("Ea/Eb centered-FD step differs")

    current_a = float(ea["result"]["baseline"]["current_A"])
    current_b = float(eb["result"]["baseline"]["current_A"])
    utility_a, utility_b = useful_currents(current_a, current_b)
    epigraph = min(utility_a, utility_b)
    point = signed_dual_objective_point(
        latent=ea["latent"],
        beta=ea["beta"],
        current_a_A=current_a,
        current_b_A=current_b,
        gradient_a_projected_A=ea["gradient"],
        gradient_b_projected_A=eb["gradient"],
        epigraph_A=epigraph,
    )
    if point["balanced_gradient_latent_A"] is None:
        raise RuntimeError("baseline utility tie prevents a unique minimum gradient")

    step = float(ea["result"]["step"])
    plus_currents = (
        float(ea["result"]["plus"]["current_A"]),
        float(eb["result"]["plus"]["current_A"]),
    )
    minus_currents = (
        float(ea["result"]["minus"]["current_A"]),
        float(eb["result"]["minus"]["current_A"]),
    )
    plus_utilities = useful_currents(*plus_currents)
    minus_utilities = useful_currents(*minus_currents)
    active_plus = "Ea" if plus_utilities[0] < plus_utilities[1] else "Eb"
    active_minus = "Ea" if minus_utilities[0] < minus_utilities[1] else "Eb"
    direction = ea["direction"]
    balanced_ad = float(np.vdot(point["balanced_gradient_latent_A"], direction))
    balanced_fd = (
        min(plus_utilities) - min(minus_utilities)
    ) / (2.0 * step)
    constraint_ad = point["constraint_gradients_latent_A"].reshape(2, -1) @ direction.ravel()
    constraint_fd = (
        epigraph_constraints(*plus_currents, epigraph)
        - epigraph_constraints(*minus_currents, epigraph)
    ) / (2.0 * step)
    balanced_error = _relative_error(balanced_ad, balanced_fd)
    constraint_errors = np.asarray(
        [_relative_error(float(ad), float(fd)) for ad, fd in zip(constraint_ad, constraint_fd)]
    )
    switching_achieved = bool(current_a > 0.0 and current_b < 0.0)
    gates = {
        "Ea_and_Eb_passed_latent_ADFD_certificates": True,
        "common_latent_projected_state_beta_step_and_direction": True,
        "latent_to_projected_state_exact": ea["mapping_error"] == 0.0
        and eb["mapping_error"] == 0.0,
        "unique_same_active_polarization_at_baseline_plus_minus": (
            point["active_polarization"] == active_plus == active_minus
        ),
        "signed_balanced_objective_AD_FD_same_nonzero_sign": bool(
            balanced_ad * balanced_fd > 0.0
        ),
        "signed_balanced_objective_AD_FD_relative_error_lt_1pct": (
            balanced_error < 1.0e-2
        ),
        "both_epigraph_constraint_AD_FD_relative_errors_lt_1pct": bool(
            np.all(constraint_errors < 1.0e-2)
        ),
        "epigraph_constraints_match_shared_objective_module": bool(
            np.array_equal(
                point["epigraph_constraints_A"],
                epigraph_constraints(current_a, current_b, epigraph),
            )
        ),
        "opposite_current_switching_status_derived_from_signed_currents": True,
    }
    raw_output = output / "signed_dual_objective_gradients.npz"
    np.savez_compressed(
        raw_output,
        latent=ea["latent"],
        projected_density=ea["projected"],
        direction=direction,
        gradient_Ia_latent_A=point["gradient_a_latent_A"],
        gradient_Ib_latent_A=point["gradient_b_latent_A"],
        gradient_balanced_utility_latent_A=point["balanced_gradient_latent_A"],
        epigraph_constraint_gradients_latent_A=point[
            "constraint_gradients_latent_A"
        ],
    )
    passed = all(gates.values())
    report = {
        "status": STATUS if passed else "FAILED_LUMERICAL_4UM_SIGNED_DUAL_OBJECTIVE",
        "passed": passed,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "one common latent direction at one Ea/Eb baseline; combines already "
            "passed certificates and performs no solver call"
        ),
        "objective": "maximize t subject to t-I_Ea<=0 and t+I_Eb<=0",
        "current_sign_convention": PTE_CURRENT_SIGN_CONVENTION,
        "beta": ea["beta"],
        "step": step,
        "currents_A": {"Ea": current_a, "Eb": current_b},
        "utilities_A": {"Ea": utility_a, "Eb": utility_b},
        "balanced_utility_A": point["balanced_utility_A"],
        "active_polarization": point["active_polarization"],
        "opposite_current_switching_achieved": switching_achieved,
        "epigraph_A": epigraph,
        "epigraph_constraints_A": point["epigraph_constraints_A"].tolist(),
        "directional_derivatives_A_per_unit_rho": {
            "balanced_AD": balanced_ad,
            "balanced_centered_FD": balanced_fd,
            "balanced_relative_error": balanced_error,
            "constraint_AD": constraint_ad.tolist(),
            "constraint_centered_FD": constraint_fd.tolist(),
            "constraint_relative_errors": constraint_errors.tolist(),
        },
        "gradient_norms_latent_A": {
            "Ia": float(np.linalg.norm(point["gradient_a_latent_A"])),
            "Ib": float(np.linalg.norm(point["gradient_b_latent_A"])),
            "balanced_utility": float(
                np.linalg.norm(point["balanced_gradient_latent_A"])
            ),
        },
        "gates": gates,
        "inputs": {
            "Ea_certificate": _artifact(ea["path"]),
            "Eb_certificate": _artifact(eb["path"]),
            "Ea_pair_manifest": _artifact(ea["pair_path"]),
            "Eb_pair_manifest": _artifact(eb["pair_path"]),
            "Ea_gradient_NPZ": _artifact(ea["gradient_path"]),
            "Eb_gradient_NPZ": _artifact(eb["gradient_path"]),
        },
        "raw_output": _artifact(raw_output),
        "Maxwell_solves": 0,
        "custom_CUDA_solves": 0,
        "Lumerical_HEAT_or_CHARGE_solves": 0,
        "FDTDX_Maxwell_solves": 0,
        "optimizer_iterations": 0,
        "optimizer_enabled": False,
        "wall_s": time.monotonic() - started,
    }
    result_path = output / "signed_dual_objective_result.json"
    result_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
