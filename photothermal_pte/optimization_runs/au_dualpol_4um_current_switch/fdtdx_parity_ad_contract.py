"""Fail-closed contract for the future FDTDX optical AD-FD certificate.

This module audits the pinned reverse-mode implementation and defines the
dimensionless absorbed-power objective and four latent-space directions.  It
does not run a field solve and does not claim that a production gradient has
passed.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_parity_contract import (
    PHYSICS,
    grid_audit,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_parity_design_mapping import (
    MAPPING,
    NODE_SHAPE,
    deterministic_gray_latent,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_parity_fixed_materials import (
    TA_A,
    TA_B,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_parity_source_calibration import (
    ETA0_OHM,
)


FDTDX_ROOT = Path(
    "/home/seunghyun200/dependencies/fdtdx-f26f84b70a8cceec9b889553955a868624736bf1"
)
EQUINOX_CHECKPOINTED = Path(
    "/home/seunghyun200/.venvs/fdtdx-fresh-py312/lib/python3.12/"
    "site-packages/equinox/internal/_loop/checkpointed.py"
)
GRADIENT_SOURCES = {
    "fdtdx_config": FDTDX_ROOT / "src/fdtdx/config.py",
    "fdtdx_dispersion": FDTDX_ROOT / "src/fdtdx/dispersion.py",
    "fdtdx_fdtd": FDTDX_ROOT / "src/fdtdx/fdtd/fdtd.py",
    "fdtdx_initialization": FDTDX_ROOT / "src/fdtdx/fdtd/initialization.py",
    "fdtdx_update": FDTDX_ROOT / "src/fdtdx/fdtd/update.py",
    "equinox_checkpointed": EQUINOX_CHECKPOINTED,
}

CHECKPOINT_CANDIDATES = (16, 32, 64, 96)
SOURCE_REFERENCE_POWER_W = 1.882400012336031e-12
EPS0_F_PER_M = 8.854_187_8128e-12
OMEGA_RAD_S = 2.0 * math.pi * 299_792_458.0 / PHYSICS.wavelength_m
Q_PREFACTOR = 0.5 * OMEGA_RAD_S * EPS0_F_PER_M * ETA0_OHM**2


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def gradient_source_audit() -> dict[str, Any]:
    missing = [str(path) for path in GRADIENT_SOURCES.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing pinned gradient sources: {missing}")
    text = {
        name: path.read_text(encoding="utf-8")
        for name, path in GRADIENT_SOURCES.items()
    }
    checks = {
        "dispersion_documents_checkpointed_only": (
            "Dispersive simulations currently support only the ``checkpointed`` gradient"
            in text["fdtdx_dispersion"]
        ),
        "reversible_rejects_ADE_state": (
            "Use GradientConfig(method='checkpointed') instead." in text["fdtdx_fdtd"]
        ),
        "checkpointed_dispatch_uses_num_checkpoints": (
            'kind="lax" if config.only_forward is None else "checkpointed"'
            in text["fdtdx_fdtd"]
            and "config.gradient_config.num_checkpoints" in text["fdtdx_fdtd"]
        ),
        "checkpointed_loop_is_reverse_mode": (
            "Reverse-mode autodifferentiable while loop" in text["equinox_checkpointed"]
        ),
        "checkpoint_memory_is_state_copies": (
            "checkpoints`-many copies of `init_val`" in text["equinox_checkpointed"]
        ),
        "c3_enters_ADE_forward_update": (
            "disp_c3 = arrays.dispersive_c3" in text["fdtdx_update"]
            and "disp_c3 * arrays.fields.E" in text["fdtdx_update"]
        ),
        "source_material_sampling_is_stop_gradient_only": (
            "the FDTD VJP itself still propagates gradient"
            in text["fdtdx_initialization"]
        ),
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "source_sha256": {
            name: file_sha256(path) for name, path in GRADIENT_SOURCES.items()
        },
        "method_required": "checkpointed",
        "reversible_allowed": False,
        "production_gradient_validated": False,
    }


def checkpoint_memory_lower_bounds(
    candidates: tuple[int, ...] = CHECKPOINT_CANDIDATES,
) -> dict[str, Any]:
    memory = grid_audit()["resources"]["memory"]
    selected = memory["pole_cases_no_c4"]["3"]
    dynamic = int(selected["one_dynamic_checkpoint_lower_bound_bytes"])
    persistent = int(selected["persistent_array_lower_bound_bytes"])
    rows = {
        str(count): {
            "checkpoint_state_lower_bound_bytes": count * dynamic,
            "checkpoint_plus_persistent_lower_bound_bytes": persistent
            + count * dynamic,
            "checkpoint_plus_persistent_lower_bound_GiB": (persistent + count * dynamic)
            / 2**30,
        }
        for count in candidates
    }
    return {
        "status": "LOWER_BOUND_ONLY_NOT_A_FEASIBILITY_CLAIM",
        "one_dynamic_checkpoint_lower_bound_bytes": dynamic,
        "persistent_array_lower_bound_bytes": persistent,
        "candidates": rows,
        "excluded": memory["excluded_from_lower_bound"],
        "required_next_gate": (
            "bounded full-grid GPU value-and-gradient probe with peak-memory "
            "and forward/backward timing"
        ),
    }


def target_au_imag_epsilon(rho_cell: Any, *, xp: Any = np) -> Any:
    rho = xp.asarray(rho_cell)
    return 57.8 * rho + 69.36 * rho * rho


def normalized_target_absorption(
    *,
    rho_cell: Any,
    e_au: Any,
    e_tairte4: Any,
    volume_au: Any,
    volume_tairte4: Any,
    xp: Any = np,
) -> Any:
    """Dimensionless Q/Pincident with direct and field-mediated rho dependence."""

    rho = xp.asarray(rho_cell)
    au_imag = target_au_imag_epsilon(rho, xp=xp)[None, :, :, None]
    ta_imag = xp.asarray(
        [TA_B.target_epsilon_imag, TA_A.target_epsilon_imag, TA_B.target_epsilon_imag]
    )[:, None, None, None]
    q_au = Q_PREFACTOR * au_imag * xp.abs(e_au) ** 2
    q_ta = Q_PREFACTOR * ta_imag * xp.abs(e_tairte4) ** 2
    power = xp.sum(q_au * xp.asarray(volume_au)) + xp.sum(
        q_ta * xp.asarray(volume_tairte4)
    )
    return power / SOURCE_REFERENCE_POWER_W


def latent_directions() -> dict[str, np.ndarray]:
    x = np.linspace(-1.0, 1.0, NODE_SHAPE[0], dtype=np.float64)[:, None]
    y = np.linspace(-1.0, 1.0, NODE_SHAPE[1], dtype=np.float64)[None, :]
    gaussian = np.exp(-((x - 0.27) ** 2 + (y + 0.19) ** 2) / (2.0 * 0.22**2))
    gaussian -= np.mean(gaussian)
    raw = {
        "uniform": np.ones(NODE_SHAPE, dtype=np.float64),
        "x_antisymmetric": np.broadcast_to(x, NODE_SHAPE).copy(),
        "y_antisymmetric": np.broadcast_to(y, NODE_SHAPE).copy(),
        "offcenter_localized_zero_mean": gaussian,
    }
    return {
        name: np.ascontiguousarray(value / np.max(np.abs(value)))
        for name, value in raw.items()
    }


def array_sha256(array: Any, *, label: str) -> str:
    value = np.ascontiguousarray(np.asarray(array, dtype="<f8"))
    digest = hashlib.sha256()
    digest.update(label.encode("utf-8"))
    digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def adfd_direction_audit(step: float = 5.0e-3) -> dict[str, Any]:
    h = float(step)
    if not math.isfinite(h) or h <= 0.0:
        raise ValueError("AD-FD step must be finite and positive")
    latent = deterministic_gray_latent()
    directions = latent_directions()
    rows = {}
    passed = True
    for name, direction in directions.items():
        plus = latent + h * direction
        minus = latent - h * direction
        feasible = bool(np.min(minus) >= 0.0 and np.max(plus) <= 1.0)
        nonzero_cell_tangent = bool(
            np.max(np.abs(MAPPING.cell_jvp(latent, direction, 4.0))) > 0.0
        )
        passed = passed and feasible and nonzero_cell_tangent
        rows[name] = {
            "direction_sha256": array_sha256(
                direction, label=f"fdtdx-parity-adfd-{name}-v1"
            ),
            "maximum_abs": float(np.max(np.abs(direction))),
            "centered_step": h,
            "minus_plus_range": [float(np.min(minus)), float(np.max(plus))],
            "feasible": feasible,
            "nonzero_cell_tangent": nonzero_cell_tangent,
        }
    return {
        "status": "PASS" if passed else "FAIL",
        "baseline": "deterministic_nonuniform_gray_beta4",
        "baseline_latent_sha256": array_sha256(
            latent, label="fdtdx-parity-adfd-baseline-latent-v1"
        ),
        "directions": rows,
        "required_polarizations": ["Ea", "Eb"],
        "required_centered_forwards": 2 * len(directions) * 2,
        "gradient_validated": False,
    }


def ad_contract_audit() -> dict[str, Any]:
    source = gradient_source_audit()
    directions = adfd_direction_audit()
    return {
        "schema": "fdtdx_4um_parity_optical_ad_contract_v1",
        "status": (
            "PASS_AUDIT_ONLY_NOT_GRADIENT"
            if source["status"] == "PASS" and directions["status"] == "PASS"
            else "BLOCKED"
        ),
        "gradient_source": source,
        "checkpoint_memory": checkpoint_memory_lower_bounds(),
        "objective": {
            "name": "target_absorbed_power_over_all_air_incident_power",
            "Au_imag_epsilon": "57.8*rho + 69.36*rho_squared",
            "TaIrTe4_solver_axis_imag_epsilon": [
                TA_B.target_epsilon_imag,
                TA_A.target_epsilon_imag,
                TA_B.target_epsilon_imag,
            ],
            "source_reference_power_W": SOURCE_REFERENCE_POWER_W,
            "includes_explicit_rho_dependence_of_Q": True,
            "includes_field_mediated_rho_dependence": True,
            "Q_clipping_allowed": False,
        },
        "adfd": directions,
        "optimizer_enabled": False,
        "production_gradient_validated": False,
    }


def main() -> int:
    import json

    payload = ad_contract_audit()
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "PASS_AUDIT_ONLY_NOT_GRADIENT" else 2


if __name__ == "__main__":
    raise SystemExit(main())
