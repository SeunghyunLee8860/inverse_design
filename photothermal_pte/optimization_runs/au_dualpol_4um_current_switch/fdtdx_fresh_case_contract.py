"""Canonical numerical-case identity for fresh FDTDX convergence runs."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_exact_binary_convergence import (
    MeshSpec,
    axis_levels,
    mesh_audit,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_pml import (
    face_parameters,
)


VERSION = "fdtdx-fresh-numerical-case-v1"
SOURCE_STARTUP_PERIODS = 4
MESH_AXES = (
    "full_domain_z",
    "design_xy",
    "outer_xy",
    "pml_xy",
    "bottom_si_buffer",
    "top_source_to_pml_gap",
    "lateral_gap",
    "lateral_pml_thickness",
    "z_pml_thickness",
)


@dataclass(frozen=True)
class TimeSpec:
    total_periods: int = 16
    window_periods: int = 4
    courant_factor: float = 0.5
    source_startup_periods: int = SOURCE_STARTUP_PERIODS

    def __post_init__(self) -> None:
        for name in ("total_periods", "window_periods", "source_startup_periods"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.total_periods <= 2 * self.window_periods:
            raise ValueError("two disjoint phasor windows must fit in the solve")
        if self.source_startup_periods != SOURCE_STARTUP_PERIODS:
            raise ValueError(
                "the pinned FDTDX source builder supports exactly four startup periods"
            )
        if self.source_startup_periods > self.total_periods - 2 * self.window_periods:
            raise ValueError("source startup must finish before the previous phasor window")
        if not math.isfinite(self.courant_factor) or not (
            0.0 < self.courant_factor <= 1.0
        ):
            raise ValueError("courant_factor must be finite and in (0, 1]")


@dataclass(frozen=True)
class FreshCaseSpec:
    mesh: MeshSpec = field(default_factory=MeshSpec)
    time: TimeSpec = field(default_factory=TimeSpec)
    pml_alpha_scale: float = 1.0
    pml_target_reflection: float = 1.0e-6

    def __post_init__(self) -> None:
        if not isinstance(self.mesh, MeshSpec) or not isinstance(self.time, TimeSpec):
            raise TypeError("mesh and time must be MeshSpec and TimeSpec instances")
        if not math.isfinite(self.pml_alpha_scale) or self.pml_alpha_scale < 0.0:
            raise ValueError("pml_alpha_scale must be finite and nonnegative")
        if not math.isfinite(self.pml_target_reflection) or not (
            0.0 < self.pml_target_reflection < 1.0
        ):
            raise ValueError("pml_target_reflection must be finite and in (0, 1)")


ANCHOR_CASE = FreshCaseSpec()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def case_contract(spec: FreshCaseSpec) -> dict[str, Any]:
    """Return a self-hashed, strict-JSON numerical solver request."""

    payload = {
        "version": VERSION,
        "mesh_spec": asdict(spec.mesh),
        "time_spec": asdict(spec.time),
        "pml_spec": {
            "alpha_scale": spec.pml_alpha_scale,
            "target_reflection": spec.pml_target_reflection,
        },
        "resolved_mesh": mesh_audit(spec.mesh),
        "resolved_pml_face_parameters": face_parameters(
            spec.mesh,
            alpha_scale=spec.pml_alpha_scale,
            target_reflection=spec.pml_target_reflection,
        ),
        "rules": {
            "both_polarizations_share_this_case": True,
            "source_pair_must_match_exactly": True,
            "per_polarization_normalization_forbidden": True,
            "optimizer_start_allowed": False,
        },
    }
    payload["case_contract_sha256"] = _canonical_sha256(payload)
    return payload


def case_from_contract(payload: Mapping[str, Any]) -> FreshCaseSpec:
    """Reconstruct only an exactly canonical contract; reject extra fields."""

    if not isinstance(payload, Mapping):
        raise TypeError("case contract must contain one JSON object")
    try:
        mesh = MeshSpec(**dict(payload["mesh_spec"]))
        time = TimeSpec(**dict(payload["time_spec"]))
        pml = dict(payload["pml_spec"])
        spec = FreshCaseSpec(
            mesh=mesh,
            time=time,
            pml_alpha_scale=pml["alpha_scale"],
            pml_target_reflection=pml["target_reflection"],
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid numerical case fields: {error}") from error
    expected = case_contract(spec)
    if dict(payload) != expected:
        raise ValueError("case contract is not the exact canonical resolved contract")
    return spec


def load_case_contract(
    path: Path,
    expected_sha256: str,
) -> tuple[FreshCaseSpec, dict[str, Any], dict[str, Any]]:
    """Load a canonical contract bound to an absolute file and byte hash."""

    supplied = path.expanduser()
    is_absolute = supplied.is_absolute()
    resolved = supplied.resolve()
    normalized_sha = expected_sha256.strip().lower()
    sha_is_hex = len(normalized_sha) == 64 and all(
        character in "0123456789abcdef" for character in normalized_sha
    )
    exists = resolved.is_file()
    actual_sha = file_sha256(resolved) if exists else None
    checks = {
        "path_is_absolute": is_absolute,
        "file_exists": exists,
        "expected_sha256_is_lowercase_hex": sha_is_hex,
        "file_sha256_matches": exists and sha_is_hex and actual_sha == normalized_sha,
    }
    if not all(checks.values()):
        raise RuntimeError(f"numerical case file audit failed: {checks}")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    spec = case_from_contract(payload)
    audit = {
        "path": str(resolved),
        "expected_sha256": normalized_sha,
        "actual_sha256": actual_sha,
        "case_contract_sha256": payload["case_contract_sha256"],
        "checks": checks,
        "ready": True,
    }
    return spec, payload, audit


def realized_time_contract(
    spec: FreshCaseSpec,
    model: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind requested periods/Courant to the solver-realized dt and step count."""

    realized_startup = int(model["source_contract"]["num_startup_periods"])
    if realized_startup != spec.time.source_startup_periods:
        raise RuntimeError("realized source startup does not match the case contract")
    return {
        "total_periods": spec.time.total_periods,
        "window_periods": spec.time.window_periods,
        "source_startup_periods": realized_startup,
        "courant_factor": spec.time.courant_factor,
        "time_step_s": float(model["config"].time_step_duration),
        "time_steps_total": int(model["config"].time_steps_total),
    }


def case_for_axis(
    axis: str,
    level: int,
    *,
    time: TimeSpec,
    pml_alpha_scale: float,
    pml_target_reflection: float,
) -> FreshCaseSpec:
    if axis == "anchor":
        if level != 0:
            raise ValueError("anchor supports only level 0")
        mesh = MeshSpec()
    else:
        if axis not in MESH_AXES:
            raise ValueError(f"unknown mesh axis {axis!r}")
        levels = axis_levels(axis, MeshSpec())
        if level not in range(len(levels)):
            raise ValueError(f"mesh level must lie in [0, {len(levels) - 1}]")
        mesh = levels[level]
    return FreshCaseSpec(
        mesh=mesh,
        time=time,
        pml_alpha_scale=pml_alpha_scale,
        pml_target_reflection=pml_target_reflection,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mesh-axis", choices=("anchor", *MESH_AXES), default="anchor")
    parser.add_argument("--mesh-level", type=int, default=0)
    parser.add_argument("--total-periods", type=int, default=16)
    parser.add_argument("--window-periods", type=int, default=4)
    parser.add_argument("--courant-factor", type=float, default=0.5)
    parser.add_argument("--pml-alpha-scale", type=float, default=1.0)
    parser.add_argument("--pml-target-reflection", type=float, default=1.0e-6)
    args = parser.parse_args()
    output = args.output.expanduser()
    if not output.is_absolute():
        parser.error("--output must be an absolute path")
    output = output.resolve()
    if not output.parent.is_dir() or output.exists():
        parser.error("--output parent must exist and the output file must not exist")
    spec = case_for_axis(
        args.mesh_axis,
        args.mesh_level,
        time=TimeSpec(
            total_periods=args.total_periods,
            window_periods=args.window_periods,
            courant_factor=args.courant_factor,
        ),
        pml_alpha_scale=args.pml_alpha_scale,
        pml_target_reflection=args.pml_target_reflection,
    )
    payload = case_contract(spec)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "path": str(output),
                "file_sha256": file_sha256(output),
                "case_contract_sha256": payload["case_contract_sha256"],
                "mesh_shape_xyz": payload["resolved_mesh"]["grid_shape_xyz"],
                "yee_cell_count": payload["resolved_mesh"]["yee_cell_count"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
