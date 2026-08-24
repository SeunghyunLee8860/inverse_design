"""Solver-independent contract for a fresh exact-binary FDTDX campaign.

The historical mesh helper changes only z.  This module instead separates the
Au design-window x/y mesh, the outer flake/gap x/y mesh, the lateral PML mesh,
and the full-domain z mesh.  It contains no FDTDX/JAX imports and performs no
solve, so its geometry and promotion rules can be audited on any host.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from typing import Any, Iterable


VERSION = "exact-binary-multiaxis-convergence-v2"
C0_M_PER_S = 299_792_458.0
EPS0_F_PER_M = 8.854_187_812_8e-12
ETA0_OHM = 376.730_313_668
WAVELENGTH_M = 4.0e-6
DESIGN_HALF_SPAN_M = 4.0e-6
FLAKE_HALF_SPAN_M = 8.0e-6
DESIGN_PITCH_M = 100.0e-9
DESIGN_CELLS = 80
BASE_OUTER_PITCH_M = 100.0e-9
BASE_PML_PITCH_M = 125.0e-9
DEFAULT_GAP_M = 1.0e-6
DEFAULT_PML_THICKNESS_M = 1.0e-6
DEFAULT_BOTTOM_SI_BUFFER_M = 1.015e-6
DEFAULT_TOP_SOURCE_TO_PML_GAP_M = 0.650e-6
DEFAULT_Z_PML_THICKNESS_M = 1.600e-6
BASE_RESOLVED_SI_PITCH_M = DEFAULT_BOTTOM_SI_BUFFER_M / 5
BASE_SOURCE_AIR_PITCH_M = DEFAULT_TOP_SOURCE_TO_PML_GAP_M / 3
BASE_Z_PML_PITCH_M = DEFAULT_Z_PML_THICKNESS_M / 8
SIO2_BOTTOM_M = -0.385e-6
TAIRTE4_BOTTOM_M = -0.100e-6
AU_TOP_M = 0.050e-6
SOURCE_Z_M = 0.750e-6


@dataclass(frozen=True)
class MeshSpec:
    """Independent mesh axes with invariant physical material boundaries."""

    design_xy_factor: int = 1
    outer_xy_factor: int = 1
    pml_xy_factor: int = 1
    z_factor: int = 4
    lateral_gap_m: float = DEFAULT_GAP_M
    lateral_pml_thickness_m: float = DEFAULT_PML_THICKNESS_M
    bottom_si_buffer_m: float = DEFAULT_BOTTOM_SI_BUFFER_M
    top_source_to_pml_gap_m: float = DEFAULT_TOP_SOURCE_TO_PML_GAP_M
    z_pml_thickness_m: float = DEFAULT_Z_PML_THICKNESS_M

    def __post_init__(self) -> None:
        for name in (
            "design_xy_factor",
            "outer_xy_factor",
            "pml_xy_factor",
            "z_factor",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        positive_lengths = (
            self.lateral_gap_m,
            self.lateral_pml_thickness_m,
            self.bottom_si_buffer_m,
            self.top_source_to_pml_gap_m,
            self.z_pml_thickness_m,
        )
        if any(value <= 0.0 for value in positive_lengths):
            raise ValueError("all domain-buffer and PML lengths must be positive")
        _integer_cells(self.lateral_gap_m, BASE_OUTER_PITCH_M, "lateral gap")
        _integer_cells(
            self.lateral_pml_thickness_m,
            BASE_PML_PITCH_M,
            "lateral PML",
        )
        _integer_cells(
            self.bottom_si_buffer_m,
            BASE_RESOLVED_SI_PITCH_M,
            "bottom Si buffer",
        )
        _integer_cells(
            self.top_source_to_pml_gap_m,
            BASE_SOURCE_AIR_PITCH_M,
            "top source-to-PML gap",
        )
        _integer_cells(
            self.z_pml_thickness_m,
            BASE_Z_PML_PITCH_M,
            "z PML",
        )


@dataclass(frozen=True)
class Segment:
    name: str
    start_m: float
    stop_m: float
    cells: int
    role: str

    @property
    def step_m(self) -> float:
        return (self.stop_m - self.start_m) / self.cells

    def audit(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "bounds_m": [self.start_m, self.stop_m],
            "cells": self.cells,
            "step_m": self.step_m,
            "role": self.role,
        }


def _integer_cells(span_m: float, pitch_m: float, label: str) -> int:
    cells = int(round(span_m / pitch_m))
    if cells < 1 or not math.isclose(cells * pitch_m, span_m, abs_tol=2.0e-18):
        raise ValueError(f"{label} must be an integer multiple of {pitch_m:.9e} m")
    return cells


def lateral_segments(spec: MeshSpec) -> tuple[Segment, ...]:
    """Return a symmetric local-refinement grid with exact physical edges."""

    gap_cells = (
        _integer_cells(spec.lateral_gap_m, BASE_OUTER_PITCH_M, "lateral gap")
        * spec.outer_xy_factor
    )
    pml_cells = (
        _integer_cells(
            spec.lateral_pml_thickness_m,
            BASE_PML_PITCH_M,
            "lateral PML",
        )
        * spec.pml_xy_factor
    )
    wing_cells = 40 * spec.outer_xy_factor
    design_cells = DESIGN_CELLS * spec.design_xy_factor
    pml_inner = FLAKE_HALF_SPAN_M + spec.lateral_gap_m
    outer = pml_inner + spec.lateral_pml_thickness_m
    return (
        Segment("left_pml", -outer, -pml_inner, pml_cells, "pml"),
        Segment(
            "left_air_gap",
            -pml_inner,
            -FLAKE_HALF_SPAN_M,
            gap_cells,
            "outer",
        ),
        Segment(
            "left_flake_wing",
            -FLAKE_HALF_SPAN_M,
            -DESIGN_HALF_SPAN_M,
            wing_cells,
            "outer",
        ),
        Segment(
            "au_design_window",
            -DESIGN_HALF_SPAN_M,
            DESIGN_HALF_SPAN_M,
            design_cells,
            "design",
        ),
        Segment(
            "right_flake_wing",
            DESIGN_HALF_SPAN_M,
            FLAKE_HALF_SPAN_M,
            wing_cells,
            "outer",
        ),
        Segment(
            "right_air_gap",
            FLAKE_HALF_SPAN_M,
            pml_inner,
            gap_cells,
            "outer",
        ),
        Segment("right_pml", pml_inner, outer, pml_cells, "pml"),
    )


def vertical_segments(spec: MeshSpec) -> tuple[Segment, ...]:
    """Return z segments while keeping every material/source edge invariant."""

    bottom_pml_inner = SIO2_BOTTOM_M - spec.bottom_si_buffer_m
    bottom_outer = bottom_pml_inner - spec.z_pml_thickness_m
    top_pml_inner = SOURCE_Z_M + spec.top_source_to_pml_gap_m
    top_outer = top_pml_inner + spec.z_pml_thickness_m
    values = (
        (
            "bottom_pml_si",
            bottom_outer,
            bottom_pml_inner,
            _integer_cells(
                spec.z_pml_thickness_m, BASE_Z_PML_PITCH_M, "z PML"
            ),
            "pml_z",
        ),
        (
            "resolved_si",
            bottom_pml_inner,
            SIO2_BOTTOM_M,
            _integer_cells(
                spec.bottom_si_buffer_m,
                BASE_RESOLVED_SI_PITCH_M,
                "bottom Si buffer",
            ),
            "bottom_si_buffer",
        ),
        ("sio2", SIO2_BOTTOM_M, TAIRTE4_BOTTOM_M, 3, "physical_stack"),
        ("tairte4", TAIRTE4_BOTTOM_M, 0.0, 5, "physical_stack"),
        ("au", 0.0, AU_TOP_M, 2, "physical_stack"),
        ("near_air", AU_TOP_M, 0.250e-6, 4, "fixed_air"),
        ("middle_air", 0.250e-6, SOURCE_Z_M, 2, "fixed_air"),
        (
            "source_air",
            SOURCE_Z_M,
            top_pml_inner,
            _integer_cells(
                spec.top_source_to_pml_gap_m,
                BASE_SOURCE_AIR_PITCH_M,
                "top source-to-PML gap",
            ),
            "top_source_to_pml_gap",
        ),
        (
            "top_pml_air",
            top_pml_inner,
            top_outer,
            _integer_cells(
                spec.z_pml_thickness_m, BASE_Z_PML_PITCH_M, "z PML"
            ),
            "pml_z",
        ),
    )
    return tuple(
        Segment(name, start, stop, base_cells * spec.z_factor, role)
        for name, start, stop, base_cells, role in values
    )


def _edge_coordinates(segments: Iterable[Segment]) -> tuple[float, ...]:
    result: list[float] = []
    for segment_index, segment in enumerate(segments):
        values = tuple(
            segment.start_m
            + (segment.stop_m - segment.start_m) * index / segment.cells
            for index in range(segment.cells + 1)
        )
        result.extend(values if segment_index == 0 else values[1:])
    if any(right <= left for left, right in zip(result[:-1], result[1:])):
        raise RuntimeError("non-monotone mesh")
    return tuple(result)


def grid_edges(spec: MeshSpec) -> tuple[tuple[float, ...], ...]:
    lateral = _edge_coordinates(lateral_segments(spec))
    vertical = _edge_coordinates(vertical_segments(spec))
    return lateral, lateral, vertical


def _edge_index(edges: tuple[float, ...], coordinate_m: float) -> int:
    matches = [
        index
        for index, value in enumerate(edges)
        if math.isclose(value, coordinate_m, rel_tol=0.0, abs_tol=2.0e-18)
    ]
    if len(matches) != 1:
        raise RuntimeError(f"required edge {coordinate_m:.9e} m is absent")
    return matches[0]


def layout(spec: MeshSpec) -> dict[str, int]:
    lateral = lateral_segments(spec)
    z = grid_edges(spec)[2]
    by_name = {segment.name: segment for segment in lateral}
    z_counts = {segment.name: segment.cells for segment in vertical_segments(spec)}
    flake_cells = (
        by_name["left_flake_wing"].cells
        + by_name["au_design_window"].cells
        + by_name["right_flake_wing"].cells
    )
    non_pml_cells = sum(segment.cells for segment in lateral if segment.role != "pml")
    closed_start = _edge_index(z, -0.588e-6)
    target_start = _edge_index(z, 0.250e-6)
    return {
        "pml_cells_xy": by_name["left_pml"].cells,
        "pml_cells_z": z_counts["bottom_pml_si"],
        "silicon_cells": z_counts["bottom_pml_si"] + z_counts["resolved_si"],
        "sio2_cells": z_counts["sio2"],
        "tairte4_cells": z_counts["tairte4"],
        "au_cells": z_counts["au"],
        "flake_xy_cells": flake_cells,
        "au_xy_cells": by_name["au_design_window"].cells,
        "source_xy_cells": flake_cells,
        "non_pml_xy_cells": non_pml_cells,
        "source_z_start": _edge_index(z, 0.750e-6),
        "target_z_start": target_start,
        "incident_z_start": _edge_index(z, 0.500e-6),
        "closed_z_start": closed_start,
        "closed_z_cells": target_start - closed_start,
    }


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def mesh_audit(spec: MeshSpec) -> dict[str, Any]:
    x, y, z = grid_edges(spec)
    lateral = lateral_segments(spec)
    vertical = vertical_segments(spec)
    value = {
        "version": VERSION,
        "spec": asdict(spec),
        "grid_shape_xyz": [len(x) - 1, len(y) - 1, len(z) - 1],
        "yee_cell_count": (len(x) - 1) * (len(y) - 1) * (len(z) - 1),
        "bounds_m": [[x[0], x[-1]], [y[0], y[-1]], [z[0], z[-1]]],
        "design_solver_pitch_m": DESIGN_PITCH_M / spec.design_xy_factor,
        "outer_solver_pitch_m": BASE_OUTER_PITCH_M / spec.outer_xy_factor,
        "pml_solver_pitch_m": BASE_PML_PITCH_M / spec.pml_xy_factor,
        "z_pml_solver_pitch_m": BASE_Z_PML_PITCH_M / spec.z_factor,
        "bottom_si_solver_pitch_m": BASE_RESOLVED_SI_PITCH_M / spec.z_factor,
        "source_air_solver_pitch_m": BASE_SOURCE_AIR_PITCH_M / spec.z_factor,
        "lateral_segments": [segment.audit() for segment in lateral],
        "vertical_segments": [segment.audit() for segment in vertical],
        "layout": layout(spec),
        "physical_edges_invariant": {
            "design_window_m": [-DESIGN_HALF_SPAN_M, DESIGN_HALF_SPAN_M],
            "flake_m": [-FLAKE_HALF_SPAN_M, FLAKE_HALF_SPAN_M],
            "sio2_m": [-0.385e-6, -0.100e-6],
            "tairte4_m": [-0.100e-6, 0.0],
            "au_m": [0.0, AU_TOP_M],
            "source_plane_z_m": SOURCE_Z_M,
        },
        "domain_extent_axes_m": {
            "lateral_gap_m": spec.lateral_gap_m,
            "lateral_pml_thickness_m": spec.lateral_pml_thickness_m,
            "bottom_si_buffer_m": spec.bottom_si_buffer_m,
            "top_source_to_pml_gap_m": spec.top_source_to_pml_gap_m,
            "z_pml_thickness_m": spec.z_pml_thickness_m,
        },
    }
    value["grid_contract_sha256"] = _canonical_sha256(value)
    return value


def axis_levels(axis: str, anchor: MeshSpec) -> tuple[MeshSpec, ...]:
    """Return a three-level ladder while holding other axes exactly fixed."""

    if axis == "design_xy":
        return tuple(replace(anchor, design_xy_factor=value) for value in (1, 2, 4))
    if axis == "outer_xy":
        return tuple(replace(anchor, outer_xy_factor=value) for value in (1, 2, 4))
    if axis == "pml_xy":
        return tuple(replace(anchor, pml_xy_factor=value) for value in (1, 2, 4))
    if axis == "full_domain_z":
        return tuple(replace(anchor, z_factor=value) for value in (2, 4, 8))
    if axis == "lateral_gap":
        return tuple(
            replace(anchor, lateral_gap_m=value)
            for value in (1.0e-6, 2.0e-6, 4.0e-6)
        )
    if axis == "lateral_pml_thickness":
        return tuple(
            replace(anchor, lateral_pml_thickness_m=value)
            for value in (1.0e-6, 1.5e-6, 2.0e-6)
        )
    if axis == "bottom_si_buffer":
        return tuple(
            replace(anchor, bottom_si_buffer_m=value)
            for value in (1.015e-6, 2.030e-6, 3.045e-6)
        )
    if axis == "top_source_to_pml_gap":
        return tuple(
            replace(anchor, top_source_to_pml_gap_m=value)
            for value in (0.650e-6, 1.300e-6, 1.950e-6)
        )
    if axis == "z_pml_thickness":
        return tuple(
            replace(anchor, z_pml_thickness_m=value)
            for value in (1.600e-6, 2.400e-6, 3.200e-6)
        )
    raise ValueError(f"unknown convergence axis {axis!r}")


def pml_parameters(
    thickness_m: float,
    *,
    alpha_scale: float = 1.0,
    target_reflection: float = 1.0e-6,
) -> dict[str, float]:
    """Return fully explicit 4-um CPML parameters for a sweep case."""

    if thickness_m <= 0.0 or alpha_scale < 0.0:
        raise ValueError("invalid PML thickness or alpha scale")
    if not 0.0 < target_reflection < 1.0:
        raise ValueError("target_reflection must lie strictly between zero and one")
    sigma_order = 3.0
    return {
        "alpha_start": alpha_scale
        * 0.01
        * 2.0
        * math.pi
        * C0_M_PER_S
        / WAVELENGTH_M
        * EPS0_F_PER_M,
        "alpha_end": 0.0,
        "alpha_order": 1.0,
        "kappa_start": 1.0,
        "kappa_end": 1.0,
        "kappa_order": 3.0,
        "sigma_start": 0.0,
        "sigma_end": -(sigma_order + 1.0)
        * math.log(target_reflection)
        / (2.0 * ETA0_OHM * thickness_m),
        "sigma_order": sigma_order,
        "alpha_reference_wavelength_m": WAVELENGTH_M,
        "target_reflection": target_reflection,
    }


REFERENCE_NAMES = (
    "empty",
    "full_design_window",
    "centered_square_2um",
    "x_bar_4um_by_1um",
    "y_bar_1um_by_4um",
    "l_shape_4um_with_1um_arms",
    "l_shape_4um_with_500nm_arms",
    "parallel_bars_4um_by_500nm_with_500nm_gap",
)

REFERENCE_POLICY = {
    "empty": {"role": "endpoint_control", "minimum_feature_m": None},
    "full_design_window": {"role": "endpoint_control", "minimum_feature_m": None},
    "centered_square_2um": {"role": "legacy_control", "minimum_feature_m": 2.0e-6},
    "x_bar_4um_by_1um": {"role": "orientation_control", "minimum_feature_m": 1.0e-6},
    "y_bar_1um_by_4um": {"role": "orientation_control", "minimum_feature_m": 1.0e-6},
    "l_shape_4um_with_1um_arms": {
        "role": "legacy_asymmetric_control",
        "minimum_feature_m": 1.0e-6,
    },
    "l_shape_4um_with_500nm_arms": {
        "role": "primary_spatial_reference",
        "minimum_feature_m": 500.0e-9,
    },
    "parallel_bars_4um_by_500nm_with_500nm_gap": {
        "role": "minimum_gap_stress",
        "minimum_feature_m": 500.0e-9,
        "minimum_gap_m": 500.0e-9,
    },
}


def reference_mask(name: str) -> tuple[tuple[int, ...], ...]:
    """Return an exact 80x80 air/Au mask; intermediate density is forbidden."""

    if name not in REFERENCE_NAMES:
        raise ValueError(f"unknown exact-binary reference {name!r}")
    mask = [[0 for _ in range(DESIGN_CELLS)] for _ in range(DESIGN_CELLS)]

    def fill(x_start: int, x_stop: int, y_start: int, y_stop: int) -> None:
        for x_index in range(x_start, x_stop):
            for y_index in range(y_start, y_stop):
                mask[x_index][y_index] = 1

    if name == "full_design_window":
        fill(0, 80, 0, 80)
    elif name == "centered_square_2um":
        fill(30, 50, 30, 50)
    elif name == "x_bar_4um_by_1um":
        fill(20, 60, 35, 45)
    elif name == "y_bar_1um_by_4um":
        fill(35, 45, 20, 60)
    elif name == "l_shape_4um_with_1um_arms":
        fill(20, 60, 20, 30)
        fill(20, 30, 20, 60)
    elif name == "l_shape_4um_with_500nm_arms":
        fill(20, 60, 20, 25)
        fill(20, 25, 20, 60)
    elif name == "parallel_bars_4um_by_500nm_with_500nm_gap":
        fill(20, 60, 20, 25)
        fill(20, 60, 30, 35)
    result = tuple(tuple(row) for row in mask)
    _require_binary_mask(result)
    return result


def _require_binary_mask(mask: tuple[tuple[int, ...], ...]) -> None:
    if len(mask) != DESIGN_CELLS or any(len(row) != DESIGN_CELLS for row in mask):
        raise ValueError("reference mask must be exactly 80x80")
    if any(value not in (0, 1) for row in mask for value in row):
        raise ValueError("reference mask must contain exact integer 0/1 values")


def upsample_mask(
    mask: tuple[tuple[int, ...], ...], factor: int
) -> tuple[tuple[int, ...], ...]:
    """Piecewise-constant topology-to-Yee map with exact edge preservation."""

    _require_binary_mask(mask)
    if not isinstance(factor, int) or isinstance(factor, bool) or factor < 1:
        raise ValueError("upsampling factor must be a positive integer")
    rows: list[tuple[int, ...]] = []
    for row in mask:
        expanded = tuple(value for value in row for _ in range(factor))
        rows.extend(expanded for _ in range(factor))
    return tuple(rows)


def mask_audit(name: str) -> dict[str, Any]:
    mask = reference_mask(name)
    payload = {
        "name": name,
        "shape": [DESIGN_CELLS, DESIGN_CELLS],
        "binary": True,
        "solid_cells": sum(value for row in mask for value in row),
        "design_pitch_m": DESIGN_PITCH_M,
        "policy": REFERENCE_POLICY[name],
        "physical_bounds_m": [
            [-DESIGN_HALF_SPAN_M, DESIGN_HALF_SPAN_M],
            [-DESIGN_HALF_SPAN_M, DESIGN_HALF_SPAN_M],
        ],
    }
    flattened = bytes(value for row in mask for value in row)
    payload["mask_sha256"] = hashlib.sha256(flattened).hexdigest()
    return payload


OPTICAL_PAIR_GATES = {
    "source_power_relative_change": 5.0e-3,
    "q_closed_flux_relative": 2.0e-2,
    "stationarity_complex_E_NRMSE": 5.0e-3,
    "total_Q_relative_change": 1.0e-2,
    "material_component_Q_max_relative_change": 2.0e-2,
    "complex_E_fixed_probe_NRMSE": 2.0e-2,
    "conservative_Q_volume_L2_NRMSE": 5.0e-2,
}
DOWNSTREAM_GATES_NOT_ACTIVE = {
    "Ta_temperature_NRMSE": 2.0e-2,
    "Tmax_relative_change": 2.0e-2,
    "current_relative_change": 1.0e-2,
    "current_absolute_change_A": 5.0e-11,
}
CURRENT_SIGN_GUARD_A = 0.5e-9


def evaluate_pair(record: dict[str, Any]) -> dict[str, Any]:
    """Evaluate one optical coarse/fine comparison; no current is accepted."""

    checks = {
        name: float(record[name]) <= limit
        for name, limit in OPTICAL_PAIR_GATES.items()
    }
    return {"pass": all(checks.values()), "checks": checks}


def endpoint_sign_gate(ea_current_A: float, eb_current_A: float) -> dict[str, Any]:
    """Apply the requested orientation with a nonzero-current guard band."""

    ea = float(ea_current_A)
    eb = float(eb_current_A)
    checks = {
        "Ea_positive_with_guard": ea >= CURRENT_SIGN_GUARD_A,
        "Eb_negative_with_guard": eb <= -CURRENT_SIGN_GUARD_A,
        "opposite_sign": ea * eb < 0.0,
    }
    return {"pass": all(checks.values()), "checks": checks}


def campaign_contract() -> dict[str, Any]:
    anchor = MeshSpec()
    axes = (
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
    alpha_scales = (0.5, 1.0, 2.0)
    amplitude_skin_depth_m = WAVELENGTH_M / (2.0 * math.pi * 28.9)
    return {
        "status": "AUDITED_EXACT_BINARY_MULTIAXIS_CONTRACT_V2_NOT_SOLVED",
        "version": VERSION,
        "scope": "solver-independent exact-binary optical convergence contract",
        "historical_optimizer_may_resume": False,
        "gray_density_allowed_in_reference_campaign": False,
        "independent_optical_thermal_electrical_rho_allowed": False,
        "references": [mask_audit(name) for name in REFERENCE_NAMES],
        "reference_execution": {
            "primary_full_ladder": "l_shape_4um_with_500nm_arms",
            "endpoint_controls": ["empty", "full_design_window"],
            "orientation_controls": [
                "x_bar_4um_by_1um",
                "y_bar_1um_by_4um",
            ],
            "minimum_gap_stress": (
                "parallel_bars_4um_by_500nm_with_500nm_gap"
            ),
            "legacy_anchor_and_selected_rechecks": [
                "centered_square_2um",
                "l_shape_4um_with_1um_arms",
            ],
        },
        "anchor_mesh": mesh_audit(anchor),
        "axis_ladders": {
            axis: [mesh_audit(spec) for spec in axis_levels(axis, anchor)]
            for axis in axes
        },
        "time_convergence": {
            "settling_ladder": {
                "spatial_contract": "anchor",
                "courant_factor": 0.5,
                "total_periods": [16, 24, 32],
                "startup_periods": 4,
                "phasor_window_periods": 4,
                "successive_pairs": [[16, 24], [24, 32]],
            },
            "courant_ladder_after_settling": {
                "total_periods": "selected_from_settling_ladder",
                "courant_factors": [0.5, 0.375, 0.25, 0.1875],
                "selected_courant_factor": 0.25,
                "confirmation_courant_factor": 0.1875,
                "failed_coarse_pair_retained": [0.5, 0.375],
                "two_successive_passing_pairs_in_fine_range_required": True,
            },
            "material_ADE_refit_and_readback_required_each_level": True,
            "source_pair_required_for_every_unique_numerical_contract": True,
            "per_polarization_power_rescaling_forbidden": True,
        },
        "pml_alpha_scale_sweep": {
            "scales": list(alpha_scales),
            "lateral_profiles": [
                pml_parameters(
                    anchor.lateral_pml_thickness_m,
                    alpha_scale=value,
                )
                for value in alpha_scales
            ],
            "z_profiles": [
                pml_parameters(anchor.z_pml_thickness_m, alpha_scale=value)
                for value in alpha_scales
            ],
        },
        "comparison_contract": {
            "complex_E": {
                "method": "component-wise complex interpolation",
                "coordinates": "fixed physical Yee-aware probe coordinates",
                "probe_plane_z_m": 0.250e-6,
                "probe_xy_bounds_m": [
                    [-DESIGN_HALF_SPAN_M, DESIGN_HALF_SPAN_M],
                    [-DESIGN_HALF_SPAN_M, DESIGN_HALF_SPAN_M],
                ],
                "array_index_comparison_forbidden": True,
            },
            "absorbed_power_density": {
                "method": "conservative restriction of cell-integrated q",
                "cell_measure": "component-specific Yee dual volume",
                "common_physical_control_volumes_required": True,
                "array_index_comparison_forbidden": True,
            },
        },
        "optical_pair_gates": OPTICAL_PAIR_GATES,
        "downstream_gates_not_active_in_optical_certificate": (
            DOWNSTREAM_GATES_NOT_ACTIVE
        ),
        "current_sign_guard_A_downstream_only": CURRENT_SIGN_GUARD_A,
        "locked_Au_scale_context": {
            "wavelength_m": WAVELENGTH_M,
            "index_at_4um": [2.2, 28.9],
            "amplitude_skin_depth_m": amplitude_skin_depth_m,
            "intensity_1e_depth_m": amplitude_skin_depth_m / 2.0,
            "anchor_Au_z_step_m": 0.050e-6 / (2 * anchor.z_factor),
            "is_convergence_evidence": False,
        },
        "rules": {
            "one_axis_changes_per_ladder": True,
            "two_successive_pair_comparisons_required": True,
            "both_polarizations_required": True,
            "primary_reference_runs_full_ladder": True,
            "every_reference_on_every_axis_required": False,
            "staged_reference_rechecks_required_before_candidate": True,
            "joint_selected_mesh_confirmation_required": True,
            "raw_fields_and_complete_solver_tree_provenance_required": True,
            "exact_binary_ordinary_Au_required": True,
            "thermal_or_current_metrics_may_certify_optical_mesh": False,
        },
        "promotion": {
            "is_mesh_certificate": False,
            "optimizer_start_allowed": False,
            "requires_generalized_hashed_mesh_time_runner": True,
            "requires_completed_optical_ladders": True,
            "requires_physical_device_contract_before_PTE_current": True,
            "requires_independent_solver_endpoint_comparison": True,
        },
    }


def main() -> None:
    print(json.dumps(campaign_contract(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
