"""50-nm design/flake x-y refinement at the fixed user-balanced z2 mesh."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Iterator

import numpy as np

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_case_contract import (
    TimeSpec,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_user_balanced_mesh import (
    Segment,
    grid_edges as baseline_grid_edges,
    lateral_segments as baseline_lateral_segments,
    pml_face_parameters,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_user_balanced_z_refinement import (
    grid_edges as z_grid_edges,
    vertical_segments,
)


VERSION = "fdtdx-user-balanced-design-flake-xy-refinement-v1"
CASE_VERSION = "fdtdx-user-balanced-design-flake-xy-case-v1"
LATERAL_FACTOR = 2
FULL_DOMAIN_Z_FACTOR = 2


@dataclass(frozen=True)
class UserBalancedLateralRefinementSpec:
    """Refine only the design window and complete flake from 100 to 50 nm."""

    design_xy_factor: int = LATERAL_FACTOR
    full_domain_z_factor: int = FULL_DOMAIN_Z_FACTOR
    lateral_pml_thickness_m: float = 1.0e-6
    z_pml_thickness_m: float = 1.6e-6

    def __post_init__(self) -> None:
        if self.design_xy_factor != LATERAL_FACTOR:
            raise ValueError("the declared design/flake lateral factor is exactly two")
        if self.full_domain_z_factor != FULL_DOMAIN_Z_FACTOR:
            raise ValueError("the lateral experiment is fixed to the z2 mesh")
        if self.lateral_pml_thickness_m != 1.0e-6:
            raise ValueError("lateral PML thickness must remain 1 um")
        if self.z_pml_thickness_m != 1.6e-6:
            raise ValueError("z PML thickness must remain 1.6 um")


def lateral_segments() -> tuple[Segment, ...]:
    """Return 50-nm flake/design segments with outer air and PML unchanged."""

    result = []
    for segment in baseline_lateral_segments():
        refine = segment.role in ("flake", "design")
        result.append(
            Segment(
                segment.name,
                segment.start_m,
                segment.stop_m,
                segment.cells * (LATERAL_FACTOR if refine else 1),
                segment.role,
                (
                    segment.requested_pitch_m / LATERAL_FACTOR
                    if refine and segment.requested_pitch_m is not None
                    else segment.requested_pitch_m
                ),
            )
        )
    return tuple(result)


def _edges(segments: tuple[Segment, ...]) -> np.ndarray:
    pieces = []
    for index, segment in enumerate(segments):
        values = np.linspace(
            segment.start_m,
            segment.stop_m,
            segment.cells + 1,
            dtype=np.float64,
        )
        pieces.append(values if index == 0 else values[1:])
    result = np.concatenate(pieces)
    if np.any(np.diff(result) <= 0.0):
        raise RuntimeError("lateral-refined mesh is not strictly increasing")
    return result


def grid_edges() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lateral = _edges(lateral_segments())
    _, _, z = z_grid_edges(FULL_DOMAIN_Z_FACTOR)
    return lateral, lateral.copy(), z


def _edge_index(edges: np.ndarray, coordinate_m: float) -> int:
    matches = np.flatnonzero(
        np.isclose(edges, coordinate_m, rtol=0.0, atol=2.0e-18)
    )
    if matches.size != 1:
        raise RuntimeError(f"required edge {coordinate_m:.9e} m is absent")
    return int(matches[0])


def layout_values() -> dict[str, int]:
    x, _, z = grid_edges()
    lateral = {segment.name: segment for segment in lateral_segments()}
    vertical = {
        segment.name: segment
        for segment in vertical_segments(FULL_DOMAIN_Z_FACTOR)
    }
    return {
        "pml_cells_xy": lateral["left_pml"].cells,
        "pml_cells_z": vertical["bottom_pml_si"].cells,
        "silicon_cells": vertical["bottom_pml_si"].cells
        + vertical["resolved_si"].cells,
        "sio2_cells": vertical["sio2"].cells,
        "tairte4_cells": vertical["tairte4"].cells,
        "au_cells": vertical["au"].cells,
        "flake_xy_cells": 160 * LATERAL_FACTOR,
        "au_xy_cells": 80 * LATERAL_FACTOR,
        "source_xy_cells": 160 * LATERAL_FACTOR,
        "non_pml_xy_cells": x.size - 1 - 2 * lateral["left_pml"].cells,
        "source_z_start": _edge_index(z, 0.750e-6),
        "target_z_start": _edge_index(z, 0.250e-6),
        "incident_z_start": _edge_index(z, 0.500e-6),
        "closed_z_start": _edge_index(z, -0.588e-6),
        "closed_z_cells": _edge_index(z, 0.250e-6)
        - _edge_index(z, -0.588e-6),
    }


def mesh_audit() -> dict[str, Any]:
    spec = UserBalancedLateralRefinementSpec()
    x, y, z = grid_edges()
    base_x, base_y, _ = baseline_grid_edges()
    _, _, z2 = z_grid_edges(FULL_DOMAIN_Z_FACTOR)
    lateral = [segment.audit() for segment in lateral_segments()]
    baseline_by_name = {
        segment.name: segment for segment in baseline_lateral_segments()
    }
    refined_by_name = {segment.name: segment for segment in lateral_segments()}
    refined_names = ("left_flake_wing", "au_design_window", "right_flake_wing")
    held_names = ("left_pml", "left_air_margin", "right_air_margin", "right_pml")
    payload: dict[str, Any] = {
        "version": VERSION,
        "spec": asdict(spec),
        "axis": "design_and_complete_flake_xy",
        "factor_from_user_baseline": LATERAL_FACTOR,
        "fixed_full_domain_z_factor": FULL_DOMAIN_Z_FACTOR,
        "grid_shape_xyz": [x.size - 1, y.size - 1, z.size - 1],
        "yee_cell_count": int((x.size - 1) * (y.size - 1) * (z.size - 1)),
        "bounds_m": [[x[0], x[-1]], [y[0], y[-1]], [z[0], z[-1]]],
        "lateral_segments": lateral,
        "vertical_segments": [
            segment.audit() for segment in vertical_segments(FULL_DOMAIN_Z_FACTOR)
        ],
        "layout": layout_values(),
        "pml_cells_each_face_xyz": [8, 8, 16],
        "invariants": {
            "physical_bounds_unchanged": bool(
                x[0] == base_x[0]
                and x[-1] == base_x[-1]
                and y[0] == base_y[0]
                and y[-1] == base_y[-1]
            ),
            "z_edges_byte_exact_to_user_z2": np.array_equal(z, z2),
            "every_baseline_x_edge_retained": all(
                np.any(np.isclose(x, edge, rtol=0.0, atol=2.0e-18))
                for edge in base_x
            ),
            "every_baseline_y_edge_retained": all(
                np.any(np.isclose(y, edge, rtol=0.0, atol=2.0e-18))
                for edge in base_y
            ),
            "only_flake_and_design_segments_refined": all(
                refined_by_name[name].cells
                == baseline_by_name[name].cells * LATERAL_FACTOR
                for name in refined_names
            )
            and all(
                refined_by_name[name].cells == baseline_by_name[name].cells
                for name in held_names
            ),
            "outer_air_pitch_remains_200nm": all(
                np.isclose(
                    refined_by_name[name].step_m,
                    200.0e-9,
                    rtol=0.0,
                    atol=2.0e-18,
                )
                for name in ("left_air_margin", "right_air_margin")
            ),
            "lateral_pml_layers_remain_8": all(
                refined_by_name[name].cells == 8
                for name in ("left_pml", "right_pml")
            ),
        },
        "rules": {
            "user_z2_contract_modified": False,
            "source_pair_required_before_material_case": True,
            "exact_binary_mask_replicated_piecewise_constant_2x2": True,
            "gray_material_allowed": False,
            "optimizer_start_allowed": False,
            "one_100nm_to_50nm_pair_selects_production_mesh": False,
        },
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    payload["grid_contract_sha256"] = hashlib.sha256(encoded).hexdigest()
    return payload


def case_contract(time_spec: TimeSpec) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "version": CASE_VERSION,
        "mesh": mesh_audit(),
        "time": {
            "total_periods": time_spec.total_periods,
            "window_periods": time_spec.window_periods,
            "courant_factor": time_spec.courant_factor,
            "source_startup_periods": time_spec.source_startup_periods,
        },
        "pml": {
            "layers_each_face_xyz": [8, 8, 16],
            "physical_thickness_m_xyz": [1.0e-6, 1.0e-6, 1.6e-6],
            "alpha_scale": 1.0,
            "target_reflection": 1.0e-6,
        },
        "rules": {
            "only_design_and_complete_flake_xy_refined": True,
            "source_pair_required_before_material_case": True,
            "per_polarization_normalization_forbidden": True,
            "optimizer_start_allowed": False,
        },
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    payload["case_contract_sha256"] = hashlib.sha256(encoded).hexdigest()
    return payload


@contextmanager
def mesh_context() -> Iterator[dict[str, Any]]:
    from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch import (
        fdtdx_4um_model as optical_model,
    )

    previous_layout = optical_model.LAYOUT
    previous_edges = optical_model.grid_edges
    edges = grid_edges()
    optical_model.LAYOUT = optical_model.GridLayout(**layout_values())
    optical_model.grid_edges = lambda: tuple(axis.copy() for axis in edges)
    try:
        yield mesh_audit()
    finally:
        optical_model.LAYOUT = previous_layout
        optical_model.grid_edges = previous_edges


def build_model(
    polarization: str,
    *,
    total_periods: int,
    window_periods: int,
    courant_factor: float,
    include_adjoint_source: bool = False,
    air_only_source_calibration: bool = False,
    dispersive_state_representation: str = "increment",
) -> dict[str, Any]:
    from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch import (
        fdtdx_4um_model as optical_model,
    )

    profiles = pml_face_parameters()
    with mesh_context():
        model = optical_model.build_model(
            polarization,
            total_periods=total_periods,
            window_periods=window_periods,
            courant_factor=courant_factor,
            include_adjoint_source=include_adjoint_source,
            air_only_source_calibration=air_only_source_calibration,
            pml_face_parameters=profiles,
            dispersive_state_representation=dispersive_state_representation,
        )
    model["fresh_mesh_audit"] = mesh_audit()
    model["pml_face_parameters"] = profiles
    return model


def material_spec() -> UserBalancedLateralRefinementSpec:
    """Return the 2x piecewise-constant exact-mask replication contract."""

    return UserBalancedLateralRefinementSpec()
