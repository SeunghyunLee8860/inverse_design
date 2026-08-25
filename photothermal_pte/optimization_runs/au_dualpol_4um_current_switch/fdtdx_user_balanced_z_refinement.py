"""Full-domain z refinement around the immutable user-balanced baseline."""

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
    UserBalancedMeshSpec,
    grid_edges as baseline_grid_edges,
    lateral_segments,
    pml_face_parameters,
    vertical_segments as baseline_vertical_segments,
)


VERSION = "fdtdx-user-balanced-full-domain-z-refinement-v1"
CASE_VERSION = "fdtdx-user-balanced-full-domain-z-case-v1"
SUPPORTED_FACTORS = (2,)


@dataclass(frozen=True)
class UserBalancedZRefinementSpec:
    factor: int = 2
    design_xy_factor: int = 1
    lateral_pml_thickness_m: float = 1.0e-6
    z_pml_thickness_m: float = 1.6e-6

    def __post_init__(self) -> None:
        if self.factor not in SUPPORTED_FACTORS:
            raise ValueError(f"supported full-domain z factors: {SUPPORTED_FACTORS}")
        if self.design_xy_factor != 1:
            raise ValueError("z refinement must hold the 100-nm design grid fixed")


def vertical_segments(factor: int = 2) -> tuple[Segment, ...]:
    spec = UserBalancedZRefinementSpec(factor=factor)
    return tuple(
        Segment(
            segment.name,
            segment.start_m,
            segment.stop_m,
            segment.cells * spec.factor,
            segment.role,
            (
                None
                if segment.requested_pitch_m is None
                else segment.requested_pitch_m / spec.factor
            ),
        )
        for segment in baseline_vertical_segments()
    )


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
        raise RuntimeError("refined mesh is not strictly increasing")
    return result


def grid_edges(factor: int = 2) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x, y, _ = baseline_grid_edges()
    return x, y, _edges(vertical_segments(factor))


def _edge_index(edges: np.ndarray, coordinate_m: float) -> int:
    matches = np.flatnonzero(np.isclose(edges, coordinate_m, rtol=0.0, atol=2.0e-18))
    if matches.size != 1:
        raise RuntimeError(f"required edge {coordinate_m:.9e} m is absent")
    return int(matches[0])


def layout_values(factor: int = 2) -> dict[str, int]:
    spec = UserBalancedZRefinementSpec(factor=factor)
    x, _, z = grid_edges(spec.factor)
    lateral = {segment.name: segment for segment in lateral_segments()}
    vertical = {segment.name: segment for segment in vertical_segments(spec.factor)}
    return {
        "pml_cells_xy": lateral["left_pml"].cells,
        "pml_cells_z": vertical["bottom_pml_si"].cells,
        "silicon_cells": vertical["bottom_pml_si"].cells
        + vertical["resolved_si"].cells,
        "sio2_cells": vertical["sio2"].cells,
        "tairte4_cells": vertical["tairte4"].cells,
        "au_cells": vertical["au"].cells,
        "flake_xy_cells": 160,
        "au_xy_cells": 80,
        "source_xy_cells": 160,
        "non_pml_xy_cells": x.size - 1 - 2 * lateral["left_pml"].cells,
        "source_z_start": _edge_index(z, 0.750e-6),
        "target_z_start": _edge_index(z, 0.250e-6),
        "incident_z_start": _edge_index(z, 0.500e-6),
        "closed_z_start": _edge_index(z, -0.588e-6),
        "closed_z_cells": _edge_index(z, 0.250e-6) - _edge_index(z, -0.588e-6),
    }


def mesh_audit(factor: int = 2) -> dict[str, Any]:
    spec = UserBalancedZRefinementSpec(factor=factor)
    x, y, z = grid_edges(spec.factor)
    base_x, base_y, base_z = baseline_grid_edges()
    vertical = [segment.audit() for segment in vertical_segments(spec.factor)]
    payload: dict[str, Any] = {
        "version": VERSION,
        "spec": asdict(spec),
        "axis": "full_domain_z",
        "factor_from_user_baseline": spec.factor,
        "grid_shape_xyz": [x.size - 1, y.size - 1, z.size - 1],
        "yee_cell_count": int((x.size - 1) * (y.size - 1) * (z.size - 1)),
        "bounds_m": [[x[0], x[-1]], [y[0], y[-1]], [z[0], z[-1]]],
        "vertical_segments": vertical,
        "layout": layout_values(spec.factor),
        "pml_cells_each_face_xyz": [8, 8, 8 * spec.factor],
        "invariants": {
            "x_edges_byte_exact_to_user_baseline": np.array_equal(x, base_x),
            "y_edges_byte_exact_to_user_baseline": np.array_equal(y, base_y),
            "z_physical_bounds_unchanged": bool(
                z[0] == base_z[0] and z[-1] == base_z[-1]
            ),
            "every_baseline_z_edge_retained": all(
                np.any(np.isclose(z, edge, rtol=0.0, atol=2.0e-18)) for edge in base_z
            ),
            "every_z_segment_cell_count_scaled": all(
                refined.cells == baseline.cells * spec.factor
                for refined, baseline in zip(
                    vertical_segments(spec.factor),
                    baseline_vertical_segments(),
                    strict=True,
                )
            ),
        },
        "rules": {
            "user_baseline_contract_modified": False,
            "source_pair_required_before_material_case": True,
            "exact_binary_only": True,
            "gray_material_allowed": False,
            "optimizer_start_allowed": False,
            "one_refinement_pair_is_not_final_mesh_convergence": True,
        },
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    payload["grid_contract_sha256"] = hashlib.sha256(encoded).hexdigest()
    return payload


def case_contract(time_spec: TimeSpec, factor: int = 2) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "version": CASE_VERSION,
        "mesh": mesh_audit(factor),
        "time": {
            "total_periods": time_spec.total_periods,
            "window_periods": time_spec.window_periods,
            "courant_factor": time_spec.courant_factor,
            "source_startup_periods": time_spec.source_startup_periods,
        },
        "pml": {
            "layers_each_face_xyz": [8, 8, 8 * factor],
            "physical_thickness_m_xyz": [1.0e-6, 1.0e-6, 1.6e-6],
            "alpha_scale": 1.0,
            "target_reflection": 1e-6,
        },
        "rules": {
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
def mesh_context(factor: int = 2) -> Iterator[dict[str, Any]]:
    from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch import (
        fdtdx_4um_model as optical_model,
    )

    previous_layout = optical_model.LAYOUT
    previous_edges = optical_model.grid_edges
    edges = grid_edges(factor)
    optical_model.LAYOUT = optical_model.GridLayout(**layout_values(factor))
    optical_model.grid_edges = lambda: tuple(axis.copy() for axis in edges)
    try:
        yield mesh_audit(factor)
    finally:
        optical_model.LAYOUT = previous_layout
        optical_model.grid_edges = previous_edges


def build_model(
    polarization: str,
    *,
    factor: int,
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

    UserBalancedZRefinementSpec(factor=factor)
    profiles = pml_face_parameters()
    with mesh_context(factor):
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
    model["fresh_mesh_audit"] = mesh_audit(factor)
    model["pml_face_parameters"] = profiles
    return model


def material_spec(factor: int = 2) -> UserBalancedMeshSpec:
    """Return the unchanged 100-nm x/y mask mapping for z refinement."""

    UserBalancedZRefinementSpec(factor=factor)
    return UserBalancedMeshSpec()
