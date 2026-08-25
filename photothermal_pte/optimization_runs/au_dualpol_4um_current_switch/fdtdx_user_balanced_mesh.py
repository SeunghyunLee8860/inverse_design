"""User-requested balanced FDTDX mesh for a new 4-um Maxwell baseline.

This profile is deliberately independent of the historical ``MeshSpec`` and
its certificates.  It keeps the existing 20 x 20 x 6 um physical domain and
material/source planes while realizing:

* 100 nm x/y cells across the complete TaIrTe4 flake and Au design window;
* 200 nm x/y cells only in the air margins outside the flake;
* 5 nm z cells in SiO2, TaIrTe4, and Au;
* 50 nm z cells in non-PML air; and
* the existing eight cells in every PML.

The frozen resolved-Si buffer is 1.015 um, which is not an integer multiple of
50 nm.  It is therefore represented by 20 uniform 50.75 nm cells rather than
moving a physical or PML boundary.  This 1.5% exception is fail-closed in the
audit and must not be described as an exact 50 nm Si mesh.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Iterator

import numpy as np

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_exact_binary_convergence import (
    pml_parameters,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_pml import (
    PML_FACES,
)


VERSION = "fdtdx-user-balanced-mesh-v1"
PROFILE = "flake100nm_outer200nm_thin5nm_air50nm_pml8"


@dataclass(frozen=True)
class UserBalancedMeshSpec:
    """Fixed mesh identity with attributes required by exact-binary helpers."""

    profile: str = PROFILE
    design_xy_factor: int = 1
    lateral_pml_thickness_m: float = 1.0e-6
    z_pml_thickness_m: float = 1.6e-6

    def __post_init__(self) -> None:
        if self.profile != PROFILE:
            raise ValueError("unexpected balanced-mesh profile")
        if self.design_xy_factor != 1:
            raise ValueError("the requested 100-nm design pitch requires factor one")
        if self.lateral_pml_thickness_m != 1.0e-6:
            raise ValueError("the frozen lateral PML thickness is 1 um")
        if self.z_pml_thickness_m != 1.6e-6:
            raise ValueError("the frozen z-PML thickness is 1.6 um")


@dataclass(frozen=True)
class Segment:
    name: str
    start_m: float
    stop_m: float
    cells: int
    role: str
    requested_pitch_m: float | None

    @property
    def step_m(self) -> float:
        return (self.stop_m - self.start_m) / self.cells

    def audit(self) -> dict[str, Any]:
        requested = self.requested_pitch_m
        return {
            "name": self.name,
            "bounds_m": [self.start_m, self.stop_m],
            "cells": self.cells,
            "role": self.role,
            "requested_pitch_m": requested,
            "realized_pitch_m": self.step_m,
            "relative_pitch_error": (
                0.0
                if requested is None
                else abs(self.step_m - requested) / requested
            ),
        }


def lateral_segments() -> tuple[Segment, ...]:
    """Return x/y segments; the complete flake remains on 100-nm cells."""

    return (
        Segment("left_pml", -10.0e-6, -9.0e-6, 8, "pml", None),
        Segment("left_air_margin", -9.0e-6, -8.0e-6, 5, "outer_air", 200e-9),
        Segment("left_flake_wing", -8.0e-6, -4.0e-6, 40, "flake", 100e-9),
        Segment("au_design_window", -4.0e-6, 4.0e-6, 80, "design", 100e-9),
        Segment("right_flake_wing", 4.0e-6, 8.0e-6, 40, "flake", 100e-9),
        Segment("right_air_margin", 8.0e-6, 9.0e-6, 5, "outer_air", 200e-9),
        Segment("right_pml", 9.0e-6, 10.0e-6, 8, "pml", None),
    )


def vertical_segments() -> tuple[Segment, ...]:
    """Return z segments with exact 5-nm thin stack and 50-nm non-PML air."""

    return (
        Segment("bottom_pml_si", -3.000e-6, -1.400e-6, 8, "pml", None),
        Segment(
            "resolved_si",
            -1.400e-6,
            -0.385e-6,
            20,
            "si_bulk",
            50e-9,
        ),
        Segment("sio2", -0.385e-6, -0.100e-6, 57, "thin_stack", 5e-9),
        Segment("tairte4", -0.100e-6, 0.0, 20, "thin_stack", 5e-9),
        Segment("au", 0.0, 0.050e-6, 10, "thin_stack", 5e-9),
        Segment("near_air", 0.050e-6, 0.250e-6, 4, "air_bulk", 50e-9),
        Segment("middle_air", 0.250e-6, 0.750e-6, 10, "air_bulk", 50e-9),
        Segment("source_air", 0.750e-6, 1.400e-6, 13, "air_bulk", 50e-9),
        Segment("top_pml_air", 1.400e-6, 3.000e-6, 8, "pml", None),
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
        raise RuntimeError("balanced mesh is not strictly increasing")
    return result


def grid_edges() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lateral = _edges(lateral_segments())
    vertical = _edges(vertical_segments())
    return lateral, lateral.copy(), vertical


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
    vertical = {segment.name: segment for segment in vertical_segments()}
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
        "closed_z_cells": _edge_index(z, 0.250e-6)
        - _edge_index(z, -0.588e-6),
    }


def pml_face_parameters(
    *, alpha_scale: float = 1.0, target_reflection: float = 1.0e-6
) -> dict[str, dict[str, float]]:
    spec = UserBalancedMeshSpec()
    return {
        face: pml_parameters(
            (
                spec.lateral_pml_thickness_m
                if face in ("minx", "maxx", "miny", "maxy")
                else spec.z_pml_thickness_m
            ),
            alpha_scale=alpha_scale,
            target_reflection=target_reflection,
        )
        for face in PML_FACES
    }


def mesh_audit() -> dict[str, Any]:
    x, y, z = grid_edges()
    lateral = [segment.audit() for segment in lateral_segments()]
    vertical = [segment.audit() for segment in vertical_segments()]
    payload: dict[str, Any] = {
        "version": VERSION,
        "spec": asdict(UserBalancedMeshSpec()),
        "requested_pitches_m": {
            "design_and_flake_xy": 100e-9,
            "outer_air_xy": 200e-9,
            "thin_stack_z": 5e-9,
            "air_and_si_bulk_z": 50e-9,
        },
        "grid_shape_xyz": [x.size - 1, y.size - 1, z.size - 1],
        "yee_cell_count": int((x.size - 1) * (y.size - 1) * (z.size - 1)),
        "bounds_m": [[x[0], x[-1]], [y[0], y[-1]], [z[0], z[-1]]],
        "lateral_segments": lateral,
        "vertical_segments": vertical,
        "layout": layout_values(),
        "pml_cells_each_face_xyz": [8, 8, 8],
        "known_pitch_exception": {
            "segment": "resolved_si",
            "requested_pitch_m": 50e-9,
            "realized_pitch_m": 50.75e-9,
            "relative_difference": 0.015,
            "reason": "the frozen 1.015-um Si buffer is not divisible by 50 nm",
            "physical_boundary_was_not_moved": True,
        },
        "rules": {
            "historical_mesh_contract_modified": False,
            "exact_binary_only": True,
            "gray_material_allowed": False,
            "optimizer_start_allowed": False,
            "one_forward_pair_is_not_mesh_convergence": True,
        },
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    payload["grid_contract_sha256"] = hashlib.sha256(encoded).hexdigest()
    return payload


@contextmanager
def mesh_context() -> Iterator[dict[str, Any]]:
    """Install the fixed grid only for one serial model build."""

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
    """Build, but do not run, one model on the user-balanced mesh."""

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
