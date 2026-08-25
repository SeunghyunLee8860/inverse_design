"""Immutable contract for the new license-free FDTDX parity route.

This module deliberately has no FDTDX, JAX, CUDA, or Lumerical import.  It
defines the physical grid and placement from SI coordinates so later solver
builders cannot inherit the historical FDTDX integer offsets.  FDTDX remains
a candidate generator; it is never final Lumerical/CV0 authority.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Callable, Iterable

import numpy as np

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_runtime_provenance import (
    audit_runtime,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_parity_resources import (
    resource_audit,
)


FDTDX_SOURCE_COMMIT = "f26f84b70a8cceec9b889553955a868624736bf1"
DEFAULT_FDTDX_SOURCE = Path(
    "/home/seunghyun200/dependencies/"
    "fdtdx-f26f84b70a8cceec9b889553955a868624736bf1"
)


@dataclass(frozen=True)
class AxisSegment:
    name: str
    start_m: float
    stop_m: float
    cells: int
    material_region: str
    pml: bool = False

    @property
    def pitch_m(self) -> float:
        return (self.stop_m - self.start_m) / self.cells

    def edges(self) -> np.ndarray:
        return np.linspace(self.start_m, self.stop_m, self.cells + 1, dtype=np.float64)


@dataclass(frozen=True)
class ParityPhysics:
    wavelength_m: float = 4.0e-6
    gaussian_waist_m: float = 4.0e-6
    reporting_incident_power_W: float = 285.0e-6
    source_z_m: float = 0.75e-6
    incident_monitor_z_m: float = 0.50e-6
    endpoint_monitor_z_m: float = 0.10e-6
    flux_box_bottom_z_m: float = -0.385e-6
    flux_box_top_z_m: float = 0.50e-6
    flake_plane_z_m: float = 0.0
    domain_half_span_xy_m: float = 10.0e-6
    domain_z_min_m: float = -3.0e-6
    domain_z_max_m: float = 3.0e-6
    flake_half_span_m: float = 8.0e-6
    design_half_span_m: float = 4.0e-6
    sio2_bottom_m: float = -0.385e-6
    flake_bottom_m: float = -0.100e-6
    flake_top_m: float = 0.0
    au_top_m: float = 0.050e-6
    courant_factor: float = 0.25
    total_periods: int = 40
    late_phasor_periods: int = 4
    pml_cells_each_face: int = 8


PHYSICS = ParityPhysics()


def lateral_segments() -> tuple[AxisSegment, ...]:
    """Return the symmetric 20-um lateral grid requested by the user."""

    return (
        AxisSegment("minus_pml", -10e-6, -9e-6, 8, "air", pml=True),
        AxisSegment("minus_outer", -9e-6, -8e-6, 5, "air"),
        AxisSegment("flake_and_design", -8e-6, 8e-6, 160, "device"),
        AxisSegment("plus_outer", 8e-6, 9e-6, 5, "air"),
        AxisSegment("plus_pml", 9e-6, 10e-6, 8, "air", pml=True),
    )


def vertical_segments() -> tuple[AxisSegment, ...]:
    """Return the exact 2.5-nm thin-stack / <=50-nm bulk-air grid."""

    return (
        AxisSegment("bottom_pml", -3.0e-6, -2.6e-6, 8, "Si", pml=True),
        AxisSegment("Si_bulk", -2.6e-6, -0.385e-6, 45, "Si"),
        AxisSegment("SiO2", -0.385e-6, -0.100e-6, 114, "SiO2"),
        AxisSegment("TaIrTe4", -0.100e-6, 0.0, 40, "TaIrTe4"),
        AxisSegment("Au_design", 0.0, 0.050e-6, 20, "Au_or_air"),
        AxisSegment("air", 0.050e-6, 2.6e-6, 51, "air"),
        AxisSegment("top_pml", 2.6e-6, 3.0e-6, 8, "air", pml=True),
    )


def _join_edges(segments: Iterable[AxisSegment]) -> np.ndarray:
    parts: list[np.ndarray] = []
    previous_stop: float | None = None
    for segment in segments:
        if segment.cells <= 0 or segment.stop_m <= segment.start_m:
            raise ValueError(f"invalid segment {segment.name!r}")
        if previous_stop is not None and not np.isclose(
            segment.start_m, previous_stop, rtol=0.0, atol=1e-18
        ):
            raise ValueError(f"noncontiguous segment {segment.name!r}")
        edges = segment.edges()
        parts.append(edges if not parts else edges[1:])
        previous_stop = segment.stop_m
    return np.concatenate(parts)


def grid_edges() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = _join_edges(lateral_segments())
    y = x.copy()
    z = _join_edges(vertical_segments())
    return x, y, z


def _edge_index(edges: np.ndarray, coordinate_m: float) -> int:
    matches = np.flatnonzero(np.isclose(edges, coordinate_m, rtol=0.0, atol=2e-18))
    if matches.size != 1:
        raise ValueError(f"{coordinate_m:.12e} m is not exactly one grid edge")
    return int(matches[0])


def _cell_slice(edges: np.ndarray, start_m: float, stop_m: float) -> slice:
    start = _edge_index(edges, start_m)
    stop = _edge_index(edges, stop_m)
    if stop <= start:
        raise ValueError("cell slice must have positive extent")
    return slice(start, stop)


def _slice_payload(value: slice) -> list[int]:
    return [int(value.start), int(value.stop)]


def placement_contract() -> dict[str, object]:
    """Translate physical object/monitor coordinates to the new grid once."""

    x, y, z = grid_edges()
    whole_x = slice(0, x.size - 1)
    whole_y = slice(0, y.size - 1)
    flake_x = _cell_slice(x, -8e-6, 8e-6)
    flake_y = _cell_slice(y, -8e-6, 8e-6)
    design_x = _cell_slice(x, -4e-6, 4e-6)
    design_y = _cell_slice(y, -4e-6, 4e-6)
    flux_x = _cell_slice(x, -8.2e-6, 8.2e-6)
    flux_y = _cell_slice(y, -8.2e-6, 8.2e-6)

    def volume(x_slice: slice, y_slice: slice, z_slice: slice) -> list[list[int]]:
        return [_slice_payload(x_slice), _slice_payload(y_slice), _slice_payload(z_slice)]

    return {
        "volumes_cell_slices": {
            "Si": volume(whole_x, whole_y, _cell_slice(z, -3.0e-6, -0.385e-6)),
            "SiO2": volume(whole_x, whole_y, _cell_slice(z, -0.385e-6, -0.100e-6)),
            "TaIrTe4": volume(flake_x, flake_y, _cell_slice(z, -0.100e-6, 0.0)),
            "Au_design": volume(design_x, design_y, _cell_slice(z, 0.0, 0.050e-6)),
            "closed_flux_box": volume(
                flux_x,
                flux_y,
                _cell_slice(z, PHYSICS.flux_box_bottom_z_m, PHYSICS.flux_box_top_z_m),
            ),
        },
        "planes_edge_indices": {
            "source": {"axis": "z", "index": _edge_index(z, PHYSICS.source_z_m)},
            "incident_power": {
                "axis": "z",
                "index": _edge_index(z, PHYSICS.incident_monitor_z_m),
            },
            "air_endpoint_field": {
                "axis": "z",
                "index": _edge_index(z, PHYSICS.endpoint_monitor_z_m),
            },
            "flake_profile": {
                "axis": "z",
                "index": _edge_index(z, PHYSICS.flake_plane_z_m),
            },
        },
        "source_aperture_cell_slices_xy": [
            _slice_payload(flake_x),
            _slice_payload(flake_y),
        ],
        "pml_cell_slices": {
            "x_minus": [0, 8],
            "x_plus": [178, 186],
            "y_minus": [0, 8],
            "y_plus": [178, 186],
            "z_minus": [0, 8],
            "z_plus": [278, 286],
        },
    }


def _array_sha256(array: np.ndarray) -> str:
    canonical = np.ascontiguousarray(np.asarray(array, dtype="<f8"))
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def grid_hashes() -> dict[str, str]:
    x, y, z = grid_edges()
    joined = np.concatenate((x, y, z))
    return {
        "x_edges_sha256": _array_sha256(x),
        "y_edges_sha256": _array_sha256(y),
        "z_edges_sha256": _array_sha256(z),
        "xyz_edges_sha256": _array_sha256(joined),
    }


def grid_audit() -> dict[str, object]:
    x, y, z = grid_edges()
    shape = (x.size - 1, y.size - 1, z.size - 1)
    dx = np.diff(x)
    dz = np.diff(z)
    required_x_edges = (-10e-6, -9e-6, -8e-6, -4e-6, 4e-6, 8e-6, 9e-6, 10e-6)
    required_z_edges = (
        -3.0e-6,
        -2.6e-6,
        -0.385e-6,
        -0.100e-6,
        0.0,
        0.050e-6,
        0.10e-6,
        0.50e-6,
        0.75e-6,
        2.6e-6,
        3.0e-6,
    )
    exact_edges = all(
        np.count_nonzero(np.isclose(x, value, rtol=0.0, atol=2e-18)) == 1
        for value in required_x_edges
    ) and all(
        np.count_nonzero(np.isclose(z, value, rtol=0.0, atol=2e-18)) == 1
        for value in required_z_edges
    )
    thin = np.concatenate((dz[53:167], dz[167:207], dz[207:227]))
    checks = {
        "shape_is_186_186_286": shape == (186, 186, 286),
        "cell_count_is_9894456": int(np.prod(shape)) == 9_894_456,
        "central_xy_pitch_is_100nm": bool(
            np.allclose(dx[13:173], 100e-9, rtol=0.0, atol=2e-18)
        ),
        "outer_xy_pitch_at_most_200nm": bool(
            np.max(np.concatenate((dx[8:13], dx[173:178]))) <= 200e-9 + 2e-18
        ),
        "thin_stack_pitch_is_2p5nm": bool(
            np.allclose(thin, 2.5e-9, rtol=0.0, atol=2e-18)
        ),
        "bulk_air_pitch_at_most_50nm": bool(
            np.max(np.concatenate((dz[8:53], dz[227:278]))) <= 50e-9 + 2e-18
        ),
        "pml_is_eight_cells_each_face": all(
            segment.cells == PHYSICS.pml_cells_each_face
            for segment in (*lateral_segments(), *vertical_segments())
            if segment.pml
        ),
        "all_required_physical_planes_are_edges": bool(exact_edges),
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "shape_cells": list(shape),
        "cell_count": int(np.prod(shape)),
        "minimum_pitch_m": float(min(dx.min(), dz.min())),
        "maximum_pitch_m": float(max(dx.max(), dz.max())),
        "segments": {
            "x_and_y": [asdict(segment) | {"pitch_m": segment.pitch_m} for segment in lateral_segments()],
            "z": [asdict(segment) | {"pitch_m": segment.pitch_m} for segment in vertical_segments()],
        },
        "placements": placement_contract(),
        "resources": resource_audit(
            shape=shape,
            min_spacings_m=(float(dx.min()), float(dx.min()), float(dz.min())),
            wavelength_m=PHYSICS.wavelength_m,
            courant_factor=PHYSICS.courant_factor,
            total_periods=PHYSICS.total_periods,
            late_periods=PHYSICS.late_phasor_periods,
            pml_cells=PHYSICS.pml_cells_each_face,
        ),
        "hashes": grid_hashes(),
    }


GitRunner = Callable[[list[str], Path], str]


def _run_git(arguments: list[str], cwd: Path) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=cwd, text=True, stderr=subprocess.STDOUT
    ).strip()


def fdtdx_runtime_audit(
    source: Path = DEFAULT_FDTDX_SOURCE,
    *,
    git_runner: GitRunner = _run_git,
    resolved_module: Path | None = None,
) -> dict[str, object]:
    """Check both the source commit and the Python import provenance."""

    return audit_runtime(
        source,
        expected_commit=FDTDX_SOURCE_COMMIT,
        git_runner=git_runner,
        resolved_module=resolved_module,
    )


def parity_contract() -> dict[str, object]:
    return {
        "schema": "fdtdx_4um_parity_contract_v1",
        "authority": {
            "role": "license_free_candidate_generator_only",
            "fdtdx_allowed_as_final_authority": False,
            "claims_cv0": False,
            "requires_later_lumerical_cv0_and_finer_mesh": True,
            "lumerical_heat_charge_calls_allowed_in_this_route": False,
        },
        "physics": asdict(PHYSICS),
        "coordinates": {
            "solver_x": "TaIrTe4_b",
            "solver_y": "TaIrTe4_a",
            "propagation": "minus_z",
            "Ea_electric_vector": [0.0, 1.0, 0.0],
            "Eb_electric_vector": [1.0, 0.0, 0.0],
            "positive_current": "plus_x_from_x_min_to_x_max",
            "target_signs": {"Ea": "positive", "Eb": "negative"},
        },
        "topology": {
            "latent_shape": [81, 81],
            "projected_nodal_shape": [81, 81],
            "physical_cell_shape": [80, 80],
            "physical_cell_pitch_m": 100e-9,
            "finite_nonperiodic_conic_filter_radius_m": 500e-9,
            "projection_eta": 0.5,
            "projection_beta_first_certificate": 4.0,
            "nodal_to_cell": "exact_four_node_average_and_committed_transpose",
            "one_shared_occupancy_for_all_physics": True,
        },
        "optical_density_law": {
            "name": "n_k_linear_then_square",
            "n": "1 + rho*(2.2-1)",
            "k": "rho*28.9",
            "epsilon": "(n + 1j*k)**2",
            "discrete_float32_ADE_relative_error_limit": 1e-5,
            "rho_cubed_allowed": False,
            "c3_only_scaling_allowed": False,
        },
        "source_normalization": {
            "calibration": "source_only_all_air_separately_for_Ea_and_Eb",
            "scale_each_result_by": "285e-6 / unscaled_incident_power_pol",
            "polarization_matching_or_empirical_rescaling_allowed": False,
            "unscaled_incident_power_relative_mismatch_limit": 0.005,
        },
        "time": {
            "courant_factor": PHYSICS.courant_factor,
            "total_carrier_periods": PHYSICS.total_periods,
            "late_phasor_window_periods": PHYSICS.late_phasor_periods,
            "field_dtype": "float32",
        },
        "objective": {
            "sense": "maximize_t",
            "constraints": ["t - I_Ea <= 0", "t + I_Eb <= 0"],
        },
        "gates": {
            "ADE_uniform_density_sweep_before_fields": True,
            "full_Ea_and_Eb_centered_AD_FD_before_optimizer": True,
            "optimizer_enabled": False,
            "first_optimizer_run_after_all_gates": "beta4_two_iterations_only",
        },
        "hard_prohibitions": [
            "legacy_scripts_10_12_13",
            "historical_80x80_optimizer_checkpoint",
            "rho_cubed_optical_law",
            "rho_times_c3_Au_called_parity",
            "independent_optical_thermal_electrical_density",
            "Q_clipping_smoothing_or_rescaling",
            "gradient_fit_or_rescale_to_FD",
        ],
        "grid": grid_audit(),
    }


def main() -> int:
    payload = parity_contract()
    payload["runtime"] = fdtdx_runtime_audit()
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["grid"]["status"] == "PASS" and payload["runtime"]["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
