"""Bridge the solver-independent fresh mesh contract into FDTDX placement."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator, Mapping

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_exact_binary_convergence import (
    MeshSpec,
    grid_edges as contract_grid_edges,
    layout as contract_layout,
    mesh_audit,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_pml import (
    face_parameters,
)


@contextmanager
def mesh_context(spec: MeshSpec) -> Iterator[dict[str, Any]]:
    """Install one fresh mesh in the historical builder for one serial build."""

    import numpy as np

    from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch import (
        fdtdx_4um_model as optical_model,
    )

    previous_layout = optical_model.LAYOUT
    previous_edges = optical_model.grid_edges
    edges = tuple(
        np.asarray(axis, dtype=np.float64) for axis in contract_grid_edges(spec)
    )
    optical_model.LAYOUT = optical_model.GridLayout(**contract_layout(spec))
    optical_model.grid_edges = lambda: tuple(axis.copy() for axis in edges)
    try:
        yield mesh_audit(spec)
    finally:
        optical_model.LAYOUT = previous_layout
        optical_model.grid_edges = previous_edges


def build_model(
    spec: MeshSpec,
    polarization: str,
    *,
    total_periods: int,
    window_periods: int,
    courant_factor: float,
    alpha_scale: float = 1.0,
    target_reflection: float = 1.0e-6,
    include_adjoint_source: bool = False,
    air_only_source_calibration: bool = False,
    material_law_contract: Mapping[str, Any] | None = None,
    dispersive_state_representation: str = "polarization",
) -> dict[str, Any]:
    """Build a fresh model; an implicit upstream PML is not an option."""

    from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch import (
        fdtdx_4um_model as optical_model,
    )

    profiles = face_parameters(
        spec,
        alpha_scale=alpha_scale,
        target_reflection=target_reflection,
    )
    with mesh_context(spec):
        model = optical_model.build_model(
            polarization,
            total_periods=total_periods,
            window_periods=window_periods,
            courant_factor=courant_factor,
            include_adjoint_source=include_adjoint_source,
            air_only_source_calibration=air_only_source_calibration,
            pml_face_parameters=profiles,
            material_law_contract=material_law_contract,
            dispersive_state_representation=dispersive_state_representation,
        )
    if model.get("pml_face_parameters") != profiles:
        raise RuntimeError("fresh FDTDX model did not preserve explicit PML provenance")
    model["fresh_mesh_audit"] = mesh_audit(spec)
    return model
