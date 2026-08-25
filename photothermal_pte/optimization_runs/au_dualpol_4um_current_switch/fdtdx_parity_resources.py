"""Analytic time-step and persistent-array accounting for FDTDX parity."""

from __future__ import annotations

import math


VACUUM_LIGHT_SPEED_M_S = 299_792_458.0


def resource_audit(
    *,
    shape: tuple[int, int, int],
    min_spacings_m: tuple[float, float, float],
    wavelength_m: float,
    courant_factor: float,
    total_periods: int,
    late_periods: int,
    pml_cells: int,
) -> dict[str, object]:
    """Return exact CFL/work counts and transparent float32 memory lower bounds.

    The memory figures follow the pinned FDTDX ``ArrayContainer`` allocation:
    real E/H fields, one broadcast inverse-permittivity component, one to three
    axis-aligned ADE poles, and four CPML auxiliary slab arrays.  They exclude
    detector buffers, XLA temporaries, cotangents, checkpoint scheduling,
    allocator overhead, and CUDA compilation workspace, so they are explicitly
    not a GPU peak-memory prediction.
    """

    nx, ny, nz = shape
    dx_min, dy_min, dz_min = min_spacings_m
    inverse_metric = dx_min**-2 + dy_min**-2 + dz_min**-2
    dt_s = courant_factor / (VACUUM_LIGHT_SPEED_M_S * math.sqrt(inverse_metric))
    period_s = wavelength_m / VACUUM_LIGHT_SPEED_M_S
    steps_per_period = period_s / dt_s
    total_steps = round(total_periods * steps_per_period)
    late_steps = round(late_periods * steps_per_period)
    cells = nx * ny * nz

    # Six CPML objects each allocate two psi_E and two psi_H slabs.  Edge and
    # corner overlap is intentionally counted for every boundary object because
    # FDTDX stores separate dictionaries for them.
    pml_slab_cells = (
        2 * pml_cells * ny * nz
        + 2 * nx * pml_cells * nz
        + 2 * nx * ny * pml_cells
    )
    float32_bytes = 4
    pml_psi_bytes = 4 * pml_slab_cells * float32_bytes
    # Six broadcast-shaped coefficient profiles (a/b/kappa for E and H) per
    # face.  Each contains only pml_cells values, not a full slab.
    pml_profile_bytes = 6 * 6 * pml_cells * float32_bytes

    pole_cases: dict[str, object] = {}
    for num_poles in (1, 2, 3):
        # Per full-domain voxel: E/H=6, broadcast inv(epsilon)=1;
        # per pole P_curr/P_prev=6 and c1/c2/c3=9.  c4 is excluded because the
        # selected carrier has not yet been chosen and certified.
        persistent_components = 6 + 1 + 15 * num_poles
        dynamic_checkpoint_components = 6 + 6 * num_poles
        persistent_bytes = (
            persistent_components * cells * float32_bytes
            + pml_psi_bytes
            + pml_profile_bytes
        )
        dynamic_checkpoint_bytes = (
            dynamic_checkpoint_components * cells * float32_bytes
            + pml_psi_bytes
        )
        pole_cases[str(num_poles)] = {
            "persistent_array_lower_bound_bytes": persistent_bytes,
            "persistent_array_lower_bound_GiB": persistent_bytes / 2**30,
            "one_dynamic_checkpoint_lower_bound_bytes": dynamic_checkpoint_bytes,
            "one_dynamic_checkpoint_lower_bound_GiB": dynamic_checkpoint_bytes / 2**30,
            "additional_c4_if_used_bytes": 3 * num_poles * cells * float32_bytes,
        }

    return {
        "status": "ANALYTIC_LOWER_BOUND_ONLY",
        "field_run_feasibility": "UNDETERMINED_UNTIL_DRY_ALLOCATION_AND_TIMED_MICROBENCHMARK",
        "time": {
            "cfl_formula": "courant/(c*sqrt(dx_min^-2+dy_min^-2+dz_min^-2))",
            "dt_s": dt_s,
            "carrier_period_s": period_s,
            "steps_per_period": steps_per_period,
            "total_steps": total_steps,
            "late_window_steps": late_steps,
            "total_simulated_time_s": total_periods * period_s,
        },
        "work": {
            "cells": cells,
            "cell_steps_per_forward": cells * total_steps,
            "estimated_wall_time_s": None,
        },
        "memory": {
            "dtype": "float32_real_fields",
            "pml_slab_cells_with_boundary_overlap": pml_slab_cells,
            "pml_psi_bytes": pml_psi_bytes,
            "pole_cases_no_c4": pole_cases,
            "excluded_from_lower_bound": [
                "detectors_and_phasor_accumulators",
                "XLA_temporaries_and_cotangents",
                "checkpoint_scheduler_storage",
                "allocator_and_CUDA_workspace",
            ],
        },
        "required_next_gate": (
            "after_ADE_selection, perform no-field setup/dry allocation and a short "
            "timed forward microbenchmark on a verified-idle permitted GPU"
        ),
    }
