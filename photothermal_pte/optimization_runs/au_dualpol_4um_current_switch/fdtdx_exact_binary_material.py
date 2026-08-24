"""Exact air/Au endpoint placement for fresh FDTDX reference calculations."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_exact_binary_convergence import (
    DESIGN_CELLS,
    MeshSpec,
    upsample_mask,
)


MATERIAL_LAW = "exact-air-or-ordinary-au-ADE-endpoints-v1"
INCREMENT_MATERIAL_LAW = "exact-air-or-ordinary-au-increment-A-C-B-endpoints-v1"
COEFFICIENT_NAMES = ("dispersive_c1", "dispersive_c2", "dispersive_c3")


def normalize_exact_mask(mask: Any) -> np.ndarray:
    """Return a contiguous uint8 80x80 mask and reject all gray/float inputs."""

    value = np.asarray(mask)
    if value.shape != (DESIGN_CELLS, DESIGN_CELLS):
        raise ValueError(f"exact mask must have shape {(DESIGN_CELLS, DESIGN_CELLS)}")
    if value.dtype.kind not in "biu":
        raise ValueError("exact mask dtype must be bool or integer; float density is forbidden")
    if not np.all((value == 0) | (value == 1)):
        raise ValueError("exact mask must contain only integer 0/1 endpoints")
    return np.ascontiguousarray(value, dtype=np.uint8)


def solver_mask(mask: Any, spec: MeshSpec) -> np.ndarray:
    """Map one design mask to the local Yee mesh without interpolation."""

    design = normalize_exact_mask(mask)
    design_tuple = tuple(tuple(int(item) for item in row) for row in design)
    expanded = upsample_mask(design_tuple, spec.design_xy_factor)
    result = np.ascontiguousarray(expanded, dtype=np.uint8)
    expected = DESIGN_CELLS * spec.design_xy_factor
    if result.shape != (expected, expected):
        raise RuntimeError(f"unexpected solver mask shape {result.shape}")
    if int(np.count_nonzero(result)) != int(np.count_nonzero(design)) * (
        spec.design_xy_factor**2
    ):
        raise RuntimeError("piecewise-constant mask replication changed Au area")
    return result


def _sha256_uint8(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value, dtype=np.uint8).tobytes()).hexdigest()


def mask_material_audit(mask: Any, spec: MeshSpec) -> dict[str, Any]:
    design = normalize_exact_mask(mask)
    expanded = solver_mask(design, spec)
    return {
        "material_law": MATERIAL_LAW,
        "gray_density_allowed": False,
        "rho_power": None,
        "design_mask_shape": list(design.shape),
        "solver_mask_shape": list(expanded.shape),
        "design_xy_factor": spec.design_xy_factor,
        "design_solid_cells": int(np.count_nonzero(design)),
        "solver_solid_cells": int(np.count_nonzero(expanded)),
        "design_mask_sha256": _sha256_uint8(design),
        "solver_mask_sha256": _sha256_uint8(expanded),
        "mapping": "integer piecewise-constant replication",
        "air_endpoint": {
            "epsilon_infinity": 1.0,
            "ADE_c1": 0.0,
            "ADE_c2": 0.0,
            "ADE_c3": 0.0,
        },
        "au_endpoint": "locked ordinary-Au finite-dt ADE coefficients",
    }


def coefficient_endpoint_matrix(
    model: dict[str, Any], material: str
) -> np.ndarray:
    """Return a finite ``(num_poles, 3)`` c1/c2/c3 endpoint matrix."""

    source = model.get("coefficient_endpoints")
    raw = (
        source[material]
        if source is not None
        else model["coefficients"][material]
    )
    value = np.asarray(raw, dtype=np.float32)
    if value.ndim == 1:
        value = value[None, :]
    if value.ndim != 2 or value.shape[0] < 1 or value.shape[1] != 3:
        raise RuntimeError(
            f"invalid ADE endpoint matrix for material {material!r}: {value.shape}"
        )
    if not np.all(np.isfinite(value)):
        raise RuntimeError(f"non-finite ADE endpoint for material {material!r}")
    return value


def arrays_for_exact_binary(model: dict[str, Any], mask: Any, spec: MeshSpec):
    """Return reset solver arrays with every Au pole masked at exact 0/1."""

    expanded = solver_mask(mask, spec)
    au_slice = model["slices"]["au_design"]
    au_shape = tuple(int(part.stop) - int(part.start) for part in au_slice)
    if au_shape[:2] != expanded.shape:
        raise RuntimeError(
            f"Au placement shape {au_shape[:2]} does not match mask {expanded.shape}"
        )
    if au_shape[2] <= 0:
        raise RuntimeError("Au placement has no z cells")

    jnp = model["jnp"]
    binary = jnp.asarray(expanded, dtype=jnp.float32)
    endpoints = coefficient_endpoint_matrix(model, "au")
    coefficient_arrays = []
    for coefficient_index, name in enumerate(COEFFICIENT_NAMES):
        coefficient = model[f"fixed_{name.removeprefix("dispersive_")}"]
        if coefficient.shape[:2] != (endpoints.shape[0], 3):
            raise RuntimeError(
                f"{name} pole/component shape does not match Au endpoints"
            )
        for pole_index, endpoint in enumerate(endpoints[:, coefficient_index]):
            for component in range(3):
                coefficient = coefficient.at[
                    (pole_index, component, *au_slice)
                ].set(float(endpoint) * binary[:, :, None])
        coefficient_arrays.append(coefficient)

    arrays = model["base"].reset()
    for name, coefficient in zip(COEFFICIENT_NAMES, coefficient_arrays, strict=True):
        arrays = arrays.aset(name, coefficient)
    return arrays


def readback_exact_binary(
    model: dict[str, Any], arrays: Any, mask: Any, spec: MeshSpec
) -> dict[str, Any]:
    """Copy only the Au window and prove every exact air/Au pole endpoint."""

    expanded = solver_mask(mask, spec)
    endpoints = coefficient_endpoint_matrix(model, "au")
    au_slice = model["slices"]["au_design"]
    nz = int(au_slice[2].stop) - int(au_slice[2].start)
    expected_shape = (
        endpoints.shape[0],
        3,
        expanded.shape[0],
        expanded.shape[1],
        nz,
    )
    binary_5d = np.broadcast_to(
        expanded[None, None, :, :, None], expected_shape
    )
    checks: dict[str, bool] = {}
    coefficients: dict[str, Any] = {}

    for coefficient_index, name in enumerate(COEFFICIENT_NAMES):
        field = getattr(arrays, name)
        observed = np.asarray(field[(slice(None), slice(None), *au_slice)])
        endpoint = np.asarray(
            endpoints[:, coefficient_index], dtype=observed.dtype
        )
        expected = endpoint[:, None, None, None, None] * binary_5d
        exact = observed.shape == expected_shape and np.array_equal(observed, expected)
        checks[f"{name}_exact_air_au_endpoints"] = exact
        coefficients[name] = {
            "endpoint": float(endpoint[0]) if endpoint.size == 1 else None,
            "endpoints_by_pole": [float(item) for item in endpoint],
            "observed_shape": list(observed.shape),
            "observed_unique": [float(item) for item in np.unique(observed)],
            "max_absolute_error": (
                float(np.max(np.abs(observed - expected)))
                if observed.shape == expected.shape
                else None
            ),
        }

    inverse_permittivity = np.asarray(
        arrays.inv_permittivities[(slice(None), *au_slice)]
    )
    checks["au_window_epsilon_infinity_inverse_is_one"] = np.array_equal(
        inverse_permittivity, np.ones_like(inverse_permittivity)
    )
    checks["solver_mask_remains_binary"] = bool(
        np.all((expanded == 0) | (expanded == 1))
    )
    checks["no_gray_material_law"] = MATERIAL_LAW.endswith("endpoints-v1")
    checks["lorentz_drude_c4_array_absent"] = (
        getattr(arrays, "dispersive_c4", None) is None
    )
    audit = mask_material_audit(mask, spec)
    state_representation = model.get(
        "dispersive_state_representation", "polarization"
    )
    if state_representation == "increment":
        audit["material_law"] = INCREMENT_MATERIAL_LAW
        audit["air_endpoint"] = {
            "epsilon_infinity": 1.0,
            "increment_A": 0.0,
            "increment_C": 0.0,
            "increment_B": 0.0,
        }
    audit["dispersive_state_representation"] = state_representation
    audit["coefficient_array_semantics"] = (
        {"dispersive_c1": "A", "dispersive_c2": "C", "dispersive_c3": "B"}
        if state_representation == "increment"
        else {
            "dispersive_c1": "classic_c1",
            "dispersive_c2": "classic_c2",
            "dispersive_c3": "classic_c3",
        }
    )
    audit.update(
        au_grid_shape=list(expected_shape[2:]),
        au_z_cells=nz,
        num_dispersive_poles=endpoints.shape[0],
        coefficient_readback=coefficients,
        inverse_permittivity_unique=[
            float(item) for item in np.unique(inverse_permittivity)
        ],
        checks=checks,
        ready=all(checks.values()),
    )
    return audit
