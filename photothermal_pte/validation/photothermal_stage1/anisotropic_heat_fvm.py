"""Conservative Cartesian finite-volume solver for diagonal thermal tensors.

The discretization stores temperature at cell centers.  Conductance between
adjacent cells is formed from the exact series resistance of the two half
cells plus an optional interfacial thermal insulance.  This preserves heat
flux across heterogeneous material interfaces and supports nonuniform grids.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg


FACE_NAMES = (
    "x_min",
    "x_max",
    "y_min",
    "y_max",
    "z_min",
    "z_max",
)


@dataclass(frozen=True)
class SteadyHeatResult:
    temperature_K: np.ndarray
    boundary_power_out_W: dict[str, float]
    source_power_W: float
    energy_balance_relative_error: float
    linear_residual_relative: float
    solver: str
    iterations: int


@dataclass(frozen=True)
class AssembledThermalSystem:
    """One immutable discrete steady-state thermal operator.

    ``matrix_W_K`` is the conductance matrix ``K_T`` on active cell-centred
    temperatures. ``source_volume_operator_m3`` is the diagonal ``M_Q`` that
    converts active-cell source density [W/m3] to cell power [W].
    ``boundary_load_W`` is ``b_T``. Forward and adjoint solves must use this
    same matrix (or its literal transpose); no second assembly is permitted.
    """

    matrix_W_K: sparse.csr_matrix
    source_volume_operator_m3: sparse.csr_matrix
    boundary_load_W: np.ndarray
    active_mask: np.ndarray
    active_ids: np.ndarray
    cell_volume_m3: np.ndarray
    diagonal_W_K: np.ndarray
    boundary_terms: dict[str, tuple[np.ndarray, np.ndarray, float]]
    shape: tuple[int, int, int]
    periodic_axes: tuple[str, ...]

    def active_source(self, source_W_m3: np.ndarray | None) -> np.ndarray:
        source = (
            np.zeros(self.shape, float)
            if source_W_m3 is None
            else np.asarray(source_W_m3, float)
        )
        if source.shape != self.shape:
            raise ValueError(f"source shape {source.shape} != {self.shape}")
        if not np.all(np.isfinite(source)):
            raise ValueError("source contains NaN or Inf")
        if np.any(source[~self.active_mask] != 0.0):
            raise ValueError("source is nonzero in inactive cells")
        return source[self.active_mask].reshape(-1)

    def full_field(self, active_values: np.ndarray) -> np.ndarray:
        values = np.asarray(active_values, float).reshape(-1)
        if values.size != self.matrix_W_K.shape[0]:
            raise ValueError(
                f"active vector length {values.size} != "
                f"{self.matrix_W_K.shape[0]}"
            )
        full = np.full(self.shape, np.nan, float)
        full[self.active_mask] = values
        return full


def _validated_edges(name: str, values: np.ndarray) -> np.ndarray:
    edges = np.asarray(values, float).reshape(-1)
    if edges.size < 2 or not np.all(np.isfinite(edges)):
        raise ValueError(f"{name} edges must contain finite values")
    if not np.all(np.diff(edges) > 0.0):
        raise ValueError(f"{name} edges must be strictly increasing")
    return edges


def _face_resistance(
    interface_resistance_m2K_W: Mapping[str, np.ndarray] | None,
    axis: str,
    shape: tuple[int, ...],
) -> np.ndarray:
    if interface_resistance_m2K_W is None or axis not in interface_resistance_m2K_W:
        return np.zeros(shape, float)
    resistance = np.asarray(interface_resistance_m2K_W[axis], float)
    if resistance.shape != shape:
        raise ValueError(
            f"{axis} interface resistance shape {resistance.shape} != {shape}"
        )
    if not np.all(np.isfinite(resistance)) or np.any(resistance < 0.0):
        raise ValueError(f"{axis} interface resistance must be finite and nonnegative")
    return resistance


def _append_internal_faces(
    *,
    ids: np.ndarray,
    kappa: np.ndarray,
    widths: tuple[np.ndarray, np.ndarray, np.ndarray],
    axis: int,
    resistance: np.ndarray,
    diagonal: np.ndarray,
    rows: list[np.ndarray],
    columns: list[np.ndarray],
    values: list[np.ndarray],
) -> None:
    resistance_per_area = _internal_face_resistance_per_area(
        kappa=kappa,
        widths=widths,
        axis=axis,
        resistance=resistance,
    )
    dx, dy, dz = widths
    if axis == 0:
        left_ids, right_ids = ids[:-1, :, :], ids[1:, :, :]
        area = dy[None, :, None] * dz[None, None, :]
    elif axis == 1:
        left_ids, right_ids = ids[:, :-1, :], ids[:, 1:, :]
        area = dx[:, None, None] * dz[None, None, :]
    else:
        left_ids, right_ids = ids[:, :, :-1], ids[:, :, 1:]
        area = dx[:, None, None] * dy[None, :, None]
    conductance = area / resistance_per_area
    connected = (left_ids >= 0) & (right_ids >= 0)
    left_flat = left_ids[connected].reshape(-1)
    right_flat = right_ids[connected].reshape(-1)
    conductance_flat = np.broadcast_to(
        conductance, left_ids.shape
    )[connected].reshape(-1)
    np.add.at(diagonal, left_flat, conductance_flat)
    np.add.at(diagonal, right_flat, conductance_flat)
    rows.extend((left_flat, right_flat))
    columns.extend((right_flat, left_flat))
    values.extend((-conductance_flat, -conductance_flat))


def _append_periodic_faces(
    *,
    ids: np.ndarray,
    kappa: np.ndarray,
    widths: tuple[np.ndarray, np.ndarray, np.ndarray],
    axis: int,
    resistance: np.ndarray,
    diagonal: np.ndarray,
    rows: list[np.ndarray],
    columns: list[np.ndarray],
    values: list[np.ndarray],
) -> None:
    """Connect the last and first cells across one periodic seam."""
    dx, dy, dz = widths
    if ids.shape[axis] == 1:
        return
    if axis == 0:
        left_ids, right_ids = ids[-1, :, :], ids[0, :, :]
        left_k, right_k = kappa[-1, :, :, 0], kappa[0, :, :, 0]
        area = dy[:, None] * dz[None, :]
        resistance_per_area = (
            0.5 * dx[-1] / left_k
            + resistance
            + 0.5 * dx[0] / right_k
        )
    elif axis == 1:
        left_ids, right_ids = ids[:, -1, :], ids[:, 0, :]
        left_k, right_k = kappa[:, -1, :, 1], kappa[:, 0, :, 1]
        area = dx[:, None] * dz[None, :]
        resistance_per_area = (
            0.5 * dy[-1] / left_k
            + resistance
            + 0.5 * dy[0] / right_k
        )
    else:
        left_ids, right_ids = ids[:, :, -1], ids[:, :, 0]
        left_k, right_k = kappa[:, :, -1, 2], kappa[:, :, 0, 2]
        area = dx[:, None] * dy[None, :]
        resistance_per_area = (
            0.5 * dz[-1] / left_k
            + resistance
            + 0.5 * dz[0] / right_k
        )
    connected = (left_ids >= 0) & (right_ids >= 0)
    left_flat = left_ids[connected].reshape(-1)
    right_flat = right_ids[connected].reshape(-1)
    conductance_flat = (
        area / resistance_per_area
    )[connected].reshape(-1)
    np.add.at(diagonal, left_flat, conductance_flat)
    np.add.at(diagonal, right_flat, conductance_flat)
    rows.extend((left_flat, right_flat))
    columns.extend((right_flat, left_flat))
    values.extend((-conductance_flat, -conductance_flat))


def _internal_face_resistance_per_area(
    *,
    kappa: np.ndarray,
    widths: tuple[np.ndarray, np.ndarray, np.ndarray],
    axis: int,
    resistance: np.ndarray,
) -> np.ndarray:
    """Return the exact two-half-cell plus interface resistance per area."""
    dx, dy, dz = widths
    if axis == 0:
        left_k, right_k = kappa[:-1, :, :, 0], kappa[1:, :, :, 0]
        left_d = dx[:-1, None, None]
        right_d = dx[1:, None, None]
    elif axis == 1:
        left_k, right_k = kappa[:, :-1, :, 1], kappa[:, 1:, :, 1]
        left_d = dy[None, :-1, None]
        right_d = dy[None, 1:, None]
    else:
        left_k, right_k = kappa[:, :, :-1, 2], kappa[:, :, 1:, 2]
        left_d = dz[None, None, :-1]
        right_d = dz[None, None, 1:]
    return (
        0.5 * left_d / left_k + resistance + 0.5 * right_d / right_k
    )


def internal_face_heat_flux_density(
    *,
    temperature_K: np.ndarray,
    x_edges_m: np.ndarray,
    y_edges_m: np.ndarray,
    z_edges_m: np.ndarray,
    kappa_W_mK: np.ndarray,
    interface_resistance_m2K_W: Mapping[str, np.ndarray] | None = None,
    active_mask: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Return signed +axis heat flux density on every internal face.

    The returned arrays use the same face shapes as
    ``interface_resistance_m2K_W``.  Positive values flow from the lower
    indexed cell toward the higher indexed cell.  The calculation uses the
    exact resistance employed by the matrix assembly, so it is also a
    conservative flux diagnostic for heterogeneous materials.
    """
    edges = (
        _validated_edges("x", x_edges_m),
        _validated_edges("y", y_edges_m),
        _validated_edges("z", z_edges_m),
    )
    widths = tuple(np.diff(item) for item in edges)
    shape = tuple(item.size for item in widths)
    temperature = np.asarray(temperature_K, float)
    kappa = np.asarray(kappa_W_mK, float)
    if temperature.shape != shape:
        raise ValueError(f"temperature shape {temperature.shape} != {shape}")
    if kappa.shape != (*shape, 3):
        raise ValueError(f"kappa shape {kappa.shape} != {(*shape, 3)}")
    active = (
        np.ones(shape, bool)
        if active_mask is None
        else np.asarray(active_mask, bool)
    )
    if active.shape != shape:
        raise ValueError(f"active mask shape {active.shape} != {shape}")
    if not np.all(np.isfinite(temperature[active])):
        raise ValueError("active-cell temperature contains NaN or Inf")
    if not np.all(np.isfinite(kappa[active])) or np.any(
        kappa[active] <= 0.0
    ):
        raise ValueError(
            "all active-cell conductivity components must be finite and positive"
        )

    fluxes: dict[str, np.ndarray] = {}
    for axis, axis_name in enumerate(("x", "y", "z")):
        face_shape = list(shape)
        face_shape[axis] -= 1
        resistance = _face_resistance(
            interface_resistance_m2K_W, axis_name, tuple(face_shape)
        )
        resistance_per_area = _internal_face_resistance_per_area(
            kappa=kappa,
            widths=widths,
            axis=axis,
            resistance=resistance,
        )
        left = np.take(temperature, np.arange(shape[axis] - 1), axis=axis)
        right = np.take(temperature, np.arange(1, shape[axis]), axis=axis)
        left_active = np.take(
            active, np.arange(shape[axis] - 1), axis=axis
        )
        right_active = np.take(
            active, np.arange(1, shape[axis]), axis=axis
        )
        connected = left_active & right_active
        flux = np.full(tuple(face_shape), np.nan, float)
        flux[connected] = (
            (left - right) / resistance_per_area
        )[connected]
        fluxes[axis_name] = flux
    return fluxes


def _boundary_geometry(
    face: str,
    ids: np.ndarray,
    kappa: np.ndarray,
    widths: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dx, dy, dz = widths
    if face == "x_min":
        return ids[0, :, :], kappa[0, :, :, 0], dy[:, None] * dz[None, :] / (0.5 * dx[0])
    if face == "x_max":
        return ids[-1, :, :], kappa[-1, :, :, 0], dy[:, None] * dz[None, :] / (0.5 * dx[-1])
    if face == "y_min":
        return ids[:, 0, :], kappa[:, 0, :, 1], dx[:, None] * dz[None, :] / (0.5 * dy[0])
    if face == "y_max":
        return ids[:, -1, :], kappa[:, -1, :, 1], dx[:, None] * dz[None, :] / (0.5 * dy[-1])
    if face == "z_min":
        return ids[:, :, 0], kappa[:, :, 0, 2], dx[:, None] * dy[None, :] / (0.5 * dz[0])
    if face == "z_max":
        return ids[:, :, -1], kappa[:, :, -1, 2], dx[:, None] * dy[None, :] / (0.5 * dz[-1])
    raise ValueError(f"unknown boundary face {face}")


def _boundary_area(
    face: str,
    widths: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    dx, dy, dz = widths
    if face in ("x_min", "x_max"):
        return dy[:, None] * dz[None, :]
    if face in ("y_min", "y_max"):
        return dx[:, None] * dz[None, :]
    if face in ("z_min", "z_max"):
        return dx[:, None] * dy[None, :]
    raise ValueError(f"unknown boundary face {face}")


def _exposed_convection_terms(
    *,
    ids: np.ndarray,
    kappa: np.ndarray,
    widths: tuple[np.ndarray, np.ndarray, np.ndarray],
    dirichlet_faces: set[str],
    excluded_exterior_faces: set[str],
    heat_transfer_W_m2K: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return cell ids and conductances for all solid-to-void surfaces."""
    exposed_ids: list[np.ndarray] = []
    exposed_conductance: list[np.ndarray] = []
    dx, dy, dz = widths
    for axis in range(3):
        if axis == 0:
            left_ids, right_ids = ids[:-1, :, :], ids[1:, :, :]
            left_k, right_k = kappa[:-1, :, :, 0], kappa[1:, :, :, 0]
            left_half_width = 0.5 * dx[:-1, None, None]
            right_half_width = 0.5 * dx[1:, None, None]
            area = dy[None, :, None] * dz[None, None, :]
        elif axis == 1:
            left_ids, right_ids = ids[:, :-1, :], ids[:, 1:, :]
            left_k, right_k = kappa[:, :-1, :, 1], kappa[:, 1:, :, 1]
            left_half_width = 0.5 * dy[None, :-1, None]
            right_half_width = 0.5 * dy[None, 1:, None]
            area = dx[:, None, None] * dz[None, None, :]
        else:
            left_ids, right_ids = ids[:, :, :-1], ids[:, :, 1:]
            left_k, right_k = kappa[:, :, :-1, 2], kappa[:, :, 1:, 2]
            left_half_width = 0.5 * dz[None, None, :-1]
            right_half_width = 0.5 * dz[None, None, 1:]
            area = dx[:, None, None] * dy[None, :, None]
        area = np.broadcast_to(area, left_ids.shape)
        left_exposed = (left_ids >= 0) & (right_ids < 0)
        right_exposed = (right_ids >= 0) & (left_ids < 0)
        for face_ids, face_k, half_width, selected in (
            (left_ids, left_k, left_half_width, left_exposed),
            (right_ids, right_k, right_half_width, right_exposed),
        ):
            half_width = np.broadcast_to(half_width, face_ids.shape)
            conductance = area / (
                half_width / face_k + 1.0 / heat_transfer_W_m2K
            )
            exposed_ids.append(face_ids[selected].reshape(-1))
            exposed_conductance.append(conductance[selected].reshape(-1))

    for face in FACE_NAMES:
        if face in dirichlet_faces or face in excluded_exterior_faces:
            continue
        face_ids, face_kappa, area_over_half_width = _boundary_geometry(
            face, ids, kappa, widths
        )
        area = _boundary_area(face, widths)
        active = face_ids >= 0
        conduction = face_kappa * area_over_half_width
        conductance = 1.0 / (
            1.0 / conduction + 1.0 / (heat_transfer_W_m2K * area)
        )
        exposed_ids.append(face_ids[active].reshape(-1))
        exposed_conductance.append(conductance[active].reshape(-1))
    return np.concatenate(exposed_ids), np.concatenate(exposed_conductance)


def assemble_steady_diagonal_kappa(
    *,
    x_edges_m: np.ndarray,
    y_edges_m: np.ndarray,
    z_edges_m: np.ndarray,
    kappa_W_mK: np.ndarray,
    dirichlet_temperature_K: Mapping[str, float],
    interface_resistance_m2K_W: Mapping[str, np.ndarray] | None = None,
    periodic_interface_resistance_m2K_W: Mapping[str, np.ndarray] | None = None,
    active_mask: np.ndarray | None = None,
    periodic_axes: tuple[str, ...] = (),
    exposed_heat_transfer_W_m2K: float = 0.0,
    ambient_temperature_K: float = 0.0,
    surface_robin_heat_transfer_W_m2K: (
        Mapping[str, float | np.ndarray] | None
    ) = None,
    surface_robin_temperature_K: Mapping[str, float] | None = None,
) -> AssembledThermalSystem:
    """Assemble ``K_T``, ``M_Q`` and ``b_T`` without solving.

    Unspecified exterior and solid-to-inactive faces are adiabatic unless
    ``exposed_heat_transfer_W_m2K`` is positive.  In that case a Robin
    condition is applied through the exact cell-center-to-surface conduction
    resistance plus ``1/h``.  Internal face resistance arrays use per-area
    units of m2 K/W and have shapes ``(nx-1, ny, nz)``, ``(nx, ny-1, nz)``,
    and ``(nx, ny, nz-1)`` for x, y, and z.

    ``surface_robin_heat_transfer_W_m2K`` implements the reduced surface law
    ``q'' = G (T_cell - T_bath)`` directly, so each selected face contributes
    ``A*G`` to the matrix diagonal and ``A*G*T_bath`` to the load.  It does
    not add a half-cell conduction resistance.  This distinct option is used
    only when the governing reduced model defines temperature on the sheet
    and supplies a surface conductance, rather than a bulk exterior domain.
    """
    x_edges = _validated_edges("x", x_edges_m)
    y_edges = _validated_edges("y", y_edges_m)
    z_edges = _validated_edges("z", z_edges_m)
    widths = (np.diff(x_edges), np.diff(y_edges), np.diff(z_edges))
    shape = tuple(item.size for item in widths)
    kappa = np.asarray(kappa_W_mK, float)
    if kappa.shape != (*shape, 3):
        raise ValueError(f"kappa shape {kappa.shape} != {(*shape, 3)}")
    if not np.all(np.isfinite(kappa)) or np.any(kappa <= 0.0):
        raise ValueError("all diagonal conductivity components must be finite and positive")
    invalid_faces = sorted(set(dirichlet_temperature_K) - set(FACE_NAMES))
    if invalid_faces:
        raise ValueError(f"unknown Dirichlet faces: {invalid_faces}")
    robin = (
        {}
        if surface_robin_heat_transfer_W_m2K is None
        else dict(surface_robin_heat_transfer_W_m2K)
    )
    robin_temperatures = (
        {}
        if surface_robin_temperature_K is None
        else dict(surface_robin_temperature_K)
    )
    invalid_robin_faces = sorted(set(robin) - set(FACE_NAMES))
    if invalid_robin_faces:
        raise ValueError(f"unknown surface Robin faces: {invalid_robin_faces}")
    missing_robin_temperatures = sorted(set(robin) - set(robin_temperatures))
    if missing_robin_temperatures:
        raise ValueError(
            "missing surface Robin bath temperatures for "
            f"{missing_robin_temperatures}"
        )
    extra_robin_temperatures = sorted(set(robin_temperatures) - set(robin))
    if extra_robin_temperatures:
        raise ValueError(
            "surface Robin temperatures without conductance for "
            f"{extra_robin_temperatures}"
        )
    conflicting_robin_faces = sorted(set(robin) & set(dirichlet_temperature_K))
    if conflicting_robin_faces:
        raise ValueError(
            "faces cannot be both Dirichlet and surface Robin: "
            f"{conflicting_robin_faces}"
        )
    periodic = tuple(dict.fromkeys(str(axis) for axis in periodic_axes))
    invalid_periodic = sorted(set(periodic) - {"x", "y", "z"})
    if invalid_periodic:
        raise ValueError(f"unknown periodic axes: {invalid_periodic}")
    for axis in periodic:
        conflicting = {
            f"{axis}_min", f"{axis}_max"
        } & set(dirichlet_temperature_K)
        if conflicting:
            raise ValueError(
                f"periodic axis {axis} conflicts with Dirichlet faces "
                f"{sorted(conflicting)}"
            )
    exposed_heat_transfer_W_m2K = float(exposed_heat_transfer_W_m2K)
    ambient_temperature_K = float(ambient_temperature_K)
    if (
        not np.isfinite(exposed_heat_transfer_W_m2K)
        or exposed_heat_transfer_W_m2K < 0.0
    ):
        raise ValueError("exposed heat-transfer coefficient must be nonnegative")
    if not np.isfinite(ambient_temperature_K):
        raise ValueError("ambient temperature must be finite")
    if (
        not dirichlet_temperature_K
        and not robin
        and exposed_heat_transfer_W_m2K == 0.0
    ):
        raise ValueError(
            "at least one Dirichlet, surface Robin, or exposed-convection "
            "boundary is required"
        )

    active = (
        np.ones(shape, bool)
        if active_mask is None
        else np.asarray(active_mask, bool)
    )
    if active.shape != shape:
        raise ValueError(f"active mask shape {active.shape} != {shape}")
    if not np.any(active):
        raise ValueError("active mask contains no solid cells")
    ids = np.full(shape, -1, dtype=np.int64)
    ids[active] = np.arange(np.count_nonzero(active), dtype=np.int64)
    diagonal = np.zeros(np.count_nonzero(active), float)
    rows: list[np.ndarray] = []
    columns: list[np.ndarray] = []
    values: list[np.ndarray] = []
    for axis, axis_name in enumerate(("x", "y", "z")):
        face_shape = list(shape)
        face_shape[axis] -= 1
        resistance = _face_resistance(
            interface_resistance_m2K_W, axis_name, tuple(face_shape)
        )
        _append_internal_faces(
            ids=ids,
            kappa=kappa,
            widths=widths,
            axis=axis,
            resistance=resistance,
            diagonal=diagonal,
            rows=rows,
            columns=columns,
            values=values,
        )
        if axis_name in periodic:
            if axis == 0:
                seam_shape = (shape[1], shape[2])
            elif axis == 1:
                seam_shape = (shape[0], shape[2])
            else:
                seam_shape = (shape[0], shape[1])
            seam_resistance = _face_resistance(
                periodic_interface_resistance_m2K_W,
                axis_name,
                seam_shape,
            )
            _append_periodic_faces(
                ids=ids,
                kappa=kappa,
                widths=widths,
                axis=axis,
                resistance=seam_resistance,
                diagonal=diagonal,
                rows=rows,
                columns=columns,
                values=values,
            )

    dx, dy, dz = widths
    volume = dx[:, None, None] * dy[None, :, None] * dz[None, None, :]
    boundary_load = np.zeros(np.count_nonzero(active), float)
    boundary_terms: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}
    for face, temperature in dirichlet_temperature_K.items():
        temperature = float(temperature)
        if not np.isfinite(temperature):
            raise ValueError(f"{face} temperature must be finite")
        face_ids, face_kappa, area_over_half_width = _boundary_geometry(
            face, ids, kappa, widths
        )
        conductance = face_kappa * area_over_half_width
        boundary_active = face_ids >= 0
        flat_ids = face_ids[boundary_active].reshape(-1)
        flat_conductance = np.broadcast_to(
            conductance, face_ids.shape
        )[boundary_active].reshape(-1)
        np.add.at(diagonal, flat_ids, flat_conductance)
        np.add.at(
            boundary_load, flat_ids, flat_conductance * temperature
        )
        boundary_terms[face] = (flat_ids, flat_conductance, temperature)
    for face, heat_transfer in robin.items():
        temperature = float(robin_temperatures[face])
        if not np.isfinite(temperature):
            raise ValueError(
                f"{face} surface Robin temperature must be finite"
            )
        face_ids, _, _ = _boundary_geometry(face, ids, kappa, widths)
        area = np.broadcast_to(
            _boundary_area(face, widths), face_ids.shape
        )
        heat_transfer_array = np.asarray(heat_transfer, float)
        if heat_transfer_array.ndim == 0:
            heat_transfer_array = np.full(
                face_ids.shape, float(heat_transfer_array)
            )
        elif heat_transfer_array.shape != face_ids.shape:
            raise ValueError(
                f"{face} surface Robin G shape "
                f"{heat_transfer_array.shape} != {face_ids.shape}"
            )
        if (
            not np.all(np.isfinite(heat_transfer_array))
            or np.any(heat_transfer_array < 0.0)
        ):
            raise ValueError(
                f"{face} surface Robin G must be finite and nonnegative"
            )
        boundary_active = face_ids >= 0
        flat_ids = face_ids[boundary_active].reshape(-1)
        flat_conductance = (
            area * heat_transfer_array
        )[boundary_active].reshape(-1)
        np.add.at(diagonal, flat_ids, flat_conductance)
        np.add.at(
            boundary_load, flat_ids, flat_conductance * temperature
        )
        boundary_terms[f"surface_robin_{face}"] = (
            flat_ids,
            flat_conductance,
            temperature,
        )
    if exposed_heat_transfer_W_m2K > 0.0:
        convection_ids, convection_conductance = _exposed_convection_terms(
            ids=ids,
            kappa=kappa,
            widths=widths,
            dirichlet_faces=set(dirichlet_temperature_K),
            excluded_exterior_faces={
                face
                for axis in periodic
                for face in (f"{axis}_min", f"{axis}_max")
            },
            heat_transfer_W_m2K=exposed_heat_transfer_W_m2K,
        )
        np.add.at(diagonal, convection_ids, convection_conductance)
        np.add.at(
            boundary_load,
            convection_ids,
            convection_conductance * ambient_temperature_K,
        )
        boundary_terms["exposed_convection"] = (
            convection_ids,
            convection_conductance,
            ambient_temperature_K,
        )
    if np.any(diagonal <= 0.0):
        raise ValueError(
            "one or more active cells have zero total conductance"
        )
    all_ids = np.arange(diagonal.size, dtype=np.int64)
    rows.append(all_ids)
    columns.append(all_ids)
    values.append(diagonal)
    matrix = sparse.coo_matrix(
        (
            np.concatenate(values),
            (np.concatenate(rows), np.concatenate(columns)),
        ),
        shape=(diagonal.size, diagonal.size),
    ).tocsr()
    source_volume_operator = sparse.diags(
        volume[active].reshape(-1), format="csr"
    )
    return AssembledThermalSystem(
        matrix_W_K=matrix,
        source_volume_operator_m3=source_volume_operator,
        boundary_load_W=boundary_load,
        active_mask=active,
        active_ids=ids,
        cell_volume_m3=volume,
        diagonal_W_K=diagonal,
        boundary_terms=boundary_terms,
        shape=shape,
        periodic_axes=periodic,
    )


def solve_assembled_thermal_system(
    system: AssembledThermalSystem,
    *,
    source_W_m3: np.ndarray | None = None,
    initial_temperature_K: np.ndarray | None = None,
    relative_tolerance: float = 1.0e-11,
    max_iterations: int = 5000,
) -> SteadyHeatResult:
    """Solve one already assembled forward system.

    The right-hand side is exactly ``M_Q @ Q + b_T``. This entry point is also
    used by the AD/FD certificate so every perturbed source shares one fixed
    thermal operator.
    """
    source_active = system.active_source(source_W_m3)
    source_power_active = (
        system.source_volume_operator_m3 @ source_active
    )
    rhs = source_power_active + system.boundary_load_W
    matrix = system.matrix_W_K
    preconditioner = sparse_linalg.LinearOperator(
        matrix.shape,
        matvec=lambda vector: vector / system.diagonal_W_K,
    )
    iterations = 0

    def count_iteration(_: np.ndarray) -> None:
        nonlocal iterations
        iterations += 1

    initial_active = None
    if initial_temperature_K is not None:
        initial = np.asarray(initial_temperature_K, float)
        if initial.shape != system.shape:
            raise ValueError(
                f"initial temperature shape {initial.shape} != {system.shape}"
            )
        initial_active = initial[system.active_mask].reshape(-1)
        if not np.all(np.isfinite(initial_active)):
            raise ValueError("initial active temperature contains NaN or Inf")

    solution, info = sparse_linalg.cg(
        matrix,
        rhs,
        x0=initial_active,
        rtol=relative_tolerance,
        atol=0.0,
        maxiter=max_iterations,
        M=preconditioner,
        callback=count_iteration,
    )
    solver_name = "scipy.sparse.linalg.cg+jacobi"
    if initial_active is not None:
        solver_name += "+warm_start"
    if info != 0:
        solution = sparse_linalg.spsolve(matrix, rhs)
        solver_name = f"scipy.sparse.linalg.spsolve_after_cg_info_{info}"
    residual = matrix @ solution - rhs
    residual_relative = float(
        np.linalg.norm(residual)
        / max(np.linalg.norm(rhs), np.finfo(float).tiny)
    )
    boundary_power = {
        face: float(
            np.sum(conductance * (solution[face_ids] - temperature))
        )
        for face, (face_ids, conductance, temperature)
        in system.boundary_terms.items()
    }
    source_power = float(np.sum(source_power_active))
    energy_error = abs(sum(boundary_power.values()) - source_power) / max(
        abs(source_power),
        max((abs(value) for value in boundary_power.values()), default=0.0),
        np.finfo(float).tiny,
    )
    return SteadyHeatResult(
        temperature_K=system.full_field(solution),
        boundary_power_out_W=boundary_power,
        source_power_W=source_power,
        energy_balance_relative_error=float(energy_error),
        linear_residual_relative=residual_relative,
        solver=solver_name,
        iterations=iterations,
    )


def solve_steady_diagonal_kappa(
    *,
    x_edges_m: np.ndarray,
    y_edges_m: np.ndarray,
    z_edges_m: np.ndarray,
    kappa_W_mK: np.ndarray,
    source_W_m3: np.ndarray | None = None,
    dirichlet_temperature_K: Mapping[str, float],
    interface_resistance_m2K_W: Mapping[str, np.ndarray] | None = None,
    periodic_interface_resistance_m2K_W: Mapping[str, np.ndarray] | None = None,
    active_mask: np.ndarray | None = None,
    periodic_axes: tuple[str, ...] = (),
    exposed_heat_transfer_W_m2K: float = 0.0,
    ambient_temperature_K: float = 0.0,
    surface_robin_heat_transfer_W_m2K: (
        Mapping[str, float | np.ndarray] | None
    ) = None,
    surface_robin_temperature_K: Mapping[str, float] | None = None,
    relative_tolerance: float = 1.0e-11,
    max_iterations: int = 5000,
) -> SteadyHeatResult:
    """Assemble and solve ``-div(K grad T) = Q``.

    This compatibility entry point preserves the validated forward API while
    delegating to the exposed operator used by the thermal adjoint.
    """
    system = assemble_steady_diagonal_kappa(
        x_edges_m=x_edges_m,
        y_edges_m=y_edges_m,
        z_edges_m=z_edges_m,
        kappa_W_mK=kappa_W_mK,
        dirichlet_temperature_K=dirichlet_temperature_K,
        interface_resistance_m2K_W=interface_resistance_m2K_W,
        periodic_interface_resistance_m2K_W=(
            periodic_interface_resistance_m2K_W
        ),
        active_mask=active_mask,
        periodic_axes=periodic_axes,
        exposed_heat_transfer_W_m2K=exposed_heat_transfer_W_m2K,
        ambient_temperature_K=ambient_temperature_K,
        surface_robin_heat_transfer_W_m2K=(
            surface_robin_heat_transfer_W_m2K
        ),
        surface_robin_temperature_K=surface_robin_temperature_K,
    )
    return solve_assembled_thermal_system(
        system,
        source_W_m3=source_W_m3,
        relative_tolerance=relative_tolerance,
        max_iterations=max_iterations,
    )
