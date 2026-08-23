"""Deterministic optical z-mesh variants with independent vertical PML cells."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import hashlib
from typing import Iterator

import numpy as np

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch import (
    fdtdx_4um_model as optical_model,
)


PARTIAL_MATERIAL_Z = "partial_material_z"
FULL_DOMAIN_Z = "full_domain_z"
Z_SEGMENTS = (
    ("bottom_pml_si", -3.000e-6, -1.400e-6, 8),
    ("resolved_si", -1.400e-6, -0.385e-6, 5),
    ("sio2", -0.385e-6, -0.100e-6, 3),
    ("tairte4", -0.100e-6, 0.000e-6, 5),
    ("au", 0.000e-6, 0.050e-6, 2),
    ("near_air", 0.050e-6, 0.250e-6, 4),
    ("middle_air", 0.250e-6, 0.750e-6, 2),
    ("source_air", 0.750e-6, 1.400e-6, 3),
    ("top_pml_air", 1.400e-6, 3.000e-6, 8),
)
MATERIAL_SEGMENTS = {"sio2", "tairte4", "au"}


def _segments(parts: tuple[tuple[float, float, int], ...]) -> np.ndarray:
    values = []
    for index, (start, stop, cells) in enumerate(parts):
        if cells <= 0 or not stop > start:
            raise ValueError((start, stop, cells))
        segment = np.linspace(start, stop, cells + 1, dtype=np.float64)
        values.append(segment if index == 0 else segment[1:])
    result = np.concatenate(values)
    if np.any(np.diff(result) <= 0.0):
        raise RuntimeError("non-monotone z-mesh variant")
    return result


def segment_cell_counts(factor: int, mode: str) -> dict[str, int]:
    if int(factor) != factor or factor < 1:
        raise ValueError("mesh factor must be a positive integer")
    if mode not in (PARTIAL_MATERIAL_Z, FULL_DOMAIN_Z):
        raise ValueError(f"unknown z-mesh mode {mode!r}")
    return {
        name: base * factor
        if mode == FULL_DOMAIN_Z or name in MATERIAL_SEGMENTS
        else base
        for name, _, _, base in Z_SEGMENTS
    }


def variant_edges(
    factor: int, mode: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    baseline_x, baseline_y, _ = optical_model.grid_edges()
    counts = segment_cell_counts(factor, mode)
    z = _segments(
        tuple((start, stop, counts[name]) for name, start, stop, _ in Z_SEGMENTS)
    )
    return baseline_x.copy(), baseline_y.copy(), z


def _edge_index(edges: np.ndarray, coordinate_m: float) -> int:
    matches = np.flatnonzero(
        np.isclose(edges, coordinate_m, rtol=0.0, atol=2.0e-18)
    )
    if matches.size != 1:
        raise RuntimeError(f"required z edge {coordinate_m:.9e} is absent")
    return int(matches[0])


def variant_layout(factor: int, mode: str):
    counts = segment_cell_counts(factor, mode)
    z = variant_edges(factor, mode)[2]
    source = _edge_index(z, 0.750e-6)
    target = _edge_index(z, 0.250e-6)
    incident = _edge_index(z, 0.500e-6)
    baseline_closed_coordinate = optical_model.grid_edges()[2][
        optical_model.LAYOUT.closed_z_start
    ]
    closed_start = _edge_index(z, float(baseline_closed_coordinate))
    return replace(
        optical_model.LAYOUT,
        pml_cells_z=counts["bottom_pml_si"],
        silicon_cells=counts["bottom_pml_si"] + counts["resolved_si"],
        sio2_cells=counts["sio2"],
        tairte4_cells=counts["tairte4"],
        au_cells=counts["au"],
        source_z_start=source,
        target_z_start=target,
        incident_z_start=incident,
        closed_z_start=closed_start,
        closed_z_cells=target - closed_start,
    )


def edges_sha256(edges: tuple[np.ndarray, np.ndarray, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for axis, values in zip("xyz", edges, strict=True):
        array = np.ascontiguousarray(values, dtype=np.float64)
        digest.update(axis.encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def variant_audit(factor: int, mode: str) -> dict[str, object]:
    edges = variant_edges(factor, mode)
    layout = variant_layout(factor, mode)
    counts = segment_cell_counts(factor, mode)
    segment_audit = {}
    for name, start, stop, _ in Z_SEGMENTS:
        count = counts[name]
        segment_audit[name] = {
            "bounds_m": [start, stop],
            "cells": count,
            "uniform_dz_m": (stop - start) / count,
        }
    return {
        "mode": mode,
        "factor": factor,
        "grid_edges_sha256": edges_sha256(edges),
        "grid_shape_xyz": [
            len(edges[0]) - 1,
            len(edges[1]) - 1,
            len(edges[2]) - 1,
        ],
        "yee_cell_count": int(
            (len(edges[0]) - 1)
            * (len(edges[1]) - 1)
            * (len(edges[2]) - 1)
        ),
        "pml_cells_each_face_xyz": [
            layout.pml_cells_xy,
            layout.pml_cells_xy,
            layout.pml_cells_z,
        ],
        "indices": {
            "source_z_start": layout.source_z_start,
            "target_z_start": layout.target_z_start,
            "incident_z_start": layout.incident_z_start,
            "closed_z_start": layout.closed_z_start,
            "closed_z_cells": layout.closed_z_cells,
        },
        "segments": segment_audit,
    }


@contextmanager
def mesh_context(factor: int, mode: str) -> Iterator[object]:
    """Temporarily install a variant for one single-threaded model build."""

    original_layout = optical_model.LAYOUT
    original_edges = optical_model.grid_edges
    layout = variant_layout(factor, mode)
    edges = variant_edges(factor, mode)
    optical_model.LAYOUT = layout
    optical_model.grid_edges = lambda: tuple(value.copy() for value in edges)
    try:
        yield layout
    finally:
        optical_model.LAYOUT = original_layout
        optical_model.grid_edges = original_edges
