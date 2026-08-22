#!/usr/bin/env python3
"""Geometry contracts for the paper-derived TaIrTe4 T/Z controls.

Only the active two-dimensional material is replaced by the project's fixed
100-nm TaIrTe4 closure.  Published numbers, figure-digitized numbers, and
numerical closures are deliberately stored as different provenance classes.
The module contains no solver calls.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from contracts import Z_PUBLISHED_DIMENSIONS_NM


@dataclass(frozen=True)
class PolygonObject:
    name: str
    material: str
    vertices_nm: tuple[tuple[float, float], ...]
    z_min_nm: float
    z_max_nm: float
    provenance_kind: str
    provenance: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MetasurfaceGeometry:
    key: str
    wavelength_nm: float
    period_x_nm: float
    period_y_nm: float
    active_material: str
    active_thickness_nm: float
    axis_mapping: dict[str, str]
    polygons: tuple[PolygonObject, ...]
    layers: tuple[dict[str, Any], ...]
    boundary_contract: dict[str, str]
    unresolved: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def inverse_t_mir_4750nm() -> MetasurfaceGeometry:
    """Return the 2024 MIR inverse-T control.

    Supplementary Fig. 14 discloses a 1500 x 1000 nm unit cell and plots the
    resonator against physical axes.  It does not provide a numeric arm table.
    The vertices below are digitized from those axes: baseline about 1200 x
    100 nm, stem about 200 x 600 nm.  These are not promoted to published CAD.
    """

    baseline_length = 1200.0
    baseline_width = 100.0
    stem_width = 200.0
    stem_length = 600.0
    y_baseline_center = -300.0
    y_baseline_top = y_baseline_center + 0.5 * baseline_width
    y_stem_top = y_baseline_top + stem_length
    vertices = (
        (-0.5 * baseline_length, y_baseline_center - 0.5 * baseline_width),
        (0.5 * baseline_length, y_baseline_center - 0.5 * baseline_width),
        (0.5 * baseline_length, y_baseline_top),
        (0.5 * stem_width, y_baseline_top),
        (0.5 * stem_width, y_stem_top),
        (-0.5 * stem_width, y_stem_top),
        (-0.5 * stem_width, y_baseline_top),
        (-0.5 * baseline_length, y_baseline_top),
    )
    return MetasurfaceGeometry(
        key="T2024_MIR_4750_FIGURE_DIGITIZED_TAIRTE4",
        wavelength_nm=4750.0,
        period_x_nm=1500.0,
        period_y_nm=1000.0,
        active_material="TaIrTe4",
        active_thickness_nm=100.0,
        axis_mapping={"x": "b", "y": "a", "z": "c=b closure"},
        polygons=(
            PolygonObject(
                name="inverse_T_effective_Au",
                material="Au",
                vertices_nm=vertices,
                z_min_nm=100.0,
                z_max_nm=133.0,
                provenance_kind="figure_digitized_geometry+paper_thickness",
                provenance=(
                    "2024 Supplementary Fig. 14 axes; 33-nm MIR resonator "
                    "thickness stated in Supplementary Note 7 scenario"
                ),
            ),
        ),
        layers=(
            {
                "name": "air",
                "z_min_nm": 133.0,
                "z_max_nm": None,
                "provenance_kind": "paper",
            },
            {
                "name": "TaIrTe4_active_substitution",
                "z_min_nm": 0.0,
                "z_max_nm": 100.0,
                "provenance_kind": "project_active_material_substitution",
            },
            {
                "name": "Al2O3_spacer",
                "z_min_nm": -35.0,
                "z_max_nm": 0.0,
                "provenance_kind": "paper_fabricated_value",
            },
            {
                "name": "Au_backplane",
                "z_min_nm": -235.0,
                "z_max_nm": -35.0,
                "provenance_kind": "200nm_numerical_opaque_closure",
            },
        ),
        boundary_contract={
            "x_min_x_max": "Periodic",
            "y_min_y_max": "Periodic",
            "z_min_z_max": "PML",
            "illumination": "normal-incidence plane wave from +z",
        },
        unresolved=(
            "arm numbers are digitized from Fig. S14, not published CAD",
            "Ti adhesion layer omitted from first optical smoke",
            "100-nm TaIrTe4 is a deliberate substitution for monolayer graphene",
            "the paper MIR control omitted top passivation",
        ),
    )


def z_m5_8um_geometry_topology_audit() -> MetasurfaceGeometry:
    """Return the 2022 M5 dimensional contract without inventing Z vertices.

    Table S1 gives every scalar dimension, but neither the article nor SI
    supplies machine-readable polygon vertices or a fixed crossing angle.  A
    fake Z polygon would therefore be less faithful than a fail-closed audit.
    The two bounding rectangles visualize the disclosed L/W values only and
    MUST NOT be sent to Maxwell as the paper Z antenna.
    """

    m5 = dict(Z_PUBLISHED_DIMENSIONS_NM[-1])
    diagnostic = (
        PolygonObject(
            name="L1_W1_dimension_envelope_NOT_CAD",
            material="Au",
            vertices_nm=(
                (-0.5 * m5["W1_nm"], 0.0),
                (0.5 * m5["W1_nm"], 0.0),
                (0.5 * m5["W1_nm"], m5["L1_nm"]),
                (-0.5 * m5["W1_nm"], m5["L1_nm"]),
            ),
            z_min_nm=100.0,
            z_max_nm=155.0,
            provenance_kind="dimension_envelope_only",
            provenance="2022 Supplementary Table 1; not a Z polygon",
        ),
        PolygonObject(
            name="L2_W2_dimension_envelope_NOT_CAD",
            material="Au",
            vertices_nm=(
                (-0.5 * m5["W2_nm"], -m5["L2_nm"]),
                (0.5 * m5["W2_nm"], -m5["L2_nm"]),
                (0.5 * m5["W2_nm"], 0.0),
                (-0.5 * m5["W2_nm"], 0.0),
            ),
            z_min_nm=100.0,
            z_max_nm=155.0,
            provenance_kind="dimension_envelope_only",
            provenance="2022 Supplementary Table 1; not a Z polygon",
        ),
    )
    return MetasurfaceGeometry(
        key="Z2022_M5_8UM_PUBLISHED_DIMENSIONS_TOPOLOGY_BLOCKED",
        wavelength_nm=m5["wavelength_nm"],
        period_x_nm=m5["P1_nm"],
        period_y_nm=m5["P2_nm"],
        active_material="TaIrTe4",
        active_thickness_nm=100.0,
        axis_mapping={"x": "b", "y": "a", "z": "c=b closure"},
        polygons=diagnostic,
        layers=(
            {"name": "TaIrTe4_active_substitution", "thickness_nm": 100.0},
            {"name": "Cr_Au_Z", "thickness_nm": 55.0},
            {"name": "Al2O3_spacer", "thickness_nm": m5["Al2O3_D_nm"]},
            {"name": "Au_backplane", "thickness_nm": 200.0},
            {"name": "thermal_SiO2", "thickness_nm": 285.0},
            {"name": "p_doped_Si", "thickness_nm": None},
        ),
        boundary_contract={
            "x_min_x_max": "Periodic",
            "y_min_y_max": "Periodic",
            "z_min_z_max": "PML",
            "illumination": "normal-incidence plane wave",
        },
        unresolved=(
            "exact Z polygon vertices/crossing angle are not disclosed",
            "TaIrTe4 conformal-versus-bridged topography is unresolved",
            "dimension envelopes are forbidden as production Maxwell geometry",
        ),
    )


def _z_m2_geometry(
    handedness: str,
    *,
    width_along_x: bool,
) -> MetasurfaceGeometry:
    if handedness not in ("LH", "RH"):
        raise ValueError(handedness)
    m2 = dict(Z_PUBLISHED_DIMENSIONS_NM[1])
    l1, l2 = float(m2["L1_nm"]), float(m2["L2_nm"])
    w1, w2 = float(m2["W1_nm"]), float(m2["W2_nm"])
    x1, x2 = (w1, w2) if width_along_x else (l1, l2)
    y1, y2 = (l1, l2) if width_along_x else (w1, w2)
    join_x = -0.5 * (x1 - x2)
    join_y = -0.5 * (y1 - y2)
    upper = np.asarray(
        [(join_x, join_y), (join_x + x1, join_y),
         (join_x + x1, join_y + y1), (join_x, join_y + y1)],
        float,
    )
    lower = np.asarray(
        [(join_x - x2, join_y - y2), (join_x, join_y - y2),
         (join_x, join_y), (join_x - x2, join_y)],
        float,
    )
    if handedness == "RH":
        upper[:, 0] *= -1.0
        lower[:, 0] *= -1.0
        upper = upper[::-1]
        lower = lower[::-1]

    version = "FIGURE_AXIS_CORRECTED_V2" if width_along_x else "LEGACY_AXIS_SWAPPED_V1"
    provenance_kind = (
        "published_dimensions+figure_axis_corrected_corner_join"
        if width_along_x
        else "published_dimensions+legacy_axis_swapped_corner_join"
    )
    axis_statement = (
        "W1/W2 horizontal and L1/L2 vertical per Fig. 1b"
        if width_along_x
        else "legacy diagnostic with L1/L2 horizontal and W1/W2 vertical"
    )
    provenance = (
        f"2022 Fig. 1b and Supplementary Table 1 M2 dimensions; {axis_statement}; "
        "inner corners joined because junction overlap/gap CAD is not disclosed"
    )
    polygons = tuple(
        PolygonObject(
            name=f"Z2022_M2_{version}_{handedness}_{label}",
            material="Au",
            vertices_nm=tuple(tuple(float(item) for item in row) for row in vertices),
            z_min_nm=100.0,
            z_max_nm=150.0,
            provenance_kind=provenance_kind,
            provenance=provenance,
        )
        for label, vertices in (("upper", upper), ("lower", lower))
    )
    # Supplementary Fig. 4 shows the long antenna direction on the long-pitch
    # lattice axis.  Consequently the Fig. 1b-corrected top view has P2 along
    # horizontal W and P1 along vertical L.  Preserve the old assignment only
    # for exact reproduction of the already-published v1 diagnostic.
    period_x_nm = float(m2["P2_nm"] if width_along_x else m2["P1_nm"])
    period_y_nm = float(m2["P1_nm"] if width_along_x else m2["P2_nm"])
    return MetasurfaceGeometry(
        key=f"Z2022_M2_5300_{handedness}_{version}",
        wavelength_nm=float(m2["wavelength_nm"]),
        period_x_nm=period_x_nm,
        period_y_nm=period_y_nm,
        active_material="TaIrTe4",
        active_thickness_nm=100.0,
        axis_mapping={"x": "b", "y": "a", "z": "c=b closure"},
        polygons=polygons,
        layers=(
            {"name": "air", "z_min_nm": 150.0, "z_max_nm": None},
            {"name": "effective_Au_Z", "z_min_nm": 100.0, "z_max_nm": 150.0},
            {"name": "TaIrTe4_active_substitution", "z_min_nm": 0.0, "z_max_nm": 100.0},
            {"name": "Al2O3_spacer", "z_min_nm": -200.0, "z_max_nm": 0.0},
            {"name": "Au_backplane", "z_min_nm": -400.0, "z_max_nm": -200.0},
            {"name": "optical_SiO2_reduced", "z_min_nm": -685.0, "z_max_nm": -400.0},
            {"name": "Si", "z_min_nm": None, "z_max_nm": -685.0},
        ),
        boundary_contract={
            "x_min_x_max": "Periodic", "y_min_y_max": "Periodic",
            "z_min_z_max": "PML", "illumination": "normal-incidence plane wave",
        },
        unresolved=(
            "junction overlap/gap is not disclosed; corner-joined figure-digitized closure retained",
            "100-nm TaIrTe4 replaces the paper's original 2-D thermoelectric material",
            "Au-above-planar-TaIrTe4 is a project endpoint, not author device CAD",
            "5-nm Cr adhesion is omitted from the first effective-Au screen",
        ),
    )


def z_m2_5300nm_corner_joined_tairte4(handedness: str = "LH") -> MetasurfaceGeometry:
    """Return the preserved, axis-swapped v1 diagnostic geometry."""
    return _z_m2_geometry(handedness, width_along_x=False)


def z_m2_5300nm_figure_corrected_tairte4_v2(
    handedness: str = "LH",
) -> MetasurfaceGeometry:
    """Return M2 with W along x and L along y, as labelled in Fig. 1b."""
    return _z_m2_geometry(handedness, width_along_x=True)


def z_m2_5300nm_figure_period_corrected_tairte4_v3(
    handedness: str = "LH",
) -> MetasurfaceGeometry:
    """Return the Fig. 1b/Table-S1 M2 reconstruction with corrected periods.

    Fig. 1b labels P1 horizontally and P2 vertically.  The scalar table does
    not disclose the relative bar offset, so the two rectangles are joined at
    x=0 and offset in y to span exactly P2.  This is a figure-constrained
    reconstruction, not the undisclosed author CAD.
    """
    if handedness not in ("LH", "RH"):
        raise ValueError(handedness)
    m2 = dict(Z_PUBLISHED_DIMENSIONS_NM[1])
    p1, p2 = float(m2["P1_nm"]), float(m2["P2_nm"])
    l1, l2 = float(m2["L1_nm"]), float(m2["L2_nm"])
    w1, w2 = float(m2["W1_nm"]), float(m2["W2_nm"])

    # Figure-constrained LH closure: long/right bar reaches the upper cell
    # edge; short/left bar reaches the lower edge.  Their y overlap is fixed
    # by L1 + L2 - P2.  The shared x edge makes one continuous Z-like metal.
    upper = np.asarray(
        [(0.0, 0.5 * p2 - l1), (w1, 0.5 * p2 - l1),
         (w1, 0.5 * p2), (0.0, 0.5 * p2)],
        float,
    )
    lower = np.asarray(
        [(-w2, -0.5 * p2), (0.0, -0.5 * p2),
         (0.0, -0.5 * p2 + l2), (-w2, -0.5 * p2 + l2)],
        float,
    )
    if handedness == "RH":
        upper[:, 0] *= -1.0
        lower[:, 0] *= -1.0
        upper = upper[::-1]
        lower = lower[::-1]

    provenance = (
        "2022 Fig. 1b and Supplementary Table 1 M2: P1 horizontal, P2 vertical, "
        "W1/W2 horizontal, L1/L2 vertical; edge-joined y-offset closure digitized "
        "from the figure because exact author CAD/relative offset is not disclosed"
    )
    polygons = tuple(
        PolygonObject(
            name=f"Z2022_M2_FIGURE_PERIOD_CORRECTED_V3_{handedness}_{label}",
            material="Au",
            vertices_nm=tuple(tuple(float(item) for item in row) for row in vertices),
            z_min_nm=100.0,
            z_max_nm=150.0,
            provenance_kind="published_dimensions+figure_period_corrected_edge_joined_v3",
            provenance=provenance,
        )
        for label, vertices in (("upper_right", upper), ("lower_left", lower))
    )
    return MetasurfaceGeometry(
        key=f"Z2022_M2_5300_{handedness}_FIGURE_PERIOD_CORRECTED_V3",
        wavelength_nm=float(m2["wavelength_nm"]),
        period_x_nm=p1,
        period_y_nm=p2,
        active_material="TaIrTe4",
        active_thickness_nm=100.0,
        axis_mapping={"x": "b", "y": "a", "z": "c=b closure"},
        polygons=polygons,
        layers=(
            {"name": "air", "z_min_nm": 150.0, "z_max_nm": None},
            {"name": "effective_Au_Z", "z_min_nm": 100.0, "z_max_nm": 150.0},
            {"name": "TaIrTe4_active_substitution", "z_min_nm": 0.0, "z_max_nm": 100.0},
            {"name": "Al2O3_spacer", "z_min_nm": -200.0, "z_max_nm": 0.0},
            {"name": "Au_backplane", "z_min_nm": -400.0, "z_max_nm": -200.0},
            {"name": "optical_SiO2_reduced", "z_min_nm": -685.0, "z_max_nm": -400.0},
            {"name": "Si", "z_min_nm": None, "z_max_nm": -685.0},
        ),
        boundary_contract={
            "x_min_x_max": "Periodic", "y_min_y_max": "Periodic",
            "z_min_z_max": "PML", "illumination": "normal-incidence plane wave",
        },
        unresolved=(
            "exact author CAD and relative two-bar offset are not disclosed",
            "edge-joined figure-constrained closure is used and explicitly audited",
            "100-nm TaIrTe4 replaces the paper's original 2-D thermoelectric material",
            "5-nm Cr adhesion is omitted from the first effective-Au screen",
        ),
    )


def signed_polygon_area_nm2(vertices: tuple[tuple[float, float], ...]) -> float:
    array = np.asarray(vertices, float)
    return 0.5 * float(
        np.sum(array[:, 0] * np.roll(array[:, 1], -1))
        - np.sum(array[:, 1] * np.roll(array[:, 0], -1))
    )
