"""Contracts for the non-periodic large-TaIrTe4 TFSF certificate.

The central 2 um square is the immutable design/PTE ROI.  The optical
background layers are deliberately larger than the FDTD region so Lumerical
extends the same vertical stack through every lateral PML cell.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np


UM = 1.0e-6
NM = 1.0e-9


@dataclass(frozen=True)
class Bounds3D:
    x: tuple[float, float]
    y: tuple[float, float]
    z: tuple[float, float]

    def validate(self, label: str) -> None:
        for axis in "xyz":
            lo, hi = getattr(self, axis)
            if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
                raise ValueError(f"invalid {label} {axis} bounds: {(lo, hi)}")

    def contains(self, other: "Bounds3D", *, strict: bool = False) -> bool:
        for axis in "xyz":
            lo, hi = getattr(self, axis)
            other_lo, other_hi = getattr(other, axis)
            if strict:
                if not lo < other_lo or not other_hi < hi:
                    return False
            elif not lo <= other_lo or not other_hi <= hi:
                return False
        return True

    def gaps_to(self, inner: "Bounds3D") -> dict[str, dict[str, float]]:
        if not self.contains(inner):
            raise ValueError("outer bounds do not contain inner bounds")
        return {
            axis: {
                "min_m": getattr(inner, axis)[0] - getattr(self, axis)[0],
                "max_m": getattr(self, axis)[1] - getattr(inner, axis)[1],
            }
            for axis in "xyz"
        }

    def to_dict(self) -> dict[str, list[float]]:
        return {
            axis: [float(value) for value in getattr(self, axis)]
            for axis in "xyz"
        }


@dataclass(frozen=True)
class LargeBackgroundOpticalContract:
    """One fully explicit numerical geometry.

    ``fdtd_outer`` is the FDTD object's user-specified outer boundary.  The
    PML cells lie inside those bounds.  Their realized inner interfaces must
    be reconstructed from the generated mesh and the per-face layer count.
    """

    fdtd_outer: Bounds3D
    tfsf: Bounds3D
    design: Bounds3D
    omega_q: Bounds3D
    flake_z: tuple[float, float] = (-100.0 * NM, 0.0)
    bottom_sio2_z: tuple[float, float] = (-385.0 * NM, -100.0 * NM)
    pml_layers: int = 24
    flake_dz_m: float = 5.0 * NM
    transverse_mesh_m: float = 50.0 * NM
    outer_z_mesh_m: float = 50.0 * NM
    background_extension_m: float = 1.0 * UM

    def validate(self) -> None:
        for label, bounds in (
            ("fdtd_outer", self.fdtd_outer),
            ("tfsf", self.tfsf),
            ("design", self.design),
            ("omega_q", self.omega_q),
        ):
            bounds.validate(label)
        if not self.fdtd_outer.contains(self.tfsf, strict=True):
            raise ValueError("TFSF must be strictly inside every FDTD outer face")
        if not self.tfsf.contains(self.design, strict=True):
            raise ValueError("design must be strictly inside the TFSF box")
        if not self.tfsf.contains(self.omega_q, strict=True):
            raise ValueError("Omega_Q/six-face box must be inside TFSF")
        if not self.omega_q.contains(self.design, strict=True):
            raise ValueError("Omega_Q/six-face box must contain the design")
        if not (
            self.omega_q.z[0] < self.flake_z[0]
            and self.flake_z[1] < self.omega_q.z[1]
        ):
            raise ValueError("Omega_Q must contain the complete TaIrTe4 thickness")
        if self.pml_layers < 8:
            raise ValueError("too few PML layers")
        for label, value in (
            ("flake_dz_m", self.flake_dz_m),
            ("transverse_mesh_m", self.transverse_mesh_m),
            ("outer_z_mesh_m", self.outer_z_mesh_m),
            ("background_extension_m", self.background_extension_m),
        ):
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{label} must be positive")

    @property
    def tfsf_injection_plane_z_m(self) -> float:
        """Backward z injection enters through the TFSF z-max face."""
        return float(self.tfsf.z[1])

    @property
    def background_lateral_bounds(self) -> tuple[float, float]:
        half_span = max(
            abs(self.fdtd_outer.x[0]),
            abs(self.fdtd_outer.x[1]),
            abs(self.fdtd_outer.y[0]),
            abs(self.fdtd_outer.y[1]),
        ) + self.background_extension_m
        return (-half_span, half_span)

    def geometry_audit_input(self) -> dict[str, object]:
        self.validate()
        background_lo, background_hi = self.background_lateral_bounds
        return {
            "fdtd_outer_input_bounds_m": self.fdtd_outer.to_dict(),
            "pml_inner_requires_realized_mesh_readback": True,
            "tfsf_bounds_m": self.tfsf.to_dict(),
            "tfsf_injection_plane_z_m": self.tfsf_injection_plane_z_m,
            "design_bounds_m": self.design.to_dict(),
            "omega_q_and_six_face_bounds_m": self.omega_q.to_dict(),
            "design_to_tfsf_gaps_m": self.tfsf.gaps_to(self.design),
            "tfsf_to_fdtd_outer_gaps_m": self.fdtd_outer.gaps_to(self.tfsf),
            "omega_q_to_tfsf_gaps_m": self.tfsf.gaps_to(self.omega_q),
            "large_background_lateral_bounds_m": {
                "x": [background_lo, background_hi],
                "y": [background_lo, background_hi],
            },
            "large_background_extends_beyond_pml_inner": True,
            "large_tairte4_background": True,
            "finite_optical_flake": False,
            "periodic_or_bloch_boundary": False,
            "pml_layers": int(self.pml_layers),
            "flake_dz_m": float(self.flake_dz_m),
            "transverse_mesh_m": float(self.transverse_mesh_m),
            "outer_z_mesh_m": float(self.outer_z_mesh_m),
            "flake_z_bounds_m": list(self.flake_z),
            "bottom_sio2_z_bounds_m": list(self.bottom_sio2_z),
        }


def baseline_contract(
    *,
    lateral_domain_um: float = 6.4,
    tfsf_span_um: float = 2.6,
    pml_layers: int = 24,
    transverse_mesh_nm: float = 50.0,
    flake_dz_nm: float = 5.0,
    z_min_um: float = -3.2,
    z_max_um: float = 3.0,
    tfsf_z_min_um: float = -1.8,
    tfsf_z_max_um: float = 1.5,
) -> LargeBackgroundOpticalContract:
    half_domain = 0.5 * lateral_domain_um * UM
    half_tfsf = 0.5 * tfsf_span_um * UM
    contract = LargeBackgroundOpticalContract(
        fdtd_outer=Bounds3D(
            (-half_domain, half_domain),
            (-half_domain, half_domain),
            (z_min_um * UM, z_max_um * UM),
        ),
        tfsf=Bounds3D(
            (-half_tfsf, half_tfsf),
            (-half_tfsf, half_tfsf),
            (tfsf_z_min_um * UM, tfsf_z_max_um * UM),
        ),
        design=Bounds3D(
            (-1.0 * UM, 1.0 * UM),
            (-1.0 * UM, 1.0 * UM),
            (0.0, 600.0 * NM),
        ),
        # A single, fixed volume for both Q and the closed six-face flux.
        # Its 150 nm lateral/top margins keep every monitor face away from
        # the 2 um design while retaining the nearby TaIrTe4 perturbation.
        omega_q=Bounds3D(
            (-1.15 * UM, 1.15 * UM),
            (-1.15 * UM, 1.15 * UM),
            (-150.0 * NM, 750.0 * NM),
        ),
        pml_layers=pml_layers,
        flake_dz_m=flake_dz_nm * NM,
        transverse_mesh_m=transverse_mesh_nm * NM,
    )
    contract.validate()
    return contract


def _strictly_increasing(values: np.ndarray, label: str) -> np.ndarray:
    coordinate = np.asarray(values, float).reshape(-1)
    if coordinate.size < 3 or not np.all(np.isfinite(coordinate)):
        raise ValueError(f"invalid realized {label} mesh")
    if not np.all(np.diff(coordinate) > 0.0):
        raise ValueError(f"realized {label} mesh is not strictly increasing")
    return coordinate


def infer_realized_pml_bounds(
    coordinates: dict[str, Iterable[float]],
    fdtd_outer: Bounds3D,
    pml_layers: int,
) -> dict[str, object]:
    """Audit actual mesh points without pretending mesh points are cell faces.

    The FDTD object's requested bounds are PML *outer* bounds.  The inner
    interface on each side is the mesh point after the requested number of
    PML cells.  The outer readback and layer-count indices are both retained.
    """

    result: dict[str, object] = {}
    outer_readback_matches = True
    for axis in "xyz":
        coordinate = _strictly_increasing(
            np.asarray(list(coordinates[axis]), float), axis
        )
        if coordinate.size <= 2 * pml_layers + 1:
            raise ValueError(
                f"{axis} mesh has {coordinate.size} points for "
                f"{pml_layers} PML layers per side"
            )
        outer_lo, outer_hi = getattr(fdtd_outer, axis)
        layer_lo_index = pml_layers
        layer_hi_index = coordinate.size - pml_layers - 1
        inferred_lo = float(coordinate[layer_lo_index])
        inferred_hi = float(coordinate[layer_hi_index])
        local_step = float(np.max(np.diff(coordinate)))
        outer_matches = bool(
            abs(float(coordinate[0]) - outer_lo) <= 1.0e-15
            and abs(float(coordinate[-1]) - outer_hi) <= 1.0e-15
        )
        outer_readback_matches = outer_readback_matches and outer_matches
        result[axis] = {
            "mesh_point_count": int(coordinate.size),
            "mesh_point_outer_bounds_m": [
                float(coordinate[0]),
                float(coordinate[-1]),
            ],
            "pml_inner_from_layer_count_m": [inferred_lo, inferred_hi],
            "fdtd_outer_input_bounds_m": [outer_lo, outer_hi],
            "layer_count_indices": [layer_lo_index, layer_hi_index],
            "fdtd_outer_mesh_readback_matches": outer_matches,
            "maximum_mesh_step_m": local_step,
            "minimum_mesh_step_m": float(np.min(np.diff(coordinate))),
        }
    result["all_axes_fdtd_outer_mesh_readback_matches"] = (
        outer_readback_matches
    )
    result["coordinate_semantics"] = (
        "Lumerical mesh points; outer cell-face coordinates are not inferred"
    )
    return result
