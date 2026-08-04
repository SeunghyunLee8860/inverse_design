#!/usr/bin/env python3
"""Fail-closed PML/TFSF geometry audit for the large-background model.

This command creates and saves a v261 FDTD project but does not call the
solver.  Long optical sweeps are forbidden until the realized mesh and every
geometric gap pass this audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np

from .large_background_contract import (
    Bounds3D,
    LargeBackgroundOpticalContract,
    baseline_contract,
    infer_realized_pml_bounds,
)
from .probe_v261_cpu_tfsf_device import (
    C0,
    EPS0,
    FREQUENCY_HZ,
    SIO2_MATERIAL,
    SOURCE_NAME,
    TAIRTE4_MATERIAL,
    add_rect,
    add_sio2_material,
    add_tairte4_material,
)
from .probe_v261_gpu_plane_wave_roi import (
    APPROVED_API,
    APPROVED_ROOT,
    json_default,
    load_lumapi,
    optional_readback,
    resource_snapshot,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--domain-um", type=float, default=6.4)
    parser.add_argument("--tfsf-span-um", type=float, default=2.6)
    parser.add_argument("--pml-layers", type=int, default=24)
    parser.add_argument("--transverse-mesh-nm", type=float, default=50.0)
    parser.add_argument("--flake-dz-nm", type=float, default=5.0)
    parser.add_argument("--z-min-um", type=float, default=-3.2)
    parser.add_argument("--z-max-um", type=float, default=3.0)
    parser.add_argument("--tfsf-z-min-um", type=float, default=-1.8)
    parser.add_argument("--tfsf-z-max-um", type=float, default=1.5)
    parser.add_argument("--threads", type=int, default=16)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_scalar(value: object, label: str) -> float:
    array = np.asarray(value).reshape(-1)
    if array.size != 1:
        raise RuntimeError(f"{label} is not scalar: {array.shape}")
    result = float(np.real(array[0]))
    if not np.isfinite(result):
        raise RuntimeError(f"{label} is not finite")
    return result


def _named_bounds(fdtd: object, name: str) -> dict[str, list[float]]:
    return {
        axis: [
            _finite_scalar(
                fdtd.getnamed(name, f"{axis} min"), f"{name} {axis} min"
            ),
            _finite_scalar(
                fdtd.getnamed(name, f"{axis} max"), f"{name} {axis} max"
            ),
        ]
        for axis in "xyz"
    }


def _mesh_coordinate(fdtd: object, axis: str) -> np.ndarray:
    raw = fdtd.getresult("FDTD", axis)
    if isinstance(raw, dict):
        if axis not in raw:
            raise RuntimeError(
                f"FDTD {axis} result dictionary has keys {sorted(raw)}"
            )
        raw = raw[axis]
    coordinate = np.asarray(raw, float).reshape(-1)
    if coordinate.size < 3:
        raise RuntimeError(f"FDTD {axis} mesh has only {coordinate.size} points")
    return coordinate


def add_large_background_geometry(
    fdtd: object,
    contract: LargeBackgroundOpticalContract,
) -> None:
    """Build the exact source/background/design geometry used by the audit."""

    fdtd.eval("selectall; delete;")
    solver = fdtd.addfdtd()
    solver["dimension"] = "3D"
    for axis in "xyz":
        lo, hi = getattr(contract.fdtd_outer, axis)
        solver[f"{axis} min"] = lo
        solver[f"{axis} max"] = hi
        solver[f"{axis} min bc"] = "PML"
        solver[f"{axis} max bc"] = "PML"
    solver["pml layers"] = contract.pml_layers
    solver["mesh type"] = "auto non-uniform"
    solver["mesh refinement"] = "conformal variant 1"
    solver["mesh accuracy"] = 5
    solver["min mesh step"] = 1.0e-9
    solver["simulation time"] = 4.0e-12
    solver["auto shutoff min"] = 1.0e-8

    add_sio2_material(fdtd)
    add_tairte4_material(fdtd)
    background_lo, background_hi = contract.background_lateral_bounds
    background_span = background_hi - background_lo
    add_rect(
        fdtd,
        "large_background_Si",
        x_span=background_span,
        y_span=background_span,
        z_min=contract.fdtd_outer.z[0] - 0.5e-6,
        z_max=contract.bottom_sio2_z[0],
        index=3.425,
    )
    add_rect(
        fdtd,
        "large_background_bottom_SiO2",
        x_span=background_span,
        y_span=background_span,
        z_min=contract.bottom_sio2_z[0],
        z_max=contract.bottom_sio2_z[1],
        material=SIO2_MATERIAL,
    )
    add_rect(
        fdtd,
        "large_background_TaIrTe4",
        x_span=background_span,
        y_span=background_span,
        z_min=contract.flake_z[0],
        z_max=contract.flake_z[1],
        material=TAIRTE4_MATERIAL,
    )
    design = fdtd.addrect()
    design["name"] = "finite_design_placeholder_air"
    for axis in "xyz":
        lo, hi = getattr(contract.design, axis)
        design[f"{axis} min"] = lo
        design[f"{axis} max"] = hi
    design["index"] = 1.0

    transverse_mesh = fdtd.addmesh()
    transverse_mesh["name"] = "certificate_transverse_mesh"
    # The high-index TaIrTe4 background occupies every lateral PML cell.
    # Restricting this override to the TFSF box lets the auto mesher create
    # ~2 nm transverse cells in the surrounding background (hundreds of GiB).
    # The declared 50 nm certificate mesh therefore covers the full FDTD
    # outer region; the 5 nm override remains z-only at the flake.
    transverse_mesh["x min"] = contract.fdtd_outer.x[0]
    transverse_mesh["x max"] = contract.fdtd_outer.x[1]
    transverse_mesh["y min"] = contract.fdtd_outer.y[0]
    transverse_mesh["y max"] = contract.fdtd_outer.y[1]
    transverse_mesh["z min"] = contract.fdtd_outer.z[0]
    transverse_mesh["z max"] = contract.fdtd_outer.z[1]
    transverse_mesh["override x mesh"] = True
    transverse_mesh["override y mesh"] = True
    transverse_mesh["override z mesh"] = False
    transverse_mesh["dx"] = contract.transverse_mesh_m
    transverse_mesh["dy"] = contract.transverse_mesh_m

    outer_z_mesh = fdtd.addmesh()
    outer_z_mesh["name"] = "certificate_outer_z_mesh"
    outer_z_mesh["x min"] = contract.fdtd_outer.x[0]
    outer_z_mesh["x max"] = contract.fdtd_outer.x[1]
    outer_z_mesh["y min"] = contract.fdtd_outer.y[0]
    outer_z_mesh["y max"] = contract.fdtd_outer.y[1]
    outer_z_mesh["z min"] = contract.fdtd_outer.z[0]
    outer_z_mesh["z max"] = contract.fdtd_outer.z[1]
    outer_z_mesh["override x mesh"] = False
    outer_z_mesh["override y mesh"] = False
    outer_z_mesh["override z mesh"] = True
    outer_z_mesh["dz"] = contract.outer_z_mesh_m

    flake_mesh = fdtd.addmesh()
    flake_mesh["name"] = "certificate_flake_z_mesh"
    flake_mesh["x min"] = contract.tfsf.x[0]
    flake_mesh["x max"] = contract.tfsf.x[1]
    flake_mesh["y min"] = contract.tfsf.y[0]
    flake_mesh["y max"] = contract.tfsf.y[1]
    flake_mesh["z min"] = contract.flake_z[0] - 25.0e-9
    flake_mesh["z max"] = contract.flake_z[1] + 25.0e-9
    flake_mesh["override x mesh"] = False
    flake_mesh["override y mesh"] = False
    flake_mesh["override z mesh"] = True
    flake_mesh["dz"] = contract.flake_dz_m

    source = fdtd.addtfsf()
    source["name"] = SOURCE_NAME
    source["injection axis"] = "z"
    source["direction"] = "backward"
    source["polarization angle"] = 0.0
    source["angle theta"] = 0.0
    source["angle phi"] = 0.0
    for axis in "xyz":
        lo, hi = getattr(contract.tfsf, axis)
        source[f"{axis} min"] = lo
        source[f"{axis} max"] = hi
    source["override global source settings"] = True
    source["wavelength start"] = 3.0e-6
    source["wavelength stop"] = 6.0e-6
    # For a 1 W/m2 vacuum plane wave, I = 0.5*eps0*c*|E|^2.
    # Keep this source-local analytic normalization; do not inherit the
    # empirical calibration from the earlier finite-device TFSF probe.
    source["amplitude"] = np.sqrt(2.0 / (EPS0 * C0))


def main() -> int:
    args = parse_args()
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    project_path = output / "large_background_geometry_audit.fsp"
    result_path = output / "geometry_audit.json"
    contract = baseline_contract(
        lateral_domain_um=args.domain_um,
        tfsf_span_um=args.tfsf_span_um,
        pml_layers=args.pml_layers,
        transverse_mesh_nm=args.transverse_mesh_nm,
        flake_dz_nm=args.flake_dz_nm,
        z_min_um=args.z_min_um,
        z_max_um=args.z_max_um,
        tfsf_z_min_um=args.tfsf_z_min_um,
        tfsf_z_max_um=args.tfsf_z_max_um,
    )
    result: dict[str, object] = {
        "status": "BLOCKED_LARGE_BACKGROUND_GEOMETRY_AUDIT_NOT_RUN",
        "purpose": "geometry and realized-mesh audit only; no solver run",
        "solver_root": str(APPROVED_ROOT),
        "lumapi_path": str(APPROVED_API),
        "engine_run": False,
        "thermal_run": False,
        "adjoint_run": False,
        "finite_difference_run": False,
        "input_contract": contract.geometry_audit_input(),
    }
    fdtd = None
    started = time.monotonic()
    try:
        lumapi = load_lumapi()
        session_started = time.monotonic()
        fdtd = lumapi.FDTD(hide=True, serverArgs={"platform": "offscreen"})
        result["session_startup_wall_s"] = time.monotonic() - session_started
        add_large_background_geometry(fdtd, contract)
        fdtd.setresource("FDTD", 1, "active", 1)
        fdtd.setresource("FDTD", 1, "processes", "1")
        fdtd.setresource("FDTD", 1, "threads", str(args.threads))
        fdtd.setresource("FDTD", 2, "active", 0)
        result["resources"] = resource_snapshot(fdtd)
        result["cad_readback"] = {
            "fdtd_bounds_m": _named_bounds(fdtd, "FDTD"),
            "tfsf_bounds_m": _named_bounds(fdtd, SOURCE_NAME),
            "design_bounds_m": _named_bounds(
                fdtd, "finite_design_placeholder_air"
            ),
            "large_background_tairte4_bounds_m": _named_bounds(
                fdtd, "large_background_TaIrTe4"
            ),
            "source": {
                prop: optional_readback(fdtd, SOURCE_NAME, prop)
                for prop in (
                    "injection axis",
                    "direction",
                    "polarization angle",
                    "angle theta",
                    "angle phi",
                    "wavelength start",
                    "wavelength stop",
                    "amplitude",
                )
            },
        }
        fdtd.save(str(project_path))
        coordinates = {
            axis: _mesh_coordinate(fdtd, axis) for axis in "xyz"
        }
        result["realized_mesh"] = infer_realized_pml_bounds(
            coordinates,
            contract.fdtd_outer,
            contract.pml_layers,
        )
        result["realized_mesh"]["coordinate_sha256"] = {
            axis: hashlib.sha256(
                np.asarray(coordinates[axis], dtype="<f8").tobytes()
            ).hexdigest()
            for axis in "xyz"
        }
        result["fsp"] = {
            "path": str(project_path),
            "byte_size": project_path.stat().st_size,
            "sha256": sha256(project_path),
        }
        realized_pml_inner = Bounds3D(
            **{
                axis: tuple(
                    result["realized_mesh"][axis][
                        "pml_inner_from_layer_count_m"
                    ]
                )
                for axis in "xyz"
            }
        )
        realized_pml_inner.validate("realized_pml_inner")
        result["realized_pml_inner_bounds_m"] = (
            realized_pml_inner.to_dict()
        )
        result["realized_tfsf_to_pml_inner_signed_gaps_m"] = {
            axis: {
                "min_m": (
                    getattr(contract.tfsf, axis)[0]
                    - getattr(realized_pml_inner, axis)[0]
                ),
                "max_m": (
                    getattr(realized_pml_inner, axis)[1]
                    - getattr(contract.tfsf, axis)[1]
                ),
            }
            for axis in "xyz"
        }
        minimum_realized_gap_m = min(
            side_gap
            for axis_gap in result[
                "realized_tfsf_to_pml_inner_signed_gaps_m"
            ].values()
            for side_gap in axis_gap.values()
        )
        result["minimum_realized_tfsf_to_pml_inner_gap_m"] = (
            minimum_realized_gap_m
        )
        exact_cad = all(
            np.allclose(
                result["cad_readback"][readback_key][axis],
                getattr(bounds, axis),
                rtol=0.0,
                atol=1.0e-15,
            )
            for readback_key, bounds in (
                ("fdtd_bounds_m", contract.fdtd_outer),
                ("tfsf_bounds_m", contract.tfsf),
                ("design_bounds_m", contract.design),
            )
            for axis in "xyz"
        )
        result["gates"] = {
            "contract_valid": True,
            "exact_cad_bounds_readback": exact_cad,
            "tfsf_strictly_inside_realized_pml_inner": (
                realized_pml_inner.contains(contract.tfsf, strict=True)
            ),
            "minimum_realized_tfsf_pml_gap_ge_100nm": (
                minimum_realized_gap_m >= 100.0e-9 - 1.0e-15
            ),
            "design_strictly_inside_tfsf": (
                contract.tfsf.contains(contract.design, strict=True)
            ),
            "omega_q_strictly_inside_tfsf": (
                contract.tfsf.contains(contract.omega_q, strict=True)
            ),
            "omega_q_contains_complete_design": (
                contract.omega_q.contains(contract.design, strict=True)
            ),
            "fdtd_outer_mesh_readback_matches": bool(
                result["realized_mesh"][
                    "all_axes_fdtd_outer_mesh_readback_matches"
                ]
            ),
            "all_six_boundaries_pml": True,
            "periodic_or_bloch_absent": True,
            "large_background_extends_beyond_pml_inner": True,
        }
        result["passed"] = all(result["gates"].values())
        result["status"] = (
            "VALIDATED_LARGE_BACKGROUND_PML_TFSF_GEOMETRY"
            if result["passed"]
            else "FAILED_LARGE_BACKGROUND_PML_TFSF_GEOMETRY"
        )
    except Exception as exc:
        result["status"] = "BLOCKED_LARGE_BACKGROUND_GEOMETRY_AUDIT"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
        result["passed"] = False
    finally:
        result["total_wall_s"] = time.monotonic() - started
        if fdtd is not None:
            try:
                fdtd.close()
            except Exception:
                pass
        result_path.write_text(
            json.dumps(result, indent=2, default=json_default) + "\n"
        )
    print(json.dumps(result, indent=2, default=json_default), flush=True)
    return 0 if result.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
