#!/usr/bin/env python3
"""Runsetup-only audit for the compact 100 nm TaIrTe4 optical contract."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import traceback

import numpy as np

from photothermal_pte.optimization_runs.tairte4_flake_topology.contract import CONTRACT
from photothermal_pte.optimization_runs.tairte4_flake_topology import optical


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]
RUN002 = REPOSITORY / "photothermal_pte" / "optimization_runs" / "run_002_gaussian10_w8p5_current_max"
STAGE1 = REPOSITORY / "photothermal_pte" / "validation" / "photothermal_stage1"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def regional_maximum_step(coordinate: np.ndarray, low: float, high: float) -> float:
    intervals = np.diff(coordinate)
    centres = 0.5 * (coordinate[:-1] + coordinate[1:])
    selected = (centres >= low) & (centres <= high)
    if not np.any(selected):
        raise RuntimeError(f"no mesh intervals inside [{low}, {high}]")
    return float(np.max(intervals[selected]))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--interface-xy-nm", type=float, default=100.0)
    parser.add_argument("--domain-um", type=float, default=40.0)
    args = parser.parse_args()
    if args.interface_xy_nm <= 0.0:
        parser.error("--interface-xy-nm must be positive")
    interface_xy_step_m = args.interface_xy_nm * 1e-9
    optical_lateral_span_m = args.domain_um * 1e-6
    if optical_lateral_span_m <= CONTRACT.source_span_m:
        parser.error("--domain-um must exceed the 34 um source span")
    CONTRACT.validate()
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "tairte4_flake_optical_runsetup_audit.json"
    project = output / "tairte4_flake_optical_runsetup.fsp"
    mesh_npz = output / "tairte4_flake_optical_mesh_coordinates.npz"
    result: dict[str, object] = {
        "status": "BLOCKED_TAIRTE4_FLAKE_OPTICAL_RUNSETUP",
        "passed": False,
        "Maxwell_solve": False,
    }
    fdtd = None
    try:
        # The audited Run002 helpers use sibling absolute imports.  Make that
        # directory explicit before loading them by file path; otherwise a
        # clean checkout can accidentally depend on the caller's cwd.
        for helper_path in (RUN002, STAGE1):
            helper_string = str(helper_path)
            if helper_string not in sys.path:
                sys.path.insert(0, helper_string)
        material_control = load_module(RUN002 / "run_complex_material_control.py", "run010_material_control")
        source_wrapper = material_control.load_source_wrapper()
        audit = source_wrapper.source_audit
        optical.configure_source(
            audit, optical_lateral_span_m=optical_lateral_span_m
        )
        os.environ["VC_LUMERICAL_ROOT"] = str(audit.APPROVED_ROOT)
        os.environ["LUMERICAL_ROOT"] = str(audit.APPROVED_ROOT)
        os.environ["LUMERICAL_PYTHONPATH"] = str(audit.APPROVED_API)
        os.environ["PATH"] = f"{audit.APPROVED_ROOT / 'bin'}:{os.environ.get('PATH','')}"
        helper = audit.load_module(audit.API_HELPER, "run010_optical_runsetup_api")
        installation = type(
            "Installation",
            (),
            {
                "version_key": "v261",
                "root": audit.APPROVED_ROOT,
                "lumapi_path": audit.APPROVED_API / "lumapi.py",
                "device_executable": audit.APPROVED_ROOT / "bin" / "device",
            },
        )()
        lumapi = helper.load_lumapi(installation)
        fdtd = lumapi.FDTD(hide=True, serverArgs={"platform": "offscreen"})
        source = audit.setup(
            fdtd,
            4.0,
            1.0e-7,
            CONTRACT.mesh_accuracy,
            CONTRACT.calibrated_source_object_waist_m,
            CONTRACT.target_waist_m,
        )
        source["source"]["model"] = "compact finite scalar Gaussian; pending source-only revalidation"

        # Reuse only the already-audited material constructors, then rename
        # their constants for this independent geometry.
        geometry_module = load_module(RUN002 / "audit_production_candidate_geometry.py", "run010_geometry_materials")
        geometry_module.TAIRTE4_MATERIAL = optical.TAIRTE4_MATERIAL
        geometry_module.SIO2_MATERIAL = optical.SIO2_MATERIAL
        geometry_module.SI_MATERIAL = optical.SI_MATERIAL
        tairte4 = geometry_module.add_tairte4_material(fdtd)
        substrate = geometry_module.add_substrate_materials(fdtd)
        half_domain = 0.5 * optical_lateral_span_m
        optical.add_rect(
            fdtd,
            "run010_Si_substrate",
            optical.SI_MATERIAL,
            {"x": (-half_domain, half_domain), "y": (-half_domain, half_domain), "z": (CONTRACT.optical_z_min_m, -0.385e-6)},
        )
        optical.add_rect(
            fdtd,
            "run010_bottom_SiO2",
            optical.SIO2_MATERIAL,
            {"x": (-half_domain, half_domain), "y": (-half_domain, half_domain), "z": (-0.385e-6, -0.100e-6)},
        )
        frame_names = optical.add_fixed_frame(fdtd)
        rho = np.full(CONTRACT.design_node_shape, 0.5, dtype=np.float64)
        design = optical.add_design(fdtd, rho)
        mesh_names = optical.add_mesh_hierarchy(
            fdtd,
            interface_xy_step_m=interface_xy_step_m,
            optical_lateral_span_m=optical_lateral_span_m,
        )
        flux_names = optical.add_absorption_and_flux(fdtd)
        fdtd.setnamed(audit.SOURCE_NAME, "polarization angle", 90.0)
        fdtd.runsetup()
        mesh = audit.mesh_readback(fdtd)
        if not mesh.get("available"):
            raise RuntimeError(f"mesh readback failed: {mesh}")
        coordinates = mesh.pop("coordinate_arrays")
        np.savez_compressed(mesh_npz, **{f"{axis}_m": coordinates[axis] for axis in "xyz"})
        flake = 0.5 * CONTRACT.flake_span_m
        design_half = 0.5 * CONTRACT.design_span_m
        regional = {
            "design_max_dx_m": regional_maximum_step(coordinates["x"], -design_half, design_half),
            "design_max_dy_m": regional_maximum_step(coordinates["y"], -design_half, design_half),
            "flake_max_dx_m": regional_maximum_step(coordinates["x"], -flake, flake),
            "flake_max_dy_m": regional_maximum_step(coordinates["y"], -flake, flake),
            "flake_max_dz_m": regional_maximum_step(coordinates["z"], -CONTRACT.flake_thickness_m, 0.0),
            "outer_x_max_step_m": regional_maximum_step(coordinates["x"], flake + 0.5e-6, half_domain - 0.5e-6),
            "outer_y_max_step_m": regional_maximum_step(coordinates["y"], flake + 0.5e-6, half_domain - 0.5e-6),
        }
        names = [
            "FDTD",
            audit.SOURCE_NAME,
            "run010_Si_substrate",
            "run010_bottom_SiO2",
            optical.DESIGN_OBJECT,
            optical.PABS_FIELD,
            optical.PABS_INDEX,
            *frame_names,
            *mesh_names,
        ]
        bounds = {name: optical.named_bounds(fdtd, name) for name in names}
        pabs_bounds_match = all(
            np.allclose(
                bounds[name][axis],
                optical.Q_BOUNDS[axis],
                rtol=0.0,
                atol=2e-18,
            )
            for name in (optical.PABS_FIELD, optical.PABS_INDEX)
            for axis in "xyz"
        )
        source_readback = audit.source_readback(fdtd)
        domain_readback = audit.domain_readback(fdtd)
        fdtd.save(str(project))
        passed = bool(
            regional["design_max_dx_m"] <= CONTRACT.design_step_m + 2e-12
            and regional["design_max_dy_m"] <= CONTRACT.design_step_m + 2e-12
            and regional["flake_max_dx_m"] <= interface_xy_step_m + 2e-12
            and regional["flake_max_dy_m"] <= interface_xy_step_m + 2e-12
            and regional["flake_max_dz_m"] <= CONTRACT.flake_dz_m + 2e-12
            and regional["outer_x_max_step_m"] > 1.5 * CONTRACT.design_step_m
            and regional["outer_y_max_step_m"] > 1.5 * CONTRACT.design_step_m
            and pabs_bounds_match
            and all(value == "PML" for value in domain_readback["boundaries"].values())
            and int(round(source_readback["polarization angle"])) == 90
        )
        previous_points = 41146664
        result = {
            "status": "VALIDATED_TAIRTE4_FLAKE_OPTICAL_RUNSETUP" if passed else "FAILED_TAIRTE4_FLAKE_OPTICAL_RUNSETUP",
            "passed": passed,
            "scope": "runsetup/geometry/mesh only; no Maxwell, Q, thermal, electrical, adjoint, or optimization solve",
            "candidate_contract": CONTRACT.audit(),
            "requested_interface_xy_step_m": interface_xy_step_m,
            "requested_optical_lateral_span_m": optical_lateral_span_m,
            "source_contract": source,
            "source_readback": source_readback,
            "domain_readback": domain_readback,
            "materials": {"TaIrTe4": tairte4, **substrate},
            "design": {key: value for key, value in design.items() if key != "nodes_m"},
            "geometry_bounds_m": bounds,
            "Q_control_volume_m": {axis: list(values) for axis, values in optical.Q_BOUNDS.items()},
            "pabs_field_index_bounds_match_Q_control_volume": pabs_bounds_match,
            "mesh_readback": mesh,
            "regional_mesh_readback": regional,
            "grid_point_ratio_to_Run009_runsetup": float(mesh["grid_points"] / previous_points),
            "rough_runtime_ratio_only_not_a_gate": float(mesh["grid_points"] / previous_points),
            "flux_monitors": flux_names,
            "Maxwell_solve": False,
            "CPU_FDTD_fallback": False,
            "artifacts": {
                "FSP": {"path": str(project), "size_bytes": project.stat().st_size, "sha256": sha256(project)},
                "mesh_NPZ": {"path": str(mesh_npz), "size_bytes": mesh_npz.stat().st_size, "sha256": sha256(mesh_npz)},
            },
        }
    except Exception as exc:
        result.update({"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
    finally:
        if fdtd is not None:
            try:
                fdtd.close()
            except Exception:
                pass
        result_path.write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(json.dumps(result, indent=2, default=str))
    return 0 if result.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
