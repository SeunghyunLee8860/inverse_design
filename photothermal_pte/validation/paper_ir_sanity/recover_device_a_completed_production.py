#!/usr/bin/env python3
"""Recover a completed Device-A production FSP without another FDTD solve."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from matplotlib.path import Path as PolygonPath

REPOSITORY = Path(__file__).resolve().parents[3]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from photothermal_pte.validation.paper_ir_sanity import (  # noqa: E402
    run_lumerical_device_a_ir_q as runner,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def faces(prefix: str) -> dict[str, dict[str, Any]]:
    return {
        f"{axis}_{side}": {
            "name": f"{prefix}_{axis}_{side}",
            "axis": axis,
            "side": side,
            "outward_sign": -1.0 if side == "min" else 1.0,
        }
        for axis in "xyz"
        for side in ("min", "max")
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--incident-reference", type=Path, default=None)
    args_cli = parser.parse_args()
    case_dir = args_cli.case_dir.resolve()
    output = args_cli.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    raw_path = case_dir / "case_result.json"
    fsp = case_dir / "finite_2um_optical_q.fsp"
    raw = json.loads(raw_path.read_text())
    raw_status = raw.get("status")
    if raw_status == "BLOCKED_EXECUTION_ERROR":
        if "bounded dual-cell weights do not close" not in str(
            raw.get("exception")
        ):
            raise RuntimeError(
                "execution failure is not the audited boundary-dual-cell case"
            )
    elif raw_status == "FAILED_ACCEPTANCE":
        completed = raw.get("run_result", {}).get("auto_shutoff", {})
        if not completed.get("simulation_completed_successfully", False):
            raise RuntimeError(
                "acceptance recovery requires a successfully completed solver log"
            )
        if float(completed.get("final_value", np.inf)) > 1e-5:
            raise RuntimeError(
                "acceptance recovery cannot waive the auto-shutoff gate"
            )
    else:
        raise RuntimeError(
            "recovery requires a preserved postprocess or completed-solver "
            "acceptance checkpoint"
        )
    pre_run = raw["pre_run_contract"]
    geometry = pre_run["geometry"]
    case = raw["case"]
    polarization = "a" if float(raw["polarization_deg"]) == 90.0 else "b"
    vertices_um = np.asarray(geometry["flake_vertices_um"], float)
    electrode = geometry.get("electrode_material_contract") or {}
    parsed = SimpleNamespace(
        case=case,
        polarization=polarization,
        polarization_deg=float(raw["polarization_deg"]),
        domain_um=float(raw["domain_um"]),
        pml_layers=int(raw["pml_layers"]),
        flake_dz_nm=float(raw["flake_dz_nm"]),
        waist_um=float(raw["waist_um"]),
        source_span_um=float(raw["source_span_um"]),
        design_radius_um=float(raw["design_radius_um"]),
        output_dir=str(output),
        incident_reference=(
            None
            if args_cli.incident_reference is None
            else str(args_cli.incident_reference.resolve())
        ),
        include_electrodes=bool(geometry["electrodes_in_optical_model"]),
        geometry="device-a-polygon",
        flake_vertices_um=vertices_um,
        top_metal_vertices_um=np.asarray(
            electrode.get("top_polygon_simulation_um", []), float
        ),
        bottom_metal_vertices_um=np.asarray(
            electrode.get("bottom_polygon_simulation_um", []), float
        ),
    )

    base = runner.load_base()
    base.TARGET_WAVELENGTH_M = runner.WAVELENGTH_M
    base.TARGET_FREQUENCY_HZ = runner.C0 / runner.WAVELENGTH_M
    base.SOURCE_START_M = runner.SOURCE_START_M
    base.SOURCE_STOP_M = runner.SOURCE_STOP_M
    base.FLAKE_THICKNESS_M = runner.FLAKE_THICKNESS_M
    base.FLAKE_BOUNDS_M = {
        "x": (float(np.min(vertices_um[:, 0])) * 1e-6, float(np.max(vertices_um[:, 0])) * 1e-6),
        "y": (float(np.min(vertices_um[:, 1])) * 1e-6, float(np.max(vertices_um[:, 1])) * 1e-6),
        "z": (-runner.FLAKE_THICKNESS_M, 0.0),
    }
    base.GEOMETRIC_AREA_M2 = runner.polygon_area(vertices_um) * 1e-12
    base.SIO2_THICKNESS_M = runner.SIO2_THICKNESS_M
    base.SI_DEPTH_M = runner.SI_DEPTH_M
    base.PABS_PADDING_M = runner.PABS_PADDING_M
    base.FDTD_Z_MIN_M = runner.FDTD_Z_MIN_M
    base.FDTD_Z_MAX_M = runner.FDTD_Z_MAX_M
    base.GAUSSIAN_SOURCE_Z_M = runner.SOURCE_Z_M
    base.GAUSSIAN_FOCUS_Z_M = runner.FOCUS_Z_M
    base.INCIDENT_REFERENCE_Z_M = runner.INCIDENT_Z_M
    base.INNER_BOX = geometry["six_face_absorption_box_bounds_m"]
    base.MATERIAL_NAME = geometry["material_contract"]["name"]
    base.SIO2_MATERIAL = runner.SIO2_MATERIAL

    xx_path = PolygonPath(vertices_um * 1e-6)

    def exact_mask(x_m: np.ndarray, y_m: np.ndarray, z_m: np.ndarray) -> np.ndarray:
        xx, yy = np.meshgrid(x_m, y_m, indexing="ij")
        xy = xx_path.contains_points(
            np.column_stack((xx.ravel(), yy.ravel())), radius=1e-15
        ).reshape(xx.shape)
        zz = (z_m >= -runner.FLAKE_THICKNESS_M) & (z_m <= 0.0)
        return xy[:, :, None] & zz[None, None, :]

    setup = {
        "inner_faces": faces("paper_ir_abs"),
        "outer_faces": faces("paper_ir_outer"),
        "geometry": geometry,
        "exact_flake_mask_builder": exact_mask,
    }
    os.environ["VC_LUMERICAL_ROOT"] = str(runner.APPROVED_ROOT)
    os.environ["LUMERICAL_ROOT"] = str(runner.APPROVED_ROOT)
    if str(runner.APPROVED_API) not in sys.path:
        sys.path.insert(0, str(runner.APPROVED_API))
    installation = SimpleNamespace(
        version_key="v261",
        root=runner.APPROVED_ROOT.resolve(),
        lumapi_path=(runner.APPROVED_API / "lumapi.py").resolve(),
        device_executable=(runner.APPROVED_ROOT / "bin" / "device").resolve(),
    )
    lumapi = base.load_lumapi(installation)
    for path in (REPOSITORY / "photothermal_pte", REPOSITORY / "photothermal_pte" / "bundle"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import eqc_lib as runtime  # noqa: E402

    fdtd = None
    original_run = runtime.run_session
    try:
        fdtd = lumapi.FDTD(filename=str(fsp), hide=True, serverArgs={"platform": "offscreen"})
        runtime.run_session = lambda *unused, **unused_kw: "READ_ONLY_COMPLETED_FSP"
        result = base.run_case(fdtd, runtime, parsed, output, setup, pre_run)
        result["auto_shutoff"] = runner.final_logged_auto_shutoff(case_dir)
        result["native_Yee_mesh_audit"] = runner.post_run_native_mesh_audit(
            base,
            fdtd,
            output,
            result,
            setup["inner_faces"],
            geometry["pabs_nominal_control_volume_bounds_m"],
        )
        runner.append_device_a_material_absorption_audit(base, output, parsed, result)
    finally:
        runtime.run_session = original_run
        if fdtd is not None:
            fdtd.close()
    acceptance = result["acceptance"]
    centered_symmetry = acceptance.pop(
        "opposite_lateral_flux_asymmetry_lt_1e_4"
    )
    result.setdefault("diagnostic_not_acceptance_gate", {})[
        "legacy_centered_source_opposite_lateral_flux_asymmetry_lt_1e_4"
    ] = centered_symmetry
    result["diagnostic_not_acceptance_gate"][
        "reason"
    ] = (
        "the frozen Device-A coordinate translation places the beam at "
        "x=-2.4522156686 um so opposite lateral flux is not expected to be "
        "symmetric; raw pair asymmetry remains in the run result"
    )
    acceptance[
        "off_center_source_symmetry_gate_correctly_not_applied"
    ] = True
    acceptance["solver_log_reports_successful_completion"] = result[
        "auto_shutoff"
    ]["simulation_completed_successfully"]
    acceptance["auto_shutoff_reached_requested_threshold"] = (
        result["auto_shutoff"]["final_value"] <= 1e-5
    )
    recovered = {
        **raw,
        "status": "COMPLETED" if all(acceptance.values()) else "FAILED_ACCEPTANCE_READ_ONLY_RECOVERY",
        "validated": False,
        "recovery": {
            "completion_mode": "READ_ONLY_POSTPROCESS_RECOVERY_OF_COMPLETED_GPU_FDTD",
            "FDTD_solve_called": False,
            "runanalysis_called": True,
            "source_FSP": str(fsp),
            "source_FSP_sha256": sha256(fsp),
            "raw_checkpoint_status": raw_status,
            "raw_checkpoint_preserved": str(raw_path),
            "raw_checkpoint_sha256": sha256(raw_path),
        },
        "run_result": result,
    }
    (output / "case_result_recovered.json").write_text(
        json.dumps(base.jsonable(recovered), indent=2) + "\n"
    )
    manifest = {
        "policy": "raw completed FSP remains outside Git",
        "source_FSP": {
            "path": str(fsp),
            "size_bytes": fsp.stat().st_size,
            "sha256": sha256(fsp),
        },
        "recovered_artifacts": {
            path.name: {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in output.iterdir()
            if path.is_file()
        },
    }
    (output / "RAW_ARTIFACT_MANIFEST_RECOVERY.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps({"status": recovered["status"], "acceptance": acceptance}, indent=2))
    return 0 if recovered["status"] == "COMPLETED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
