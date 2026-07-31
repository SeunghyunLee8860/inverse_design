#!/usr/bin/env python3
"""Recover a completed planar empty-stack reference without rerunning FDTD.

The production empty-stack solve can finish before a later mesh-audit error
causes the original wrapper to fail closed.  This tool reopens the saved FSP,
re-evaluates only monitor data and the pabs analysis group, and writes a new
recovery directory.  It never calls the FDTD ``run`` command and never edits
the original raw artifact directory.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from photothermal_pte.validation.paper_ir_sanity import (  # noqa: E402
    run_lumerical_device_a_ir_q as runner,
)


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


def configure_base(base: Any, raw: dict[str, Any]) -> None:
    geometry = raw["pre_run_contract"]["geometry"]
    absorption_bounds = geometry["absorption_analysis_bounds_m"]
    inner_box = geometry["six_face_absorption_box_bounds_m"]
    base.TARGET_WAVELENGTH_M = runner.WAVELENGTH_M
    base.TARGET_FREQUENCY_HZ = runner.C0 / runner.WAVELENGTH_M
    base.SOURCE_START_M = runner.SOURCE_START_M
    base.SOURCE_STOP_M = runner.SOURCE_STOP_M
    base.FLAKE_THICKNESS_M = runner.FLAKE_THICKNESS_M
    base.FLAKE_BOUNDS_M = absorption_bounds
    base.GEOMETRIC_AREA_M2 = (
        (absorption_bounds["x"][1] - absorption_bounds["x"][0])
        * (absorption_bounds["y"][1] - absorption_bounds["y"][0])
    )
    base.SIO2_THICKNESS_M = runner.SIO2_THICKNESS_M
    base.SI_DEPTH_M = runner.SI_DEPTH_M
    base.PABS_PADDING_M = runner.PABS_PADDING_M
    base.FDTD_Z_MIN_M = runner.FDTD_Z_MIN_M
    base.FDTD_Z_MAX_M = runner.FDTD_Z_MAX_M
    base.GAUSSIAN_SOURCE_Z_M = runner.SOURCE_Z_M
    base.GAUSSIAN_FOCUS_Z_M = runner.FOCUS_Z_M
    base.INCIDENT_REFERENCE_Z_M = runner.INCIDENT_Z_M
    base.INNER_BOX = inner_box
    base.MATERIAL_NAME = runner.PRODUCTION_MATERIAL_NAME
    base.SIO2_MATERIAL = runner.SIO2_MATERIAL


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument(
        "--output-dir",
        help="must be new; defaults to ARTIFACT_DIR/readonly_recovery_v1",
    )
    args = parser.parse_args()

    artifact_dir = Path(args.artifact_dir).expanduser().resolve()
    raw_path = artifact_dir / "case_result.json"
    fsp_path = artifact_dir / "finite_2um_optical_q.fsp"
    if not raw_path.is_file() or not fsp_path.is_file():
        raise FileNotFoundError("case_result.json and completed FSP are required")
    output = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else artifact_dir / "readonly_recovery_v1"
    )
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"recovery output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    if raw.get("case") != "empty-stack":
        raise RuntimeError("recovery is restricted to an empty-stack case")
    if raw.get("status") != "BLOCKED_EXECUTION_ERROR":
        raise RuntimeError("input is not the preserved failed wrapper result")
    if "bounded dual-cell weights do not close" not in raw.get(
        "exception", ""
    ):
        raise RuntimeError("input did not fail at the known post-run audit")
    pre_run = raw["pre_run_contract"]
    if not pre_run["checks"]["all"]:
        raise RuntimeError("saved case did not pass its pre-run contract")

    os.environ["VC_LUMERICAL_ROOT"] = str(runner.APPROVED_ROOT)
    os.environ["LUMERICAL_ROOT"] = str(runner.APPROVED_ROOT)
    os.environ["LUMERICAL_PYTHONPATH"] = str(runner.APPROVED_API)
    if str(runner.APPROVED_API) not in sys.path:
        sys.path.insert(0, str(runner.APPROVED_API))

    base = runner.load_base()
    configure_base(base, raw)
    installation = SimpleNamespace(
        version_key="v261",
        root=runner.APPROVED_ROOT.resolve(),
        lumapi_path=(runner.APPROVED_API / "lumapi.py").resolve(),
        device_executable=(
            runner.APPROVED_ROOT / "bin" / "device"
        ).resolve(),
    )
    lumapi = base.load_lumapi(installation)
    parsed = SimpleNamespace(
        case="empty-stack",
        polarization_deg=float(raw["polarization_deg"]),
        domain_um=float(raw["domain_um"]),
        pml_layers=int(raw["pml_layers"]),
        flake_dz_nm=float(raw["flake_dz_nm"]),
        source_span_um=float(raw["source_span_um"]),
        waist_um=float(raw["waist_um"]),
    )
    setup = {
        "inner_faces": faces("paper_ir_abs"),
        "outer_faces": faces("paper_ir_outer"),
        "geometry": pre_run["geometry"],
    }
    runtime = SimpleNamespace(
        run_session=lambda *_args, **_kwargs: (
            "SAVED_GPU_RESULT_READ_ONLY_NO_FDTD_RERUN"
        )
    )

    fdtd = None
    try:
        fdtd = lumapi.FDTD(
            str(fsp_path),
            hide=True,
            serverArgs={"platform": "offscreen"},
        )
        run_result = base.run_case(
            fdtd,
            runtime,
            parsed,
            output,
            setup,
            pre_run,
        )
        run_result["native_Yee_mesh_audit"] = (
            runner.post_run_native_mesh_audit(
                base,
                fdtd,
                output,
                run_result,
                setup["inner_faces"],
                pre_run["geometry"][
                    "pabs_nominal_control_volume_bounds_m"
                ],
            )
        )
        auto_shutoff = runner.final_logged_auto_shutoff(artifact_dir)
        run_result["auto_shutoff"] = auto_shutoff
        run_result["acceptance"][
            "auto_shutoff_reached_requested_threshold"
        ] = auto_shutoff["final_value"] <= 1.0e-5
    finally:
        if fdtd is not None:
            fdtd.close()

    acceptance = run_result["acceptance"]
    all_pass = bool(acceptance) and all(bool(value) for value in acceptance.values())
    recovered = {
        **{
            key: value
            for key, value in raw.items()
            if key
            not in (
                "exception",
                "exception_type",
                "traceback",
                "run_result",
            )
        },
        "status": "COMPLETED" if all_pass else "FAILED_ACCEPTANCE",
        "generation_commit": base.git_commit(),
        "run_result": run_result,
        "recovery_provenance": {
            "classification": (
                "READ_ONLY_RECOVERY_OF_COMPLETED_GPU_FDTD_MONITORS"
            ),
            "FDTD_solve_called": False,
            "runanalysis_called": True,
            "original_case_result": {
                "path": str(raw_path),
                "size_bytes": raw_path.stat().st_size,
                "sha256": base.sha256(raw_path),
            },
            "completed_FSP": {
                "path": str(fsp_path),
                "size_bytes": fsp_path.stat().st_size,
                "sha256": base.sha256(fsp_path),
            },
            "completed_GPU_engine_log": {
                "path": auto_shutoff["log_path"],
                "size_bytes": Path(auto_shutoff["log_path"]).stat().st_size,
                "sha256": base.sha256(Path(auto_shutoff["log_path"])),
            },
            "original_generation_commit": raw["generation_commit"],
            "original_raw_artifacts_modified": False,
        },
    }
    recovered_path = output / "case_result_recovered.json"
    base.write_json(recovered_path, recovered)
    print(json.dumps(base.jsonable(recovered), indent=2))
    return 0 if all_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
