#!/usr/bin/env python3
"""Read-only production-Q recovery after an incident-reference mismatch.

The completed FSP is loaded and postprocessed with the matching empty-stack
polarization reference.  This command never calls the FDTD solver.  It may
run the saved pabs analysis group against already-computed monitor fields.
The failed raw case result and completed FSP remain immutable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from photothermal_pte.validation.paper_ir_sanity import (  # noqa: E402
    run_lumerical_device_a_ir_q as runner,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fsp", type=Path, required=True)
    parser.add_argument("--failed-case-result", type=Path, required=True)
    parser.add_argument("--incident-reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path: Path, role: str) -> dict[str, Any]:
    return {
        "role": role,
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


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


def exact_halfplane_mask(
    x_m: np.ndarray,
    y_m: np.ndarray,
    z_m: np.ndarray,
) -> np.ndarray:
    return (
        (y_m[None, :, None] <= x_m[:, None, None])
        & (z_m[None, None, :] >= -runner.FLAKE_THICKNESS_M)
        & (z_m[None, None, :] <= 0.0)
    )


def main() -> int:
    args = parse_args()
    fsp = args.fsp.resolve()
    failed_path = args.failed_case_result.resolve()
    reference_path = args.incident_reference.resolve()
    output = args.output_dir.resolve()
    for path in (fsp, failed_path, reference_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)

    failed = json.loads(failed_path.read_text())
    if failed.get("status") != "BLOCKED_EXECUTION_ERROR":
        raise RuntimeError("input is not a preserved failed postprocess case")
    if "incident reference contract mismatch" not in str(
        failed.get("exception", "")
    ):
        raise RuntimeError("failure is not the audited reference mismatch")
    auto = runner.final_logged_auto_shutoff(fsp.parent)
    if not auto["simulation_completed_successfully"]:
        raise RuntimeError("saved FSP does not have a successful solver log")

    reference = json.loads(reference_path.read_text())
    expected = {
        "case": "empty-stack",
        "polarization_deg": float(failed["polarization_deg"]),
        "source_span_um": float(failed["source_span_um"]),
        "waist_um": float(failed["waist_um"]),
        "status": "COMPLETED",
    }
    mismatches = {
        key: (reference.get(key), value)
        for key, value in expected.items()
        if (
            reference.get(key) != value
            if isinstance(value, str)
            else not np.isclose(
                float(reference.get(key, np.nan)),
                float(value),
            )
        )
    }
    if mismatches:
        raise RuntimeError(f"replacement reference mismatch: {mismatches}")

    base = runner.load_base()
    base.TARGET_WAVELENGTH_M = runner.WAVELENGTH_M
    base.TARGET_FREQUENCY_HZ = runner.C0 / runner.WAVELENGTH_M
    base.SOURCE_START_M = runner.SOURCE_START_M
    base.SOURCE_STOP_M = runner.SOURCE_STOP_M
    base.FLAKE_THICKNESS_M = runner.FLAKE_THICKNESS_M
    base.FLAKE_BOUNDS_M = failed["pre_run_contract"]["geometry"][
        "absorption_analysis_bounds_m"
    ]
    span_x = (
        base.FLAKE_BOUNDS_M["x"][1] - base.FLAKE_BOUNDS_M["x"][0]
    )
    span_y = (
        base.FLAKE_BOUNDS_M["y"][1] - base.FLAKE_BOUNDS_M["y"][0]
    )
    base.GEOMETRIC_AREA_M2 = 0.5 * span_x * span_y
    base.MATERIAL_NAME = failed["pre_run_contract"]["material"]["name"]
    base.SIO2_MATERIAL = runner.SIO2_MATERIAL

    os.environ["VC_LUMERICAL_ROOT"] = str(runner.APPROVED_ROOT)
    os.environ["LUMERICAL_ROOT"] = str(runner.APPROVED_ROOT)
    os.environ["LUMERICAL_PYTHONPATH"] = str(runner.APPROVED_API)
    if str(runner.APPROVED_API) not in sys.path:
        sys.path.insert(0, str(runner.APPROVED_API))
    installation = SimpleNamespace(
        version_key="v261",
        root=runner.APPROVED_ROOT.resolve(),
        lumapi_path=(runner.APPROVED_API / "lumapi.py").resolve(),
        device_executable=(runner.APPROVED_ROOT / "bin" / "device").resolve(),
    )
    lumapi = base.load_lumapi(installation)

    parsed = SimpleNamespace(
        case="finite-flake",
        polarization_deg=float(failed["polarization_deg"]),
        domain_um=float(failed["domain_um"]),
        pml_layers=int(failed["pml_layers"]),
        flake_dz_nm=float(failed["flake_dz_nm"]),
        source_span_um=float(failed["source_span_um"]),
        waist_um=float(failed["waist_um"]),
        incident_reference=str(reference_path),
    )
    setup = {
        "inner_faces": faces("paper_ir_abs"),
        "outer_faces": faces("paper_ir_outer"),
        "exact_flake_mask_builder": exact_halfplane_mask,
        "geometry": failed["pre_run_contract"]["geometry"],
    }
    runtime = SimpleNamespace(
        run_session=lambda *_args, **_kwargs: "READ_ONLY_COMPLETED_FSP"
    )
    fdtd = None
    try:
        fdtd = lumapi.FDTD(hide=True, serverArgs={"platform": "offscreen"})
        fdtd.load(str(fsp))
        run_result = base.run_case(
            fdtd,
            runtime,
            parsed,
            output,
            setup,
            failed["pre_run_contract"],
        )
        run_result["auto_shutoff"] = auto
        run_result["acceptance"][
            "solver_log_reports_successful_completion"
        ] = True
        run_result["acceptance"][
            "auto_shutoff_reached_requested_threshold"
        ] = auto["final_value"] <= 1.0e-5
        run_result["native_Yee_mesh_audit"] = (
            runner.post_run_native_mesh_audit(
                base,
                fdtd,
                output,
                run_result,
                setup["inner_faces"],
                failed["pre_run_contract"]["geometry"][
                    "pabs_nominal_control_volume_bounds_m"
                ],
            )
        )
        run_result["material_epsilon_readback"] = failed[
            "pre_run_contract"
        ]["material"]["epsilon_readback"]
    finally:
        if fdtd is not None:
            fdtd.close()

    all_pass = all(bool(value) for value in run_result["acceptance"].values())
    recovered = {
        **{
            key: value
            for key, value in failed.items()
            if key not in ("exception", "exception_type", "traceback")
        },
        "status": "COMPLETED" if all_pass else "FAILED_ACCEPTANCE",
        "validated": False,
        "project": str(fsp),
        "run_result": run_result,
        "read_only_recovery": {
            "FDTD_solve_called": False,
            "runanalysis_on_saved_monitor_fields": True,
            "source_FSP_immutable": True,
            "failed_case_result_immutable": True,
            "corrected_item": (
                "matching b-polarized empty-stack incident reference"
            ),
            "generation_command": shlex.join([sys.executable, *sys.argv]),
        },
    }
    result_path = output / "case_result.json"
    write_json(result_path, recovered)
    artifacts = [
        record(fsp, "completed_source_FSP"),
        record(failed_path, "preserved_failed_case_result"),
        record(reference_path, "matching_incident_reference"),
        record(output / "finite_q_on_artifact.npz", "recovered_raw_Q_NPZ"),
        record(result_path, "recovered_case_result"),
        record(
            output / "native_yee_mesh_coordinates.npz",
            "recovered_native_Yee_coordinates",
        ),
    ]
    for optional, role in (
        (output / "field_slices_raw.npz", "recovered_field_slices"),
        (output / "Q_component_xy_slices.png", "recovered_Q_components"),
        (output / "Q_cross_section_slices.png", "recovered_Q_cross_sections"),
    ):
        if optional.is_file():
            artifacts.append(record(optional, role))
    manifest = {
        "status": recovered["status"],
        "read_only_recovery": True,
        "FDTD_solve_called": False,
        "raw_artifacts_committed_to_Git": False,
        "generation_command": recovered["read_only_recovery"][
            "generation_command"
        ],
        "artifacts": artifacts,
    }
    write_json(output / "RAW_ARTIFACT_MANIFEST.json", manifest)
    print(json.dumps(recovered, indent=2))
    return 0 if recovered["status"] == "COMPLETED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
