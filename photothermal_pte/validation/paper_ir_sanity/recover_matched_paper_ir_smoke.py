#!/usr/bin/env python3
"""Recover matched-control-volume metrics from one completed diagnostic FSP.

This is a read-only recovery path for a solver run that completed before the
original Python postprocessor rejected a staggered Yee sample.  It never calls
``run`` or ``runanalysis`` and never changes the saved FSP or raw case result.
"""

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

REPOSITORY = Path(__file__).resolve().parents[3]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from photothermal_pte.validation.paper_ir_sanity import (
    run_lumerical_device_a_ir_q as runner,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def dual_cell_contract(
    coordinate: np.ndarray,
    bounds: list[float],
) -> dict[str, Any]:
    weights = runner.bounded_dual_cell_weights(
        coordinate,
        float(bounds[0]),
        float(bounds[1]),
    )
    active = np.flatnonzero(weights > 0.0)
    return {
        "sample_bounds_m": [
            float(coordinate[0]),
            float(coordinate[-1]),
        ],
        "realized_face_bounds_m": [float(bounds[0]), float(bounds[1])],
        "sample_count": int(coordinate.size),
        "positive_weight_sample_count": int(active.size),
        "first_positive_weight_sample_index": int(active[0]),
        "last_positive_weight_sample_index": int(active[-1]),
        "zero_weight_sample_count": int(np.count_nonzero(weights == 0.0)),
        "weight_sum_m": float(np.sum(weights)),
        "realized_span_m": float(bounds[1] - bounds[0]),
        "dual_cell_support_closes_on_realized_faces": bool(
            np.isclose(
                float(np.sum(weights)),
                float(bounds[1] - bounds[0]),
                rtol=1.0e-13,
                atol=1.0e-18,
            )
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fsp", required=True)
    parser.add_argument("--case-result", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    fsp = Path(args.fsp).expanduser().resolve()
    case_result_path = Path(args.case_result).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not fsp.is_file() or not case_result_path.is_file():
        raise FileNotFoundError("completed FSP and raw case result are required")
    if output_dir != fsp.parent:
        raise RuntimeError("recovery output must remain beside the source FSP")

    raw = json.loads(case_result_path.read_text(encoding="utf-8"))
    if raw.get("status") != "BLOCKED_EXECUTION_ERROR":
        raise RuntimeError("recovery is restricted to the postprocess failure")
    if "Q sample coordinate lies outside realized flux faces" not in str(
        raw.get("exception", "")
    ):
        raise RuntimeError("raw failure is not the known staggered-grid error")

    base = runner.load_base()
    base.TARGET_WAVELENGTH_M = runner.WAVELENGTH_M
    base.TARGET_FREQUENCY_HZ = runner.C0 / runner.WAVELENGTH_M
    base.FLAKE_THICKNESS_M = runner.FLAKE_THICKNESS_M

    os.environ["VC_LUMERICAL_ROOT"] = str(runner.APPROVED_ROOT)
    os.environ["LUMERICAL_ROOT"] = str(runner.APPROVED_ROOT)
    os.environ["LUMERICAL_PYTHONPATH"] = str(runner.APPROVED_API)
    if str(runner.APPROVED_API) not in sys.path:
        sys.path.insert(0, str(runner.APPROVED_API))
    installation = SimpleNamespace(
        version_key="v261",
        root=runner.APPROVED_ROOT.resolve(),
        lumapi_path=(runner.APPROVED_API / "lumapi.py").resolve(),
        device_executable=(
            runner.APPROVED_ROOT / "bin" / "device"
        ).resolve(),
    )
    lumapi = base.load_lumapi(installation)

    faces = {
        f"{axis}_{side}": {
            "name": f"paper_ir_abs_{axis}_{side}",
            "axis": axis,
            "side": side,
            "outward_sign": -1.0 if side == "min" else 1.0,
        }
        for axis in "xyz"
        for side in ("min", "max")
    }
    fdtd = None
    try:
        fdtd = lumapi.FDTD(hide=True, serverArgs={"platform": "offscreen"})
        fdtd.load(str(fsp))
        source_power = base.scalar(
            fdtd.sourcepower(
                base.TARGET_FREQUENCY_HZ,
                2,
                base.SOURCE_NAME,
            ),
            "saved native source power",
        )
        six_face = base.face_fluxes(fdtd, faces, source_power, 1.0)
        realized = runner.realized_six_face_control_volume(fdtd, faces)
        bounds = realized["bounds_m"]
        auto_shutoff = runner.final_logged_auto_shutoff(output_dir)

        q_data = base.common_grid_component_q(
            fdtd,
            base.TARGET_FREQUENCY_HZ,
        )
        coordinates = {
            axis: np.asarray(q_data[f"{axis}_m"], float)
            for axis in "xyz"
        }
        common = {
            axis: np.asarray(q_data[f"Q{axis}_native_W_m3"], float)
            for axis in "xyz"
        }
        common_total = common["x"] + common["y"] + common["z"]
        common_power = {
            axis: runner.integrate_xyz_bounded(
                common[axis],
                coordinates,
                bounds,
            )
            for axis in "xyz"
        }
        native = runner.native_component_absorption_bounded(
            base,
            fdtd,
            base.TARGET_FREQUENCY_HZ,
            bounds,
        )
        native_power = native["component_power_W"]
        p_common = float(sum(common_power.values()))
        p_native = float(native["total_power_W"])
        p_six = float(six_face["net_inward_power_W"])
        denominator = max(abs(p_six), np.finfo(float).tiny)

        artifact_path = output_dir / "diagnostic_q_common_grid_artifact.npz"
        np.savez(
            artifact_path,
            x_m=coordinates["x"],
            y_m=coordinates["y"],
            z_m=coordinates["z"],
            Qx_common_grid_W_m3=common["x"],
            Qy_common_grid_W_m3=common["y"],
            Qz_common_grid_W_m3=common["z"],
            Q_common_grid_W_m3=common_total,
            source_power_native_W=np.asarray([source_power]),
            P_Q_common_grid_bounded_W=np.asarray([p_common]),
            P_Q_native_component_bounded_W=np.asarray([p_native]),
            P_six_native_W=np.asarray([p_six]),
            **{
                f"realized_control_volume_{axis}_bounds_m": np.asarray(
                    bounds[axis],
                    float,
                )
                for axis in "xyz"
            },
        )

        hotspot_index = np.unravel_index(
            int(np.argmax(common_total)),
            common_total.shape,
        )
        payload = {
            "status": (
                "FAILED_MATCHED_CONTROL_VOLUME_SMOKE_AUTO_SHUTOFF_UNRESOLVED"
            ),
            "validated": False,
            "classification": (
                "READ_ONLY_RECOVERY_OF_ONE_COMPLETED_DIAGNOSTIC_GPU_SMOKE"
            ),
            "FDTD_solve_called": False,
            "runanalysis_called": False,
            "thermal_run": False,
            "PTE_run": False,
            "adjoint_run": False,
            "optimization_run": False,
            "raw_failure_preserved": {
                "case_result": str(case_result_path),
                "status": raw["status"],
                "exception": raw["exception"],
            },
            "input": {
                "fsp_path": str(fsp),
                "fsp_size_bytes": fsp.stat().st_size,
                "fsp_sha256": sha256(fsp),
                "case_result_path": str(case_result_path),
                "case_result_size_bytes": case_result_path.stat().st_size,
                "case_result_sha256": sha256(case_result_path),
                "generation_commit": raw.get("generation_commit"),
                "generation_command": raw.get("generation_command"),
            },
            "normalization": {
                "source_power_native_W": source_power,
                "incident_intensity_normalization_applied": False,
                "empirical_flux_gain": False,
            },
            "realized_control_volume": realized,
            "common_grid_dual_cell_contract": {
                axis: dual_cell_contract(coordinates[axis], bounds[axis])
                for axis in "xyz"
            },
            "component_power_common_grid_bounded_W": common_power,
            "component_power_native_Yee_bounded_W": native_power,
            "P_Q_common_grid_bounded_W": p_common,
            "P_Q_native_Yee_bounded_W": p_native,
            "P_six_face_native_W": p_six,
            "six_face_native_source_amplitude": six_face,
            "common_grid_six_face_relative_closure": (
                abs(p_common - p_six) / denominator
            ),
            "native_Yee_six_face_relative_closure": (
                abs(p_native - p_six) / denominator
            ),
            "common_native_total_relative_difference": (
                abs(p_common - p_native)
                / max(abs(p_native), np.finfo(float).tiny)
            ),
            "independent_field_index_pairing": native[
                "independent_field_index_pairing"
            ],
            "maximum_independent_field_index_coordinate_mismatch_m": (
                native["maximum_coordinate_mismatch_m"]
            ),
            "auto_shutoff": auto_shutoff,
            "Q_hotspot": {
                "x_m": float(coordinates["x"][hotspot_index[0]]),
                "y_m": float(coordinates["y"][hotspot_index[1]]),
                "z_m": float(coordinates["z"][hotspot_index[2]]),
                "Q_common_grid_W_m3": float(common_total[hotspot_index]),
            },
            "Q_operations": {
                "clipped": False,
                "smoothed": False,
                "gain_applied": False,
                "globally_rescaled": False,
                "tiled": False,
                "source_deleted": False,
            },
            "acceptance": {
                "solver_log_reports_successful_completion": auto_shutoff[
                    "simulation_completed_successfully"
                ],
                "auto_shutoff_reached_lt_1e_minus_5": (
                    auto_shutoff["final_value"] < 1.0e-5
                ),
                "common_grid_six_face_closure_lt_0p5_percent": (
                    abs(p_common - p_six) / denominator
                    < base.POWER_CLOSURE_LIMIT
                ),
                "native_Yee_six_face_closure_lt_0p5_percent": (
                    abs(p_native - p_six) / denominator
                    < base.POWER_CLOSURE_LIMIT
                ),
                "independent_coordinate_mismatch_lt_1fm": (
                    native["maximum_coordinate_mismatch_m"] < 1.0e-15
                ),
                "all_dual_cell_axes_close": all(
                    dual_cell_contract(
                        coordinates[axis],
                        bounds[axis],
                    )["dual_cell_support_closes_on_realized_faces"]
                    for axis in "xyz"
                ),
                "no_Q_clipping_smoothing_gain_rescaling_tiling_or_deletion": (
                    True
                ),
            },
            "artifact": {
                "path": str(artifact_path),
                "size_bytes": artifact_path.stat().st_size,
                "sha256": sha256(artifact_path),
            },
        }
        output_path = output_dir / "read_only_matched_smoke_recovery.json"
        write_json(output_path, payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
    finally:
        if fdtd is not None:
            fdtd.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
