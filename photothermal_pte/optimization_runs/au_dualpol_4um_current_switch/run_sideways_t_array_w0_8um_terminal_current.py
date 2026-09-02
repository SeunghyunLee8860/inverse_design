#!/usr/bin/env python3
"""Forward-only 4-um sideways-T array terminal-current experiment.

This is a separate, non-optimizing scenario.  Maxwell propagation is solved
with Lumerical FDTD and the absorbed power is passed to the custom CUDA 3-D
thermal/electrical PDE.  It never invokes FDTDX or Lumerical HEAT/CHARGE.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import numpy as np


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch import (
    contract as contract_module,
)


GPU_INDEX_DEFAULT = 6
SOURCE_OBJECT_W0_UM = 7.99
FLAKE_SPAN_UM = 24.0
DESIGN_SPAN_UM = 18.0
ARRAY_NX = 15
ARRAY_NY = 11
ARRAY_PITCH_X_UM = 1.0
ARRAY_PITCH_Y_UM = 1.5
OUTPUT_DEFAULT = Path(
    "/home/seunghyun200/tairte4_raw_artifacts/"
    "sideways_t_array_w0_8um_100uW"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    temporary.replace(path)


def scenario_contract():
    return replace(
        contract_module.CONTRACT,
        gaussian_waist_m=8.0e-6,
        optical_lateral_span_m=30.0e-6,
        source_aperture_span_m=24.0e-6,
        reporting_incident_power_W=100.0e-6,
        flake_span_x_m=FLAKE_SPAN_UM * 1.0e-6,
        flake_span_y_m=FLAKE_SPAN_UM * 1.0e-6,
        design_span_x_m=DESIGN_SPAN_UM * 1.0e-6,
        design_span_y_m=DESIGN_SPAN_UM * 1.0e-6,
        measurement_electrode_span_y_m=FLAKE_SPAN_UM * 1.0e-6,
        measurement_electrode_contact_S_m2=1.0e13,
        electrical_contact_S_m2=1.0e13,
        final_geometry_identity=(
            "exact 15x11 right-facing T array, 100-nm raster, separated "
            "from fixed left/right top-Au electrodes"
        ),
    )


def build_mask(contract) -> tuple[np.ndarray, dict[str, Any]]:
    nx, ny = contract.design_shape
    if (nx, ny) != (180, 180):
        raise RuntimeError(f"unexpected custom design shape: {(nx, ny)}")
    x = (np.arange(nx) + 0.5) * contract.design_pitch_m
    y = (np.arange(ny) + 0.5) * contract.design_pitch_m
    x -= 0.5 * contract.design_span_x_m
    y -= 0.5 * contract.design_span_y_m
    xx, yy = np.meshgrid(x, y, indexing="ij")
    mask = np.zeros((nx, ny), dtype=bool)
    centers_x = (
        np.arange(ARRAY_NX) - 0.5 * (ARRAY_NX - 1)
    ) * ARRAY_PITCH_X_UM * 1.0e-6
    centers_y = (
        np.arange(ARRAY_NY) - 0.5 * (ARRAY_NY - 1)
    ) * ARRAY_PITCH_Y_UM * 1.0e-6
    # Clockwise-rotated paper-digitized T: vertical bar on the left and arm
    # pointing right, exactly matching the orientation in the supplied image.
    for center_x in centers_x:
        for center_y in centers_y:
            vertical = (
                (xx >= center_x - 0.40e-6 - 1e-15)
                & (xx < center_x - 0.20e-6 - 1e-15)
                & (yy >= center_y - 0.60e-6 - 1e-15)
                & (yy < center_y + 0.60e-6 - 1e-15)
            )
            arm = (
                (xx >= center_x - 0.20e-6 - 1e-15)
                & (xx < center_x + 0.40e-6 - 1e-15)
                & (yy >= center_y - 0.10e-6 - 1e-15)
                & (yy < center_y + 0.10e-6 - 1e-15)
            )
            mask |= vertical | arm
    occupied = np.argwhere(mask)
    x_min = float(x[occupied[:, 0]].min() - 0.5 * contract.design_pitch_m)
    x_max = float(x[occupied[:, 0]].max() + 0.5 * contract.design_pitch_m)
    y_min = float(y[occupied[:, 1]].min() - 0.5 * contract.design_pitch_m)
    y_max = float(y[occupied[:, 1]].max() + 0.5 * contract.design_pitch_m)
    electrode_inner_x = 0.5 * contract.flake_span_x_m - (
        contract.measurement_electrode_overlap_x_m
    )
    geometry = {
        "orientation": "right-facing sideways T (clockwise 90-degree rotation)",
        "array_count": int(ARRAY_NX * ARRAY_NY),
        "array_shape": [ARRAY_NX, ARRAY_NY],
        "array_pitch_um": [ARRAY_PITCH_X_UM, ARRAY_PITCH_Y_UM],
        "single_T_dimensions_um": {
            "vertical_bar_height": 1.2,
            "vertical_bar_width": 0.2,
            "right_arm_length_from_bar_left": 0.6,
            "right_arm_width": 0.2,
        },
        "raster_pitch_nm": contract.design_pitch_m * 1.0e9,
        "mask_shape": list(mask.shape),
        "occupied_cell_count": int(np.count_nonzero(mask)),
        "occupied_area_um2": float(np.count_nonzero(mask)) * 0.01,
        "array_physical_bounds_um": [
            x_min * 1.0e6,
            x_max * 1.0e6,
            y_min * 1.0e6,
            y_max * 1.0e6,
        ],
        "minimum_x_gap_array_to_electrode_um": (
            electrode_inner_x - max(abs(x_min), abs(x_max))
        )
        * 1.0e6,
    }
    if geometry["minimum_x_gap_array_to_electrode_um"] <= 0.0:
        raise RuntimeError("T array touches a measurement electrode")
    return mask.astype(np.uint8), geometry


def plot_geometry(path: Path, mask: np.ndarray, contract, geometry: dict) -> None:
    fig, axis = plt.subplots(figsize=(10, 9), constrained_layout=True)
    extent = (-9.0, 9.0, -9.0, 9.0)
    axis.imshow(mask.T, origin="lower", extent=extent, cmap="YlOrBr", alpha=0.9)
    half_flake = 0.5 * contract.flake_span_x_m * 1.0e6
    overlap = contract.measurement_electrode_overlap_x_m * 1.0e6
    axis.add_patch(
        Rectangle(
            (-half_flake, -half_flake), overlap, 2 * half_flake,
            color="#355cde", alpha=0.85, label="left Au electrode",
        )
    )
    axis.add_patch(
        Rectangle(
            (half_flake - overlap, -half_flake), overlap, 2 * half_flake,
            color="#d13a32", alpha=0.85, label="right Au electrode",
        )
    )
    axis.add_patch(
        Rectangle(
            (-half_flake, -half_flake), 2 * half_flake, 2 * half_flake,
            fill=False, edgecolor="black", linewidth=2.0, label="TaIrTe4 flake",
        )
    )
    axis.add_patch(
        Circle(
            (0.0, 0.0), 8.0, fill=False, edgecolor="#00a36c",
            linewidth=2.2, linestyle="--", label="Gaussian w0=8 um",
        )
    )
    axis.set_xlim(-13, 13)
    axis.set_ylim(-13, 13)
    axis.set_aspect("equal")
    axis.set_xlabel("x=b (um), left/right terminal direction")
    axis.set_ylabel("y=a (um)")
    axis.set_title(
        f"{geometry['array_count']} right-facing floating Au T elements; "
        "24 um flake, w0=8 um"
    )
    axis.legend(loc="upper center", ncol=2)
    fig.savefig(path, dpi=220, facecolor="white")
    plt.close(fig)


def one_json(directory: Path) -> tuple[Path, dict[str, Any]]:
    paths = sorted(directory.glob("*.json"))
    if len(paths) != 1:
        raise RuntimeError(f"expected one result JSON in {directory}: {paths}")
    return paths[0], json.loads(paths[0].read_text())


def run_lumerical_case(runner, output: Path, arguments: list[str]) -> tuple[Path, dict]:
    attempt = 0
    while True:
        destination = output / f"attempt_{attempt:03d}"
        argv = arguments + ["--output-dir", str(destination)]
        sys.argv = [str(HERE / "25_run_lumerical_4um_exact_au_control.py"), *argv]
        code = int(runner.main())
        result_path, result = one_json(destination)
        if code == 0 and result.get("all_gates_passed") is True:
            return result_path, result
        message = json.dumps(result).lower()
        transient = any(
            token in message
            for token in (
                "license", "resource", "activation", "users already reached",
                "failed to start", "solver exited with a non-zero exit code",
            )
        )
        if not transient:
            raise RuntimeError(
                f"non-transient Lumerical failure in {destination}: "
                f"{result.get('error', result.get('status'))}"
            )
        attempt += 1
        time.sleep(60.0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-index", type=int, default=GPU_INDEX_DEFAULT)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    output = args.output_root.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing non-empty output root: {output}")
    output.mkdir(parents=True, exist_ok=True)

    contract = scenario_contract()
    contract_module.CONTRACT = contract
    mask, geometry = build_mask(contract)
    mask_path = output / "sideways_t_array_exact_mask.npz"
    np.savez_compressed(mask_path, binary_mask=mask)
    geometry_plot = output / "sideways_t_array_geometry.png"
    plot_geometry(geometry_plot, mask, contract, geometry)
    scenario = {
        "status": "PREPARED_SIDEWAYS_T_ARRAY_W0_8UM_FORWARD_SCENARIO",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "physics": {
            "Maxwell": "Lumerical FDTD R1.2 build 4522 GPU",
            "thermal_electrical": "custom CUDA explicit 3-D volumetric PDE",
            "FDTDX": False,
            "Lumerical_HEAT_CHARGE": False,
        },
        "illumination": {
            "wavelength_um": 4.0,
            "target_waist_w0_um": 8.0,
            "w0_definition": "1/e^2 intensity radius at z=0",
            "incident_power_uW": 100.0,
            "propagation": "normal incidence along -z",
            "polarizations": ["Ea (Lumerical y)", "Eb (Lumerical x)"],
        },
        "device": {
            "flake_um": [24.0, 24.0, 0.1],
            "stack": "air / 50-nm top Au / 100-nm TaIrTe4 / 285-nm SiO2 / Si",
            "electrodes": "1-um-wide fixed Au strips at x-min/x-max",
            "terminal_sign": "positive conventional current along +x, left to right",
            "Au_Ta_electrical_contact_S_m2": 1.0e13,
            "Au_bulk_conductivity_S_m": contract.au_bulk_electrical_conductivity_S_m,
            "array": geometry,
        },
        "contract": contract.audit(),
        "artifacts": {"mask": artifact(mask_path), "geometry_plot": artifact(geometry_plot)},
    }
    write_json(output / "scenario.json", scenario)
    if args.prepare_only:
        print(json.dumps(scenario, indent=2, default=str))
        return 0

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)
    from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch import (
        lumerical_4um_forward,
    )
    from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch import (
        lumerical_4um_exact_au,
    )
    lumerical_4um_forward.CONTRACT = contract
    lumerical_4um_exact_au.CONTRACT = contract
    import importlib.util
    runner_path = HERE / "25_run_lumerical_4um_exact_au_control.py"
    spec = importlib.util.spec_from_file_location("sideways_t_lumerical_runner", runner_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {runner_path}")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)

    common = [
        "--gpu-index", str(args.gpu_index), "--accelerator-policy", "b200",
        "--source-object-w0-um", str(SOURCE_OBJECT_W0_UM),
        "--mesh-label", "sideways_t_w0_8um_xy100_z5_cv0_pml8_span30",
        "--flake-dxy-nm", "100", "--stack-dz-nm", "5",
        "--bulk-dz-nm", "50", "--outer-dxy-nm", "250",
        "--mesh-accuracy", "3", "--au-max-coefficients", "6",
        "--au-fit-tolerance", "0", "--mesh-refinement", "conformal variant 0",
        "--pml-layers", "8", "--lateral-span-um", "30",
        "--z-min-um", "-3", "--z-max-um", "3",
        "--simulation-time-ps", "1", "--auto-shutoff-min", "1e-7",
        "--threads", str(args.threads),
    ]
    source_results = {}
    for polarization in ("Ea", "Eb"):
        path, result = run_lumerical_case(
            runner,
            output / "source_calibration" / polarization,
            ["--case", "source_only", "--polarization", polarization, *common],
        )
        source_results[polarization] = (path, result)

    forward_results = {}
    for polarization in ("Ea", "Eb"):
        path, result = run_lumerical_case(
            runner,
            output / "maxwell" / polarization,
            [
                "--case",
                "exact_binary",
                "--binary-mask-file",
                str(mask_path),
                "--binary-mask-key",
                "binary_mask",
                "--polarization",
                polarization,
                "--source-calibration-json",
                str(source_results[polarization][0]),
                *common,
            ],
        )
        forward_results[polarization] = (path, result)

    from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch import (
        multiphysics_4um,
    )
    from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch import (
        volumetric_electrical_4um,
    )

    multiphysics_4um.CONTRACT = contract
    volumetric_electrical_4um.CONTRACT = contract
    evaluator_path = HERE / "42_evaluate_lumerical_4um_exact_binary.py"
    evaluator_spec = importlib.util.spec_from_file_location(
        "sideways_t_exact_evaluator", evaluator_path
    )
    if evaluator_spec is None or evaluator_spec.loader is None:
        raise RuntimeError(f"cannot load {evaluator_path}")
    evaluator = importlib.util.module_from_spec(evaluator_spec)
    evaluator_spec.loader.exec_module(evaluator)

    currents = {}
    pde_artifacts = {}
    for polarization, (result_path, result) in forward_results.items():
        raw_rows = [
            row
            for row in result["raw_artifacts"]
            if row["path"].endswith("_raw.npz")
        ]
        if len(raw_rows) != 1:
            raise RuntimeError(f"{polarization} raw Q artifact is ambiguous")
        raw_path = Path(raw_rows[0]["path"])
        with np.load(raw_path, allow_pickle=False) as raw:
            coordinates = evaluator.component_coordinates_from_raw(raw)
            q = evaluator.component_q_from_raw(raw)
        reporting_scale = float(
            result["reporting_normalization"]["scalar_reporting_factor"]
        )
        public, arrays = evaluator._pde_resolution(
            mask=mask,
            component_coordinates=coordinates,
            q=q,
            reporting_scale=reporting_scale,
            core_step_m=100.0e-9,
        )
        raw_pde = output / f"{polarization}_custom_cuda_pde.npz"
        np.savez_compressed(raw_pde, **arrays)
        currents[polarization] = public
        pde_artifacts[polarization] = artifact(raw_pde)

    summary = {
        **scenario,
        "status": (
            "PASSED_SIDEWAYS_T_ARRAY_W0_8UM_TERMINAL_CURRENT"
            if all(row["passed"] for row in currents.values())
            else "FAILED_SIDEWAYS_T_ARRAY_W0_8UM_TERMINAL_CURRENT"
        ),
        "GPU_index": args.gpu_index,
        "source_calibration": {
            key: {
                "result": artifact(value[0]),
                "realized_w0_um": value[1]["target_plane_metrics"][
                    "fitted_waist_effective_m"
                ]
                * 1.0e6,
            }
            for key, value in source_results.items()
        },
        "terminal_currents": currents,
        "forward_results": {
            key: artifact(value[0]) for key, value in forward_results.items()
        },
        "pde_artifacts": pde_artifacts,
    }
    write_json(output / "terminal_current_summary.json", summary)
    print(json.dumps(summary, indent=2, default=str))
    return 0 if summary["status"].startswith("PASSED") else 2


if __name__ == "__main__":
    raise SystemExit(main())
