#!/usr/bin/env python3
"""Fresh dual-polarization evaluation of one exact dispersive-Au cell mask."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Any

import numpy as np

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.contract import (
    CONTRACT,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.dfm import (
    exact_500nm_audit,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.lumerical_4um_gray_q_coupling import (
    GrayYeeQCoupling,
    component_coordinates_from_raw,
    component_q_from_raw,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.multiphysics_4um import (
    evaluate_fixed_source,
    thermal_edges,
)


HERE = Path(__file__).resolve().parent
REPOSITORY = Path(__file__).resolve().parents[3]
MESH_LABEL = "fine_z2p5_bulk50_xy100_cv0_pml8_span20_z6_t1ps"
SOURCE_OBJECT_W0_UM = 3.9561433030461415


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary-mask-npz", required=True, type=Path)
    parser.add_argument("--binary-mask-key", default="binary_mask")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--gpu-index", required=True, type=int)
    parser.add_argument(
        "--accelerator-policy", choices=("development", "b200"), required=True
    )
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--ea-source-calibration", required=True, type=Path)
    parser.add_argument("--eb-source-calibration", required=True, type=Path)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact(path: Path) -> dict[str, Any]:
    value = path.resolve()
    return {
        "path": str(value),
        "size_bytes": value.stat().st_size,
        "sha256": _sha256(value),
    }


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _matching_artifact(result: dict[str, Any], suffix: str) -> Path:
    matches = [
        Path(row["path"]).resolve()
        for row in result.get("raw_artifacts", [])
        if str(row.get("path", "")).endswith(suffix)
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one exact-forward {suffix} artifact")
    return matches[0]


def _forward_command(
    *,
    args: argparse.Namespace,
    polarization: str,
    source_calibration: Path,
    output: Path,
) -> list[str]:
    return [
        sys.executable,
        str(HERE / "25_run_lumerical_4um_exact_au_control.py"),
        "--case",
        "exact_binary",
        "--binary-mask-file",
        str(args.binary_mask_npz.resolve()),
        "--binary-mask-key",
        args.binary_mask_key,
        "--polarization",
        polarization,
        "--gpu-index",
        str(args.gpu_index),
        "--accelerator-policy",
        args.accelerator_policy,
        "--output-dir",
        str(output),
        "--source-calibration-json",
        str(source_calibration.resolve()),
        "--source-object-w0-um",
        str(SOURCE_OBJECT_W0_UM),
        "--mesh-label",
        MESH_LABEL,
        "--flake-dxy-nm",
        "100",
        "--stack-dz-nm",
        "2.5",
        "--bulk-dz-nm",
        "50",
        "--outer-dxy-nm",
        "200",
        "--mesh-accuracy",
        "3",
        "--au-max-coefficients",
        "6",
        "--au-fit-tolerance",
        "0",
        "--mesh-refinement",
        "conformal variant 0",
        "--pml-layers",
        "8",
        "--lateral-span-um",
        "20",
        "--z-min-um",
        "-3",
        "--z-max-um",
        "3",
        "--simulation-time-ps",
        "1",
        "--auto-shutoff-min",
        "1e-7",
        "--threads",
        str(args.threads),
    ]


def _evaluate_polarization(
    *,
    args: argparse.Namespace,
    polarization: str,
    source_calibration: Path,
    mask: np.ndarray,
    output: Path,
) -> dict[str, Any]:
    output.mkdir(parents=True)
    command = _forward_command(
        args=args,
        polarization=polarization,
        source_calibration=source_calibration,
        output=output,
    )
    log_path = output.parent / f"exact_forward_{polarization}.log"
    with log_path.open("w", encoding="utf-8", errors="replace") as stream:
        completed = subprocess.run(
            command,
            cwd=REPOSITORY,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0:
        tail = "\n".join(
            log_path.read_text(errors="replace").splitlines()[-80:]
        )
        raise RuntimeError(
            f"{polarization} exact forward failed with exit {completed.returncode}\n{tail}"
        )
    json_paths = sorted(output.glob("*.json"))
    if len(json_paths) != 1:
        raise RuntimeError(f"expected one {polarization} exact-forward JSON")
    forward = json.loads(json_paths[0].read_text(encoding="utf-8"))
    if (
        forward.get("all_gates_passed") is not True
        or forward.get("case") != "exact_binary"
        or forward.get("polarization") != polarization
    ):
        raise RuntimeError(f"{polarization} exact-forward gates failed")
    raw_path = _matching_artifact(forward, "_raw.npz")
    with np.load(raw_path, allow_pickle=False) as raw:
        coupling = GrayYeeQCoupling.from_component_coordinates(
            component_coordinates_from_raw(raw), thermal_edges()
        )
        q = component_q_from_raw(raw)
        source_power_raw, mapping = coupling.map_power(q)
    reporting_scale = float(
        forward["reporting_normalization"]["scalar_reporting_factor"]
    )
    source_power = source_power_raw * reporting_scale
    evaluated = evaluate_fixed_source(
        np.asarray(mask, dtype=np.float64),
        source_power,
        0,
        need_gradient=False,
    )
    thermal = evaluated["thermal_audit"]
    electrical = evaluated["electrical_audit"]
    gates = {
        "ordinary_dispersive_exact_Au_forward": True,
        "Q_mapping_conservation_lt_1e_12": float(
            mapping["relative_power_error"]
        )
        < 1.0e-12,
        "thermal_residual_lt_1e_8": float(thermal["relative_residual"])
        < 1.0e-8,
        "thermal_energy_balance_lt_1pct": float(
            thermal["energy_balance_relative"]
        )
        < 1.0e-2,
        "electrical_residual_lt_1e_8": float(
            electrical["relative_residual"]
        )
        < 1.0e-8,
        "electrical_terminal_balance_lt_1pct": float(
            electrical["terminal_balance_relative"]
        )
        < 1.0e-2,
        "finite_nonzero_current": bool(
            np.isfinite(evaluated["objective_A"])
            and float(evaluated["objective_A"]) != 0.0
        ),
    }
    return {
        "passed": all(gates.values()),
        "polarization": polarization,
        "current_A": float(evaluated["objective_A"]),
        "current_nA": 1.0e9 * float(evaluated["objective_A"]),
        "mapped_source_power_W_reporting": float(np.sum(source_power)),
        "mapping": mapping,
        "thermal": thermal,
        "electrical": electrical,
        "gates": gates,
        "forward_result": _artifact(json_paths[0]),
        "forward_raw": _artifact(raw_path),
        "forward_log": _artifact(log_path),
    }


def main() -> int:
    args = _parse_args()
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing non-empty exact-binary output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "exact_binary_dualpol_result.json"
    result: dict[str, Any] = {
        "status": "FAILED_LUMERICAL_4UM_EXACT_BINARY_DUALPOL_EVALUATION",
        "passed": False,
        "Lumerical_HEAT_or_CHARGE_solves": 0,
        "FDTDX_Maxwell_solves": 0,
    }
    started = time.monotonic()
    try:
        with np.load(args.binary_mask_npz, allow_pickle=False) as data:
            mask = np.asarray(data[args.binary_mask_key])
        if (
            mask.shape != CONTRACT.design_shape
            or not np.all((mask == 0) | (mask == 1))
        ):
            raise ValueError("binary candidate must be exact 80x80 zero/one")
        mask = np.asarray(mask, dtype=np.uint8)
        exact = exact_500nm_audit(
            mask,
            spacing_m=CONTRACT.design_pitch_m,
            minimum_feature_m=250.0e-9,
        )
        if not bool(exact["solid_pass"] and exact["void_pass"]):
            raise RuntimeError("exact mask failed 250 nm solid/void audit")
        rows = {
            "Ea": _evaluate_polarization(
                args=args,
                polarization="Ea",
                source_calibration=args.ea_source_calibration,
                mask=mask,
                output=output / "forward_Ea",
            ),
            "Eb": _evaluate_polarization(
                args=args,
                polarization="Eb",
                source_calibration=args.eb_source_calibration,
                mask=mask,
                output=output / "forward_Eb",
            ),
        }
        currents = {key: row["current_A"] for key, row in rows.items()}
        switching = bool(currents["Ea"] > 0.0 and currents["Eb"] < 0.0)
        numerical_pass = all(row["passed"] for row in rows.values())
        result = {
            "status": "PASSED_LUMERICAL_4UM_EXACT_BINARY_DUALPOL_EVALUATION"
            if numerical_pass
            else "FAILED_LUMERICAL_4UM_EXACT_BINARY_DUALPOL_EVALUATION",
            "passed": numerical_pass,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "material_geometry": (
                "ordinary dispersive Au rectangles coalesced from the exact 80x80 mask"
            ),
            "binary_mask": _artifact(args.binary_mask_npz.resolve()),
            "exact_250nm_audit": {
                key: value
                for key, value in exact.items()
                if key not in {"binary", "bad_solid", "bad_void"}
            },
            "currents_A": currents,
            "currents_nA": {key: 1.0e9 * value for key, value in currents.items()},
            "balanced_utility_A": min(currents["Ea"], -currents["Eb"]),
            "opposite_current_switching_achieved": switching,
            "polarizations": rows,
            "Maxwell_solver": "Lumerical FDTD 2026 R1.2 build 4522",
            "Lumerical_Maxwell_solves": {"forward": 2, "adjoint": 0},
            "custom_CUDA_thermal_solves": {"forward": 2, "adjoint": 0},
            "custom_CUDA_electrical_solves": {"forward": 2, "adjoint": 0},
            "Lumerical_HEAT_or_CHARGE_solves": 0,
            "FDTDX_Maxwell_solves": 0,
            "wall_s": time.monotonic() - started,
        }
    except Exception as error:
        result.update(
            error=f"{type(error).__name__}: {error}",
            traceback=traceback.format_exc(),
            wall_s=time.monotonic() - started,
        )
    _write_json(result_path, result)
    print(json.dumps(result, indent=2, default=str))
    return 0 if result.get("passed") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
