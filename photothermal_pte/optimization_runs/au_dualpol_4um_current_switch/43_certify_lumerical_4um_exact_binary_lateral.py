#!/usr/bin/env python3
"""Certify one exact Au mask on 100/50-nm Lumerical lateral meshes.

This is the terminal numerical certificate for the Lumerical continuation.
It runs four fresh exact-binary Maxwell forwards (Ea/Eb on the 100-nm and
50-nm flake/design meshes), requires both 100-to-50-nm Maxwell comparisons,
and only then evaluates the 50-nm raw Q through the adaptive custom-CUDA
thermal/electrical mesh-convergence path in script 42.
"""

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
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.lumerical_4um_control_comparison import (
    compare_control_pair,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.lumerical_4um_mesh_contract import (
    LumericalMeshSpec,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.lumerical_maxwell_contract import (
    binary_mask_sha256,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.objective import (
    opposite_current_switching_achieved,
)


HERE = Path(__file__).resolve().parent
REPOSITORY = Path(__file__).resolve().parents[3]
SOURCE_OBJECT_W0_UM = 3.9561433030461415
COARSE_MESH_LABEL = "fine_z2p5_bulk50_xy100_cv0_pml8_span20_z6_t1ps"
FINE_MESH_LABEL = "fine_z2p5_bulk50_xy50_cv0_pml8_span20_z6_t1ps"


def _mesh(label: str, flake_dxy_nm: float) -> LumericalMeshSpec:
    return LumericalMeshSpec(
        label=label,
        flake_dxy_m=flake_dxy_nm * 1.0e-9,
        stack_dz_m=2.5e-9,
        bulk_dz_m=50.0e-9,
        outer_dxy_m=200.0e-9,
        mesh_accuracy=3,
        pml_layers=8,
        lateral_span_m=20.0e-6,
        z_min_m=-3.0e-6,
        z_max_m=3.0e-6,
        simulation_time_s=1.0e-12,
        auto_shutoff_min=1.0e-7,
        conformal_mesh="conformal variant 0",
    )


COARSE_MESH = _mesh(COARSE_MESH_LABEL, 100.0)
FINE_MESH = _mesh(FINE_MESH_LABEL, 50.0)


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
    parser.add_argument("--ea-coarse-source-calibration", required=True, type=Path)
    parser.add_argument("--eb-coarse-source-calibration", required=True, type=Path)
    parser.add_argument("--ea-fine-source-calibration", required=True, type=Path)
    parser.add_argument("--eb-fine-source-calibration", required=True, type=Path)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact(path: Path) -> dict[str, Any]:
    value = path.expanduser().resolve()
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


def _git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "UNKNOWN"


def _matching_raw(result: dict[str, Any]) -> Path:
    records = [
        row
        for row in result.get("raw_artifacts", [])
        if str(row.get("path", "")).endswith("_raw.npz")
    ]
    if len(records) != 1:
        raise RuntimeError("exact forward must name exactly one raw NPZ")
    record = records[0]
    path = Path(record["path"]).expanduser().resolve()
    if not path.is_file():
        raise RuntimeError(f"missing exact-forward raw NPZ: {path}")
    if int(record.get("size_bytes", -1)) != path.stat().st_size:
        raise RuntimeError("exact-forward raw NPZ size changed")
    if str(record.get("sha256", "")) != _sha256(path):
        raise RuntimeError("exact-forward raw NPZ SHA256 changed")
    return path


def _forward_command(
    *,
    args: argparse.Namespace,
    polarization: str,
    source_calibration: Path,
    mesh: LumericalMeshSpec,
    output: Path,
) -> list[str]:
    audit = mesh.audit()
    return [
        sys.executable,
        str(HERE / "25_run_lumerical_4um_exact_au_control.py"),
        "--case",
        "exact_binary",
        "--binary-mask-file",
        str(args.binary_mask_npz.expanduser().resolve()),
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
        str(source_calibration.expanduser().resolve()),
        "--source-object-w0-um",
        str(SOURCE_OBJECT_W0_UM),
        "--mesh-label",
        str(audit["label"]),
        "--flake-dxy-nm",
        str(float(audit["flake_dxy_m"]) * 1.0e9),
        "--stack-dz-nm",
        str(float(audit["stack_dz_m"]) * 1.0e9),
        "--bulk-dz-nm",
        str(float(audit["bulk_dz_m"]) * 1.0e9),
        "--outer-dxy-nm",
        str(float(audit["outer_dxy_m"]) * 1.0e9),
        "--mesh-accuracy",
        str(audit["mesh_accuracy"]),
        "--au-max-coefficients",
        "6",
        "--au-fit-tolerance",
        "0",
        "--mesh-refinement",
        str(audit["conformal_mesh"]),
        "--pml-layers",
        str(audit["pml_layers"]),
        "--lateral-span-um",
        str(float(audit["lateral_span_m"]) * 1.0e6),
        "--z-min-um",
        str(float(audit["z_min_m"]) * 1.0e6),
        "--z-max-um",
        str(float(audit["z_max_m"]) * 1.0e6),
        "--simulation-time-ps",
        str(float(audit["simulation_time_s"]) * 1.0e12),
        "--auto-shutoff-min",
        str(audit["auto_shutoff_min"]),
        "--threads",
        str(args.threads),
    ]


def _run_forward(
    *,
    args: argparse.Namespace,
    polarization: str,
    source_calibration: Path,
    mesh: LumericalMeshSpec,
    output: Path,
) -> dict[str, Any]:
    output.mkdir(parents=True)
    command = _forward_command(
        args=args,
        polarization=polarization,
        source_calibration=source_calibration,
        mesh=mesh,
        output=output,
    )
    log_path = output.parent / f"{output.name}.log"
    with log_path.open("w", encoding="utf-8", errors="replace") as stream:
        completed = subprocess.run(
            command,
            cwd=REPOSITORY,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0:
        tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-80:])
        raise RuntimeError(
            f"{output.name} failed with exit {completed.returncode}\n{tail}"
        )
    json_paths = sorted(output.glob("*.json"))
    if len(json_paths) != 1:
        raise RuntimeError(f"{output.name} produced {len(json_paths)} JSON files")
    result_path = json_paths[0].resolve()
    result = json.loads(result_path.read_text(encoding="utf-8"))
    geometry = (
        result.get("layout", {}).get("geometry", {}).get("exact_au_geometry", {})
    )
    processing = result.get("Q_processing", {})
    gates = {
        "all_forward_gates_passed": result.get("all_gates_passed") is True,
        "exact_binary_case": result.get("case") == "exact_binary",
        "polarization_matches": result.get("polarization") == polarization,
        "mesh_spec_matches": result.get("mesh_spec") == mesh.audit(),
        "accelerator_policy_matches": result.get("accelerator_policy")
        == args.accelerator_policy,
        "git_commit_matches": result.get("git_commit") == _git_commit(),
        "binary_mask_matches": geometry.get("mask_payload_sha256")
        == binary_mask_sha256(_load_mask(args)),
        "raw_Q_unmodified": all(
            processing.get(name) is False
            for name in ("clipping", "smoothing", "gain", "field_or_Q_rescaling")
        ),
    }
    if not all(gates.values()):
        raise RuntimeError(f"{output.name} provenance gates failed: {gates}")
    raw = _matching_raw(result)
    return {
        "result": result,
        "result_path": result_path,
        "raw_path": raw,
        "log_path": log_path.resolve(),
        "gates": gates,
    }


def _load_mask(args: argparse.Namespace) -> np.ndarray:
    with np.load(args.binary_mask_npz.expanduser().resolve(), allow_pickle=False) as data:
        mask = np.asarray(data[args.binary_mask_key])
    if mask.shape != CONTRACT.design_shape or not np.all((mask == 0) | (mask == 1)):
        raise ValueError("binary candidate must be exact 80x80 zero/one")
    return np.ascontiguousarray(mask, dtype=np.uint8)


def _fine_pde_command(
    *,
    args: argparse.Namespace,
    mask_path: Path,
    forwards: dict[str, dict[str, dict[str, Any]]],
    output: Path,
) -> list[str]:
    return [
        sys.executable,
        str(HERE / "42_evaluate_lumerical_4um_exact_binary.py"),
        "--binary-mask-npz",
        str(mask_path),
        "--binary-mask-key",
        args.binary_mask_key,
        "--output-dir",
        str(output),
        "--gpu-index",
        str(args.gpu_index),
        "--accelerator-policy",
        args.accelerator_policy,
        "--threads",
        str(args.threads),
        "--ea-forward-result",
        str(forwards["fine"]["Ea"]["result_path"]),
        "--ea-raw-npz",
        str(forwards["fine"]["Ea"]["raw_path"]),
        "--eb-forward-result",
        str(forwards["fine"]["Eb"]["result_path"]),
        "--eb-raw-npz",
        str(forwards["fine"]["Eb"]["raw_path"]),
        "--mesh-label",
        FINE_MESH_LABEL,
        "--flake-dxy-nm",
        "50",
        "--outer-dxy-nm",
        "200",
        "--stack-dz-nm",
        "2.5",
        "--bulk-dz-nm",
        "50",
    ]


def main() -> int:
    args = _parse_args()
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing non-empty final-certificate output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "final_exact_binary_certificate.json"
    result: dict[str, Any] = {
        "schema": "au-lumerical-exact-binary-lateral-pde-certificate-v1",
        "status": "FAILED_LUMERICAL_4UM_EXACT_BINARY_LATERAL_PDE_CERTIFICATE",
        "passed": False,
        "git_commit": _git_commit(),
        "Lumerical_HEAT_or_CHARGE_solves": 0,
        "FDTDX_Maxwell_solves": 0,
    }
    started = time.monotonic()
    try:
        mask = _load_mask(args)
        exact = exact_500nm_audit(
            mask,
            spacing_m=CONTRACT.design_pitch_m,
            minimum_feature_m=250.0e-9,
        )
        if not bool(exact["solid_pass"] and exact["void_pass"]):
            raise RuntimeError("exact mask failed 250 nm solid/void audit")
        sources = {
            "coarse": {
                "Ea": args.ea_coarse_source_calibration,
                "Eb": args.eb_coarse_source_calibration,
            },
            "fine": {
                "Ea": args.ea_fine_source_calibration,
                "Eb": args.eb_fine_source_calibration,
            },
        }
        for group in sources.values():
            for path in group.values():
                if not path.expanduser().resolve().is_file():
                    raise FileNotFoundError(f"missing source calibration: {path}")
        meshes = {"coarse": COARSE_MESH, "fine": FINE_MESH}
        forwards: dict[str, dict[str, dict[str, Any]]] = {}
        for mesh_name in ("coarse", "fine"):
            forwards[mesh_name] = {}
            for polarization in ("Ea", "Eb"):
                forwards[mesh_name][polarization] = _run_forward(
                    args=args,
                    polarization=polarization,
                    source_calibration=sources[mesh_name][polarization],
                    mesh=meshes[mesh_name],
                    output=output / f"{mesh_name}_{polarization}_exact_forward",
                )

        comparisons = {
            polarization: compare_control_pair(
                forwards["coarse"][polarization]["result_path"],
                forwards["fine"][polarization]["result_path"],
                refinement_axis="xy",
            )
            for polarization in ("Ea", "Eb")
        }
        comparison_artifacts: dict[str, dict[str, Any]] = {}
        for polarization, comparison in comparisons.items():
            path = output / f"{polarization}_xy100_to_xy50_maxwell_comparison.json"
            _write_json(path, comparison)
            comparison_artifacts[polarization] = _artifact(path)
        optical_passed = all(
            comparison["all_gates_passed"] for comparison in comparisons.values()
        )

        fine_pde_result: dict[str, Any] | None = None
        fine_pde_artifact: dict[str, Any] | None = None
        fine_pde_returncode: int | None = None
        if optical_passed:
            fine_pde_output = output / "fine_xy50_adaptive_pde"
            command = _fine_pde_command(
                args=args,
                mask_path=args.binary_mask_npz.expanduser().resolve(),
                forwards=forwards,
                output=fine_pde_output,
            )
            log_path = output / "fine_xy50_adaptive_pde.log"
            with log_path.open("w", encoding="utf-8", errors="replace") as stream:
                completed = subprocess.run(
                    command,
                    cwd=REPOSITORY,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            fine_pde_returncode = completed.returncode
            fine_pde_path = fine_pde_output / "exact_binary_dualpol_result.json"
            if not fine_pde_path.is_file():
                tail = "\n".join(
                    log_path.read_text(errors="replace").splitlines()[-80:]
                )
                raise RuntimeError(
                    "fine adaptive PDE evaluator produced no result "
                    f"(exit {completed.returncode})\n{tail}"
                )
            fine_pde_result = json.loads(
                fine_pde_path.read_text(encoding="utf-8")
            )
            fine_pde_artifact = _artifact(fine_pde_path)

        numerical_pde_passed = bool(
            fine_pde_result is not None
            and fine_pde_returncode == 0
            and fine_pde_result.get("passed") is True
        )
        currents = (
            fine_pde_result.get("currents_A", {})
            if fine_pde_result is not None
            else {}
        )
        switching = bool(
            set(currents) == {"Ea", "Eb"}
            and opposite_current_switching_achieved(
                float(currents["Ea"]), float(currents["Eb"])
            )
        )
        passed = bool(optical_passed and numerical_pde_passed and switching)
        b200_forward_gate = bool(
            all(
                row["result"].get("B200_promotion_certified") is True
                for group in forwards.values()
                for row in group.values()
            )
        )
        forward_evidence = {
            mesh_name: {
                polarization: {
                    "result": _artifact(row["result_path"]),
                    "raw": _artifact(row["raw_path"]),
                    "log": _artifact(row["log_path"]),
                    "provenance_gates": row["gates"],
                }
                for polarization, row in group.items()
            }
            for mesh_name, group in forwards.items()
        }
        result.update(
            status=(
                "PASSED_LUMERICAL_4UM_EXACT_BINARY_LATERAL_PDE_NUMERICAL_CERTIFICATE"
                if passed
                else "FAILED_LUMERICAL_4UM_EXACT_BINARY_LATERAL_PDE_CERTIFICATE"
            ),
            passed=passed,
            generated_at_utc=datetime.now(timezone.utc).isoformat(),
            accelerator_policy=args.accelerator_policy,
            B200_promotion_certified=bool(passed and b200_forward_gate),
            binary_mask=_artifact(args.binary_mask_npz.expanduser().resolve()),
            binary_mask_payload_sha256=binary_mask_sha256(mask),
            exact_250nm_audit={
                key: value
                for key, value in exact.items()
                if key not in {"binary", "bad_solid", "bad_void"}
            },
            optical_lateral_meshes={
                "coarse": COARSE_MESH.audit(),
                "fine": FINE_MESH.audit(),
            },
            optical_lateral_comparisons=comparisons,
            optical_lateral_comparison_artifacts=comparison_artifacts,
            both_polarizations_optical_xy_converged=optical_passed,
            fine_xy50_adaptive_PDE_result=fine_pde_result,
            fine_xy50_adaptive_PDE_result_artifact=fine_pde_artifact,
            numerical_PDE_and_sign_gates_passed=numerical_pde_passed,
            currents_A=currents,
            currents_nA={
                key: 1.0e9 * float(value) for key, value in currents.items()
            },
            opposite_current_switching_achieved=switching,
            forward_evidence=forward_evidence,
            solver_counts={
                "Lumerical_forward": 4,
                "Lumerical_adjoint": 0,
                "Lumerical_HEAT_or_CHARGE": 0,
                "FDTDX_Maxwell": 0,
            },
            wall_s=time.monotonic() - started,
        )
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
