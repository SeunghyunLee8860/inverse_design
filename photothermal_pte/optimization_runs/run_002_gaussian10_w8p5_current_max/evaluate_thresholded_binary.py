#!/usr/bin/env python3
"""Evaluate the exact thresholded-binary design with GPU Maxwell/CUDA thermal."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]
for path in (HERE, REPOSITORY):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from beta_continuation_support import exact_binary_audit  # noqa: E402
from run_production_combined_adfd_smoke import (  # noqa: E402
    boundary_energy,
    compact_forward,
    contract_configuration,
    map_q,
    open_fdtd,
    run_forward,
    solve_base_thermal,
)


STATUS = "VALIDATED_THRESHOLDED_BINARY_GPU_MAXWELL_CUDA_THERMAL"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, object]:
    path = path.expanduser().resolve()
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-fsp", type=Path, required=True)
    parser.add_argument("--base-sha256", required=True)
    parser.add_argument("--binary-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu-device", default="GPU 0")
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--incident-power-W", type=float, default=1.3822950233084244e-13)
    parser.add_argument("--polarization-angle-deg", type=float, default=None)
    parser.add_argument("--polarization-label", default=None)
    parser.add_argument(
        "--completed-fsp",
        type=Path,
        default=None,
        help=(
            "Reuse a completed GPU-forward FSP and run only result extraction, "
            "conservative remap, and CUDA thermal/PTE evaluation."
        ),
    )
    args = parser.parse_args()
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "thresholded_binary_evaluation_result.json"
    result: dict[str, object] = {
        "status": "FAILED_THRESHOLDED_BINARY_GPU_MAXWELL_CUDA_THERMAL",
        "passed": False,
    }
    fdtd = None
    started = time.monotonic()
    try:
        base_fsp = args.base_fsp.expanduser().resolve()
        if sha256(base_fsp) != args.base_sha256:
            raise RuntimeError("base FSP SHA mismatch")
        binary_path = args.binary_npz.expanduser().resolve()
        binary_data = np.load(binary_path)
        key = "rho_binary" if "rho_binary" in binary_data else "rho"
        rho = np.asarray(binary_data[key], float)
        if rho.shape != (373, 373):
            raise RuntimeError(f"binary density shape changed: {rho.shape}")
        if not np.all((rho == 0.0) | (rho == 1.0)):
            raise RuntimeError("final density is not exactly binary")
        exact = exact_binary_audit(rho, 50.0e-9)
        if not exact["solid_pass"] or not exact["void_pass"]:
            raise RuntimeError("exact 500 nm solid/void DRC did not pass")
        config = contract_configuration("selected_production")
        fdtd, session_audit, runtime = open_fdtd(args.gpu_device)
        forward = run_forward(
            fdtd,
            session_audit,
            runtime,
            base_fsp=base_fsp,
            rho=rho,
            role=(
                "thresholded_binary_final"
                if args.polarization_label is None
                else f"thresholded_binary_final_{args.polarization_label}"
            ),
            output=output,
            imported_object=str(config["imported_object"]),
            nodes=config["nodes"],
            polarization_angle_deg=args.polarization_angle_deg,
            completed_project=(
                args.completed_fsp.expanduser().resolve()
                if args.completed_fsp is not None
                else None
            ),
        )
        # The electric/epsilon arrays are needed only by Maxwell-adjoint work,
        # not by this exact-binary forward evaluation.  Release them and the
        # Lumerical session before the conservative material-overlap remap so
        # the two large solver representations never coexist in memory.
        forward.pop("electric", None)
        forward.pop("epsilon", None)
        if fdtd is not None:
            fdtd.close()
            fdtd = None
        thermal_data, mapping = map_q(
            forward["q"], design_half_span_m=float(config["design_half_span_m"])
        )
        state, pair, objective, _, _ = solve_base_thermal(
            thermal_data,
            rho,
            args.cuda_device,
            config["density_forward"],
            config["density_transpose"],
        )
        residual = max(
            pair.forward.explicit_relative_residual,
            pair.adjoint.explicit_relative_residual,
        )
        energy = boundary_energy(state, pair.forward.solution)
        passed = bool(
            forward["closure"] < 0.005
            and mapping["internal_relative_power_error"] < 0.005
            and float(forward["log_audit"]["final_auto_shutoff"]) < 1.0e-5
            and residual < 1.0e-8
            and energy < 0.01
        )
        raw = output / "thresholded_binary_evaluation.npz"
        temperature_grid = np.full(state.active.shape, np.nan, dtype=float)
        temperature_grid[state.active] = np.asarray(pair.forward.solution, float)
        np.savez_compressed(
            raw,
            rho_binary=rho.astype(np.uint8),
            thermal_temperature_active_K=np.asarray(pair.forward.solution, float),
            thermal_temperature_grid_K=temperature_grid,
            thermal_adjoint_active=np.asarray(pair.adjoint.solution, float),
            Q_total_W_m3=np.asarray(thermal_data["Q_total_W_m3"], float),
            x_edges_m=np.asarray(thermal_data["x_edges_m"], float),
            y_edges_m=np.asarray(thermal_data["y_edges_m"], float),
            z_edges_m=np.asarray(thermal_data["z_edges_m"], float),
        )
        result = {
            "status": STATUS if passed else "FAILED_THRESHOLDED_BINARY_GPU_MAXWELL_CUDA_THERMAL",
            "passed": passed,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "objective_A": objective,
            "objective_A_per_incident_W": objective / args.incident_power_W,
            "incident_power_W": args.incident_power_W,
            "polarization": {
                "label": args.polarization_label or "legacy_default",
                "requested_angle_deg": args.polarization_angle_deg,
                "readback_angle_deg": forward["source_polarization_angle_deg"],
            },
            "binary_density_values": np.unique(rho).tolist(),
            "exact_binary_audit": {
                key: value for key, value in exact.items() if not isinstance(value, np.ndarray)
            },
            "base_forward": compact_forward(forward),
            "base_mapping": mapping,
            "gates": {
                "optical_closure": forward["closure"],
                "Q_mapping_error": mapping["internal_relative_power_error"],
                "base_auto_shutoff": forward["log_audit"]["final_auto_shutoff"],
                "thermal_residual": residual,
                "thermal_energy_balance": energy,
            },
            "inputs": {
                "base_FSP": artifact(base_fsp),
                "binary_density": artifact(binary_path),
            },
            "raw_artifact": artifact(raw),
            "Maxwell_forward_solves": 1,
            "Maxwell_adjoint_solves": 0,
            "thermal_forward_solves": 1,
            "thermal_adjoint_solves": 1,
            "CPU_FDTD_fallback": False,
            "CPU_thermal_solve_fallback": False,
            "posthoc_density_repair": False,
            "wall_s": time.monotonic() - started,
        }
    except Exception as error:
        result.update({
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
            "wall_s": time.monotonic() - started,
        })
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
