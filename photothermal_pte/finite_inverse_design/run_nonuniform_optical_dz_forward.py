#!/usr/bin/env python3
"""Run one nonuniform 81x81 physical-density optical dz checkpoint."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import traceback

from .audit_v261_large_background_tfsf_geometry import (
    add_large_background_geometry,
)
from .large_background_contract import baseline_contract
from .probe_v261_gpu_plane_wave_roi import (
    APPROVED_API,
    APPROVED_ROOT,
    json_default,
    load_lumapi,
)
from .run_combined_physical_rho_pte_adfd import (
    compact_forward,
    physical_state,
    run_forward_density,
    set_imported_density,
)
from .run_v261_large_background_mixed_optical_adfd import (
    FIELD_REGION,
    add_adjoint_monitors,
    configure_pml_profile,
)
from .run_v261_large_background_tfsf_forward import (
    add_monitors,
    configure_design,
    save_native_npz,
    sha256,
)


STATUS_PASS = "GENERATED_NONUNIFORM_OPTICAL_DZ_FORWARD"
STATUS_FAIL = "FAILED_NONUNIFORM_OPTICAL_DZ_FORWARD"
CLOSURE_LIMIT = 5.0e-3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--flake-dz-nm", required=True, type=float)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--simulation-time-ps", type=float, default=4.0)
    args = parser.parse_args()
    if args.flake_dz_nm <= 0.0:
        parser.error("flake-dz-nm must be positive")
    return args


def main() -> int:
    args = parse_args()
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    tag = f"{args.flake_dz_nm:g}".replace(".", "p")
    project = output / f"nonuniform_physical_rho_dz{tag}nm_cpu_tfsf.fsp"
    raw_npz = output / f"nonuniform_physical_rho_dz{tag}nm_native_q.npz"
    result_path = output / "nonuniform_optical_dz_forward_result.json"
    result: dict[str, object] = {
        "status": "BLOCKED_NONUNIFORM_OPTICAL_DZ_FORWARD_NOT_RUN",
        "passed": False,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "nonuniform 81x81 physical-rho CPU-TFSF forward at one "
            "TaIrTe4 optical dz; no thermal, adjoint, FD, or optimization"
        ),
        "solver_root": str(APPROVED_ROOT),
        "lumapi_path": str(APPROVED_API),
        "flake_dz_nm": args.flake_dz_nm,
        "thermal_run": False,
        "adjoint_run": False,
        "finite_difference_run": False,
        "optimization_run": False,
        "forbidden_operations": {
            "clipping": False,
            "smoothing": False,
            "gain": False,
            "global_rescaling": False,
            "tiling": False,
            "optical_Q_source_deletion": False,
        },
    }
    fdtd = None
    started = time.monotonic()
    try:
        rho, _ = physical_state()
        contract = baseline_contract(
            lateral_domain_um=7.2,
            z_min_um=-3.6,
            z_max_um=3.4,
            pml_layers=32,
            flake_dz_nm=args.flake_dz_nm,
        )
        lumapi = load_lumapi()
        fdtd = lumapi.FDTD(hide=True, serverArgs={"platform": "offscreen"})
        add_large_background_geometry(fdtd, contract)
        pml = configure_pml_profile(fdtd, "stabilized-xy")
        design = configure_design(
            fdtd, "gray", 0.5, "imported-permittivity"
        )
        set_imported_density(fdtd, rho)
        fdtd.setnamed(
            "FDTD", "simulation time", args.simulation_time_ps * 1.0e-12
        )
        flux_signs = add_monitors(fdtd, contract)
        add_adjoint_monitors(fdtd, contract)
        fdtd.setnamed(FIELD_REGION, "source mode", False)
        forward = run_forward_density(
            fdtd,
            rho=rho,
            project=project,
            threads=args.threads,
            flux_signs=flux_signs,
        )
        save_native_npz(raw_npz, forward["native"], fdtd)
        closure = float(forward["six_face_closure_relative_error"])
        passed = bool(
            closure < CLOSURE_LIMIT
            and raw_npz.is_file()
            and project.is_file()
        )
        result.update(
            {
                "status": STATUS_PASS if passed else STATUS_FAIL,
                "passed": passed,
                "optical_contract": {
                    "forward_engine": "CPU TFSF",
                    "periodic_or_bloch": False,
                    "lateral_domain_um": 7.2,
                    "z_min_um": -3.6,
                    "z_max_um": 3.4,
                    "pml_layers": 32,
                    "pml_profile": "stabilized-xy",
                    "pml_profile_readback": pml,
                    "simulation_time_ps": args.simulation_time_ps,
                    "flake_dz_nm": args.flake_dz_nm,
                    "design_representation": design,
                    "design_density": (
                        "fixed nonuniform physical_state(), 81x81 nodal"
                    ),
                },
                "forward": compact_forward(forward),
                "gates": {
                    "six_face_closure_relative_error": closure,
                    "six_face_closure_limit": CLOSURE_LIMIT,
                },
                "raw_artifacts": {
                    "FSP": {
                        "path": str(project),
                        "byte_size": project.stat().st_size,
                        "sha256": sha256(project),
                    },
                    "native_Q_NPZ": {
                        "path": str(raw_npz),
                        "byte_size": raw_npz.stat().st_size,
                        "sha256": sha256(raw_npz),
                    },
                },
            }
        )
    except Exception as exc:
        result.update(
            {
                "status": STATUS_FAIL,
                "passed": False,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        if fdtd is not None:
            try:
                fdtd.close()
            except Exception:
                pass
        result["wall_s"] = time.monotonic() - started
        result_path.write_text(
            json.dumps(result, indent=2, default=json_default) + "\n"
        )
    print(
        json.dumps(
            {
                "status": result["status"],
                "passed": result["passed"],
                "flake_dz_nm": args.flake_dz_nm,
                "result_path": str(result_path),
            }
        ),
        flush=True,
    )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
