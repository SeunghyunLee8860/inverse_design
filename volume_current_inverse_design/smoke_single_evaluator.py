#!/usr/bin/env python3
"""Single x-evaluator smoke: prove the API/engine pair and the whole adjoint chain.

Run this BEFORE any production launch.  It exercises exactly the production
entrypoint's imports and evaluator, on one GPU, for one gradient evaluation:

    forward solve -> coherent FOM -> Ex/Ez volume-current adjoint solves
    (importdataset round trip) -> 27-color measured rho->epsilon transpose
    -> physical gradient

Pass criteria (all fail-closed, exit 2 on any miss):
  * lumapi resolved under the approved r12 root
  * FOM finite and strictly positive
  * both adjoint profile round trips exactly 0.0
  * periodic-source pairing relative error <= 1e-13
  * gradient all-finite and not identically zero

Usage:  smoke_single_evaluator.py <output-dir> [--pol x|y]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for _p in (HERE, HERE / "bundle"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("output")
    ap.add_argument("--pol", choices=("x", "y"), default="x")
    ap.add_argument("--beta", type=float, default=2.0)
    ap.add_argument("--rho-step", type=float, default=0.001)
    args = ap.parse_args()

    out = Path(args.output).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "smoke_report.json"

    import eqc_lib as lib

    print("[provenance] before load_model: "
          + json.dumps(lib.import_provenance(), default=str), flush=True)
    model = lib.load_model()
    provenance = lib.import_provenance()
    print("[provenance] after  load_model: "
          + json.dumps(provenance, default=str), flush=True)
    lib.assert_approved_lumapi()

    from volume_current_evaluator import VolumeCurrentEvaluator

    checks: dict[str, bool] = {}
    report: dict[str, object] = {
        "polarization": args.pol,
        "beta": args.beta,
        "import_provenance": provenance,
    }
    try:
        evaluator = VolumeCurrentEvaluator(
            out / f"solver_{args.pol}", args.rho_step, args.pol
        )
        t0 = time.time()
        contract = evaluator.prepare(force_rebuild=False)
        report["prepare_seconds"] = time.time() - t0
        report["production_contract"] = contract

        rho = lib.physical_seed(model, beta=args.beta)
        report["rho_shape"] = list(np.asarray(rho).shape)
        report["rho_range"] = [float(np.min(rho)), float(np.max(rho))]

        t0 = time.time()
        evaluation = evaluator.value_and_gradient(
            rho, label=f"smoke_{args.pol}", density_mode="probe_safe"
        )
        report["evaluate_seconds"] = time.time() - t0

        gradient = np.asarray(evaluation.gradient_physical, float)
        meta = evaluation.metadata
        source = meta["source"]
        roundtrips = {k: float(v["roundtrip_max_abs_error"]) for k, v in source.items()}
        pairing = float(meta["periodic_source_pairing_relative_error"])
        leakage = float(meta["rho_epsilon_transpose"]["max_owner_leakage_fraction"])

        report.update({
            "lumapi_file": provenance["lumapi_file"],
            "fdtd_engine": provenance["fdtd_engine"],
            "fom": float(evaluation.fom),
            "adjoint_roundtrip_max_abs_error": roundtrips,
            "periodic_source_pairing_relative_error": pairing,
            "max_owner_leakage_fraction": leakage,
            "gradient_l2": float(np.linalg.norm(gradient)),
            "gradient_absmax": float(np.max(np.abs(gradient))),
            "gradient_finite": bool(np.all(np.isfinite(gradient))),
            "solver_safe_affine": meta["solver_safe_affine"],
        })
        np.savez_compressed(out / "smoke_gradient.npz",
                            gradient=gradient, fom=np.array(evaluation.fom))

        checks["lumapi_approved"] = bool(provenance["lumapi_approved"])
        checks["fom_finite_positive"] = bool(
            np.isfinite(evaluation.fom) and evaluation.fom > 0.0)
        checks["adjoint_roundtrip_exact_zero"] = all(
            value == 0.0 for value in roundtrips.values())
        checks["pairing_within_1e-13"] = bool(pairing <= 1e-13)
        checks["gradient_finite"] = bool(np.all(np.isfinite(gradient)))
        checks["gradient_nonzero"] = bool(np.any(gradient != 0.0))
    except Exception as exc:  # noqa: BLE001
        import traceback
        report["status"] = "failed"
        report["exception"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
        report["checks"] = checks
        report_path.write_text(json.dumps(report, indent=2, default=str) + "\n")
        print(json.dumps(report, indent=2, default=str), flush=True)
        return 1

    report["checks"] = checks
    report["status"] = "passed" if all(checks.values()) else "failed_checks"
    report_path.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps(report, indent=2, default=str), flush=True)
    if not all(checks.values()):
        failed = [name for name, ok in checks.items() if not ok]
        print(f"SMOKE FAILED: {failed}", flush=True)
        return 2
    print("SMOKE PASSED", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
