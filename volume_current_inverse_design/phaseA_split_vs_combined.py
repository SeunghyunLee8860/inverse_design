#!/usr/bin/env python3
"""Phase A: split vs combined FieldRegion adjoint on the SAME latent.

split    (certified baseline): per objective evaluation
         x fwd + x adj-inplane + x adj-z + y fwd + y adj-inplane + y adj-z = 6 solves
combined (candidate):
         x fwd + x vector-adj + y fwd + y vector-adj                      = 4 solves

The two are mathematically identical by linearity of the adjoint solve in its
source and of the sensitivity in the adjoint field; this driver PROVES it on
the production geometry before any further runtime optimisation.

Pass criteria (fail-closed, exit 2 on miss):
    FOM rel difference        < 1e-6   (per polarization and F_sum)
    gradient cosine           > 0.9999 (physical and latent, per polarization)
    gradient rel L2 diff      < 1e-2
    adjoint roundtrip         == 0     (every solve, both modes)
    pairing rel error         < 1e-13
    owner leakage fraction    == 0

Usage: phaseA_split_vs_combined.py <outdir> [--beta 4.0] [--rho-step 0.001]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for _p in (HERE, HERE / "bundle"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

TINY = np.finfo(float).tiny


def rel_diff(a: float, b: float) -> float:
    return abs(a - b) / max(abs(a), abs(b), TINY)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1)
    b = b.reshape(-1)
    return float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), TINY))


def rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm((a - b).reshape(-1))
                 / max(np.linalg.norm(a.reshape(-1)),
                       np.linalg.norm(b.reshape(-1)), TINY))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("output")
    ap.add_argument("--beta", type=float, default=4.0)
    ap.add_argument("--rho-step", type=float, default=0.001)
    ap.add_argument("--direction-seed", type=int, default=20260726)
    args = ap.parse_args()
    out = Path(args.output).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    import eqc_lib as lib

    model = lib.load_model()
    lib.assert_approved_lumapi()
    provenance = lib.import_provenance()
    print("[provenance] " + json.dumps(
        {k: provenance[k] for k in ("lumapi_file", "fdtd_engine")}), flush=True)

    from autograd import tensor_jacobian_product
    from volume_current_evaluator import VolumeCurrentEvaluator

    mapping = model.mapping
    shape = (model.Nx, model.Ny, model.Nz)
    latent = np.asarray(model.x0, float).reshape(-1)
    phys = np.asarray(mapping(latent, args.beta), float).reshape(shape)

    rng = np.random.default_rng(args.direction_seed)
    dir_phys = rng.standard_normal(phys.size)
    dir_phys /= np.linalg.norm(dir_phys)
    dir_lat = rng.standard_normal(latent.size)
    dir_lat /= np.linalg.norm(dir_lat)

    evaluators = {}
    for pol in ("x", "y"):
        ev = VolumeCurrentEvaluator(out / f"solver_{pol}", args.rho_step, pol)
        ev.prepare(force_rebuild=False)
        evaluators[pol] = ev

    results: dict[str, dict] = {}
    for mode in ("split", "combined"):
        os.environ["VC_ADJOINT_COMPONENT_MODE"] = mode
        results[mode] = {}
        for pol in ("x", "y"):
            t0 = time.time()
            ev = evaluators[pol].value_and_gradient(
                phys, label=f"phaseA_{mode}_{pol}", density_mode="probe_safe")
            wall = time.time() - t0
            g_phys = np.asarray(ev.gradient_physical, float)
            g_lat = np.asarray(tensor_jacobian_product(
                lambda z: mapping(z, args.beta))(latent, g_phys.reshape(-1)),
                float).reshape(-1)
            meta = ev.metadata
            roundtrips = {k: float(v["roundtrip_max_abs_error"])
                          for k, v in meta["source"].items()}
            results[mode][pol] = {
                "fom": float(ev.fom),
                "gradient_physical": g_phys,
                "gradient_latent": g_lat,
                "gradient_l2": float(np.linalg.norm(g_phys)),
                "gradient_absmax": float(np.max(np.abs(g_phys))),
                "latent_gradient_l2": float(np.linalg.norm(g_lat)),
                "dir_deriv_phys": float(np.dot(g_phys.reshape(-1), dir_phys)),
                "dir_deriv_latent": float(np.dot(g_lat, dir_lat)),
                "roundtrips": roundtrips,
                "pairing": float(meta["periodic_source_pairing_relative_error"]),
                "leakage": float(
                    meta["rho_epsilon_transpose"]["max_owner_leakage_fraction"]),
                "em_solves": int(meta["electromagnetic_solves"]),
                "wall_time_s": meta["wall_time_s"],
                "eval_total_s": wall,
            }
            print(f"[phaseA] {mode}/{pol}: fom={ev.fom:.9e} solves="
                  f"{meta['electromagnetic_solves']} wall={wall:.1f}s "
                  f"times={json.dumps(meta['wall_time_s'])}", flush=True)

    # ---- comparison -------------------------------------------------------
    split, comb = results["split"], results["combined"]
    comparison: dict[str, dict] = {}
    for pol in ("x", "y"):
        s, c = split[pol], comb[pol]
        comparison[pol] = {
            "fom_split": s["fom"], "fom_combined": c["fom"],
            "fom_rel_diff": rel_diff(s["fom"], c["fom"]),
            "grad_cosine_phys": cosine(s["gradient_physical"], c["gradient_physical"]),
            "grad_rel_l2_phys": rel_l2(s["gradient_physical"], c["gradient_physical"]),
            "grad_cosine_latent": cosine(s["gradient_latent"], c["gradient_latent"]),
            "grad_rel_l2_latent": rel_l2(s["gradient_latent"], c["gradient_latent"]),
            "grad_l2": [s["gradient_l2"], c["gradient_l2"]],
            "grad_absmax": [s["gradient_absmax"], c["gradient_absmax"]],
            "dir_deriv_phys": [s["dir_deriv_phys"], c["dir_deriv_phys"]],
            "dir_deriv_phys_rel_diff": rel_diff(s["dir_deriv_phys"], c["dir_deriv_phys"]),
            "dir_deriv_latent": [s["dir_deriv_latent"], c["dir_deriv_latent"]],
            "dir_deriv_latent_rel_diff": rel_diff(
                s["dir_deriv_latent"], c["dir_deriv_latent"]),
        }
    fsum_s = split["x"]["fom"] + split["y"]["fom"]
    fsum_c = comb["x"]["fom"] + comb["y"]["fom"]

    all_roundtrips = [v for m in results.values() for p in m.values()
                      for v in p["roundtrips"].values()]
    all_pairing = [p["pairing"] for m in results.values() for p in m.values()]
    all_leakage = [p["leakage"] for m in results.values() for p in m.values()]

    checks = {
        "fom_rel_diff_lt_1e-6": all(
            comparison[p]["fom_rel_diff"] < 1e-6 for p in "xy"
        ) and rel_diff(fsum_s, fsum_c) < 1e-6,
        "grad_cosine_gt_0.9999": all(
            comparison[p]["grad_cosine_phys"] > 0.9999
            and comparison[p]["grad_cosine_latent"] > 0.9999 for p in "xy"),
        "grad_rel_l2_lt_1pct": all(
            comparison[p]["grad_rel_l2_phys"] < 1e-2
            and comparison[p]["grad_rel_l2_latent"] < 1e-2 for p in "xy"),
        "roundtrip_exact_zero": all(v == 0.0 for v in all_roundtrips),
        "pairing_lt_1e-13": all(v <= 1e-13 for v in all_pairing),
        "leakage_zero": all(v == 0.0 for v in all_leakage),
    }

    solves = {
        "split": split["x"]["em_solves"] + split["y"]["em_solves"],
        "combined": comb["x"]["em_solves"] + comb["y"]["em_solves"],
    }
    report = {
        "status": "passed" if all(checks.values()) else "failed_checks",
        "beta": args.beta,
        "latent": "model.x0 (production seed, ID_SEED env)",
        "checks": checks,
        "comparison": comparison,
        "F_sum": {"split": fsum_s, "combined": fsum_c,
                  "rel_diff": rel_diff(fsum_s, fsum_c)},
        "solve_count_per_objective_eval": solves,
        "wall_times": {
            m: {p: {"eval_total_s": results[m][p]["eval_total_s"],
                    **results[m][p]["wall_time_s"]} for p in "xy"}
            for m in ("split", "combined")},
        "import_provenance": provenance,
    }
    for mode in ("split", "combined"):
        for pol in ("x", "y"):
            np.savez_compressed(
                out / f"gradient_{mode}_{pol}.npz",
                gradient_physical=results[mode][pol]["gradient_physical"],
                gradient_latent=results[mode][pol]["gradient_latent"],
                fom=np.array(results[mode][pol]["fom"]))
    (out / "phaseA_report.json").write_text(
        json.dumps(report, indent=2, default=float) + "\n")
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("import_provenance",)},
                     indent=2, default=float), flush=True)
    if not all(checks.values()):
        print("PHASE A FAILED: " + ", ".join(
            k for k, v in checks.items() if not v), flush=True)
        return 2
    print("PHASE A PASSED", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
