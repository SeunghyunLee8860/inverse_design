#!/usr/bin/env python3
"""Phase B: simulation-time convergence (4 ps baseline vs 3 ps vs 2 ps).

Runs each condition in a FRESH subprocess (eqc_lib reads VC_SIM_TIME_S at
import), always in COMBINED adjoint mode, on the same latent as Phase A
(model.x0 at the same beta).  The 4 ps baseline is reused from a passed
Phase-A run (--baseline-dir) so it is not re-solved.

Convergence is judged against the 4 ps baseline -- NOT by AD/FD (a short
simulation truncates the same physics in both AD and FD, so AD/FD alone can
pass while the answer is wrong):

    F_sum rel difference            < 1%
    gradient cosine (phys+latent)   > 0.995
    directional-derivative rel diff < 2%

The shortest passing simulation time is recommended.  Engine-log tails
(auto-shutoff/termination info) and per-solve wall times are recorded for
every condition.

Usage:
  phaseB_simtime_convergence.py <outdir> --baseline-dir <phaseA outdir>
                                [--times 3e-12,2e-12] [--beta 4.0]
  (internal) --single <sim_time_s>  : run one condition in this process
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
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


def engine_log_tails(root: Path, lines: int = 12) -> dict:
    tails = {}
    for log in sorted(root.rglob("*_p0.log")):
        try:
            content = log.read_text(errors="replace").splitlines()
            tails[str(log.relative_to(root))] = content[-lines:]
        except Exception as exc:  # noqa: BLE001
            tails[str(log)] = [f"<unreadable: {exc}>"]
    return tails


def run_single(args) -> int:
    """One condition, current process; VC_SIM_TIME_S already in the env."""
    import eqc_lib as lib

    sim_time = float(os.environ["VC_SIM_TIME_S"])
    assert abs(lib.SIM_TIME_S - sim_time) < 1e-18, \
        f"eqc_lib.SIM_TIME_S={lib.SIM_TIME_S} != env {sim_time}"
    model = lib.load_model()
    lib.assert_approved_lumapi()

    from autograd import tensor_jacobian_product
    from volume_current_evaluator import VolumeCurrentEvaluator

    out = Path(args.output).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    mapping = model.mapping
    shape = (model.Nx, model.Ny, model.Nz)
    latent = np.asarray(model.x0, float).reshape(-1)
    phys = np.asarray(mapping(latent, args.beta), float).reshape(shape)

    os.environ["VC_ADJOINT_COMPONENT_MODE"] = "combined"
    record: dict = {
        "sim_time_s": sim_time,
        "expected_steps": int(round(sim_time / lib.EXPECTED_DT_S)),
        "auto_shutoff_min": lib.AUTO_SHUTOFF_MIN,
        "beta": args.beta,
    }
    for pol in ("x", "y"):
        ev = VolumeCurrentEvaluator(out / f"solver_{pol}", args.rho_step, pol)
        ev.prepare(force_rebuild=False)
        t0 = time.time()
        evaluation = ev.value_and_gradient(
            phys, label=f"phaseB_{sim_time:.0e}_{pol}", density_mode="probe_safe")
        g_phys = np.asarray(evaluation.gradient_physical, float)
        g_lat = np.asarray(tensor_jacobian_product(
            lambda z: mapping(z, args.beta))(latent, g_phys.reshape(-1)),
            float).reshape(-1)
        np.savez_compressed(out / f"gradient_{pol}.npz",
                            gradient_physical=g_phys, gradient_latent=g_lat,
                            fom=np.array(evaluation.fom))
        meta = evaluation.metadata
        record[pol] = {
            "fom": float(evaluation.fom),
            "eval_total_s": time.time() - t0,
            "wall_time_s": meta["wall_time_s"],
            "em_solves": int(meta["electromagnetic_solves"]),
            "pairing": float(meta["periodic_source_pairing_relative_error"]),
            "roundtrips": {k: float(v["roundtrip_max_abs_error"])
                           for k, v in meta["source"].items()},
        }
        print(f"[phaseB {sim_time:.0e}] {pol}: fom={evaluation.fom:.9e} "
              f"times={json.dumps(meta['wall_time_s'])}", flush=True)
    record["engine_log_tails"] = engine_log_tails(out)
    (out / "condition_report.json").write_text(
        json.dumps(record, indent=2, default=float) + "\n")
    return 0


def load_condition(directory: Path) -> dict:
    report = json.loads((directory / "condition_report.json").read_text())
    for pol in ("x", "y"):
        z = np.load(directory / f"gradient_{pol}.npz")
        report[pol]["gradient_physical"] = np.asarray(z["gradient_physical"])
        report[pol]["gradient_latent"] = np.asarray(z["gradient_latent"])
    return report


def load_phaseA_baseline(directory: Path) -> dict:
    """Adapt a passed Phase-A run (combined mode @4ps) as the 4 ps baseline."""
    report = json.loads((directory / "phaseA_report.json").read_text())
    if report.get("status") != "passed":
        raise SystemExit(f"Phase-A baseline at {directory} did not pass")
    baseline: dict = {"sim_time_s": 4e-12, "source": str(directory)}
    for pol in ("x", "y"):
        z = np.load(directory / f"gradient_combined_{pol}.npz")
        baseline[pol] = {
            "fom": float(np.asarray(z["fom"])),
            "gradient_physical": np.asarray(z["gradient_physical"]),
            "gradient_latent": np.asarray(z["gradient_latent"]),
            "wall_time_s": report["wall_times"]["combined"][pol],
        }
    return baseline


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("output")
    ap.add_argument("--baseline-dir")
    ap.add_argument("--times", default="3e-12,2e-12")
    ap.add_argument("--beta", type=float, default=4.0)
    ap.add_argument("--rho-step", type=float, default=0.001)
    ap.add_argument("--direction-seed", type=int, default=20260726)
    ap.add_argument("--single", action="store_true",
                    help="internal: run ONE condition (env VC_SIM_TIME_S)")
    args = ap.parse_args()
    if args.single:
        return run_single(args)
    if not args.baseline_dir:
        raise SystemExit("--baseline-dir (passed Phase-A run) is required")

    out = Path(args.output).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    baseline = load_phaseA_baseline(Path(args.baseline_dir).resolve())

    rng = np.random.default_rng(args.direction_seed)
    dir_phys = rng.standard_normal(
        baseline["x"]["gradient_physical"].size)
    dir_phys /= np.linalg.norm(dir_phys)
    dir_lat = rng.standard_normal(baseline["x"]["gradient_latent"].size)
    dir_lat /= np.linalg.norm(dir_lat)

    conditions = {}
    for spec in args.times.split(","):
        sim_time = float(spec)
        cond_dir = out / f"T{spec}"
        env = dict(os.environ)
        env["VC_SIM_TIME_S"] = spec
        env["VC_ADJOINT_COMPONENT_MODE"] = "combined"
        print(f"[phaseB] launching condition {spec} -> {cond_dir}", flush=True)
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), str(cond_dir),
             "--single", "--beta", str(args.beta),
             "--rho-step", str(args.rho_step)],
            env=env, cwd=str(HERE))
        if proc.returncode != 0:
            raise SystemExit(f"condition {spec} failed rc={proc.returncode}")
        conditions[spec] = load_condition(cond_dir)

    def dd(vec, direction):
        return float(np.dot(vec.reshape(-1), direction))

    fsum_base = baseline["x"]["fom"] + baseline["y"]["fom"]
    table = []
    for spec, cond in conditions.items():
        fsum = cond["x"]["fom"] + cond["y"]["fom"]
        row = {
            "sim_time_s": float(spec),
            "expected_steps": cond["expected_steps"],
            "Fx": cond["x"]["fom"], "Fy": cond["y"]["fom"], "F_sum": fsum,
            "F_sum_rel_diff_vs_4ps": rel_diff(fsum, fsum_base),
            "wall_time_s": {p: cond[p]["wall_time_s"] for p in "xy"},
            "eval_total_s": {p: cond[p]["eval_total_s"] for p in "xy"},
        }
        for pol in ("x", "y"):
            b, c = baseline[pol], cond[pol]
            row[f"grad_cosine_phys_{pol}"] = cosine(
                b["gradient_physical"], c["gradient_physical"])
            row[f"grad_cosine_latent_{pol}"] = cosine(
                b["gradient_latent"], c["gradient_latent"])
            row[f"dir_deriv_rel_diff_phys_{pol}"] = rel_diff(
                dd(b["gradient_physical"], dir_phys),
                dd(c["gradient_physical"], dir_phys))
            row[f"dir_deriv_rel_diff_latent_{pol}"] = rel_diff(
                dd(b["gradient_latent"], dir_lat),
                dd(c["gradient_latent"], dir_lat))
        row["pass"] = bool(
            row["F_sum_rel_diff_vs_4ps"] < 1e-2
            and all(row[f"grad_cosine_phys_{p}"] > 0.995 for p in "xy")
            and all(row[f"grad_cosine_latent_{p}"] > 0.995 for p in "xy")
            and all(row[f"dir_deriv_rel_diff_phys_{p}"] < 2e-2 for p in "xy")
            and all(row[f"dir_deriv_rel_diff_latent_{p}"] < 2e-2 for p in "xy"))
        table.append(row)

    passing = [row for row in table if row["pass"]]
    recommended = min((row["sim_time_s"] for row in passing), default=4e-12)
    report = {
        "baseline_4ps": {
            "source": baseline["source"],
            "Fx": baseline["x"]["fom"], "Fy": baseline["y"]["fom"],
            "F_sum": fsum_base,
            "wall_time_s": {p: baseline[p]["wall_time_s"] for p in "xy"},
        },
        "criteria": {"F_sum_rel": 1e-2, "grad_cosine": 0.995,
                     "dir_deriv_rel": 2e-2},
        "conditions": table,
        "recommended_sim_time_s": recommended,
        "engine_log_tails": {spec: conditions[spec]["engine_log_tails"]
                             for spec in conditions},
    }
    (out / "phaseB_report.json").write_text(
        json.dumps(report, indent=2, default=float) + "\n")
    slim = {k: v for k, v in report.items() if k != "engine_log_tails"}
    print(json.dumps(slim, indent=2, default=float), flush=True)
    print(f"PHASE B RECOMMENDED SIM TIME: {recommended:g} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
