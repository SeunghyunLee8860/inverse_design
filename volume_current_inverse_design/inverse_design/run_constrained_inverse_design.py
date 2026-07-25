#!/usr/bin/env python3
"""Constrained inverse design (NLopt MMA/CCSA) with length-scale constraints.

Replaces the Adam continuous-anneal runner.  Adam cannot honour nonlinear
geometry constraints; MMA/CCSA can.  Formulation (nominal / default):

    maximize    Fx(rho_nominal) + Fy(rho_nominal)
    subject to  g_solid(latent, beta) <= 0
                g_void (latent, beta) <= 0
                0 <= latent_i <= 1

* objective gradient = FieldRegion adjoint (VolumeCurrentEvaluator, probe-safe
  density) pulled back through the mapping VJP.
* constraint gradients = autograd (LengthScaleConstraints).
* beta is a continuation parameter across stages (NOT a success target).
* one objective evaluation = 6 EM solves (x: fwd+2 adj, y: fwd+2 adj).

Robust mode (--robust) adds an epigraph variable and the three-field worst-case
constraints (18 EM solves per objective evaluation).

Provenance (spec 10): immutable per-attempt contract, append-only history.jsonl,
best-FEASIBLE checkpoint, resume refuses a changed code/config hash.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bundle"))


def _preflight_nlopt():
    try:
        import nlopt  # noqa: F401
        return nlopt
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "nlopt is required (pip install 'nlopt>=2.7,<3'). "
            f"import failed: {exc}"
        )


def _code_hash(paths) -> str:
    h = hashlib.sha256()
    for p in sorted(paths):
        try:
            h.update(Path(p).read_bytes())
        except OSError:
            pass
    return h.hexdigest()[:16]


def _config_hash(cfg: dict) -> str:
    return hashlib.sha256(
        json.dumps(cfg, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]


def _beta_stages(spec: str):
    return [float(v) for v in spec.split(",") if v.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--objective", choices=["sum"], default="sum")
    ap.add_argument("--mfs-um", type=float, default=0.5)
    ap.add_argument("--mgs-um", type=float, default=0.5)
    ap.add_argument("--beta-schedule", default="2,4,8,16,32,64")
    ap.add_argument("--maxeval-per-stage", type=int, default=12)
    ap.add_argument("--gpu", default=os.environ.get("CL_GPU_DEVICE", "GPU 1"))
    ap.add_argument("--rho-step", type=float, default=0.001)
    ap.add_argument("--constraint-c", type=float, default=None)
    ap.add_argument("--tol-solid", type=float, default=None)
    ap.add_argument("--tol-void", type=float, default=None)
    ap.add_argument("--robust", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    os.environ.setdefault("MSOPT_MAPPING", "periodic_constrained")
    os.environ["MFS_UM"] = str(args.mfs_um)
    os.environ["MGS_UM"] = str(args.mgs_um)
    os.environ["CL_GPU_DEVICE"] = args.gpu
    os.environ["LUMERICAL_SESSION_GPU_DEVICE"] = args.gpu
    os.environ["TARGET_WL_UM"] = "4.0"
    os.environ["SOURCE_WL_START_UM"] = "3.0"
    os.environ["SOURCE_WL_STOP_UM"] = "6.0"
    os.environ["MATERIAL_FIT_START_UM"] = "2.7"
    os.environ["MATERIAL_FIT_STOP_UM"] = "13.2"
    os.environ["BULK_MESH_MODE"] = "auto"
    os.environ["MESH_ACCURACY"] = "5"
    os.environ["VC_MESH_REFINEMENT"] = "conformal variant 1"
    nlopt = _preflight_nlopt()

    import eqc_lib as lib
    from geometric_constraints import LengthScaleConstraints
    from volume_current_evaluator import VolumeCurrentEvaluator
    from autograd import tensor_jacobian_product

    model = lib.load_model()
    if os.environ["MSOPT_MAPPING"] != "periodic_constrained":
        raise SystemExit("this runner requires MSOPT_MAPPING=periodic_constrained")
    mapping = model.mapping
    nlat = model.Nux * model.Nuy
    shape = (model.Nx, model.Ny, model.Nz)

    run_root = (HERE / args.output).resolve()
    (run_root / "attempts").mkdir(parents=True, exist_ok=True)
    (run_root / "checkpoints").mkdir(exist_ok=True)

    contract = {
        "mapping_version": getattr(
            __import__("periodic_constrained_mapping"), "MAPPING_VERSION", "?"),
        "mapping_config": mapping.config.to_dict(),
        "physical_shape": list(shape),
        "unique_shape": [model.Nux, model.Nuy],
        "objective": args.objective,
        "robust": args.robust,
        "mfs_um": args.mfs_um, "mgs_um": args.mgs_um,
        "beta_schedule": _beta_stages(args.beta_schedule),
        "rho_step": args.rho_step,
        "source_wavelength_range_um": [3.0, 6.0],
        "analysis_wavelength_um": list(np.asarray(model.target_wl, float)),
        "material_fit_range_um": [2.7, 13.2],
        "mesh_type": "auto non-uniform",
        "mesh_refinement": lib.MESH_REFINEMENT,
        "mesh_accuracy": lib.MESH_ACCURACY,
        "global_uniform_mesh": "absent (asserted from realized FSP)",
        "flake_dz_nm": 5.0,
        "period_um": model.period_xy,
    }
    code_files = [
        ROOT / "bundle/periodic_constrained_mapping.py",
        ROOT / "bundle/periodic_filter.py",
        ROOT / "volume_current_evaluator.py",
        HERE / "geometric_constraints.py",
        HERE / "run_constrained_inverse_design.py",
    ]
    contract["code_hash"] = _code_hash(code_files)
    contract["config_hash"] = _config_hash(contract)

    # attempt bookkeeping + resume guard
    existing = sorted((run_root / "attempts").glob("attempt_*"))
    latent = np.asarray(model.x0, float).reshape(-1).copy()
    if args.resume and existing:
        prev = json.loads((existing[-1] / "contract.json").read_text())
        if prev.get("code_hash") != contract["code_hash"] or \
           prev.get("config_hash") != contract["config_hash"]:
            raise SystemExit(
                "resume refused: code/config hash changed -> start a new attempt")
        ckpt = run_root / "checkpoints" / "best_feasible.npz"
        if ckpt.exists():
            latent = np.asarray(np.load(ckpt)["latent"], float).reshape(-1).copy()
    attempt_id = len(existing) + 1
    attempt = run_root / "attempts" / f"attempt_{attempt_id:04d}"
    attempt.mkdir(parents=True, exist_ok=True)
    (attempt / "contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    history = (run_root / "history.jsonl").open("a")

    ev_x = VolumeCurrentEvaluator(attempt / "solver_x", args.rho_step, "x")
    ev_y = VolumeCurrentEvaluator(attempt / "solver_y", args.rho_step, "y")
    ev_x.prepare(force_rebuild=False)
    ev_y.prepare(force_rebuild=False)
    constraints = LengthScaleConstraints(
        mapping, c_decay=args.constraint_c,
        tol_solid=args.tol_solid, tol_void=args.tol_void,
    )

    state = {"beta": 0.0, "iter": 0, "best_feasible": None, "best_feasible_obj": -np.inf}

    def objective_and_grad(x, beta):
        phys = np.asarray(mapping(x, beta), float).reshape(shape)
        Fx = ev_x.value_and_gradient(phys, label=f"it{state['iter']}_x",
                                     density_mode="probe_safe")
        Fy = ev_y.value_and_gradient(phys, label=f"it{state['iter']}_y",
                                     density_mode="probe_safe")
        g_phys = (Fx.gradient_physical + Fy.gradient_physical).reshape(-1)
        dlat = tensor_jacobian_product(lambda z: mapping(z, beta))(x, g_phys)
        return float(Fx.fom + Fy.fom), np.asarray(dlat, float).reshape(-1), \
            float(Fx.fom), float(Fy.fom)

    def log(rec):
        history.write(json.dumps(rec, default=float) + "\n")
        history.flush()

    for beta in _beta_stages(args.beta_schedule):
        state["beta"] = beta
        opt = nlopt.opt(nlopt.LD_MMA, nlat)
        opt.set_lower_bounds(np.zeros(nlat))
        opt.set_upper_bounds(np.ones(nlat))
        opt.set_maxeval(int(args.maxeval_per_stage))
        opt.set_xtol_rel(1e-4)

        def nlopt_obj(x, grad):
            f, dlat, fx, fy = objective_and_grad(x, beta)
            gs, gv = constraints.residuals(x, beta)
            feasible = gs <= 0 and gv <= 0
            if feasible and f > state["best_feasible_obj"]:
                state["best_feasible_obj"] = f
                state["best_feasible"] = np.array(x, copy=True)
                np.savez_compressed(run_root / "checkpoints" / "best_feasible.npz",
                                    latent=x, beta=np.array(beta), objective=np.array(f))
            state["iter"] += 1
            log({"attempt": attempt_id, "iter": state["iter"], "beta": beta,
                 "Fx": fx, "Fy": fy, "objective": f,
                 "g_solid": gs, "g_void": gv, "feasible": bool(feasible)})
            if grad.size:
                grad[:] = -dlat  # minimising -(Fx+Fy)
            return -f

        def c_solid(x, grad):
            val, g = constraints.solid_residual_and_grad(x, beta)
            if grad.size:
                grad[:] = g
            return val

        def c_void(x, grad):
            val, g = constraints.void_residual_and_grad(x, beta)
            if grad.size:
                grad[:] = g
            return val

        opt.set_min_objective(nlopt_obj)
        opt.add_inequality_constraint(c_solid, 0.0)
        opt.add_inequality_constraint(c_void, 0.0)
        t0 = time.time()
        try:
            latent = opt.optimize(latent)
        except Exception as exc:
            log({"attempt": attempt_id, "beta": beta,
                 "stage_error": f"{type(exc).__name__}: {exc}"})
        log({"attempt": attempt_id, "beta": beta, "stage_done": True,
             "stop_reason": int(opt.last_optimize_result()),
             "stage_seconds": time.time() - t0})

    # finalise: best feasible if any, else last latent (flagged)
    best = state["best_feasible"] if state["best_feasible"] is not None else latent
    np.savez_compressed(run_root / "final_design.npz", latent=best,
                        beta=np.array(_beta_stages(args.beta_schedule)[-1]),
                        had_feasible=np.array(state["best_feasible"] is not None))
    stop = {
        "category": "completed" if state["best_feasible"] is not None
        else "geometry_infeasible",
        "best_feasible_objective": state["best_feasible_obj"]
        if np.isfinite(state["best_feasible_obj"]) else None,
        "attempt": attempt_id,
    }
    (attempt / "stop.json").write_text(json.dumps(stop, indent=2) + "\n")
    history.close()
    print(json.dumps(stop, indent=2))


if __name__ == "__main__":
    main()
