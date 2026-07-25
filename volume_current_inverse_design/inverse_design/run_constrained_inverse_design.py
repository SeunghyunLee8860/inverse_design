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
    for p in sorted(str(x) for x in paths):
        h.update(Path(p).read_bytes())  # strict: a missing/unreadable file must raise
    return h.hexdigest()[:16]


def _config_hash(cfg: dict) -> str:
    return hashlib.sha256(
        json.dumps(cfg, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]


def _beta_stages(spec: str):
    return [float(v) for v in spec.split(",") if v.strip()]


def production_code_files(root: Path, here: Path):
    """Every source file whose contents change the result (shared with the
    finalizer so its provenance check hashes the exact same set)."""
    return [
        root / "bundle/periodic_constrained_mapping.py",
        root / "bundle/periodic_filter.py",
        root / "bundle/tairte4_volume_model.py",
        root / "bundle/msopt/Filters.py",
        root / "bundle/msopt/Sub_Mapping.py",
        root / "bundle/msopt/Lumerical_utill.py",
        root / "bundle/msopt/Mapping.py",
        root / "eqc_lib.py",
        root / "volume_current_evaluator.py",
        root / "volume_current_adjoint_core.py",
        root / "volume_current_colored_jacobian.py",
        root / "volume_current_yee_metric.py",
        root / "collocated_coherent_fom.py",
        here / "geometric_constraints.py",
        here / "geometry_drc.py",
        here / "final_projection.py",
        here / "run_constrained_inverse_design.py",
    ]


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

    if args.robust:
        # P1-1: the epigraph three-field robust formulation is NOT implemented
        # (optimizer dim, eroded/nominal/dilated constraints, 18-solve loop all
        # absent).  Fail loudly rather than silently run the nominal problem.
        raise SystemExit(
            "--robust is not implemented yet (would need an epigraph variable and "
            "the eroded/nominal/dilated worst-case constraints; 18 EM solves/eval). "
            "Run the nominal formulation without --robust.")

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

    # P0-1: resolve --output relative to the CURRENT WORKING DIRECTORY (where the
    # launcher runs), NOT relative to this file's dir, so the shell's OUT and the
    # runner's run_root are the same path.
    run_root = Path(args.output).expanduser().resolve()
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
        # P1-5: record RESOLVED config that changes results
        "maxeval_per_stage": int(args.maxeval_per_stage),
        "optimizer": "nlopt.LD_MMA",
        "nlopt_version": getattr(nlopt, "__version__", "unknown"),
        "resolved_constraint": {
            "c_decay": float(args.constraint_c if args.constraint_c is not None
                             else os.environ.get("VC_CONSTRAINT_C", 400.0)),
            "tol_solid": float(args.tol_solid if args.tol_solid is not None
                               else os.environ.get("VC_TOL_SOLID", 1e-5)),
            "tol_void": float(args.tol_void if args.tol_void is not None
                              else os.environ.get("VC_TOL_VOID", 1e-5)),
            "pnorm_p": float(os.environ.get("VC_CONSTRAINT_PNORM", 8.0)),
        },
        "safety_slack_env": os.environ.get("VC_SAFETY_SLACK"),
        "gpu": args.gpu, "fdtd_threads": os.environ.get("FDTD_THREADS"),
    }
    # P1-5: hash EVERY file whose contents change the result, and fail loudly if
    # any is unreadable (a silently-skipped file would make the hash meaningless).
    code_files = production_code_files(ROOT, HERE)
    missing = [str(p) for p in code_files if not Path(p).exists()]
    if missing:
        raise SystemExit(f"code hash incomplete; missing files: {missing}")
    contract["code_hash"] = _code_hash(code_files)
    contract["config_hash"] = _config_hash(contract)

    # attempt bookkeeping + resume guard
    existing = sorted((run_root / "attempts").glob("attempt_*"))
    latent = np.asarray(model.x0, float).reshape(-1).copy()
    state = {"beta": 0.0, "iter": 0, "best_feasible": None,
             "best_feasible_obj": -np.inf, "best_beta": None}
    if args.resume and existing:
        prev = json.loads((existing[-1] / "contract.json").read_text())
        if prev.get("code_hash") != contract["code_hash"] or \
           prev.get("config_hash") != contract["config_hash"]:
            raise SystemExit(
                "resume refused: code/config hash changed -> start a new attempt")
        ckpt = run_root / "checkpoints" / "best_feasible.npz"
        if ckpt.exists():
            z = np.load(ckpt)
            latent = np.asarray(z["latent"], float).reshape(-1).copy()
            # P1-2/P1-3: fully restore best-feasible state so a resume never
            # regresses to "infeasible" or loses the winning beta/objective.
            state["best_feasible"] = latent.copy()
            state["best_feasible_obj"] = float(z["objective"]) if "objective" in z else -np.inf
            state["best_beta"] = float(z["beta"]) if "beta" in z else None

    # P1-6: attempt id = max numeric + 1 (gaps never collide); contract immutable
    # via exclusive create (mkdir exist_ok=False).
    nums = [int(p.name.split("_")[-1]) for p in existing if p.name.split("_")[-1].isdigit()]
    attempt_id = (max(nums) + 1) if nums else 1
    attempt = run_root / "attempts" / f"attempt_{attempt_id:04d}"
    attempt.mkdir(parents=True, exist_ok=False)
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

    def _classify(exc) -> str:
        text = f"{type(exc).__name__}: {exc}".lower()
        if any(k in text for k in ("licen", "ansysli", "flexnet")):
            return "license_failure"
        if any(k in text for k in ("no result", "no d-card", "session run failed",
                                   "getresult", "solver")):
            return "solver_failure"
        return "numerical_failure"

    stage_failure = None       # None => no deterministic stage failure
    completed_all_stages = True
    for beta in _beta_stages(args.beta_schedule):
        state["beta"] = beta
        opt = nlopt.opt(nlopt.LD_MMA, nlat)
        opt.set_lower_bounds(np.zeros(nlat))
        opt.set_upper_bounds(np.ones(nlat))
        opt.set_maxeval(int(args.maxeval_per_stage))
        opt.set_xtol_rel(1e-4)

        def nlopt_obj(x, grad, _beta=beta):
            f, dlat, fx, fy = objective_and_grad(x, _beta)
            gs, gv = constraints.residuals(x, _beta)
            constraint_feasible = gs <= 0 and gv <= 0
            if constraint_feasible and f > state["best_feasible_obj"]:
                state["best_feasible_obj"] = f
                state["best_feasible"] = np.array(x, copy=True)
                state["best_beta"] = float(_beta)
                # atomic checkpoint carrying latent+beta+objective together.
                # P0-2: the temp name MUST end in .npz, else np.savez_compressed
                # appends ".npz" (best_feasible.tmp.npz.npz) and replace() fails.
                tmp = run_root / "checkpoints" / "best_feasible.tmp.npz"
                np.savez_compressed(tmp, latent=x, beta=np.array(_beta),
                                    objective=np.array(f))
                tmp.replace(run_root / "checkpoints" / "best_feasible.npz")
            state["iter"] += 1
            log({"attempt": attempt_id, "iter": state["iter"], "beta": _beta,
                 "Fx": fx, "Fy": fy, "objective": f,
                 "g_solid": gs, "g_void": gv,
                 # NOTE: constraint_feasible != DRC feasible; DRC is the final gate
                 "constraint_feasible": bool(constraint_feasible)})
            if grad.size:
                grad[:] = -dlat  # minimising -(Fx+Fy)
            return -f

        def c_solid(x, grad, _beta=beta):
            val, g = constraints.solid_residual_and_grad(x, _beta)
            if grad.size:
                grad[:] = g
            return val

        def c_void(x, grad, _beta=beta):
            val, g = constraints.void_residual_and_grad(x, _beta)
            if grad.size:
                grad[:] = g
            return val

        opt.set_min_objective(nlopt_obj)
        opt.add_inequality_constraint(c_solid, 0.0)
        opt.add_inequality_constraint(c_void, 0.0)
        t0 = time.time()
        try:
            latent = opt.optimize(latent)
        except Exception as exc:  # P1-4: classify and STOP; do not mislabel completed
            stage_failure = _classify(exc)
            completed_all_stages = False
            log({"attempt": attempt_id, "beta": beta, "stage_error":
                 f"{type(exc).__name__}: {exc}", "category": stage_failure})
            break
        log({"attempt": attempt_id, "beta": beta, "stage_done": True,
             "stop_reason": int(opt.last_optimize_result()),
             "stage_seconds": time.time() - t0})

    # finalise: prefer best-feasible; record its OWN beta (P1-3), not the last.
    had_feasible = state["best_feasible"] is not None
    best = state["best_feasible"] if had_feasible else latent
    best_beta = state["best_beta"] if had_feasible else _beta_stages(args.beta_schedule)[-1]
    # P0-6: persist the mapping/mode identity so the finalizer can refuse a
    # design produced under a different mapping/isolation/MFS/mode.
    mapping_identity = json.dumps({
        "mapping_config": mapping.config.to_dict(),
        "isolation_gap_um": float(getattr(mapping, "isolation_gap_um", 0.0)),
        "mfs_um": args.mfs_um, "mgs_um": args.mgs_um,
        "mapping_mode": os.environ.get("MSOPT_MAPPING", "periodic_constrained"),
    }, sort_keys=True)
    np.savez_compressed(run_root / "final_design.npz", latent=best,
                        beta=np.array(best_beta),
                        objective=np.array(state["best_feasible_obj"] if had_feasible else np.nan),
                        had_feasible=np.array(had_feasible),
                        code_hash=np.array(contract["code_hash"]),
                        config_hash=np.array(contract["config_hash"]),
                        mapping_identity=np.array(mapping_identity),
                        attempt=np.array(attempt_id))
    if stage_failure is not None:
        category = stage_failure                 # deterministic/solver/license failure
    elif not had_feasible:
        category = "geometry_infeasible"
    elif not completed_all_stages:
        category = "incomplete"
    else:
        category = "completed"
    stop = {
        "category": category,
        "had_feasible": bool(had_feasible),
        "best_feasible_objective": state["best_feasible_obj"] if had_feasible else None,
        "best_beta": best_beta if had_feasible else None,
        "attempt": attempt_id,
        "note": "constraint-feasible only; 500 nm is certified by final_projection DRC",
    }
    (attempt / "stop.json").write_text(json.dumps(stop, indent=2) + "\n")
    history.close()
    print(json.dumps(stop, indent=2))
    # P0-4: a stage failure must make the PROCESS fail so the launcher aborts and
    # never auto-finalises after an optimizer failure.
    if stage_failure is not None:
        raise SystemExit(6)


if __name__ == "__main__":
    main()
