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

Stage scheduling (2026-08-07 rework -- "beta=2 must actually FIND a topology"):

* --beta-schedule entries are ``beta[:maxeval[:min_evals]]`` so the low-beta
  stages get a LARGE budget (topology forms there) while high-beta stages stay
  short.  Plain ``2,4,8`` still works (falls back to --maxeval-per-stage).
* constraint warm-up: stages with beta < --constraint-start-beta run WITHOUT
  the length-scale inequality constraints (pure FOM ascent).  Enforcing Zhou
  penalties on a gray low-beta field only throttles the FOM; the constraints
  join the MMA problem once projection is meaningful.  g_solid/g_void are
  still evaluated and logged every iteration, and best-FEASIBLE bookkeeping
  still requires real g<=0, so the DRC/finalize invariants are unchanged.
* binarization polish: after the ladder, if the nominal density still has more
  than --gray-tol gray nodes, extra beta-doubling stages are appended (up to
  --beta-cap) so the returned design is actually near-binary and the exact
  binary projection does not change the FOM behind the optimiser's back.

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
from dataclasses import replace as dc_replace
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


def parse_beta_schedule(spec: str, default_maxeval: int, default_min_evals: int):
    """Parse ``beta[:maxeval[:min_evals]]`` comma entries into stage dicts.

    Plain ``2,4,8`` keeps working: such entries take ``default_maxeval`` and
    ``default_min_evals`` (the latter clamped to the stage maxeval, so a short
    smoke schedule like BETA_SCHEDULE=2,4 MAXEVAL=2 stays valid).  An EXPLICIT
    min_evals larger than the stage maxeval is a configuration error.
    """
    stages = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        bits = part.split(":")
        try:
            beta = float(bits[0])
            maxeval = int(bits[1]) if len(bits) > 1 and bits[1] else int(default_maxeval)
            explicit_min = len(bits) > 2 and bool(bits[2])
            min_evals = int(bits[2]) if explicit_min else int(default_min_evals)
        except (ValueError, IndexError):
            raise SystemExit(f"bad beta-schedule entry {part!r} "
                             "(expected beta[:maxeval[:min_evals]])")
        if len(bits) > 3:
            raise SystemExit(f"bad beta-schedule entry {part!r} (too many fields)")
        if beta <= 0 or maxeval < 1 or min_evals < 1:
            raise SystemExit(f"bad beta-schedule entry {part!r} (non-positive value)")
        if explicit_min and min_evals > maxeval:
            raise SystemExit(f"bad beta-schedule entry {part!r} (min_evals > maxeval)")
        stages.append({"beta": beta, "maxeval": maxeval,
                       "min_evals": min(min_evals, maxeval)})
    if not stages:
        raise SystemExit("empty beta schedule")
    if any(a["beta"] >= b["beta"] for a, b in zip(stages, stages[1:])):
        raise SystemExit(f"beta schedule must be strictly increasing: {spec!r}")
    return stages


def _beta_stages(spec: str):
    return [s["beta"] for s in parse_beta_schedule(spec, 1, 1)]


# Gray-node band for binarization accounting: a nominal-density node outside
# (GRAY_LO, GRAY_HI) counts as committed solid/void.
GRAY_LO = 0.02
GRAY_HI = 0.98


def gray_fraction_unique(mapping, latent, beta,
                         lo: float = GRAY_LO, hi: float = GRAY_HI) -> float:
    """Fraction of unique nominal-density nodes that are neither solid nor void."""
    field = np.asarray(
        mapping.field_unique(np.asarray(latent, float).reshape(-1), beta), float
    )
    return float(np.mean((field > lo) & (field < hi)))


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
        here / "adaptive_stage.py",
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
    # Per-stage budgets: beta[:maxeval[:min_evals]].  The first stage carries
    # the big budget on purpose -- the topology is FOUND at beta=2; later
    # stages only harden it.  --maxeval-per-stage is the fallback for plain
    # entries.
    ap.add_argument("--beta-schedule",
                    default=os.environ.get(
                        "BETA_SCHEDULE", "2:40:12,4:16,8:16,16:12,32:10,64:10"))
    ap.add_argument("--maxeval-per-stage", type=int, default=12)
    # THE effective MMA warm-up knob (measured in the runtime-opt workstream,
    # probe_mma_step): CCSA starts its inner penalty at rho_init and relaxes
    # ~10x per outer iteration.  nlopt's default rho_init=1.0 throttles the
    # first ~5 evaluations of EVERY stage to step RMS 4.45e-6 -> 4.45e-5 ->
    # 6.5e-4 (each a ~15 min FDTD evaluation wasted).  rho_init=1e-2 starts at
    # ~4.4e-4 and reaches real movement by evaluation 2-3 without violent
    # first steps.
    ap.add_argument("--rho-init", type=float,
                    default=float(os.environ.get("VC_RHO_INIT", "1e-2")))
    # Constraint warm-up: stages with beta < this run WITHOUT the length-scale
    # inequality constraints (pure FOM ascent); constraints join at this beta.
    ap.add_argument("--constraint-start-beta", type=float,
                    default=float(os.environ.get("VC_CONSTRAINT_START_BETA", "8")))
    # Binarization polish: after the ladder, keep doubling beta (small budget)
    # while the nominal density has more than gray-tol gray nodes.
    ap.add_argument("--gray-tol", type=float,
                    default=float(os.environ.get("VC_GRAY_TOL", "0.05")))
    ap.add_argument("--beta-cap", type=float,
                    default=float(os.environ.get("VC_BETA_CAP", "256")))
    ap.add_argument("--polish-maxeval", type=int,
                    default=int(os.environ.get("VC_POLISH_MAXEVAL", "8")))
    # Adaptive beta continuation (see adaptive_stage.py).  Env defaults let the
    # launcher steer a pilot without editing this file.
    ap.add_argument("--min-evals-per-stage", type=int,
                    default=int(os.environ.get("MIN_EVALS_PER_STAGE", "3")))
    ap.add_argument("--convergence-window", type=int,
                    default=int(os.environ.get("VC_CONV_WINDOW", "3")))
    ap.add_argument("--objective-rel-tol", type=float,
                    default=float(os.environ.get("VC_OBJ_REL_TOL", "5e-3")))
    ap.add_argument("--latent-rms-tol", type=float,
                    default=float(os.environ.get("VC_LATENT_RMS_TOL", "1e-3")))
    ap.add_argument("--latent-max-tol", type=float,
                    default=float(os.environ.get("VC_LATENT_MAX_TOL", "1e-2")))
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
    from adaptive_stage import (
        ABORT_REASONS, CONTINUE, AdaptiveConfig, StageController,
        is_intentional_stop,
    )
    from geometric_constraints import LengthScaleConstraints
    from geometry_drc import geometry_drc
    from iteration_plots import density_metrics, save_iteration_plots
    from volume_current_evaluator import VolumeCurrentEvaluator
    from autograd import tensor_jacobian_product

    adaptive_cfg = AdaptiveConfig(
        min_evals_per_stage=args.min_evals_per_stage,
        max_evals_per_stage=args.maxeval_per_stage,
        convergence_window=args.convergence_window,
        objective_rel_tol=args.objective_rel_tol,
        latent_rms_tol=args.latent_rms_tol,
        latent_max_tol=args.latent_max_tol,
    ).validate()
    stages = parse_beta_schedule(
        args.beta_schedule, args.maxeval_per_stage, args.min_evals_per_stage)
    if args.gray_tol <= 0 or args.beta_cap < stages[-1]["beta"] \
            or args.polish_maxeval < 1:
        raise SystemExit("invalid binarization-polish configuration "
                         "(gray-tol>0, beta-cap>=last beta, polish-maxeval>=1)")
    if not (args.rho_init > 0 and np.isfinite(args.rho_init)):
        raise SystemExit(f"--rho-init must be a finite positive number, "
                         f"got {args.rho_init}")

    # Import provenance BEFORE the model is built (load_model imports lumapi).
    # A wrong Python API only explodes later, inside the adjoint, so record what
    # was actually loaded at both points and fail closed on a mismatch.
    print("[provenance] before load_model: "
          + json.dumps(lib.import_provenance(), default=str), flush=True)
    model = lib.load_model()
    provenance = lib.import_provenance()
    print("[provenance] after  load_model: "
          + json.dumps(provenance, default=str), flush=True)
    lib.assert_approved_lumapi()
    print(f"[provenance] lumapi OK: {provenance['lumapi_file']}", flush=True)
    print(f"[provenance] engine   : {provenance['fdtd_engine']}", flush=True)
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
        # Per-stage budgets are part of the configuration identity.
        "beta_schedule": [dict(s) for s in stages],
        "constraint_start_beta": float(args.constraint_start_beta),
        "binarization_policy": {
            "gray_lo": GRAY_LO, "gray_hi": GRAY_HI,
            "gray_tol": float(args.gray_tol),
            "beta_cap": float(args.beta_cap),
            "polish_maxeval": int(args.polish_maxeval),
        },
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
        # Adaptive continuation policy is part of the configuration identity:
        # any change lands in config_hash and refuses to resume older attempts.
        "adaptive": adaptive_cfg.to_dict(),
        "nlopt_rho_init": float(args.rho_init),
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
    # Recorded AFTER config_hash on purpose: which lumapi/engine was loaded is
    # audit evidence, not part of the configuration identity.  Hashing it would
    # make config_hash depend on PYTHONPATH and break every resume.
    contract["import_provenance"] = provenance

    # attempt bookkeeping + resume guard
    existing = sorted((run_root / "attempts").glob("attempt_*"))
    latent = np.asarray(model.x0, float).reshape(-1).copy()
    state = {"beta": 0.0, "iter": 0, "best_feasible": None,
             "best_feasible_obj": -np.inf, "best_beta": None,
             "records": []}   # in-memory eval records for the plot dashboard
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
        nominal_unique = np.array(phys[:-1, :-1, 0], copy=True)
        Fx = ev_x.value_and_gradient(phys, label=f"it{state['iter']}_x",
                                     density_mode="probe_safe")
        Fy = ev_y.value_and_gradient(phys, label=f"it{state['iter']}_y",
                                     density_mode="probe_safe")
        g_phys = (Fx.gradient_physical + Fy.gradient_physical).reshape(-1)
        dlat = tensor_jacobian_product(lambda z: mapping(z, beta))(x, g_phys)
        return float(Fx.fom + Fy.fom), np.asarray(dlat, float).reshape(-1), \
            float(Fx.fom), float(Fy.fom), nominal_unique

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

    def _stage_drc_diagnostic(latent_like, beta, source_label):
        """Exact-binary DRC snapshot of the current best -- DIAGNOSTIC ONLY.

        Never used as an optimizer constraint or a gate; a diagnostics hiccup
        must not kill a multi-hour production run, so errors are recorded
        instead of raised.
        """
        path = run_root / f"stage_drc_beta{beta:g}.json"
        try:
            mask = (np.asarray(
                mapping.filter_unique(np.asarray(latent_like, float))
            ) >= 0.5).astype(np.uint8)
            gap_env = os.environ.get("PERIODIC_ISOLATION_GAP_UM")
            min_gap = float(gap_env) if gap_env and float(gap_env) > 0 else None
            report = geometry_drc(
                mask, spacing_um=mapping.config.dx_um,
                min_solid_width_um=args.mfs_um, min_void_width_um=args.mgs_um,
                min_gap_um=min_gap,
            )
            report.update({
                "diagnostic_only": True, "beta": beta, "source": source_label,
            })
        except Exception as exc:  # noqa: BLE001 - diagnostics must not abort a run
            report = {"diagnostic_only": True, "beta": beta,
                      "source": source_label,
                      "error": f"{type(exc).__name__}: {exc}"}
        path.write_text(json.dumps(report, indent=2, default=float) + "\n")
        return report

    stage_failure = None       # None => no deterministic stage failure
    adaptive_abort = None      # set to an ABORT_REASONS value by the controller
    completed_all_stages = True
    stage_summaries = []
    schedule = [dict(s) for s in stages]   # may grow: binarization polish
    polish_rounds = 0
    last_beta_run = float(schedule[0]["beta"])
    stage_index = 0
    while stage_index < len(schedule):
        stage_spec = schedule[stage_index]
        beta = float(stage_spec["beta"])
        # Constraint warm-up: below constraint_start_beta the length-scale
        # constraints are LOGGED but not handed to MMA -- the low-beta stages
        # exist to let the FOM find a topology, not to chase feasibility of a
        # gray field.  Feasibility bookkeeping below still uses real residuals.
        constraints_active = beta >= args.constraint_start_beta
        state["beta"] = beta
        last_beta_run = beta
        stage_cfg = dc_replace(
            adaptive_cfg,
            max_evals_per_stage=int(stage_spec["maxeval"]),
            min_evals_per_stage=int(stage_spec["min_evals"]),
        )
        controller = StageController(stage_cfg, beta,
                                     constraints_active=constraints_active)
        opt = nlopt.opt(nlopt.LD_MMA, nlat)
        opt.set_lower_bounds(np.zeros(nlat))
        opt.set_upper_bounds(np.ones(nlat))
        # nlopt's own maxeval stays as a backstop; the controller normally
        # force-stops at exactly the stage budget itself.
        opt.set_maxeval(int(stage_spec["maxeval"]))
        # xtol MUST be off: nlopt's own xtol can end a stage after one tiny
        # step, bypassing min_evals_per_stage AND the feasibility gating.
        # Stage termination belongs to the adaptive controller alone (maxeval
        # is the backstop); the controller's latent-quiet test replaces xtol.
        opt.set_xtol_rel(0.0)
        # CCSA inner-penalty warm start (see --rho-init); verify nlopt took it.
        opt.set_param("rho_init", float(args.rho_init))
        realized_rho = float(opt.get_param("rho_init", -1.0))
        if not np.isclose(realized_rho, args.rho_init, rtol=0, atol=0):
            raise SystemExit(
                f"nlopt did not accept rho_init={args.rho_init} "
                f"(realized {realized_rho})")

        def nlopt_obj(x, grad, _beta=beta, _controller=controller, _opt=opt,
                      _active=constraints_active):
            f, dlat, fx, fy, rho_unique = objective_and_grad(x, _beta)
            gray = float(np.mean((rho_unique > GRAY_LO) & (rho_unique < GRAY_HI)))
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
            decision = _controller.record(f, gs, gv, x)
            previous = state["records"][-1] if state["records"] else None
            step_rms = float("nan")
            if previous is not None and "latent" in previous:
                delta = np.asarray(x, float) - previous["latent"]
                step_rms = float(np.sqrt(np.mean(delta * delta)))
            rec = {"attempt": attempt_id, "iter": state["iter"], "beta": _beta,
                   "Fx": fx, "Fy": fy, "objective": f,
                   "g_solid": gs, "g_void": gv,
                   # NOTE: constraint_feasible != DRC feasible; DRC is the final gate
                   "constraint_feasible": bool(constraint_feasible),
                   "constraints_active": bool(_active),
                   "gray_fraction": gray,
                   "latent_step_rms": step_rms,
                   "adaptive_decision": decision,
                   **density_metrics(rho_unique)}
            rec["frac_rails"] = rec["frac_below_0.01"] + rec["frac_above_0.99"]
            log(rec)
            rec["latent"] = np.array(x, copy=True)   # in-memory only, for step_rms
            state["records"].append(rec)
            try:
                save_iteration_plots(run_root, rho_unique, state["iter"],
                                     _beta, state["records"])
            except Exception as plot_exc:  # noqa: BLE001 - plots never kill a run
                print(f"[plots] skipped for iter {state['iter']}: {plot_exc}",
                      flush=True)
            if decision != CONTINUE:
                # Intentional stop: nlopt raises ForcedStop from optimize();
                # is_intentional_stop() keeps it out of the failure classifier.
                _opt.force_stop()
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
        if constraints_active:
            opt.add_inequality_constraint(c_solid, 0.0)
            opt.add_inequality_constraint(c_void, 0.0)
        t0 = time.time()
        try:
            latent = opt.optimize(latent)
        except Exception as exc:
            if is_intentional_stop(exc, controller):
                # Adaptive stop: hand the LAST OBSERVED latent to the next
                # stage (optimize() raised, so it returned nothing).  The
                # best-feasible checkpoint was already persisted per-eval.
                latent = np.array(controller.last_latent, copy=True)
            else:
                # P1-4: classify and STOP; do not mislabel completed
                import traceback
                stage_failure = _classify(exc)
                completed_all_stages = False
                tb = traceback.format_exc()
                (attempt / f"stage_error_beta{beta:g}.txt").write_text(tb)
                traceback.print_exc()
                log({"attempt": attempt_id, "beta": beta, "stage_error":
                     f"{type(exc).__name__}: {exc}", "category": stage_failure})
                break

        try:
            nlopt_code = int(opt.last_optimize_result())
        except Exception:  # noqa: BLE001
            nlopt_code = None
        stage_reason = controller.stop_reason or f"nlopt_stop_{nlopt_code}"
        stage_gray = gray_fraction_unique(mapping, latent, beta)
        summary = {
            **controller.summary(stage_reason),
            "nlopt_result_code": nlopt_code,
            "stage_maxeval": int(stage_spec["maxeval"]),
            "stage_min_evals": int(stage_spec["min_evals"]),
            "polish_stage": bool(stage_spec.get("polish", False)),
            "stage_end_gray_fraction": stage_gray,
            "best_feasible_objective": (
                state["best_feasible_obj"]
                if state["best_feasible"] is not None else None),
            "stage_seconds": time.time() - t0,
        }
        drc_source = ("best_feasible" if state["best_feasible"] is not None
                      else "last_latent")
        drc_latent = (state["best_feasible"]
                      if state["best_feasible"] is not None else latent)
        drc = _stage_drc_diagnostic(drc_latent, beta, drc_source)
        summary["stage_drc_pass"] = drc.get("pass")
        stage_summaries.append(summary)
        log({"attempt": attempt_id, "beta": beta, "stage_done": True,
             "stage_summary": summary})
        print(f"[stage] beta={beta:g} reason={stage_reason} "
              f"evals={controller.evaluations} constrained={constraints_active} "
              f"gray={stage_gray:.4f} drc_pass={drc.get('pass')}",
              flush=True)
        if stage_reason in ABORT_REASONS:
            adaptive_abort = stage_reason
            completed_all_stages = False
            break
        stage_index += 1
        # Binarization polish: the ladder alone often ends visibly gray.  Keep
        # doubling beta with a short constrained stage until the nominal field
        # is near-binary or beta hits the cap (recorded either way; the exact
        # binary projection remains the final authority).
        if stage_index == len(schedule) and stage_gray > args.gray_tol \
                and beta * 2.0 <= args.beta_cap:
            polish_rounds += 1
            polish = {"beta": beta * 2.0,
                      "maxeval": int(args.polish_maxeval),
                      "min_evals": min(2, int(args.polish_maxeval)),
                      "polish": True}
            schedule.append(polish)
            log({"attempt": attempt_id, "polish_stage_appended": polish,
                 "gray_fraction": stage_gray})
            print(f"[polish] gray={stage_gray:.4f} > tol={args.gray_tol:g} "
                  f"-> appending beta={polish['beta']:g} stage", flush=True)

    # finalise: prefer best-feasible; record its OWN beta (P1-3), not the last.
    had_feasible = state["best_feasible"] is not None
    best = state["best_feasible"] if had_feasible else latent
    best_beta = state["best_beta"] if had_feasible else last_beta_run
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
    elif adaptive_abort is not None:
        category = adaptive_abort                # stage_stalled_infeasible / maxeval_infeasible
    elif not had_feasible:
        category = "geometry_infeasible"
    elif not completed_all_stages:
        category = "incomplete"
    else:
        category = "completed"
    final_gray = gray_fraction_unique(mapping, best, best_beta)
    stop = {
        "category": category,
        "had_feasible": bool(had_feasible),
        "best_feasible_objective": state["best_feasible_obj"] if had_feasible else None,
        "best_beta": best_beta if had_feasible else None,
        "attempt": attempt_id,
        "adaptive": adaptive_cfg.to_dict(),
        "constraint_start_beta": float(args.constraint_start_beta),
        "polish_rounds": polish_rounds,
        "final_gray_fraction": final_gray,
        "binarization_converged": bool(final_gray <= args.gray_tol),
        "stage_summaries": stage_summaries,
        "note": "constraint-feasible only; 500 nm is certified by final_projection DRC",
    }
    (attempt / "stop.json").write_text(
        json.dumps(stop, indent=2, default=float) + "\n")
    history.close()
    print(json.dumps(stop, indent=2, default=float))
    # P0-4: a stage failure must make the PROCESS fail so the launcher aborts and
    # never auto-finalises after an optimizer failure.
    if stage_failure is not None:
        raise SystemExit(6)
    # An adaptive abort is not a solver failure, but the ladder did NOT finish:
    # exit nonzero (distinct code) so the launcher never auto-finalises it.
    if adaptive_abort is not None:
        raise SystemExit(7)


if __name__ == "__main__":
    main()
