"""FDTD-free constraint repair (feasibility restoration) for gated stages.

The length-scale constraints are pure autograd (milliseconds per evaluation),
while one objective evaluation costs ~915 s of GPU FDTD.  Production run r4
measured the consequence of ignoring that: its beta=16 stage burned 12 FDTD
evaluations moving g by -3% and aborted maxeval_infeasible, while an offline
constraint-only solve moved the SAME latent's g by -99.4% in six minutes of
CPU.  So: before a gated (beta >= feasibility_gate_beta) FDTD stage starts,
project the latent into the feasible set here, for free.

Formulation: epigraph MMA  --  min t  s.t.  g_solid <= t, g_void <= t,
0 <= x <= 1, t in [-1, 1] -- which drives max(gs, gv) down directly (a plain
sum objective was measured to park gs at the -1e-5 floor while gv crept for
thousands of evaluations).  Stops as soon as both residuals <= -margin.

This is a numerical assist inside the SAME constrained problem the optimizer
already solves; the certified evaluator/gradient chain is untouched and the
final independent DRC + exact-binary FDTD SUCCESS gates are unchanged.
"""

from __future__ import annotations

import time

import numpy as np

REPAIR_VERSION = "feasibility_repair/v1-epigraph"


def repair_to_feasible(constraints, latent, beta, *, margin: float = 1e-6,
                       maxeval: int = 40000, rho_init: float = 1e-2,
                       log=None) -> dict:
    """Drive (g_solid, g_void) at `beta` to <= -margin without any FDTD.

    Returns a dict with feasible flag, repaired latent, eval count, wall time
    and before/after residuals.  Never raises on non-convergence -- the caller
    decides (production aborts fail-closed on infeasible repair).
    """
    latent = np.asarray(latent, float).reshape(-1)
    gs0, gv0 = constraints.residuals(latent, beta)
    result = {
        "version": REPAIR_VERSION,
        "beta": float(beta),
        "margin": float(margin),
        "maxeval": int(maxeval),
        "g_before": [float(gs0), float(gv0)],
    }
    if gs0 <= -margin and gv0 <= -margin:
        result.update({"feasible": True, "latent": latent,
                       "evaluations": 0, "seconds": 0.0,
                       "g_after": [float(gs0), float(gv0)],
                       "note": "already feasible; repair skipped"})
        return result

    import nlopt  # lazy: keeps this module importable without nlopt

    n = latent.size
    opt = nlopt.opt(nlopt.LD_MMA, n + 1)      # variables: (latent, t)
    lower = np.zeros(n + 1)
    upper = np.ones(n + 1)
    lower[-1], upper[-1] = -1.0, 1.0
    opt.set_lower_bounds(lower)
    opt.set_upper_bounds(upper)
    opt.set_maxeval(int(maxeval))
    opt.set_param("rho_init", float(rho_init))

    state = {"n": 0, "solution": None, "t0": time.time()}

    def objective(v, grad):
        if grad.size:
            grad[:] = 0.0
            grad[-1] = 1.0
        return float(v[-1])

    def make_constraint(which):
        def constraint(v, grad):
            state["n"] += 1
            x, t = v[:-1], v[-1]
            if which == "solid":
                value, gradient = constraints.solid_residual_and_grad(x, beta)
            else:
                value, gradient = constraints.void_residual_and_grad(x, beta)
            if grad.size:
                grad[:-1] = gradient
                grad[-1] = -1.0
            if which == "void":
                gs, _ = constraints.solid_residual_and_grad(x, beta)
                if gs <= -margin and value <= -margin and state["solution"] is None:
                    state["solution"] = np.array(x, copy=True)
                    opt.force_stop()
                if log is not None and state["n"] % 8000 == 0:
                    log(f"[repair] beta={beta:g} eval~{state['n'] // 2}: "
                        f"gs={gs:.3e} gv={value:.3e} "
                        f"({time.time() - state['t0']:.0f}s)")
            return value - t
        return constraint

    opt.set_min_objective(objective)
    opt.add_inequality_constraint(make_constraint("solid"), 0.0)
    opt.add_inequality_constraint(make_constraint("void"), 0.0)
    start = np.concatenate([latent, [max(gs0, gv0, 1e-3)]])
    try:
        opt.optimize(start)
    except Exception:  # noqa: BLE001 - ForcedStop / maxeval end both fine
        pass

    repaired = state["solution"]
    feasible = repaired is not None
    if not feasible:
        repaired = latent                     # never hand back a worse vector
    gs1, gv1 = constraints.residuals(repaired, beta)
    result.update({
        "feasible": bool(feasible),
        "latent": np.asarray(repaired, float),
        "evaluations": int(state["n"]),
        "seconds": float(time.time() - state["t0"]),
        "g_after": [float(gs1), float(gv1)],
    })
    return result
