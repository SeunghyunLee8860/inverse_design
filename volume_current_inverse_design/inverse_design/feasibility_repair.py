"""FDTD-free constraint repair (feasibility restoration) for gated stages.

The length-scale constraints are pure autograd (milliseconds per evaluation),
while one objective evaluation costs ~915 s of GPU FDTD.  Production run r4
measured the consequence of ignoring that: its beta=16 stage burned 12 FDTD
evaluations moving g by -3% and aborted maxeval_infeasible, while an offline
constraint-only solve moved the SAME latent's g by -99.4% in six minutes of
CPU.  So: before a gated (beta >= feasibility_gate_beta) FDTD stage starts,
project the latent into the feasible set here, for free.

Formulation: unconstrained MMA on the SUM  min (g_solid + g_void), stopping
the moment both residuals are <= -margin.  An epigraph form (min t s.t.
g_i <= t) looked cleaner on paper and passed on a 61-node problem, but on the
57,600-variable production problem it HUNG inside nlopt's dual solve --
2h22m at 99.5% CPU with ZERO callback invocations -- because the epigraph
variable's O(1) gradient sits seven orders of magnitude above the O(1e-7)
per-element latent gradients.  The sum form is the one measured to work at
production scale (20k evaluations in 40 min, callbacks firing, monotone
descent to gs=-7.5e-6 / gv=+1.5e-6 and still closing).

This is a numerical assist inside the SAME constrained problem the optimizer
already solves; the certified evaluator/gradient chain is untouched and the
final independent DRC + exact-binary FDTD SUCCESS gates are unchanged.
"""

from __future__ import annotations

import time

import numpy as np

REPAIR_VERSION = "feasibility_repair/v2-sum"


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
    opt = nlopt.opt(nlopt.LD_MMA, n)
    opt.set_lower_bounds(np.zeros(n))
    opt.set_upper_bounds(np.ones(n))
    opt.set_maxeval(int(maxeval))
    opt.set_param("rho_init", float(rho_init))

    state = {"n": 0, "solution": None, "t0": time.time()}

    def objective(x, grad):
        state["n"] += 1
        gs, dgs = constraints.solid_residual_and_grad(x, beta)
        gv, dgv = constraints.void_residual_and_grad(x, beta)
        if grad.size:
            grad[:] = dgs + dgv
        if gs <= -margin and gv <= -margin and state["solution"] is None:
            state["solution"] = np.array(x, copy=True)
            opt.force_stop()
        if log is not None and state["n"] % 4000 == 0:
            log(f"[repair] beta={beta:g} eval {state['n']}: "
                f"gs={gs:.3e} gv={gv:.3e} "
                f"({time.time() - state['t0']:.0f}s)")
        return gs + gv

    opt.set_min_objective(objective)
    try:
        opt.optimize(np.array(latent, copy=True))
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
