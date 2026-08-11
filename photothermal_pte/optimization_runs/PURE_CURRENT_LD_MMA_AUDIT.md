# Pure-terminal-current NLopt LD_MMA audit

## Purpose

Run030/Run031 use the contact-anchored TaIrTe4 device with a top and a bottom
electrode. They optimize the signed, full-flake terminal PTE current only.
They deliberately do **not** require a minimum terminal conductance or a
graph/connectivity constraint.

The finite flake is \(24\times24\,\mu\mathrm m^2\). The design region spans
the full \(x=b\) width and \(20\,\mu\mathrm m\) in \(y=a\); the remaining
two \(2\,\mu\mathrm m\)-deep strips are fixed TaIrTe4 terminal-contact
regions. They define the two electrical electrodes used for the weighting
potential; they are not a connectivity inequality.

The historical Run020/Run021 `LD_MMA` path is retained unchanged. It imposed
the additional inequality

\[
1-G_{\rm terminal}/G_{\rm min}\le 0,
\qquad G_{\rm min}=0.1G_{\rm full-solid},
\]

which is an optional numerical guardrail, not a required part of the
top/bottom-electrode weighting-field physics. It is absent from Run030/Run031.

## Verified electrical contract

The production electrical solver in `tairte4_flake_topology/electrical.py`:

- constructs the finite full-flake mesh in Lumerical coordinates `x=b`, `y=a`;
- fixes every lower-contact node to \(\psi=0\) and every upper-contact node to
  \(\psi=1\);
- solves the weighting potential inside that material-dependent mesh;
- evaluates the signed **full-flake terminal** PTE current; and
- differentiates both the direct material term and the implicit
  weighting-potential response.

The objective/gradient entry point
`tairte4_flake_topology/evaluate_objective_gradient.py` writes the sum of the
optical, thermal, and electrical material derivatives. Its reported objective
is explicitly `signed full-flake terminal PTE current`.

`terminal_conductance_S` and its derivative are still calculated in every
evaluation. In Run030/Run031 they are stored in history/manifest only; they
are not passed to `NLopt.add_inequality_mconstraint`.

## Exact optimizer path

`tairte4_flake_topology/run_pure_current_ld_mma_optimization.py` creates

```python
nlopt.opt(nlopt.LD_MMA, variable_count)
```

with latent box bounds \(0\le x\le1\) and the analytic adjoint gradient. It
does not implement a hand-written MMA update, move limit, Adam state,
gradient-direction normalization, post-update clipping, symmetry constraint,
volume constraint, or connectivity constraint.

At \(\beta<8\), there are no inequality constraints. From \(\beta=8\), the
only two active inequalities are the existing differentiable 500-nm solid and
void morphology constraints. They are unrelated to electrical connectivity.
The beta stage does not advance after `MAXEVAL_REACHED`; that condition is
fail-closed rather than a forced continuation.

## Native-MMA scale and continuation corrections

Run030's uncalibrated diagnostic is preserved and is not promoted. Its first
five full-physics evaluation points reproduced the rapid Run020 bound-seeking
trajectory even after removal of the conductance inequality. Therefore the
conductance constraint is not the cause of that behavior.

The replacement driver explicitly sets the native NLopt parameters at every
beta stage:

- `initial_step` corresponds to a target initial **physical-density** change
  of 0.025, divided by the midpoint derivative of the active tanh projection;
- `rho_init` begins at 10 for beta=1 and scales with the square of that same
  projection derivative, which is the local chain-rule curvature factor;
- `always_improve=1` and `inner_gradients=1` are explicit and logged rather
  than implicit library defaults.

These are initialization parameters, not fixed move limits. LD_MMA retains
control of every subsequent moving asymptote. The `rho_init` scale is a
conservative CCSA prior, not a claimed Hessian measurement; no AD-FD check is
run as part of this optimizer correction.

The former morphology continuation opened beta=8 at
`max(target, 0.9 * current)`, which can make the incoming point 11.1% infeasible.
The pure-current driver now uses `max(target, current)`, making the first point
of each morphology stage feasible. Every JSON/PNG is labelled a
`full_physics_evaluation`, not an MMA outer iteration.

The continuation now follows the documented Ansys LumOpt pattern: beta=1 is
the initial grayscale phase, subsequent beta values are multiplied by 1.2,
and each binarization stage has a fixed budget of 20 full-physics evaluations.
The beta=1 grayscale budget remains 40 evaluations. A positive NLopt FTOL,
XTOL, or MAXEVAL stop with a feasible returned design completes the stage.
The callback history contains trial and repeated CCSA points, not an
accepted-iterate sequence, so a raw callback-objective plateau is explicitly
not used as a beta-transition veto. A beta transition may perturb the FOM;
the next fixed-budget stage re-optimizes the perturbed physical design.

Finally, the driver checks `PURE_CURRENT_LD_MMA_CODE_MANIFEST.json` before a
GPU session opens. It fail-closes if the audited pure driver, shared LD_MMA
callback, shared constraint code, or paired supervisor no longer match their
recorded SHA-256 values. This closes optimizer-code provenance; it does not
pretend to replace the separate physical-solver preflight.

## Reproduction and publication

Run the paired sequence with:

```bash
TAIRTE4_PURE_CURRENT_LD_MMA_GPU=<physical GPU index> \
  /home/eidl/miniconda3/envs/EIDL-Lumapi/bin/python \
  photothermal_pte/optimization_runs/run_pure_current_ld_mma_dual_supervisor.py
```

It verifies the immutable base-FSP SHA, passed component-Yee Jacobian
certificate, and passed optical/thermal/electrical preflight before opening a
GPU FDTD session. `Ea` runs first; `Eb` starts only after the `Ea` final
certificate passes. Raw FSP/NPZ remain outside Git; their paths, byte counts,
and SHA-256 values are emitted to each raw-artifact manifest.
