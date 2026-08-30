# Phase 3B-1 — smooth Robin to hard-electrode convergence

## Verdict

```text
PASS at the fixed 0.5 um electrical mesh.
The nodal-lumped Robin model converges to the hard node contact in both I and psi.
Selected differentiable optimization relaxation: g = 1e12 S/m2.
No optimizer was called by this validation.
```

## Fixed validation case

- actual saved center-beam temperature, beam center `(0,0) um`
- mesh step `0.5 um`, flake size `24 x 24 um`, perimeter `P=96 um`
- geometry `(c0,L0,c1,L1)=(6.1,7.3,58.4,10.7) um`
- transition width `0.75 um`, boundary quadrature order 5
- hard current `I_hard=-2.3502688395134244e-10 A`
- hard terminal conductance `0.022205891810750847 S`
- hard contact node counts `(15,21)`

The script sweeps

```text
g = 1e10, 1e11, 1e12, 1e13, 1e14, 1e15,
    1e16, 3e16, 1e17, 3e17, 1e18 S/m2.
```

## Why the contact discretization was changed

The first implementation integrated a smooth mask with a consistent P1 edge
mass matrix.  On a fixed mesh, a partially covered edge penalizes both of its
end nodes as `g -> infinity`.  The legacy hard model instead fixes only those
boundary nodes whose perimeter coordinate lies inside the nominal electrode.
Therefore those two discretizations do not have the same fixed-mesh limit.

This was measured, not assumed.  For the old `consistent_edge` mode, the
current relative error versus the legacy hard result grows to `17.42%` at
`g=1e18 S/m2` instead of approaching zero.

Production now uses a nodal mass-lumped Robin term,

```text
B_e = t g sum_i w_i m_i e_i e_i^T,
b_e = t g sum_i w_i m_i V_e e_i,
```

where `w_i` is the cyclic trapezoidal perimeter weight and `m_i` is the smooth
periodic contact mask evaluated at boundary node `i`.  Generic electrode
endpoints then converge to exactly the same node set as the hard contact.  The
analytic derivatives of `m_i` with respect to center and length are unchanged.
After this correction the full four-variable adjoint/central-FD check was rerun
and passed with worst component relative error `4.58e-6`.

## Measured hard-limit convergence

| g (S/m2) | relative error in I | relative L2 error in psi | absolute Linf error in psi |
|---:|---:|---:|---:|
| 1e15 | 8.653e-3 | 1.776e-3 | 1.331e-2 |
| 1e16 | 1.009e-3 | 2.062e-4 | 1.545e-3 |
| 3e16 | 3.404e-4 | 6.955e-5 | 5.213e-4 |
| 1e17 | 1.026e-4 | 2.095e-5 | 1.570e-4 |
| 3e17 | 3.423e-5 | 6.993e-6 | 5.241e-5 |
| 1e18 | 1.027e-5 | 2.099e-6 | 1.573e-5 |

At `g=1e18`, all three hard-limit tolerances are `1e-4`, and the state
relative residual is `1.87e-16`.  Thus the asymptotic connection is a PASS.
The mild non-monotonic behavior at lower `g` is caused by the staggered
penalization of nodal mask values; it does not persist in the high-`g` tail.

## Why optimization uses g=1e12 rather than the largest g

The largest `g` proves the limit but is a poor differentiable relaxation:
the hard node-selection staircase is recovered and geometry gradients collapse.
At `g=1e18`, the scaled gradient norm is only `1.54e-11 A`, and the smallest
absolute component is `3.78e-4` of the largest.

At `g=1e12`:

- relative current error versus hard: `0.440%` (criterion: `<=1%`)
- relative `L2` error in `psi`: `3.03%` (criterion: `<=5%`)
- state relative residual: `2.40e-15`
- scaled gradient norm: `8.01e-9 A`
- smallest/largest absolute gradient component ratio: `0.0612`

This is the measured compromise between physical fidelity and a usable smooth
sensitivity.  `g=1e13` was not chosen because its current error is `3.50%` in
this fixed case.  Every smooth optimum must therefore be re-evaluated with the
hard contact, and candidates are ranked by final `abs(I_hard)` rather than by
the relaxed objective alone.

## Reproduce

```bash
cd /home/seunghyun/tairte4/pte_electrode_boundary_adjoint
/home/eidl/miniconda3/envs/EIDL-Lumapi/bin/python \
  validation/phase3_robin_hard_convergence.py
```

Machine-readable results and the convergence figure are
`phase3_robin_hard_convergence.json`, `.csv`, and `.png` in this directory.
