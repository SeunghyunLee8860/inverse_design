# Phase 3A — actual baseline temperature gradient check

## Verdict

```text
PASS: all four components of grad_x(I) agree with central finite differences.
No optimizer was imported or called.
```

This is the first Phase-3 blocker gate only.  It validates the implemented
state derivative, contact derivative assembly, transpose adjoint, and
`p=P*x` chain rule at one strictly feasible generic design using the actual
saved baseline temperature field.

## Fixed problem

- temperature: `per_beam_500nm_final/per_beam_fields.npz`, beam index 4,
  center `(0,0) um`, SHA-256
  `129c1b2ee91cbb43e5defd4316f770a446a8953c1d319a5185c4177c87265f24`
- electrical mesh: 0.5 um, 49 x 49 nodes
- physical design `(c0,L0,c1,L1)=(6.1,7.3,58.4,10.7) um`
- scaled design `(u0,l0,u1,l1)=(0.0635417,0.0760417,0.6083333,0.1114583)`
- Robin conductance: `1e12 S/m2`
- smoothing transition: `0.75 um`
- boundary quadrature order: 5
- contact discretization: `nodal_lumped`
- raw current: `-2.339921812929894e-10 A`

The point is away from overlap, constraint boundaries, symmetry points, and
corners.  Its two scaled separation/packing constraints are positive:
`(1.7735, 0.80208)`.

## Adjoint result

Variable order is `(u0,l0,u1,l1)=(c0/P,L0/P,c1/P,L1/P)`.  The adjoint gives

```text
grad_x(I) = [
  +7.653141616792487e-09,
  +2.112904562173362e-09,
  -4.686048388635981e-10,
  +9.104957793172376e-10
] A.
```

State and adjoint relative residuals are `2.46e-15` and `6.36e-14`.

## Central-FD convergence

| scaled h | physical P*h | vector relative error | max component relative error |
|---:|---:|---:|---:|
| 1.0e-2 | 0.960 um | 2.566e-1 | 7.526e-1 |
| 5.0e-3 | 0.480 um | 2.374e-1 | 6.920e-1 |
| 2.0e-3 | 0.192 um | 1.002e-1 | 5.891e-1 |
| 1.0e-3 | 0.096 um | 2.481e-2 | 2.046e-1 |
| 5.0e-4 | 0.048 um | 7.350e-3 | 4.379e-2 |
| 2.0e-4 | 0.0192 um | 1.196e-3 | 7.271e-3 |
| 1.0e-4 | 0.0096 um | 2.999e-4 | 1.828e-3 |
| 5.0e-5 | 0.0048 um | 7.502e-5 | 4.575e-4 |
| 2.0e-5 | 0.00192 um | 1.201e-5 | 7.323e-5 |
| 1.0e-5 | 0.00096 um | 3.002e-6 | 1.831e-5 |
| 5.0e-6 | 0.00048 um | 7.504e-7 | 4.577e-6 |

At the best tested step, central FD is

```text
[
  +7.653146647398762e-09,
  +2.112902076873708e-09,
  -4.686069836510626e-10,
  +9.104957744075053e-10
] A.
```

The four component relative errors are

```text
[6.573e-7, 1.176e-6, 4.577e-6, 5.392e-9].
```

Over the last two step halvings, the maximum component error falls by almost
exactly a factor of four each time, which is the expected second-order central
FD truncation behavior.  The observed orders are `1.99992` and `2.00005`.

## What this does and does not establish

This result removes the immediate derivative-assembly blocker at this design:

```text
adjoint grad_x(I) == central-FD grad_x(I)
```

The signed production objectives inherit this result by multiplication with
the design-independent factor `-b/I_ref`.

This check was repeated after changing the production contact assembly from
consistent edge integration to nodal mass lumping.  That change was necessary
so the smooth Robin model and the node-snapped hard model have the same
fixed-mesh contact set as `g -> infinity`.

It does not certify mesh, smoothing-width, or all symmetry/robustness gates.
The separate Phase-3B Robin-to-hard test establishes the required fixed-mesh
connection before the first 0.5 um optimization run.
