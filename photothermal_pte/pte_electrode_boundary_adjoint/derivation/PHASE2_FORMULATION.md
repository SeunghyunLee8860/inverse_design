# Phase 2 — Differentiable perimeter-contact formulation

## 1. Design space

Let the rectangle perimeter be one counter-clockwise periodic coordinate
`s in [0,P)`, with `P=2(W+H)=96 um`:

```text
s=0                    : bottom-left
0 -> W                 : bottom, left to right
W -> W+H               : right, bottom to top
W+H -> 2W+H            : top, right to left
2W+H -> P              : left, top to bottom
```

The design is

```text
p = (c0, L0, c1, L1),
```

where centers are periodic modulo `P` and each contact is one contiguous arc.
An arc may cross a rectangle corner without changing topology.

## 2. Compact differentiable contact mask

A logistic mask is strictly positive on the entire perimeter.  Consequently,
its `g -> infinity` limit would eventually impose contact everywhere and would
not be the hard finite-segment limit.  We therefore reject the plain logistic
mask and use a compact C2 mask.

For one electrode define

```text
theta = 2*pi*(s-c)/P
ell   = pi*L/P
z     = cos(theta) - cos(ell)
delta = delta_floor + (2*pi*epsilon/P)*sin(ell)
u     = z/delta.
```

The mask is the quintic smootherstep

```text
m = 0                         u <= 0
m = 6u^5 - 15u^4 + 10u^3     0 < u < 1
m = 1                         u >= 1.
```

Its support is exactly the periodic arc whose shortest arclength distance from
`c` is less than `L/2`.  It is identically zero outside that arc, has an
approximately fixed transition width `epsilon` inside each endpoint, and is
C2 at the moving support/core boundaries.  Corner crossing is automatic.

In the transition,

```text
H'(u) = 30 u^2 (u-1)^2
z_c   = (2*pi/P) sin(theta)
z_L   = (pi/P) sin(ell)
delta_L = (2*pi*epsilon/P)*(pi/P)*cos(ell)
u_c   = z_c/delta
u_L   = (z_L*delta - z*delta_L)/delta^2
m_c   = H'(u) u_c
m_L   = H'(u) u_L.
```

The derivatives are zero in the exact-zero and exact-one regions.  Since
`H'` vanishes at both endpoints, these piecewise expressions are continuous.

## 3. Robin optimization state

The bulk equation and anisotropic tensor are unchanged.  On the perimeter we
use the finite contact-conductance law

```text
n . sigma grad(psi)
  = g*m0*(V0-psi) + g*m1*(V1-psi),
V0=0, V1=1,
```

where `g` has units S/m^2 for the vertical flake-edge contact area.  The P1
weak form is

```text
int_Omega t grad(v).sigma.grad(psi) dA
+ sum_k int_Gamma t g mk v psi ds
= sum_k int_Gamma t g mk Vk v ds.
```

With Gaussian quadrature on every boundary edge, define

```text
B_k(p) = t g int_Gamma mk N^T N ds
b_k(p) = t g int_Gamma mk N ds.
```

Then the exact assembled state is

```text
K(p) psi = f(p)
K(p) = K_bulk + B_0(p) + B_1(p)
f(p) = V0*b_0(p) + V1*b_1(p) = b_1(p).
```

`K_bulk` is the audited baseline matrix.  For the current symmetric diagonal
conductivity tensor, `K_bulk`, `B0`, `B1`, and `K` are symmetric.  The code
will measure symmetry and always solve the transpose in the adjoint so the
derivation remains correct if a future assembly is not symmetric.

Any nonzero contact conductance removes the constant nullspace.  The finite-g
matrix is symmetric positive definite in the present model.

## 4. Exact discrete current and signed-branch objectives

The fixed temperature defines the audited current vector

```text
q_i = -t sum_e A_e grad(N_i)^T alpha grad(T)_e,
alpha = sigma S.
```

It is independent of electrode parameters because thermal physics is frozen
during electrode optimization.  The exact discrete current is

```text
I(p) = q^T psi(p).
```

Production optimization does **not** maximize `I^2`.  Squaring suppresses the
gradient by a factor `2I`, creates a stationary point at `I=0`, and gives an
unhelpfully tiny dimensional objective.  Instead, for each beam we run two
independent branches `b in {-1,+1}`:

```text
maximize J_b(p) = b I(p) / I_ref.
```

The `+1` branch finds the largest positive current and the `-1` branch finds
the most negative current.  Both candidates are re-evaluated with hard
contacts and compared by `abs(I_hard)`.  `I^2` may still be reported as a
diagnostic, but it is not passed to the production optimizer.

`I_ref` is positive and fixed for one beam; it must not depend on the trial
design.  The default is `I_ref=norm(q,1)`, a design-independent current scale,
with a documented user override allowed.  Raw current in amperes is always
stored alongside the dimensionless objective.

## 5. Exact discrete adjoint

Differentiate the state:

```text
K dpsi/dp_i = df/dp_i - dK/dp_i psi.
```

First differentiate signed current itself.  Define the current adjoint by

```text
K^T lambda_I = q.
```

Then

```text
dI/dp_i = lambda_I^T (df/dp_i - dK/dp_i psi).
```

For the dimensionless maximization branch,

```text
dJ_b/dp_i = (b/I_ref) dI/dp_i.
```

SciPy is a minimizer, so the implemented function and gradient are
`phi_b=-J_b` and `grad(phi_b)=-b grad(I)/I_ref`.  This single current adjoint
also permits `2I*grad(I)` to be reported as the diagnostic `grad(I^2)` without
another linear solve.

For a parameter belonging to contact `k`,

```text
dK/dp_i = t g int_Gamma (dmk/dp_i) N^T N ds
df/dp_i = t g Vk int_Gamma (dmk/dp_i) N ds.
```

These matrices/vectors use the same edge quadrature as the forward Robin
terms; no continuous/discrete mismatch is allowed.

## 6. Anisotropy

Anisotropy enters only through

```text
K_bulk = t int B^T sigma B dA
q       = -t int B^T (sigma S) grad(T) dA.
```

The boundary weak form does not assume isotropic bulk conductivity.  A scalar
contact conductance relates the normal anisotropic bulk flux to the metal
potential.  Thus the formulation remains valid for the audited diagonal
anisotropy.  A tensorial/contact-direction-dependent interface law would be a
different physical model and is not introduced here.

## 7. Electrode swap identity

Swapping `(m0,V0=0)` and `(m1,V1=1)` leaves `K` unchanged.  Because
`K*1 = b0+b1`, the swapped solution is `psi'=1-psi`.  The P1 gradient
partition of unity gives `q^T*1=0`, hence

```text
I' = -I.
```

Consequently, swapping terminal labels maps the `+I` branch to the `-I`
branch.  This will be tested numerically for both forward and adjoint
gradients; final hard geometries are compared without treating a terminal
swap as a new physical layout.

## 8. Smooth geometry constraints

Length bounds impose `Lmin <= Lk <= Lmax`.  Let the required hard gap be
`gap`,

```text
r     = pi*(L0+L1+2*gap)/P
Delta = 2*pi*(c1-c0)/P.
```

For total occupied length below the perimeter, two arcs are separated by at
least `gap` iff their shortest center separation is at least
`(L0+L1)/2+gap`.  A smooth periodic inequality is

```text
h_sep = cos(r) - cos(Delta) >= 0.
```

Its analytic Jacobian is

```text
dh/dc0 = -(2*pi/P) sin(Delta)
dh/dc1 = +(2*pi/P) sin(Delta)
dh/dL0 = dh/dL1 = -(pi/P) sin(r).
```

We also impose the explicit packing inequality

```text
P - L0 - L1 - 2*gap >= 0
```

so the cosine test is used only on its monotone `[0,pi]` branch.

The current 0.5 um baseline has `Lmin=1 um`, `gap=0.5 um`, a 0.5 um
corner clearance, and `max_fraction=0.9`.  On one 24 um side this produces a
largest different-side contact of `0.9*(24-2*0.5)=20.7 um`.  For an
apples-to-apples first comparison, `Lmax=20.7 um` is the recommended new
default.  The old corner clearance itself cannot coexist with unrestricted
corner-crossing arcs: either corner crossing is accepted and the clearance is
retired, or extra smooth corner-exclusion constraints must be introduced.
That is a design-space decision and will be exposed in configuration rather
than changed silently.

## 9. Nondimensional variables and a seam-free periodic center

SLSQP must not see center bounds `[0,P]`: those bounds turn the physically
identical `0/P` seam into an artificial line-search barrier.  Use the lifted,
dimensionless variables

```text
x = (u0,l0,u1,l1) = (c0/P,L0/P,c1/P,L1/P).
```

`u0,u1` live on the real line and have no optimizer bounds.  All forward terms
and constraints use periodic sine/cosine functions, so `u` and `u+n` are
exactly equivalent for integer `n`.  Centers are wrapped with `u mod 1` only
for reporting, clustering, and hard-contact conversion.  Length variables
have the dimensionless bounds `Lmin/P <= lk <= Lmax/P`.

In these coordinates the constraints are

```text
r       = pi*(l0+l1+2*gap/P)
Delta   = 2*pi*(u1-u0)
h_sep   = cos(r)-cos(Delta) >= 0
h_pack  = 1-l0-l1-2*gap/P >= 0.
```

Their Jacobian entries are order unity.  The optimizer gradient follows by
the chain rule:

```text
dI/dx_i = P dI/dp_i,
dphi_b/dx_i = -(b*P/I_ref) dI/dp_i.
```

Thus both variables and objective are nondimensionalized before SLSQP sees
them.  No internal length is supplied in meters and no objective in `A^2`.

## 10. Optimizer selection after gradient validation

The problem has only four dimensionless variables, length bounds, two
unbounded lifted centers, and two smooth nonlinear inequalities.  SLSQP
directly supports these constraints and analytic
Jacobians without introducing a penalty weight.  It is therefore the first
candidate *after* adjoint/FD validation.  L-BFGS-B is rejected for production
because it cannot enforce periodic separation without a penalty or a more
restrictive parameterization.  `trust-constr` remains a validation fallback
if SLSQP constraint residuals or line searches prove unreliable.

Both signed branches and multiple deterministic starts are mandatory.
Electrode-swap-equivalent and periodically equivalent results will be
canonicalized before clustering.

The interface conductance `g` is a relaxation/continuation parameter unless a
measured metal--TaIrTe4 edge contact resistivity is supplied.  It is not taken
from the thermal boundary conductance table.  A characteristic electrical
scale is `sigma/h_mesh`; Phase 3 must sweep `g`, monitor conditioning, and
demonstrate convergence to the hard solution before selecting a production
value.  No particular finite `g` is declared physical at this stage.

## 11. Phase-3 numerical gate

Before enabling SLSQP, the exact comparison

```text
adjoint gradient == central finite-difference gradient
```

must pass for all four scaled variables and both signed branches over multiple
FD steps.  The gate also contains three distinct discretization studies:

1. bulk mesh refinement (`1.0, 0.5, 0.25 um` initially);
2. boundary Gaussian quadrature-order convergence (for example orders
   `3,5,7,9`, extending until current and gradient changes meet tolerance);
3. smoothing transition-width convergence, paired with adequate quadrature
   and boundary resolution (initially `epsilon/h = 2,1,0.5`, refined as
   needed).

Quadrature order and `epsilon` cannot be certified independently: a narrow
transition sampled by too few edge quadrature points can give an apparently
smooth but wrong derivative.  The report must therefore include `I`, all four
adjoint derivatives, all four FD derivatives, and their relative errors as a
joint `(mesh,quadrature,epsilon,h_FD)` table.  The production transition is
chosen only in a resolved plateau and must also preserve the projected hard
current.

The existing `g` continuation, electrode swap, reflection, zero-temperature-
gradient, and old-DE equivalence tests remain mandatory.

## 12. Hard validation

The final `(c0,L0,c1,L1)` is converted to exact periodic boundary-node sets,
including corner-crossing segments, and solved by the original Dirichlet
elimination algebra.  Finite-g optimization performance is never reported as
the final physical current without this hard re-evaluation.

## 13. Per-beam contract

This formulation is run separately for each Gaussian center.  For beam index
`b`, the thermal solve produces one fixed `T_b` and hence one fixed `q_b`; the
optimizer then finds its own `p_b=(c0,L0,c1,L1)`.  There is no mean-current
objective across beam positions in this workflow.  The two lengths are
independent and may converge to different values.  A shared-electrode design
for several beam positions would be a different multi-load optimization and
is not used here.
