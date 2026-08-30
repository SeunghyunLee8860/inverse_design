# Phase 1 — Current implementation audit

## Scope and immutable baseline

The audited baseline is `/home/seunghyun/tairte4/pte_electrode_optimizer`.
No baseline source was modified.  The audit used its 0.5 um center-beam
temperature from `results/per_beam_500nm_final/per_beam_fields.npz`.

For reproducibility, the audited SHA-256 values are:

```text
electrical.py       85330d8982dbea598e8da2680f03ee581b9f20ae21184a8abf84d2e0a141f118
optimize.py         d0cb64087d01db1a26c5f1dc4fc25193f253095d7c47e287c0d2b574b78f5862
thermal.py          9b5ed7d1a58978f3f47bd7382afbbdf340795961c5481f37177360ea19357d62
per_beam_fields.npz 129c1b2ee91cbb43e5defd4316f770a446a8953c1d319a5185c4177c87265f24
per_beam_results    08dcdcf3156b89ce0ff8b7e26737aec2f7206b0bc8eeafe0cfe1ce220ef505ca
```

## A. Current FEM formulation

### Mesh and unknown

- `electrical.py:49-82` builds a structured rectangular nodal grid and splits
  every rectangle into two positively oriented P1 triangles.
- For the 24 um square at 0.5 um step, the measured mesh has 49 x 49 = 2401
  nodes and 4608 triangles.
- The scalar weighting potential `psi` is defined at mesh nodes.  Its gradient
  is constant in every P1 triangle.

### Conductivity assembly

The bulk matrix is assembled as

```text
K_e = t A_e B_e^T sigma B_e
```

where `t` is the TaIrTe4 thickness, `A_e` the triangle area, and `B_e`
contains the physical-coordinate gradients of the three shape functions.
The configured tensor is diagonal in the required `x=b, y=a` coordinates:

```text
sigma = diag(1.10e5, 4.91e5) S/m.
```

The thickness factor is present exactly once in `K` (`electrical.py:92-101`).
The measured full and Dirichlet-reduced matrix relative asymmetries are both
exactly zero.  This follows from the actually assembled symmetric diagonal
tensor; the future adjoint nevertheless must solve `K.T` rather than assume
symmetry.

### Boundary conditions

- Contact nodes are removed by exact Dirichlet elimination.
- Electrode 0 nodes receive `psi=0`; electrode 1 nodes receive `psi=1`.
- The reduced right-hand side is `-K_free,fixed psi_fixed`.
- No boundary term is assembled on the rest of the perimeter.  Therefore the
  remaining flake boundary has the natural zero-normal-current Neumann
  condition.

### Contact node selection

For a side-local `Electrode(side, center, length)`, a boundary node is selected
iff

```text
abs(tangent_coordinate - center) <= length/2 + 1e-12*mesh_step.
```

At least two nodes are required.  Overlap is rejected if the two integer node
sets intersect.

### Temperature and local PTE source

The explicit 3-D FVM temperature is averaged through the TaIrTe4 thickness
with z-cell-width weights.  Each 2-D thermal cell value is then distributed to
its four electrical corner nodes and divided by the number of contributing
cells.  Boundary nodes therefore inherit one- or two-cell averages, while
interior nodes inherit four-cell averages (`thermal.py:115-127,250-255`).

On each electrical triangle,

```text
grad(T)_e = sum_i T_i grad(N_i)
alpha = sigma S
j_PTE,e = -alpha grad(T)_e.
```

No additional interpolation is performed inside a triangle.

### Shockley-Ramo current and thickness

The implemented current is

```text
I = -t sum_e A_e grad(psi)_e^T alpha grad(T)_e.
```

Thus the 2-D integral represents the 3-D volume integral through a
uniform-through-thickness sheet.  Defining

```text
q_i = -t sum_{e containing i} A_e grad(N_i)^T alpha grad(T)_e,
```

the exact discrete identity is `I = q^T psi`.  The numerical reconstruction
relative error is `4.17e-16`.

Doubling only the configured flake thickness leaves `psi` unchanged and makes
both `I` and terminal conductance exactly 2.0 times larger.  This verifies that
the thickness factor is neither absent nor double-counted.

## Literature cross-check

The local sources checked were the 2026 Advanced Functional Materials article,
its `adfm75986-sup-0001-suppmat-2.pdf`, and the Blevins thesis.  The most direct
reference is Supplement S5, especially Table S2 and Equations S1--S7.

### What agrees

- Supplement Eq. S5/S6 defines the local source as
  `Jloc=-sigma S grad(T)` and the collected current as
  `integral Jloc.grad(psi)`.  The code implements precisely this sign and
  contraction after setting terminal 1 to `psi=1`.
- Supplement S5.D says the contact boundaries are Dirichlet 0/1, other sample
  boundaries are zero-flux, and the final current is a volumetric integral.
  These agree with the code.  The explicit factor `t` is the thickness part of
  that volume integral after reducing the electrical calculation to 2-D.
- Table S2 lists `(kappa_a,kappa_b,kappa_c)=(14.4,3.8,1.0) W/m/K`,
  `(sigma_a,sigma_b)=(4.91e5,1.1e5) S/m`, and
  `(S_a,S_b)=(-6,27) uV/K`.  With the code contract `x=b,y=a,z=c`, its arrays
  `[3.8,14.4,1.0]`, `[1.1e5,4.91e5]`, and `[27,-6]` are the same values under
  a consistent axis permutation.
- `G_TaIrTe4-thermal-SiO2=7.37e6 W/m2/K` matches Table S2.
- The Gaussian lateral factor used by `thermal.py` is the Supplement S1 form
  `exp(-2 r^2/w0^2)`.  When a beam is clipped by the finite flake, the code
  deposits only the analytic in-flake fraction rather than renormalizing it
  back to the full incident power.

### What is an implementation assumption, not an exact paper identity

- Supplement Eq. S7 is printed as the scalar Laplace equation
  `laplacian(psi)=0`, whereas the code solves
  `div(sigma grad(psi))=0`.  The latter is the conductivity-weighted reciprocal
  problem consistent with anisotropic charge continuity, but it is **not
  literally the printed S7 operator**.  The supplement does not specify enough
  COMSOL detail to prove which operator was used there.  This distinction must
  remain explicit in any paper comparison.
- Eq. S6 is written as a 2-D integral, while S5.D calls the COMSOL operation a
  volumetric integral.  The code assumes `psi` and the thickness-averaged
  temperature gradient are uniform through the 100 nm electrical sheet and
  therefore uses `t*dA`.  This is a controlled thin-sheet reduction, not a
  full 3-D electrical solve.
- The production configuration uses a uniform z heat profile.  Supplement S2
  uses Beer--Lambert attenuation.  This is an intentional consequence of the
  present optical-free Gaussian-Q problem, not a reproduction of optical
  absorption in the paper.
- `waist=8.5 um` and one-microwatt fully converted heat are configuration
  choices; Table S2 does not establish them as universal TaIrTe4 material
  values.
- The code resolves an air layer with `k_air=0.026 W/m/K` and applies
  `h=10 W/m2/K` only at the remote top of that air domain.  It does not apply
  Table S2's `G_TaIrTe4-air=1 W/m2/K` as a separate flake/air interface
  resistance.  It also adds `G_SiO2-Si=1.10e9 W/m2/K`, which is not listed in
  Table S2.  These thermal choices are inherited and are outside the present
  electrode-only change, but they should not be described as exact Table-S2
  replication.

Therefore the electrical source, axis-permuted transport values, boundary
roles, and volume-current normalization are consistent with the stated PTE
model.  Exact equivalence to every printed paper equation cannot be claimed
because of the anisotropic weighting operator and the listed thermal-model
choices.

### Terminal conductance

`G = psi^T K psi`.  Since the imposed terminal voltage difference is one volt,
this is the discrete Joule power divided by `1 V^2`, hence the two-terminal
conductance in siemens.  `Voc=-I/G` and `Pmatched=I^2/(4G)` follow.

### Electrode swap

For the audited hard geometry:

- `I(1,0) = -I(0,1)` to relative error `9.87e-15`.
- `psi_swapped = 1-psi` to max absolute error `7.85e-14`.

## B. Current electrode parameterization

### Different-side contacts

For each fixed side, normalized variables map affinely to an allowed length
and then to a center interval that keeps the whole segment beyond the corner
clearance.  The two lengths are independent.

### Same-side contacts

The usable side length first reserves the required minimum gap.  Candidate
extra lengths are scaled together only if their sum exceeds the remaining
length budget.  The residual slack is split into a leading margin, extra
inter-contact gap, and trailing margin.  This construction is feasible before
node snapping and preserves independent nominal lengths subject to the shared
packing constraint.

### Nominal versus realized geometry

The optimizer stores continuous nominal centers and lengths, but physics sees
only the selected integer boundary-node sets.  The result files therefore
report both nominal geometry and realized node-coordinate min/max/span.

### Cache key

Within one side-pair DE run, the cache key is exactly

```text
(tuple(electrode_0_node_ids), tuple(electrode_1_node_ids)).
```

Continuous candidates that snap to the same ordered node sets reuse one
objective.  Swapping the two tuples is not the same key because it changes the
terminal sign, although `|I|` or `I^2` is invariant.

## C. Objective smoothness audit

The audit swept one top-contact parameter while retaining a fixed bottom
contact and the center-beam temperature.

- Center sweep: 321 nominal samples at 0.025 um spacing over 8 um produced only
  33 unique node sets.
- Length sweep: 321 nominal samples at 0.025 um spacing over 8 um produced only
  9 unique node sets.
- Central FD steps from 0.001 through 0.05 um stayed in one plateau and returned
  exactly zero derivative.
- At larger steps the plus/minus node sets changed and the finite difference
  measured a mesh-dependent jump divided by the chosen step, not a convergent
  derivative.

Therefore the hard-Dirichlet/node-snapped objective is piecewise constant in
the nominal parameters with discontinuities at node-selection thresholds.
There is no useful classical derivative inside a plateau.  An exact adjoint of
the existing hard problem would correctly return zero almost everywhere and
cannot drive continuous endpoint optimization.

Artifacts:

- `audit_current.py`: reproducible audit script.
- `audit_current.json`: matrix, current, swap, thickness, and FD metrics.
- `snapping_center.csv`, `snapping_length.csv`: raw sweeps.
- `snapping_sweeps.png`: staircase plots.

## Phase-1 conclusion

A differentiable optimization requires changing only the *optimization-stage
boundary representation*, while keeping the bulk FEM, fixed temperature,
anisotropic tensors, and Shockley-Ramo current vector unchanged.  The final
candidate must be projected back to a hard perimeter segment and re-evaluated
with exact Dirichlet elimination.
