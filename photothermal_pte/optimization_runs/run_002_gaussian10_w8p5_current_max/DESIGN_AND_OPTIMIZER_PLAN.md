# Run 002 design, constraint, and optimizer plan

This document remains the reviewed optimization plan. Source-only, production
forward, component-Yee mapping, literal material-Q attribution, exact 3D
thermal deposition, production CUDA thermal/PTE, and fixed-Q thermal-material
AD-FD gates have passed. Combined Maxwell/thermal AD-FD and the one-time
coarse-gradient window selection still block optimizer execution.

## Physical layout contract

- Scalar Gaussian: λ=10 µm, target-plane w0=8.5 µm.
- Calibrated Lumerical source-object waist: 8.36043075475035 µm.
- Source/FDTD lateral spans: 40/48 µm; 56 and 64 µm are convergence audits.
- Six PML boundaries, no periodic/Bloch boundary.
- Optical TaIrTe4 background extends through the transverse PML. The thermal
  flake remains finite (32 µm initial span) inside a 64 µm thermal domain.
- The bottom SiO2 interface and design-contact interface independently take
  thermally-grown (7.37e6 W m^-2 K^-1) or evaporated (7.37e4 W m^-2 K^-1)
  values. These are four named scenarios, not gray interpolation between two
  fabrication processes.
- At 10 µm, SiO2 uses the repository Kitamura closure, epsilon =
  7.3490019303 + 1.9899687287 i (n = 2.7352020978 + 0.3637699624 i).
  Thermally-grown and evaporated SiO2 are not assigned different optical
  constants without process-specific complex-index data.

## Design thickness

The baseline is 1.0 µm, with 0.6 and 1.5 µm mandatory pre-optimization
sensitivity cases. At the 10 µm Kitamura index these correspond to approximate
air-relative phase delays of 0.208π, 0.347π, and 0.521π. Their bulk propagation
intensity factors are about 0.760, 0.633, and 0.504, respectively.

Thus 1.0 µm is a compromise: it gives useful phase authority without starting
from the highest-aspect-ratio and most lossy candidate. It is not asserted to
be the optimal height. At a 500 nm minimum feature, the three aspect ratios are
1.2, 2.0, and 3.0.

## Why the design region is not the full illuminated plane

The 8.5 µm waist makes a full 40–48 µm design plane unnecessarily expensive.
The first material forward/adjoint smoke uses a 20×20 µm, 100 nm coarse
sensitivity canvas. We then select the smallest reviewed asymmetric window
that retains at least 90% of the absolute physical-density gradient L1 norm.

Candidate 12×6 µm strips are placed on the ±a and ±b sides; a centered 10×10
µm window is a control. Production variables use 50 nm nodes. Window selection
is gradient-driven and performed once before optimization; the window is not
moved during the run.

## BPVE constraint audit

The useful ideas in `volume_current_inverse_design` are retained:

1. NLopt LD_MMA with explicit 0≤latent≤1 bounds;
2. a conic density filter plus tanh projection and beta continuation;
3. separate differentiable minimum-solid and minimum-void penalties;
4. a high-order smooth aggregate rather than a global mean that hides one
   narrow violation;
5. append-only history, immutable code/config hashes, atomic best-feasible
   checkpointing, and final exact-binary independent DRC.

They cannot be copied literally. The BPVE mapping wraps both axes on a torus;
Run 002 is finite and must use a nonperiodic filter with explicit edge
normalization. The BPVE `--robust` option also fails closed because its
three-field epigraph is not implemented. Run 002 therefore starts nominal and
adds eroded/nominal/dilated worst-case optimization only after nominal progress
and a calibrated pixel-to-length-scale fixture.

Final DRC targets are 500 nm minimum solid and void. Differentiable steering is
set slightly higher, 525 nm, because the smooth constraint is guidance; exact
binary DRC remains the authority.

## Initial-FOM stagnation controls

The old BPVE-style start can stagnate here for three independent reasons: the
PTE objective is signed and may nearly cancel under a symmetric uniform
density; current in amperes is numerically tiny; and high beta/strong geometry
constraints can flatten the mapping before a useful topology appears.

Run 002 therefore uses:

- two independent smooth objectives, maximize +I/Pin and maximize -I/Pin,
  instead of |I|;
- one fixed nondimensional objective scale derived once from all-air and
  all-SiO2 endpoint controls—never per-iteration gradient rescaling;
- beta=2 asymmetric sensitivity seeds plus deterministic small perturbations;
- a nominal stage before late three-field robustness;
- several deterministic starts and an atomic best-feasible checkpoint;
- diagnostics of physical-gradient norm, latent-gradient norm, filter and
  projection attenuation, constraint-gradient norm, and accepted MMA step.

If the first two accepted steps improve the scaled objective by less than 0.1%
while the physical gradient is nonzero, the run stops for scaling/mapping
diagnosis rather than spending more FDTD solves. If the physical gradient is
near-null, the opposite signed objective and the asymmetric windows are tested
before changing any physics.

## Runtime strategy

- Optical: local 50 nm mesh only around the design/absorber and illuminated
  region; 100/200 nm intermediate/outer regions; TaIrTe4 dz=10 nm candidate
  with 5 nm audit. The large air/Si/SiO2 domain is not uniformly fine.
- Thermal: assemble the existing conservative anisotropic/finite-G FVM once,
  transfer its CSR matrix to CUDA float64 once, then reuse it for arbitrary
  forward and implicit-adjoint right-hand sides.
- The CUDA implementation differentiates the discrete equation, not the PCG
  iterations. Explicit residual and forward-adjoint reciprocity are gates.

The production `362×362×91` FVM now passes on CUDA float64 for all four named
interface scenarios. Fixed-Q thermal-material derivatives also pass for the
grown/grown and evaporated/evaporated endpoints. Host matrix assembly remains
the expensive part of FD certification; optimization uses one forward/adjoint
pair rather than repeated ±FD solves.

## Remaining fail-closed sequence

1. build the thermal-adjoint pullback from material-resolved thermal Q to the
   component-specific native Yee absorption grids;
2. run one representative Gaussian combined physical-density Maxwell/thermal
   AD-FD smoke without gradient rescaling;
3. use the validated physical gradient on the coarse canvas to select and
   freeze the smallest reviewed asymmetric design window;
4. certify the finite nonperiodic filter/projection JVP/VJP and exact-binary
   DRC fixtures on that window;
5. only then enable a short nominal signed-objective MMA pilot.
