# Periodic optical array versus finite PTE device

## What the present inverse-T calculation proves

The 2024 paper-derived inverse-T calculation is a normal-incidence optical
unit-cell problem:

- `k=-z`;
- `x/y Periodic`, `z PML`;
- one 1500 nm x 1000 nm cell repeated infinitely;
- absorbed optical power and TaIrTe4 heat-source shape per cell.

All four published cases (`T_Ea`, `T_Eb`, `bare_Ea`, and `bare_Eb`) use this
Bloch/Periodic plane wave.  **No Gaussian beam was used in these results.**
In particular, no physical Gaussian spot was fitted inside the 1500 nm x
1000 nm unit cell.

The raw pabs monitor stores duplicated/staggered endpoint planes.  A thin
line at the plotted endpoint is a numerical periodic seam, not a physical
flake edge.  The raw data and power integrals are preserved.  The
`periodic_canonical/` figures provide a separate display-only canonical cell
and a 3x3 tiling so the physical periodic interpretation is unambiguous.

## What it cannot prove

A periodic optical cell does not define two finite collection electrodes,
a unique weighting potential, or a net terminal PTE current.  Under a fully
periodic, laterally symmetric thermal/electrical model, opposite current
contributions cancel and the terminal current is not the finite-device
observable.

Therefore the present result is an optical absorption-screening result.  It
is not a PTE inverse-design certificate.

## Required finite-device calculation

To isolate the resonance of a single Au T without introducing a flake-air
lateral edge, the next optical model must contain:

1. one finite Au T on a laterally extended TaIrTe4/Al2O3/Au-mirror stack;
2. the TaIrTe4 and all planar underlayers continuing through the x/y PML,
   without terminating at the PML entrance;
3. optical PML on all six Maxwell boundaries;
4. a finite Gaussian or validated TFSF normal-incidence source (not an
   infinite plane wave extending directly into lateral PML);
5. a matched bare multilayer control so the T-induced local `Delta Q` can be
   separated from area-dependent background absorption;
6. explicit optical closure and domain/PML/mesh convergence.

This optical-local model does not itself define terminal PTE current.  The
subsequent finite physical-device model must add:

7. conservative volumetric-Q transfer into a finite thermal domain;
8. physical thermal boundary conditions and finite electrode contacts;
9. an electrical weighting-potential solve with `psi=1/0` at the two
   terminals and insulating non-contact flake edges;
10. the full terminal current integral over the finite TaIrTe4 volume.

## Substrate below the Au mirror

The fabricated 2024 device wafer is intrinsic Si with 1.5 um of thermally
grown SiO2 before the reflector recess is etched and filled with Au.  A
physical full-stack schematic must therefore show thermal SiO2 and intrinsic
Si below the Au mirror; the region below the mirror is not air.

The four existing optical-forward artifacts did **not** include that physical
substrate.  They extended Au from the nominal backplane surface through the
bottom PML as an explicit opaque numerical closure.  This is acceptable only
as an optical-above-an-opaque-mirror control.  It must not be presented as the
fabricated substrate or reused as a thermal model.  The next physical-stack
check must terminate the Au mirror, place thermal SiO2 and intrinsic Si below
it, and put the bottom z-PML inside the continuing Si substrate.  Because the
Au mirror is optically opaque, the old and physical-substrate optical results
are expected to be close, but that equivalence still requires a matched
numerical comparison rather than an assumption.

The finite source type and size are not yet selected.  A Gaussian is allowed
only if its realized waist and boundary intensity are measured in a domain
large enough that the field is negligible before the lateral PML.  At the
present wavelength of 4.75 um, its footprint must not be drawn or assumed to
fit inside the 1500 nm x 1000 nm periodic cell.  A validated TFSF source is the
alternative when a finite structure under plane-wave illumination is the
intended experiment.

Optical PML applies only to Maxwell.  Thermal and electrical problems use
their own physical boundaries; they do not use PML.

The finite flake size, beam waist/footprint, and electrode geometry must be
fixed before the GPU run.  They are not inferred from the periodic paper
unit cell.
