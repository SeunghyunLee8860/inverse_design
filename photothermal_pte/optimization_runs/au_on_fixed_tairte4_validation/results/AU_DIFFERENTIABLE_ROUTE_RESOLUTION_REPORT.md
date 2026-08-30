# Au differentiable-route resolution

Status: `BLOCKED_AU_PRODUCTION_GRADIENT_REQUIRES_DISPERSIVE_DISCRETE_ADJOINT`

## What is now resolved

The failure is **not** a PML problem and is **not** repaired by a smaller
Courant factor.  The all-Metal control and the `dt=0.5` PML control both
diverged.  Global Conformal Variant 0 also diverged.  Most decisively, an
exact scalar-Au base with zero endpoint perturbation still diverged as soon as
the v261 Index-perturbation/temperature-grid wrapper was active.

The exact scalar Au forward model itself is stable, and the fixed-geometry
material derivative passed AD--FD with relative error
`0.003896%`.  The blocker is therefore the
**differentiable moving-metal representation**, not Maxwell propagation through
Au in general.

## Why the boundary derivative fails

At 10 um, the frozen Ordal endpoint is

`n+ik = 12.1+69.2i`, `epsilon = -4642.23+1674.64i`.

Its field-amplitude and intensity e-folding depths are only
`23.00 nm` and `11.50 nm`.
The old `50 nm x 50 nm x 25 nm` Au control mesh is therefore not an
interface-converged production mesh.

The same-session complex diagonal-epsilon derivative changes sign only below
about 1 nm.  At the finest tested 0.1 nm CAD step it has the correct sign
relative to the older 50 nm Maxwell FD, but still differs by
`38.501%` and its sub-nm tail changes by
`7.637%`.  The coordinate mismatch is
only `5.082e-21 m`, so an
ordinary coordinate pairing error is excluded.

The decisive equal-step tests are now complete.  The independently re-solved
Maxwell central FD is `-2.916216e-30 J/um` at `h=1 nm` and
`-2.918610e-30 J/um` at `h=0.5 nm`; their relative change is only
`0.0820%`.  The Maxwell local derivative is therefore on a
sub-1% step plateau.  By contrast, the matching complex diagonal-epsilon
contractions miss those FDs by
`66.981%` and
`39.196%`, respectively.  Both signs agree, but both magnitude
gates fail.  Comparing unlike parameter steps and an unconverged Maxwell FD
are therefore excluded.  The diagonal volume `d epsilon` term is not the
complete derivative of the conformal moving-metal operator.

Independently, the installed v261
`lumopt2` implementation explicitly applies `real(index_c**2)` and later takes
the real part of the sparse difference.  Its wavelength-remapping helper also
clips negative real epsilon and fits a lossless Cauchy model under the source
assumption `n >> k approximately 0`.  That generic path discards Au loss and
cannot be promoted for Au.

The legacy bundled `lumopt` geometry path, unlike `lumopt2`, retains complex
`index_c**2`.  The same-session control deliberately reproduces that part of
the legacy contract.  It is not promoted merely because the source looks
better: the independent equal-step Maxwell FD remains the deciding numerical
test.

This is consistent with the documented scope rather than a hidden PML setting.
Ansys describes `FunctionDefinedPolygon` as using a *shape derivative
approximation* and documents `eps_in`/`eps_out` as scalar permittivities; the
official examples are ordinary dielectrics, not high-loss Au.  Ansys also
states that Precise Volume Average evaluates dispersive materials at one mesh
frequency and makes the averaged cell non-dispersive.  That mesh operation is
useful for forward geometry sensitivity, but it is not a causal dispersive
material Jacobian with auxiliary states.

The Ansys page discussing an `enable conformal meshing` property explicitly
says its tips do not apply to np-density and Temperature attributes.  The
installed v261 Temperature object was queried directly and exposes no such
property.  Therefore that switch is not an available repair for this carrier;
global CV0 was tested separately and still diverged.

## Resolution, not a rescaling

There are three physically defensible routes:

1. **Immediate few-parameter exact-binary route.** Keep exact scalar Au and
   use independently re-solved central differences (or a derivative-free
   trust-region method) for only a small number of bounded geometric
   parameters.  First converge 10/5/2.5 nm Au-interface meshes.  This is a
   practical exact-Au fallback, but it is not free-form topology optimization.
   The current lumopt2 documentation explicitly supports gradient-free SciPy
   methods and states that the adjoint solve is skipped for those methods.
2. **Fixed-Au coupled inverse design.** Treat the electrode geometry as fixed;
   the validated fixed-geometry material/field chain can then be coupled to
   TaIrTe4/dielectric design variables without differentiating a moving Au
   boundary.  The complete coupled PTE AD--FD must still pass.
3. **Production free-form metal topology route.** Use a discrete dispersive
   FDTD adjoint with Drude/CCPR auxiliary states.  Density must interpolate
   causal dispersive parameters, and the auxiliary-state gradient terms must
   be included.  Exact binary endpoints must be cross-validated against
   Lumerical.  This is the route demonstrated in the plasmonic inverse-design
   literature; it is not provided by the tested v261 GPU carrier and must be
   implemented in a solver that exposes its discrete update equations.

A PEC/surface-impedance model is only an optional reduced approximation.  The
semi-infinite estimate is `Rs = 0.9237 ohm`, but it needs
finite-thickness, substrate, forward-field and shape-gradient validation before
use.

No sign flip, empirical normalization, or gradient scaling was used.  No
thermal, PTE, or optimization stage was run.

## Primary references

- [Ansys conformal-mesh selection](https://optics.ansys.com/hc/en-us/articles/360034382614-Selecting-the-best-mesh-refinement-option-in-the-FDTD-simulation-object)
- [Ansys grid-attribute limitations](https://optics.ansys.com/hc/en-us/articles/360034915193-Tips-and-background-information-when-using-grid-attributes)
- [Ansys GPU material limitations](https://optics.ansys.com/hc/en-us/articles/17518942465811-Getting-started-with-running-FDTD-on-GPU)
- [Ansys optimizable-geometry d-epsilon contract](https://optics.ansys.com/hc/en-us/articles/360052044913-Optimizable-Geometry-Python-API)
- [Ansys lumopt2 optimization and gradient-free fallback](https://lumerical.docs.pyansys.com/version/dev/user_guide/lumopt2/optimization_session.html)
- [Zeng et al., discrete plasmonic FDTD adjoint](https://arxiv.org/abs/2007.11442)
- [Hassan and Calà Lesina, dispersive Drude-ADE topology](https://arxiv.org/abs/2203.01462)
- [Gedeon et al., CCPR-ADE power-dissipation topology](https://arxiv.org/abs/2407.05994)
