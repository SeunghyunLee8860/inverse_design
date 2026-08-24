# Au/TaIrTe4 4 um inverse-design physics audit

Status: **production blocked**. The repository can run numerical diagnostics,
but the historical topology is not a validated physical design and a new
optimization must not start yet.

Superseding correction: the shared-linear gray law discussed below is a
historical FDTDX consistency diagnostic, not the selected Lumerical optical
law. The authorized route is density topology with one projected occupancy,
the nonlinear Christiansen `n-k` optical relaxation, documented custom-CUDA
thermal/electrical maps, and an independent exact-binary ordinary
dispersive-Au final reevaluation. See `LUMERICAL_MAXWELL_GPU_PDE_ROUTE.md` and
`NP_DENSITY_ROUTE_REJECTED.md`.

## Intended observable

The requested switch is a signed zero-bias current: `Ia > 0` for `E || a` and
`Ib < 0` for `E || b`. The code uses a Shockley-Ramo functional equivalent to
`I = integral J_local . grad(psi) dA`, with
`J_local = -sigma S grad(T)`.

This is the correct class of PTE observable. The signs and material values in
the current axis convention are internally consistent: solver `x=b`, `y=a`,
`Sb=+27 uV/K`, `Sa=-6 uV/K`, `sigma_b=1.1e5 S/m`, and
`sigma_a=4.91e5 S/m`.

With `psi=0` at `x_min` and `psi=1` at `x_max`, positive implemented current
is the `+x` component of internal conventional current, from `x_min` to
`x_max`. The epigraph target `Ia>0`, `Ib<0` therefore requests opposite
directions `x_min -> x_max` and `x_max -> x_min`, respectively. Earlier
contract prose that called positive current right-to-left was reversed; the
corrected Shockley-Ramo code and its regression test were already using the
sign stated here.

The local literature files support those values and the Shockley-Ramo model:

- `papers/Adv Funct Materials - 2026 - Blevins - Large Transverse Thermoelectric Effect in Weyl Semimetal TaIrTe4 Engineered for-2.pdf`, pp. 3-4 and Eq. (6).
- `papers/adfm75986-sup-0001-suppmat-2.pdf`, Table S2 and Sec. S5B-D.

They also show why the present geometry cannot be assumed: the weighting field
is device dependent, actual electrode boundaries are Dirichlet boundaries,
sample-air boundaries are insulating, and the transverse example uses an
`a`-axis/electrode angle of 45 degrees.

## P0 blockers

1. **Historical O3/TE1 topology is invalid for production.** Optical Au used
   `rho^3` while thermal and electrical Au used `rho`. The robust run also
   omitted nominal `eta=0.50`; its nominal `Ib` had the wrong sign. The later
   shared-linear/all-projection correction is diagnostic only. The selected
   Lumerical `n-k` law is now implemented solver-free. Its nonuniform
   density-to-component-Yee material Jacobian passes an actual RTX development
   FSP certificate, but B200 endpoint, resonance, complete AD-FD, and mesh
   gates remain open.

   The nonuniform gray optical-Q connection to the custom CUDA PDEs now also
   passes. Every native `Qx/Qy/Qz` sample is overlap-deposited without an
   exact-material equality mask, so relaxed design-layer loss is not
   discarded; its transpose agrees to machine precision. This does not yet
   validate the Maxwell field adjoint or the end-to-end density gradient.

2. **No converged mesh exists.** The historical partial-z sweep refined only
   Au, TaIrTe4, and SiO2, and its tables are stale because they used O3/TE1,
   the old current sign, and an under-specified cache key. Its numerical
   changes are not evidence for the exact-Au route. Si, air, PML,
   optical x/y, thermal, and electrical meshes were not certified.
   The replacement full-domain factor-1/2/4 sweep has now also failed: every
   one of its six final factor-2 to factor-4 comparisons missed the 0.5% gate.
   Worst changes were 3.314% in total Q, 34.072% in the remapped Q field,
   3.634% in TaIrTe4 temperature-field NRMSE, 30.150% in Tmax, and 37.664% in
   PTE current. These are stable spatial-discretization changes, not a temporal
   instability: all 18 individual material runs passed their independent
   physics gates.

3. **The time/material blocker is closed, but no spatial mesh is certified.**
   Material isolation traced the factor-8, Courant-0.5 long-time failure to Au
   dispersion; substrates-only and TaIrTe4-only stayed stationary. At the
   identical partial factor-8 grid, Courant 0.25 passed Au-only/full-dispersion
   32/40-period gates. The full case had 0.1878% field NRMSE, 0.00223% late
   energy change, 0.3567% Q/phasor-flux mismatch, and only 0.000678% Q change
   from 32 to 40 periods. This establishes the 40-period/Courant-0.25 starting
   contract for spatial sweeps, not z/x/y convergence. Float32 ADE coefficients
   are fitted against their realized carrier response with <1e-5 relative
   complex-permittivity error.

4. **The target physical device is not defined.** The code assumes a square
   16 um flake, 100 nm thickness, no crystal rotation, full left/right edge
   terminals, a centered floating Au patch, 285 nm SiO2, and a centered beam.
   The 2026 paper's Device A is 130 nm thick and its collection depends on
   actual flake edges/electrodes. None of the code assumptions has yet been
   confirmed for the user's target device.

5. **Au contact parameters are scenarios, not measurements.** Au/TaIrTe4
   thermal conductance and electrical contact conductance materially affect
   heat spreading and the weighting field. They require a sweep or measured
   bounds before an experimental prediction claim.

## P1 numerical and modeling risks

- The baseline optical z grid has only 2 Au cells (25 nm), 5 TaIrTe4 cells
  (20 nm), 3 SiO2 cells (95 nm), and 5 resolved-Si cells over 1.015 um
  (about 203 nm). These values are starting points, not accepted meshes.
- The thermal model uses 100 nm lateral cells in the device, with progressively
  coarser substrate/air cells and ambient-temperature lateral boundaries at
  +/-32 um. Domain-size, z-mesh, and boundary-condition convergence are absent.
- The electrical model is a 100 nm 2-D network. Nominal void retains tiny Au
  sheet and vertical-contact conductances to regularize otherwise disconnected
  nodes. The final current needs floor-to-zero sensitivity and lateral mesh
  convergence.
- Si and SiO2 are implemented as lossless uniform FDTDX materials. The current
  4 um readback is zero/negligible loss; code now fails if a future material
  contract has non-negligible substrate loss instead of silently discarding it.
- The existing 4-um material certificate reads only the center-frequency n,k;
  it is not a Lumerical time-domain dispersion certificate. The endpoint/final
  exact-Au control builder now constructs sampled Ordal-Au,
  anisotropic-TaIrTe4, and Kitamura-SiO2 inputs over a guard band, but must
  still pass Lumerical fitted-material readback across that band before any
  Maxwell result is promoted. The continuous optimizer carrier has separate
  bandwidth, uniform-density resonance, and AD-FD gates.
- TaIrTe4 uses `epsilon_c=epsilon_b` because no independent c-axis table is in
  the current contract. This approximation needs explicit acceptance or c-axis
  data if out-of-plane absorption is important.
- The present electrical network supports only grid-aligned diagonal tensors
  (`x=b`, `y=a`) and cannot represent an arbitrary crystal/electrode angle or
  the resulting off-diagonal in-plane transport tensors. If the target device
  is rotated (the paper's transverse example uses 45 degrees), the rotated
  optical, thermal, Seebeck, and conductivity tensors must be implemented
  before device or mesh certification.
- The optical-to-thermal remap is conservative and has a discrete transpose,
  but conservation alone does not establish field convergence.
- The current 500 nm production path uses filter/projection, per-projection
  grayness constraints, and an exact binary opening audit. The filter alone is
  not a proof of minimum solid and void feature size.
- AD-FD validates differentiation of a fixed discrete operator. It cannot
  compensate for a wrong geometry, material law, contact value, mesh, or time
  window.

## Historical gray-path corrections already pushed

- One shared linear Au material fraction in the legacy Maxwell, thermal,
  electrical, and direct/adjoint code. This fixed O3/TE1 consistency only; it
  is now production-blocked and must not be mistaken for the selected
  Lumerical `n-k` density relaxation or final exact Au.
- Robust constraints for `eta=0.35, 0.50, 0.65`, including both signed-current
  constraints and grayness at every projection.
- Production entry points blocked until hash-linked certificates exist.
- Production runners now regenerate the mesh selected by the certificate and
  enforce its grid hash, Courant factor, and time windows. Ea and Eb use
  separate same-grid source-power calibrations; a passing certificate can no
  longer silently fall back to the coarse baseline runner.
- Combined-adjoint material placement now derives TaIrTe4 and Au source slices
  from each runner's realized placed-object slices. The former implementation
  used baseline z-cell counts and would have misplaced the adjoint source on a
  refined-z production grid even when the forward runner used that fine grid.
- The new z sweep refines all nine vertical segments (Si, SiO2, TaIrTe4, Au,
  three air regions, and both z-PMLs) together at factors 1/2/4 under the
  validated Courant-0.25/40-period contract. It uses independent Ea/Eb source
  calibration per grid and requires every material case to pass time and
  absorption-flux gates before comparing the final 2-to-4 spatial pair.
- That complete sweep returned
  `BLOCKED_SHARED_LINEAR_FULL_DOMAIN_Z_CONVERGENCE`: physics gates passed but
  convergence passed in 0/6 final-pair cases. This closes the question of
  whether the legacy shared-gray factor-4 grid was adequate; it was not. It
  does not certify or choose a grid for exact-Au Lumerical. The next optical
  z/x/y/PML convergence hierarchy belongs to that superseding route.
- Source calibration bound to exact grid/source/time metadata.
- Stale validation artifacts rejected by status, material-law, grid, input,
  and SHA checks.
- Material interpolation range checks and a fail-closed substrate-loss check.
- Portable launchers/raw paths and explicit single-GPU contracts.
- Correct Shockley-Ramo sign in the electrical objective, current-density map,
  thermal pullback, and all downstream gradients: the implemented edge term is
  now `-sigma*S*DeltaT*Deltapsi`, matching `J_local=-sigma*S*grad(T)`.
  Historical signed-current artifacts made before this correction are stale
  and must be recomputed; they cannot be relabeled as physical current.
- Heat generation and its direct-loss derivative now use the realized float32
  discrete-ADE susceptibility, not the continuous target epsilon. The time
  diagnostic measured a 1.11-1.14% target/discrete Q mismatch. Existing
  optical-gradient/phase tables are marked stale until stability is fixed and
  AD-FD is reissued on this discrete-loss objective.
- New shared-linear Eb optical-gradient split: 0.0158% total AD-FD error at
  step 0.0025. This is diagnostic only, on the unconverged baseline mesh.
- New discrete adjoint phase diagnostic: production `exp(+i omega dt)` is the
  best tested phase, with 0.00937% field-term AD-FD error.

## Required validation order

1. Confirm and freeze `physical_device_contract.json` from the target device.
2. Establish optical time-window stationarity and Q/closed-flux closure.
3. Converge the full optical z domain, including Si, air, and z-PML.
4. Converge optical x/y and PML/domain clearance.
5. Converge thermal x/y/z/domain/boundary placement and electrical lateral mesh.
6. Sweep electrical void floors and uncertain Au/TaIrTe4 contacts.
7. Issue one combined multi-direction AD-FD certificate on the chosen mesh,
   hash-linked to the mesh and device certificates.
8. Validate the 4-um Lumerical `n-k` density carrier: ordinary-Au 4-um
   endpoint field/absorption/Q parity, quantified finite-source-band error,
   uniform-density resonance sweep, selected-mesh/B200 repetition of the now
   passed RTX component-Yee Jacobian, and multi-direction latent AD-FD for Ea
   and Eb. All four planned common latent directions now pass for both
   polarizations, and their exact signed epigraph passes in all four, only on
   the RTX development mesh. The fail-closed Lumerical evaluation driver,
   selected mesh, and B200 repetition remain. Do not resume either O3/TE1
   or the shared-linear FDTDX LD_MMA history.
9. Revalidate the final exact-binary design on the finer meshes and parameter
   sensitivity corners before claiming the polarization current switch.
