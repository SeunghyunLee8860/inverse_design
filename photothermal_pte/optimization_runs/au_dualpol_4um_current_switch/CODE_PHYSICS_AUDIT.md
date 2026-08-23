# Au/TaIrTe4 4 um inverse-design physics audit

Status: **production blocked**. The repository can run numerical diagnostics,
but the historical topology is not a validated physical design and a new
optimization must not start yet.

## Intended observable

The requested switch is a signed zero-bias current: `Ia > 0` for `E || a` and
`Ib < 0` for `E || b`. The code uses a Shockley-Ramo functional equivalent to
`I = integral J_local . grad(psi) dA`, with
`J_local = -sigma S grad(T)`.

This is the correct class of PTE observable. The signs and material values in
the current axis convention are internally consistent: solver `x=b`, `y=a`,
`Sb=+27 uV/K`, `Sa=-6 uV/K`, `sigma_b=1.1e5 S/m`, and
`sigma_a=4.91e5 S/m`.

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
   omitted nominal `eta=0.50`; its nominal `Ib` had the wrong sign. Production
   now uses one shared linear fraction and all three robust projections, but no
   full optimization has been rerun under that corrected contract.

2. **No converged mesh exists.** The historical partial-z sweep refined only
   Au, TaIrTe4, and SiO2, and its tables are stale because they used O3/TE1,
   the old current sign, and an under-specified cache key. Its numerical
   changes are not evidence for the current shared-linear code. Si, air, PML,
   optical x/y, thermal, and electrical meshes were not certified.

3. **Time error and spatial error are mixed.** The partial-z sweep did not
   compare the previous and late phasor windows. Some fine-grid Q/closed-flux
   errors reached 0.84-1.96%, so time stationarity and discrete ADE absorption
   closure must be isolated before another z conclusion is possible.

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
- TaIrTe4 uses `epsilon_c=epsilon_b` because no independent c-axis table is in
  the current contract. This approximation needs explicit acceptance or c-axis
  data if out-of-plane absorption is important.
- The optical-to-thermal remap is conservative and has a discrete transpose,
  but conservation alone does not establish field convergence.
- The current 500 nm production path uses filter/projection, per-projection
  grayness constraints, and an exact binary opening audit. The filter alone is
  not a proof of minimum solid and void feature size.
- AD-FD validates differentiation of a fixed discrete operator. It cannot
  compensate for a wrong geometry, material law, contact value, mesh, or time
  window.

## Corrections already pushed

- One shared linear Au material fraction in Maxwell, thermal, electrical, and
  all direct/adjoint derivatives.
- Robust constraints for `eta=0.35, 0.50, 0.65`, including both signed-current
  constraints and grayness at every projection.
- Production entry points blocked until hash-linked certificates exist.
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
8. Start a new shared-law robust optimization; do not resume O3/TE1 history.
9. Revalidate the final exact-binary design on the finer meshes and parameter
   sensitivity corners before claiming the polarization current switch.
