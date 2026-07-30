# Latest photothermal validation status

## Paper-IR source-only certification

- Status: `BLOCKED_LUMERICAL_LICENSE_UNAVAILABLE_BEFORE_SOURCE_ONLY`.
- The paper/SI beam audit is complete. The paper reports a 7–13 µm
  Block LaserTune QCL, NA=0.4 reflective objective, approximately 9–16 µm
  diffraction-limited spot, and 11 µm / 285 µW for Figure 3, but it does
  not define the spot as radius/diameter/FWHM/1/e² or publish the exact
  11-µm waist plane and pupil fill.
- The optical source contract is now fixed as the
  **paper-like scalar-Gaussian scenario with an explicitly assumed waist**:
  Gaussian 1/e² radius `w0=12 µm`, source span `50 µm`, lateral domain
  `60 µm`. Its analytic square capture is `99.992914%`; analytic boundary
  maximum/mean are `1.93379e-4 / 5.86049e-5`. This is not called an
  experimentally reproduced or paper-certified beam.
- A vectorial thin-lens comparison is no longer a gate or blocker and is
  retained only as an optional future diagnostic. The legacy nominal
  `w0=2 µm` case remains
  `DIAGNOSTIC_ONLY_INVALID_FOR_PAPER_LIKE_BEAM`.
- Three contract-only probes, including the new fixed-scalar-contract probe
  at `f89c7b7`, failed before an FDTD session opened:
  `ANSYSLI exited or could not read server port`. Neither `runsetup` nor
  a GPU solve started; no CPU fallback was used. Consequently there is no
  realized-beam, grid, memory, runtime, closure, or auto-shutoff result.
- Historical 12/48-µm material runs are recorded only as non-certifying
  resource context. Total expected time for the new 60-µm homogeneous-air
  contract is unresolved until license/session startup and actual
  `runsetup` readback succeed.
- Per the fail-closed order, planar-a/b, finite-edge-a/b, thermal,
  weighting potential, PTE, adjoint, gradient, and optimization were not
  run. The previous
  `VALIDATED_OFFLINE_MASKED_PLANAR_AND_SOURCE_AUDIT` status below is
  preserved.
- When license startup succeeds, exactly one homogeneous-air GPU source-only
  case is run first. A passing scalar source-only gate directly authorizes
  planar-a/b then straight-45-degree finite-edge-a/b with identical source
  geometry and incident-power normalization. No polarization-specific raw-Q
  rescaling is allowed.
- Report, beam audit, summary, CSV tables, aperture figure, and manifest:
  `reports/paper_ir_source_only_certification/`.

## Offline w0=2 µm masked-planar and source-contract audit

- Status: `VALIDATED_OFFLINE_MASKED_PLANAR_AND_SOURCE_AUDIT`.
- No new FDTD, thermal, PTE, weighting-potential, adjoint, gradient,
  optimization, or w0=6.5 µm calculation was executed. The analysis reads
  the existing 4-ps planar-b and finite-edge-b artifacts only.
- The exact bounded-dual-cell overlap with `y<=x` removes `49.950862%` of
  planar power and explains `98.760752%` of the observed planar-to-edge
  power drop. The remaining signed redistribution is
  `D_EM/P_planar=+0.626783%`; the signed decomposition closes exactly.
- Equal-power NRMSE is `99.664762%` for full planar versus analytic
  half-plane-masked planar, `12.183080%` for analytic masked planar versus
  finite edge, and `100.185592%` for full planar versus finite edge.
  These pairwise metrics are not added or interpreted as a decomposition.
- The component diagnostic
  `f_c=Im(epsilon_edge,c)/Im(epsilon_planar,c)` is explicitly not
  occupancy and is not clipped. It exceeds one in 268 x-component cells
  and 268 y-component cells, reaching about `1.99951`; no negative values
  occur. Near-floor denominator cells are separately inventoried.
- Native E/index component-coordinate mismatch is at most `8.47e-22 m`.
  Planar/edge coordinates agree, and the exact bounded-dual-cell
  component-to-common remap has maximum relative power error `1.61e-16`.
- The saved source-object field, not a paraxial formula, is primary source
  evidence. Its square-boundary maximum is `73.5030%` of peak, its fitted
  infinite-Gaussian square captured fraction is only `32.4850%`, and
  `lambda/(pi*w0)=1.750704`; the nominal w0=2 µm scalar-source setup is
  therefore an aperture-truncation diagnostic, not a paper-like beam
  certificate.
- Analysis implementation is checkpointed at `b231267`; 12 focused tests and the full
  63-test finite-inverse-design suite pass. Report, JSON, CSV tables,
  figures, and manifest:
  `reports/paper_ir_w2_masked_planar_offline/`.

## Nominal-w0=2 µm planar/finite-edge optical diagnostic

- Status:
  `PARTIAL_W2_EDGE_ISOLATION_OBSERVABLE_Q_VALIDATED_AUTO_SHUTOFF_FAILED`.
- The approved GPU-only stage is complete for planar-a, planar-b, and
  finite straight-45-degree-edge-b at 1.2 ps and 4 ps. No `w0=6.5 µm`
  calculation was started. No CPU FDTD fallback, thermal, PTE, adjoint,
  gradient, or optimization calculation was used.
- All six completed solves use the same 12 µm domain, six PML boundaries,
  24 PML layers, 10 nm flake-region `dz`, 130 nm TaIrTe4 thickness,
  11 µm wavelength, nominal 2 µm scalar-Gaussian waist, and the explicit
  `epsilon_c=epsilon_b` 3D closure with `x=b`, `y=a`.
- Matched-volume common/native six-face closure is below 0.5% in every
  4 ps case. Common-grid closure is `0.119337% / 0.102457% / 0.015313%`
  for planar-a / planar-b / finite-edge-b.
- Each 1.2→4 ps observable-Q comparison passes: P_Q changes
  `0.000294% / 0.000514% / 0.000595%` and normalized spatial-Q NRMSE is
  `0.000076% / 0.000116% / 0.000336%`.
- Auto-shutoff remains above `1e-5` in every pair
  (`1.56401e-5`, `1.95295e-5`, `2.00846e-5` at 4 ps), so the artifacts
  are not promoted to production Q. This failure is kept separate from
  the observable-Q pass.
- The requested waist is 2 µm, but the fitted field-plane effective
  waists are `6.437 / 6.369 / 6.369 µm`. This is therefore a
  nominal-w0=2 µm diagnostic, not a realized 2 µm beam certificate and
  not a paper-like result.
- At 4 ps, raw P_Q is
  `1.335218e-16 / 1.955253e-16 / 9.663323e-17 W`. Planar-b versus
  finite-edge-b changes raw power by `50.5776%`; after equal-power
  normalization its spatial-Q NRMSE is `100.1856%`, demonstrating a
  large finite-edge spatial effect in this diagnostic.
- Raw NPZ/FSP files remain external and are SHA-256 inventoried. Report,
  JSON, CSV, figures, and manifest:
  `reports/paper_ir_w2_planar_edge_diagnostic/`.

## Offline paper-IR Q, thermal, and remap controls

- Overall status:
  `PARTIAL_OFFLINE_PAPER_IR_VALIDATION_BLOCKED_PLANAR_Q_AND_FIG3HI`.
- No new FDTD solve was run. Saved 1.2-ps and 4-ps matched-control-volume
  Q artifacts have observable convergence: P_Q changes `0.000123%`,
  normalized spatial-Q NRMSE is `0.000738%`, spatial correlation is
  `0.999999999965`, and the hotspot does not move.
- This is diagnostic-Q convergence only. Auto-shutoff remains
  `1.81076e-5 / 1.80982e-5`, above the `1e-5` gate, so neither artifact is
  promoted to production Q.
- The paper analytic Gaussian--Beer--Lambert source plus the reduced
  TaIrTe4 Robin model reproduces the Figure-3F/G thermal trend. The robust
  exact-edge x-gradient b/a ratio is `1.440200 / 1.440259 / 1.440266` on
  200/100/50-nm meshes; the 100-to-50-nm robust metric changes about
  `0.454%`. Raw cell-gradient maxima change about `9%` and Tmax changes
  about `1.3%`, so those diagnostics remain unresolved.
- Exact-cut-cell direct analytic Q versus 33.898-nm Yee-like layout plus
  the current conservative remap passes its field gates: worst Q_T/T/grad
  NRMSE are `0.494% / 0.040% / 0.334%`. The worst raw-cell peak changes
  `1.891%` and remains diagnostic.
- The required edge-free planar TaIrTe4-stack Q artifact does not exist.
  Empty-stack is not TaIrTe4, finite-centre is an edged polygon, and the
  saved straight-edge artifact is legacy `epsilon_c=16` with Qz=0.
  Therefore the three-source decomposition is
  `BLOCKED_PLANAR_STACK_Q_ARTIFACT_UNAVAILABLE`; per the approved order,
  Figure 3H/I was not started.
- Report, JSON, CSV, figures, and manifest:
  `reports/paper_ir_offline_q_thermal_controls/`.

## Matched-control-volume paper-IR GPU smoke

- Official status remains:
  `PARTIAL_PAPER_IR_CONTROL_VALIDATION_BLOCKED_OPTICAL_RUNTIME_AND_UNRESOLVED_EDGE_METRIC`
- New smoke substatus:
  `FAILED_MATCHED_CONTROL_VOLUME_SMOKE_AUTO_SHUTOFF_UNRESOLVED`
- The one approved 12 x 12 um, `a`-polarized GPU rerun completed normally
  on GPU 4: 131,247 iterations, 4.000005 ps, and 640.481187 s wall time.
  No CPU FDTD fallback was used.
- Q and all six flux faces were read back on one realized volume:
  x/y=`+/-4.542372881356 um`, z=`[-180,+50] nm`.
- `P_Q(native Yee)=8.715867473376e-17 W`,
  `P_Q(common)=8.701470836178e-17 W`, and
  `P_six=8.717844152299e-17 W`. Native-Yee and common-grid closure are
  `0.022674%` and `0.187814%`; both pass the `<0.5%` gate.
- Therefore the earlier `9.18%` value was an unmatched-control-volume
  comparison, not a 9.18% FDTD energy-conservation error.
- Field and index coordinates are now independently read from their saved
  monitors. Their maximum component-specific mismatch is
  `8.47033e-22 m`; the earlier copied exact-zero claim is retracted.
- The final auto-shutoff is `1.80982e-5`, still above the requested
  `<1e-5` gate after 4 ps. The Q artifact is therefore not promoted.
- The original postprocess-failure JSON is preserved. A separate read-only
  recovery called neither FDTD `run` nor `runanalysis`; it did not alter Q.
  No thermal, PTE, adjoint, gradient, or optimization calculation ran.
- Report, JSON, face CSV, and manifest:
  `reports/paper_ir_edge_material_gradient_controls/`.

## Reduced one-polarization paper-IR GPU diagnostic

- Official status remains:
  `PARTIAL_PAPER_IR_CONTROL_VALIDATION_BLOCKED_OPTICAL_RUNTIME_AND_UNRESOLVED_EDGE_METRIC`
- Diagnostic substatus:
  `FAILED_DIAGNOSTIC_ONE_POL_GPU_SMOKE_SIX_FACE_CLOSURE`
- This was a separately labeled 12 x 12 um, `a`-polarized, GPU-only smoke;
  it is not the 48-um paper-like production case and is not a paper result.
- The v261 engine completed normally on GPU 4 in `155.807 s` solver wall
  time (`142.705 s` GPU stepping), using a logged
  `402 x 402 x 161 = 26,018,244` grid and `1.336 GiB` precise GPU memory
  estimate. No CPU FDTD fallback was used.
- The material and coordinate readbacks pass, including
  `epsilon_x=epsilon_z=epsilon_b`; Qx, Qy, and nonzero Qz were exported with
  finite values, and component-specific native Yee coordinates were saved.
- In native source-amplitude units,
  `P_Q(common)=8.701460132991e-17 W`,
  `P_Q(native components)=8.704063329997e-17 W`, and
  `P_six=9.580832894734e-17 W`. The common-grid closure is `9.178458%`,
  so the required `<0.5%` gate fails.
- Native/common Q integration differs by only `0.029908%`, excluding
  component interpolation as the main 9.18% error. Two confirmed issues
  remain: the Q output ends near x/y=`+/-4.542 um` while the flux box is
  x/y=`+/-5 um`, and the 1.2-ps run ended at auto-shutoff
  `1.81076e-5`, above the requested `1e-5`.
- The face powers are strongly cancelling: the Q/flux mismatch is only
  `0.369893%` of the sum of absolute face powers but `9.15%` of the small
  net absorbed flux. Existing data cannot separate unmatched-volume loss
  from finite-time DFT error.
- Per the fail-closed one-smoke contract, no second FDTD solve was started.
  No Q correction, thermal, PTE, adjoint, gradient, or optimization ran.
- This historical unmatched-volume interpretation is superseded by the
  matched-control-volume smoke above. Its raw values remain provenance only.
- Report, JSON, face CSV, and raw-artifact manifest:
  `reports/paper_ir_edge_material_gradient_controls/`.

## Existing paper-IR checkpoint and GPU-failure audit

- Status:
  `AUDITED_EXISTING_PAPER_IR_CHECKPOINTS_UNRESOLVED_ENGINE_TERMINATION_AND_EDGE_METRIC`
- The audit began from clean local HEAD
  `651797fefbcaa254737bcec3cac854979ae2bfef`; the four existing checkpoints
  remain unsquashed and unchanged.
- The a/b 200/100/50 nm paper-reduced cases use the same geometry, Robin
  boundary, polarization-specific source, and conservative remap contract.
  Only the intended lateral core step and grid shape differ.
- From 100 to 50 nm, analytic-source fitted-x strip mean changes by about
  `0.61%`, but fitted edge-normal strip mean changes by about `2.99%`.
  Legacy Maxwell-Q fitted-x strip mean changes by about `0.40%` for a and
  `12.85%` for b using the existing symmetric relative metric. Raw maxima
  remain diagnostic only, so the local edge-gradient gate is not promoted.
- Retry 4 acquired a GPU solve license, meshed
  `1461 x 1461 x 161 = 343,657,881` gridpoints, and began 39,362 time
  steps. The log stops at `3.3357%`; the kernel records a contemporaneous
  `fdtd-solutions-app` remote-messenger segfault. The external engine exit
  code was not captured, so the formal classification remains
  `UNRESOLVED_ENGINE_TERMINATION`.
- No contemporaneous OOM, GPU Xid/reset, timeout, or retry-4 license failure
  was found. Precise GPU memory estimate was `15.169 GiB` against
  `49,140 MiB` capacity; host-available memory was `903.354 GiB`.
- There is no x/y override. The large high-index TaIrTe4 half-plane spans
  the 48-um domain, causing the accuracy-5 auto mesh to use approximately
  `33.97 nm` lateral monitor spacing over tens of micrometres. Native solver
  coordinates were not recovered; incomplete-HDF5 coordinates are explicitly
  labeled monitor sampling grids.
- Full table, audit JSON, and partial-HDF5 coordinate table:
  `reports/paper_ir_edge_material_gradient_controls/`.

## Paper-like IR material/source/edge controls

- Status:
  `PARTIAL_PAPER_IR_CONTROL_VALIDATION_BLOCKED_OPTICAL_RUNTIME_AND_UNRESOLVED_EDGE_METRIC`
- The production 3D TaIrTe4 optical closure is now
  `epsilon_c(lambda)=epsilon_b(lambda)`. This is the paper-consistent
  finite-edge 3D implementation extending the reported in-plane
  `epsilon_a,epsilon_b` data; it is not described as a directly measured
  c-axis property. The old lossless `epsilon_c=16` artifact is preserved as
  legacy diagnostic only.
- At 11 um, requested
  `epsilon_x=epsilon_z=13.2681477+26.1817959i` and
  `epsilon_y=-42.9662316+204.5326948i`. The v261 material fit and
  finite-time-step readbacks give exact x/z equality; relative fit errors
  against the requested x/y/z values are
  `4.95e-6 / 1.91e-5 / 4.95e-6`.
- Three same-contract analytic controls were run at 200/100/50 nm. At
  50 nm the paper-like absorbed-power source gives
  `max|dT/dx|_b/max|dT/dx|_a=1.446954`; the equal-power shape control gives
  `0.967078`; exact-identical Q gives `1.000000` with zero field difference.
  Equal-power normalization was used only for that analytic control.
- The exact-coordinate robust comparator uses
  `n=(-x+y)/sqrt(2), t=(x+y)/sqrt(2)` and retains raw maxima as diagnostics.
  Analytic fitted-x strip mean changes `0.6096%` from 100 to 50 nm, but the
  legacy Maxwell-Q b case changes `12.851%`; fit-band sensitivity also
  exceeds 10%. No 50- or 100-nm local edge-gradient mesh is promoted.
- The v261 contract-only GPU session passes all geometry, PML, source,
  material, and resource readbacks. Three GPU-only solve attempts stopped
  before timestepping because only 4 of 9 requested `lum_fdtd_solve` tasks
  were available. A fourth acquired the licenses, meshed a
  `1461 x 1461 x 161` grid, and started GPU timestepping, but Lumerical
  API/engine communication failed after `3.3357%`. Its incomplete HDF5 is
  provenance only. CPU FDTD fallback was not used. Therefore production
  Qx/Qy/Qz, P_Q, six-face closure, native Yee coordinates, and edge-normal
  Q remain blocked; legacy `epsilon_c=16` has exactly `Qz=0`.
- The historical scalar-versus-thin-lens comparison remains an optional,
  unexecuted diagnostic only and is no longer a finite-edge gate.
- No raw Lumerical Q modification, PTE current, adjoint, gradient, or
  optimization was performed.
- Report and machine-readable outputs:
  `reports/paper_ir_edge_material_gradient_controls/`.

## Straight-edge spatial-Q/remap/gradient audit

- Status:
  `UNRESOLVED_STRAIGHT_EDGE_OPTICAL_AND_THERMAL_SPATIAL_CONVERGENCE`
- The earlier `x/y/z/x` support projection is coordinate-order dependent.
  A symmetric Gaussian regression gives a 50.0% relative L1 difference
  against `y/x/z/y` with identical total power. The straight-edge path now
  uses one physical-3D-nearest support projection with symmetric tie
  splitting; conservation, transpose, and reflection tests pass.
- The 50.0% value is a structural synthetic regression, not the actual-Q
  error. On the saved raw Q, x-first versus y-first differs by only
  `0.002586% / 0.002235%` for a/b; historical versus physical-nearest differs
  by `0.019838% / 0.004467%`. The old operator is invalid in principle, but
  it does not explain the roughly 20% gradient-order reversal.
- Area/volume averages now use literal cell measures. The report retains
  `max|dT/dx|`, `max|dT/dy|`, `max|grad T|`, `max|dT/dn|`, and
  `max|dT/dt|` separately. The Figure-3G comparator is `max|dT/dx|`.
- Paper analytic Gaussian--Beer--Lambert Q plus the Eq.-S4 reduced Robin
  model gives `max|dT/dx| b/a = 1.44677`, reproducing the requested
  polarization order.
- Saved finite-edge Lumerical Q on the same reduced thermal operator gives
  `b/a = 0.805447` at 100 nm and `0.881330` at 50 nm; the expanded 80-um
  FVM gives `0.817054`. At 50 nm, four of five gradient ratios remain below
  one while `max|dT/dy|=1.000035` is numerically near-null; none is promoted
  because the peak-gradient mesh gate fails.
- The paper-reduced 100-to-50-nm thermal refinement does not pass:
  the worst change among the five edge-gradient observables is `67.8619%`
  (limit `1%`). `Tmax` changes by up to `2.735%` and the fixed-ROI average
  by only `0.224%`, demonstrating that local derivatives, not the integrated
  temperature response, remain mesh unresolved.
- Expanded-model 48-to-80-um relative changes are at most `0.0882%` for
  `Tmax`, `0.444%` for the fixed 24-um ROI average, and `0.00772%` for the
  five edge-gradient metrics. Lateral/bottom Dirichlet powers are reported
  as numerical truncation-boundary fluxes, not physical path fractions.
- The common Q artifact grid is `33.9703/33.9703/10 nm`, but native Yee
  lateral-mesh and fitted sampled-epsilon readbacks remain absent.
  `epsilon_c=16+0i` forces `Qz=0` and remains an edge-model blocker.
- No new FDTD was run. Report and data:
  `reports/paper_ir_straight_45_edge_spatial_q_audit/`.

## Corner-free straight 45-degree edge optical/thermal control

- Legacy status: `FAILED_STRAIGHT_45_EDGE_PAPER_GRADIENT_TREND`
- This is preserved as the pre-audit checkpoint; its unweighted averages and
  axis-ordered remap must not be used as promoted values.
- A single `y=x` TaIrTe4/air edge replaces the approximate polygon; TaIrTe4
  occupies `y<=x`, and all remote polygon faces lie outside the 48-um optical
  domain. There is no physical corner in the calculation.
- Independent 11-um `E||a` and `E||b` v261 GPU-FDTD runs use a centred
  `w0=6.5 um` Gaussian and pass six-face closure at
  `0.249998% / 0.411927%`.
- At 285-uW incident power, `P_abs(a/b)=30.0635/37.2358 uW`;
  `P_abs,b/P_abs,a=1.23857`.
- The explicit anisotropic/multimaterial thermal FVM was run at 200- and
  100-nm lateral meshes. Q mapping, energy balance, and linear residual pass
  in all four cases.
- On the 100-nm mesh, flake-average
  `DeltaT(a/b)=0.0341940/0.0422899 K`, but flake
  `Tmax(a/b)=0.239721/0.228105 K` and max edge-normal gradient
  `35.972/29.022 kK/m`.
- Thus the requested Figure-3F ordering
  `|grad T_b|>|grad T_a|` is not reproduced. The same reversal occurs at
  200 nm, although the peak gradient magnitude itself is not mesh converged.
- The earlier concave polygon corner is therefore not the sole cause of the
  failed trend. No weighting field or PTE current was applied, and no
  empirical gain, fitting, AD-FD, or optimization was used.
- Report and machine-readable outputs:
  `reports/paper_ir_straight_45_edge/`.

## Paper-like Device-A 11-um coupled sanity check

- Status: `FAILED_COUPLED_DEVICE_A_IR_PTE_SANITY_GEOMETRY_UNRESOLVED`
- This is an actual
  `v261 GPU Lumerical Gaussian Q -> conservative thermal remap ->
  Cartesian FVM -> solved approximate-contact weighting potential -> PTE`
  calculation, not a parameter-only equation replay.
- Published quantities held fixed:
  130-nm TaIrTe4, 285-nm SiO2/Si,
  `kappa_a,b,c=14.4/3.8/1.0 W/(m K)`,
  `sigma_a,b=4.91e5/1.10e5 S/m`,
  `S_a,b=-6/+27 uV/K`, and the paper interface conductances.
- Central finite-Gaussian Lumerical absorption at 11 um,
  `E||a / E||b = 17.350% / 25.114%`;
  the independent TMM gives `17.673% / 26.329%` and the paper Fig. 3D is
  approximately `18% / 26%`.
- Central and edge six-face optical closure ranges from
  `0.0103%` to `0.0636%`.
- At the selected off-axis edge point and 285-uW incident power,
  Lumerical absorbed power is `28.597 / 35.578 uW` for `E||a / E||b`.
- Both the current expanded thermal model and a separate paper-Eq.-S4
  reduced Robin model pass Q-power, thermal energy, and linear-residual
  numerical gates.
- The coupled PTE polarization trend does not pass:
  after the local contact-cell-width correction, expanded
  `|I_a|/|I_b|=1.18855` and paper-reduced `1.22589`, whereas the paper
  reports approximately `0.8`.
- The corrected expanded currents are `24.6549/20.7436 nA` for a/b;
  the old `24.0479/20.2183 nA` values remain legacy diagnostics. The current
  changes by `2.52%/2.60%`, confirming that the contact-width bug was
  material even though the polarization ratio changes only slightly.
- The initial diagnostic was a strong `E||a` hotspot at the concave corner
  of the approximate Fig.-2A polygon. The newer corner-free straight-edge
  control above still reverses the gradient ordering, so that corner is no
  longer considered the sole cause. Exact Device-A CAD, electrode mask,
  beam center, and wavelength-specific beam radius remain unpublished.
- No empirical gain/current rescaling, transient, AD-FD, or optimization
  was used in this separate paper sanity check.

## Scale-adaptive near-null combined AD-FD diagnostic

- Status: `FAILED_SCALE_ADAPTIVE_NEAR_NULL_COMBINED_ADFD`
- The immutable five-direction failure remains unchanged.
- A first recovered `h=0.02` central-plus orphan FSP was rejected:
  its byte size, six-face closure, and objective trend disagreed with the
  clean pairs. It remains provenance-only and is not used below.
- Clean replacement central-plus FSP:
  `122773042` bytes,
  SHA-256
  `b3e2f235ee52fe03843c30614d8bd55d05d5934d6ad3605b7b7c9af0c1513807`.
- Clean new-pair worst optical closure / Q mapping / thermal energy /
  linear residual:
  `2.01905e-4 / 2.38618e-16 / 3.16622e-12 / 1.02229e-11`;
  all pass their existing gates.
- `h=0.02` AD-FD relative error, central 4/6 um:
  `0.0105939% / 0.00986133%`.
- Selected `h=0.005` AD-FD relative error, central 4/6 um:
  `0.144072% / 0.194467%`.
- Selected `h=0.005` AD-FD relative error, fixed-random 4/6 um:
  `0.0265038% / 0.00232609%`.
- Strict `0.02 -> 0.01 -> 0.005` step-plateau values,
  central 4 um / random 4 um / central 6 um / random 6 um:
  `0.170038% / 0.137484% / 0.144701% / 0.0745501%`
  (limit `0.1%`; only the final case passes).
- No clipping, empirical normalization, or gradient rescaling was used.
- Combined physical-density promotion, gray-law sensitivity, full latent
  AD-FD, and optimization remain blocked.
- A 17-figure validation suite is published under
  `photothermal_pte/reports/inverse_design_pte_adfd/figures/`.

## Corrected combined five-direction diagnostic

- Status: `FAILED_CORRECTED_COMBINED_PHYSICAL_RHO_PTE_ADFD`
- Full combined certificate: not validated
- Worst strong error: `2.42739e-5` (limit `1%`)
- Worst multidirection normalized error: `3.15606e-5` (limit `1%`)
- Worst directional-subspace angle: `0.00119228 deg` (limit `1 deg`)
- Closure, Q mapping, energy, residual, and transpose gates: passed
- Unresolved FD step plateau:
  central 4 um `0.139509%`, random 4 um `0.509411%`,
  random 6 um `0.242508%` (limit `0.1%`)
- Failed raw JSON SHA-256:
  `566e601759e044b48f0b723f02123b4615fbf78577942e388deca3fc76f645c3`
- No empirical normalization or gradient rescaling
- Gray-law sensitivity, full latent AD-FD, and optimization remain blocked
- Next gate: reduce the offending-direction FDTD finite-difference noise
  floor and repeat only the unresolved subgate

## Corrected combined strong-direction PTE AD-FD

- Status:
  `DIAGNOSTIC_PASSED_CORRECTED_COMBINED_STRONG_DIRECTION_ADFD`
- Scope: new corrected adjoint-aligned direction at
  `h=0.01, 0.005, 0.0025`; not the final five-direction certificate
- Worst strong-direction relative error: `2.427386e-5` (limit `1%`)
- 4 um / 6 um FD step-plateau relative change:
  `9.23008e-6 / 1.23328e-5` (limit `0.1%`)
- Strict monotone difference reduction: `false / false`, explicitly retained
- Worst optical closure: `2.16546e-4` (limit `0.5%`)
- Worst Q mapping error: `2.38626e-16` (limit `0.5%`)
- Worst thermal energy balance: `3.17640e-12` (limit `1%`)
- Worst linear residual: `1.02135e-11` (limit `1e-8`)
- Mapping transpose error: `2.98326e-15` (limit `1e-12`)
- Empirical normalization and gradient rescaling: absent
- Original Stage 10 failure and original monotonic-heuristic raw JSON:
  immutable
- Next gate: central-localized, design-edge-localized, smooth/asymmetric,
  and fixed-seed-random combined physical-rho AD-FD
- Gray-law sensitivity, full latent AD-FD, and optimization remain blocked

## Full Yee dual-cell gradient-measure correction

- Status: `VALIDATED_FULL_YEE_DUAL_CELL_GRADIENT_MEASURE`
- Root cause: `J_c=d epsilon_Yee,c/d rho` already encodes conformal fill and
  exact design support; clipping `dV_c` again to the nominal design box
  double-counted the support fraction
- Correct Maxwell bilinear measure: complete component-specific Yee
  dual-cell volume
- Preserved old clipped combined errors at `h=0.005`, 4 um / 6 um:
  `2.78106% / 2.82633%`
- Corrected full-Yee combined errors at `h=0.005`, 4 um / 6 um:
  `8.49222e-6 / 1.56977e-5`
- Worst corrected optical/combined error across 4/6 um and
  `h=0.01/0.005`: `1.56977e-5` (limit `1%`)
- Old clipped computation reproduction error: `0`
- Active `J_x/J_y/J_z` rows changed by erroneous clipping:
  `5520 / 5520 / 6161`
- Exact component-source GPU diagnostic reproduced the official FieldRegion
  gradient; source staggering/collocation was not the root cause
- Maximum forward/adjoint coordinate mismatch: `4.23516e-22 m`
- Stage 10 failure and inverse-collocation strong failure remain immutable
  diagnostics; neither is relabeled as passing
- Empirical normalization, gradient rescaling, gray-law sensitivity,
  latent AD--FD, and optimization: absent
- This checkpoint validates the corrected measure on the existing smooth
  direction only; it is not the final multi-direction combined certificate
- Next gate:
  `CORRECTED_STRONG_AND_FIVE_DIRECTION_PHYSICAL_RHO_PTE_ADFD`

## Large-background non-periodic inverse-design AD–FD

- PTE/nodal-contract audit:
  `AUDITED_PTE_DISCRETE_OPERATOR_AND_81X81_NODAL_CONTRACT`
- Physical design variables: `81 × 81` nodes on exact
  `[-1,1] um × [-1,1] um`, `25 nm` spacing; not 81 finite-width pixels
- Physical density: 2D nodal field extruded from `z=0` to `600 nm`
- PTE weighting surrogate:
  `dpsi/dx=dpsi/dy=1/(4 um)`; periodic derivative wrap absent
- PTE affine analytic / forward-source / temperature-source FD errors:
  `0 / 0 / 5.42831e-11`
- PTE meaning: uniform-45-degree surrogate only; not a solved finite-contact
  terminal weighting potential
- Explicit thermal grid-parameter audit:
  `AUDITED_EXPLICIT_THERMAL_INDEPENDENT_GRID_PARAMETERS`
- Independent parameters:
  `core_xy_cell_size_m`, `flake_dz_m`, `design_dz_m`
- Legacy 100 nm baseline versus explicit `100/25/100 nm` parameterization:
  bitwise-equal grid, material, kappa, and z-interface arrays
- Realized baseline / xy-refined / flake-z-refined / design-z-refined
  TaIrTe4/design z-cell counts:
  `4/6`, `4/6`, `8/6`, `4/12`
- This parameter audit does not claim thermal convergence
- v261 FDTD license/API gate:
  `PASSED_V261_FDTD_LICENSE_API_PROBE`
- FDTD application version / session / script / save / reload:
  `8.35.4522 / passed / passed / passed / passed`
- License/API probe solver and optimization execution:
  `false / false`
- Initial direct-probe failure cause:
  restricted sandbox blocked the localhost Ansys license socket; not seat
  exhaustion or missing entitlement
- A separate pre-existing GPU FDTD optimization process was observed and was
  neither started nor modified by this checkpoint; it must not overlap the
  timed matched CPU-TFSF gate
- Matched uniform-rho optical forward:
  `VALIDATED_MATCHED_RHO05_CPU_TFSF_FORWARD`
- Matched case: rho `0.5`, PML `32`, x/y stabilized and z standard, flake
  `dz=2.5 nm`
- PML-32 outer x/y expansion `6.4 -> 7.2 um` preserved realized PML-inner
  x/y at approximately `[-2,2] um`; ROI/TFSF/Q bounds were unchanged
- Matched `P_Q / P_six / closure`:
  `1.6887880194040323e-12 W / 1.6893345559747856e-12 W / 3.23522e-4`
- Matched `Qx / Qy / Qz`:
  `1.6885593488584841e-12 / 2.286705455481133e-16 / 0 W`
- Matched native Q SHA-256:
  `711c4c93589603f32bfc0525e1b63b36fd773a0ace561509ed0391cb2604ddb2`
- Matched native/complete wall time:
  `583.974 / 590.597 s`, contended reference only
- Support-remap spatial-deposition status:
  `VALIDATED_SUPPORT_REMAP_SPATIAL_CONVERGENCE`
- Matched optical flake `dz=5 -> 2.5 nm` mapped-power difference /
  volume-weighted spatial-Q NRMSE:
  `0.0179774% / 0.488139%`
- Lateral-integrated / depth-integrated energy NRMSE:
  `0.165346% / 0.0230061%`
- Support-remap coarse/fine exact-TaIrTe4 exterior nonzero counts:
  `0 / 0`
- Coarse/fine mapping SHA-256:
  `9971edd6bc61c0028d7fad7a86958099b6bcbe698aa1eedbf6a80a0c903eb290` /
  `d6691afe8034ffca10e058b9bb008d63f449d1de351b6d7bf70e54cd1a3c8145`
- The one-cell mapped-hotspot shift is between reflection-symmetric central
  thermal cells; the spatial NRMSE, not a chosen peak cell, is the primary
  convergence gate
- Fixed-Q thermal domain/depth/mesh status:
  `VALIDATED_FIXED_Q_THERMAL_DOMAIN_DEPTH_MESH_CONVERGENCE`
- Named TaIrTe4 footprint scenarios: `4 um / 6 um`; neither is promoted as
  fabrication truth
- Native thermal grid:
  `32 um lateral / 20 um Si / 100-25-100 nm core-flake-design`
- Independent controls:
  `40 um lateral`, `30 um Si`, and
  `50-12.5-50 nm core-flake-design`
- Worst fixed-Q thermal common-field/scalar convergence metric:
  `0.314485%` (limit `1%`)
- Worst Q-mapping / energy-balance / linear-residual errors:
  `3.58746e-16 / 5.63685e-12 / 1.95372e-11`
- Native 4 um / 6 um Tmax:
  `1.069958441e-7 / 1.051871964e-7 K per 1 W/m2`
- Refined 4 um / 6 um Tmax change:
  `0.283335% / 0.314485%`
- Thermal physical law held fixed in this checkpoint:
  TaIrTe4 `diag(14.4,3.8,1.0) W/(m K)`,
  bottom `G=7.37e6`, deposited-design endpoint `G=7.37e4`,
  air `G=1`, SiO2/Si `G=1.1e9`, exposed-SiO2/air
  `h=10 W/(m2 K)`
- Lateral/bottom reported powers are numerical truncation-boundary fluxes,
  not intrinsic physical heat-path fractions
- Fixed-Q PTE thermal adjoint/FD, 81x81 mapping, combined AD-FD, latent
  AD-FD, transient, and optimization were not executed by this checkpoint
- Fixed-local-Q PTE thermal-only AD-FD status:
  `VALIDATED_FIXED_LOCAL_Q_PTE_THERMAL_ONLY_ADFD`
- Objective: uniform-45-degree PTE current surrogate; not a solved
  finite-contact terminal current
- Thermal density in this operator checkpoint:
  native `20x20` cell-centered rho at `0.5`; not yet the approved
  `81x81` nodal mapping
- The matched native optical Q is bitwise fixed in every baseline/plus/minus
  thermal solve; Maxwell and optical-Q gradients are absent
- Thermal adjoint components:
  bulk design kappa, internal TaIrTe4/design G, and exposed-surface
  half-cell-kappa contribution
- Selected FD step / worst conditioned AD-FD error:
  `0.005 / 2.06536e-6`
- Worst thermal PTE energy / forward-or-adjoint residual /
  gradient-component-sum error:
  `3.46728e-12 / 1.01751e-11 / 4.03793e-16`
- Raw thermal PTE artifacts, 4 um / 6 um SHA-256:
  `381e2c11bbf456cd9ed321c9fa83e0efccb20621e7c736f003009e1a702de77a` /
  `c3f4ac851d73c5f1159e95e40a9e1f15d1177234dbd6ed9a152ad3c071e85688`
- 81x81 mapping, combined Maxwell-thermal AD-FD, latent AD-FD, transient,
  and optimization remain unexecuted
- Nodal physical-density coupling status:
  `VALIDATED_81X81_NODAL_OPTICAL_THERMAL_MAPPING_JVP_VJP`
- Physical coordinates:
  exact `81x81` nodes on `[-1,1] um` at `25 nm`; nonperiodic and not
  81 finite-width pixels
- Optical density target:
  identity x-y nodes extruded exactly to `81x81x13` nodes over
  `z=[0,600] nm`
- Thermal density targets:
  exact piecewise-bilinear area averages on native `20x20` (100 nm) and
  refined `40x40` (50 nm) control cells
- Worst nodal JVP-FD / JVP-VJP dot / endpoint / area-integral errors:
  `6.18584e-11 / 2.93535e-16 / 5.99520e-15 / 2.01948e-16`
- Opposite-boundary leakage and optical z-extrusion error:
  `0 / 0`
- Raw nodal coupling NPZ SHA-256:
  `47f10d96683c11be0168f45b780392947db210afcef918894c33eabf4862c53f`
- Imported-permittivity endpoint status:
  `VALIDATED_IMPORTED_PERMITTIVITY_ENDPOINT_EQUIVALENCE`
- Imported object: exact `81x81x13` samples on
  `x,y=[-1,1] um`, `z=[0,600] nm`
- Scalar/imported endpoint comparison at rho `0/1` and flake
  `dz=5/2.5 nm`: worst gated relative metric `6.13849e-17`
- Imported-object readback bounds error: `0 m`
- Rho1 raw spatial-Q convergence trace:
  `1.04065%` (5->2.5 nm), `0.530395%` (2.5->1.25 nm), and
  `0.107027%` (1.25->0.625 nm)
- Promoted finest-pair mesh metric:
  `0.107027%` (limit `0.5%`)
- The two coarse raw-Q failures are preserved in the report; the endpoint
  representation equivalence is independently exact at both matched
  5 and 2.5 nm meshes
- Finest rho1 P_Q / P_six relative changes:
  `2.30908e-7 / 1.43701e-5`
- Uniform rho=0.5 representation status:
  `VALIDATED_RHO05_IMPORTED_PERMITTIVITY_EQUIVALENCE`
- Matched scalar/imported `P_Q`, `P_six`, complex-field NRMSE, and
  spatial-Q NRMSE differences:
  `0 / 0 / 0 / 0`
- Matched scalar/imported index NRMSE:
  `4.15226e-17`
- Imported rho=0.5 object:
  exact `81x81x13`, exact `x,y=[-1,1] um`, `z=[0,600] nm`,
  bounds error `0 m`
- Imported rho=0.5 `P_Q / P_six / closure`:
  `1.6887880194040323e-12 W / 1.6893345559747856e-12 W / 3.23522e-4`
- The scalar completed FSP was SHA-pinned and re-read to extract the
  missing native E/index arrays; no additional scalar electromagnetic solve
  was run
- Combined physical-rho PTE AD-FD diagnostic status:
  `DIAGNOSTIC_FAILED_COMBINED_PHYSICAL_RHO_PTE_ADFD`
- Combined 4 um / 6 um selected-step errors at `h=0.005`:
  `2.14938% / 2.88520%` (original gate `<0.5%`; failed)
- The diagnostic used
  `dI/dQ_thermal -> R_Q^T -> native-Yee vector source`; it did not reuse
  the scalar `P_Q` adjoint source
- No empirical normalization or gradient rescaling was used
- Passing diagnostic controls: Q mapping `2.38678e-16`, six-face closure
  `2.02707e-4`, thermal energy `3.16735e-12`, residual `1.02050e-11`,
  and CPU/GPU adjoint field NRMSE `4.00238e-5`
- Previously unresolved gate, now resolved: component-specific nonuniform
  `rho_81x81 -> epsilon_Yee,{x,y,z}` Jacobian and exact E/index Yee
  collocation
- Component-wise Yee material-Jacobian status:
  `VALIDATED_COMPONENT_WISE_YEE_MATERIAL_JACOBIAN`
- Production chain:
  `rho81x81 -> epsilon -> n -> importnk2(81x81x13) -> v261 conformal
  index_detail -> epsilon_Yee,{x,y,z}`
- Explicit sparse `Jx/Jy/Jz` shapes:
  `194392 x 6561` each; nonzeros `53960 / 53960 / 48749`
- v261 completed-solver versus layout `index_detail` epsilon error:
  `0`
- Maximum forward/adjoint/index component-coordinate mismatch:
  `4.23516e-22 m`
- Five-direction worst mapping-only FD / JVP-VJP transpose errors:
  `2.62931e-10 / 8.81466e-15`
- Construction used 25-color local layout perturbations, zero Maxwell
  solves, no per-pixel Maxwell solves, no empirical normalization, and no
  gradient rescaling
- The earlier separate `DESIGN_FIELD`/`DESIGN_INDEX` same-array-index
  multiplication path is removed from the promoted component-Jacobian path
- Optical-dz downstream status:
  `VALIDATED_OPTICAL_DZ_DOWNSTREAM_PTE_GRADIENT_CONVERGENCE`
- Nonuniform physical-rho `P_Q` at `dz=2.5/1.25/0.625 nm`:
  `1.692523974e-12 / 1.692526176e-12 / 1.692537364e-12 W`
- Six-face closure at `dz=2.5/1.25/0.625 nm`:
  `2.01444e-4 / 2.19647e-4 / 2.27224e-4`
- Direct `2.5 -> 0.625 nm` remapped-Q NRMSE:
  `0.302051%` for both named thermal footprints
- Direct `2.5 -> 0.625 nm` TaIrTe4 temperature-field NRMSE:
  `0.0133913% / 0.0139121%` for the 4 um / 6 um scenarios
- Direct `2.5 -> 0.625 nm` raw-PTE relative change:
  `0.0280247% / 0.0280219%`
- Direct `2.5 -> 0.625 nm` optical directional-gradient change:
  `0.00557631% / 0.0136565%`
- Direct `2.5 -> 0.625 nm` combined directional-gradient change:
  `0.00577677% / 0.0138882%`
- Production optical flake mesh fixed at `dz=2.5 nm`: this is the coarsest
  mesh whose raw PTE, optical gradient, and combined gradient are all within
  `0.5%` of the `0.625 nm` reference for both named thermal footprints
- Worst downstream component-J transpose error: `9.56908e-16`
- No empirical normalization, gradient rescaling, clipping, smoothing, gain,
  global Q rescaling, tiling, or Q-source deletion was used
- Thermal raw-PTE/localized-ADFD subgate status:
  `VALIDATED_THERMAL_RAW_PTE_AND_LOCALIZED_ADFD_SUBGATES`
- Historical 6 um native-to-50 nm refined raw-PTE change remains
  `0.629536%` and is preserved as
  `RAW_PTE_LT_0P5PCT_UNRESOLVED`; it was not rewritten
- New 6 um fixed-Q raw-PTE changes:
  `0.184374%` for 50→40 nm, `0.336745%` for 40→33.333 nm, and
  `0.152652%` for the direct 50→33.333 nm comparison
- New successive-mesh TaIrTe4 common-field NRMSE:
  `0.0422701% / 0.0480763%`
- Fixed-Q thermal-only AD-FD now includes adjoint-aligned, fixed-seed
  random, asymmetric-smooth, central-localized, and design-edge-localized
  directions at `h=0.01/0.005/0.0025`
- Worst selected five-direction thermal-only AD-FD error at `h=0.0025`:
  `1.91862e-6`
- Central-localized / design-edge-localized `h=0.0025` errors, 4 um:
  `1.90768e-6 / 8.70987e-8`; 6 um:
  `1.91862e-6 / 1.81202e-7`
- Next gate: rerun combined physical-rho PTE AD-FD using the promoted
  component-specific Yee Jacobian/collocation and production optical
  `dz=2.5 nm`
- Gray-law sensitivity, filter/projection latent AD-FD, transient, and
  optimization remain unexecuted
- Status: `VALIDATED_MIXED_CPU_TFSF_GPU_FIELDREGION_OPTICAL_ADFD`
- Protected design/PTE ROI: exactly `x,y=[-1,1] µm`
- Optical TaIrTe4: 100 nm thick and extended through lateral PML as the
  large-background model; this does not set the finite thermal flake footprint
- Inverse-designed material: actual SiO2, `2 µm × 2 µm × 600 nm`
- Optical boundaries: six PML faces; periodic boundaries forbidden
- Requested illumination: normal-incidence ideal plane wave; Gaussian and
  periodic/Bloch boundaries are not substituted
- Installed v261 GPU TFSF probe:
  `BLOCKED_GPU_TFSF_UNSUPPORTED`
- Explicit engine error:
  `GPU simulation does not support the use of TFSF sources`
- Bloch/periodic source crossing transverse PML: rejected as an invalid
  source/boundary pairing
- Official all-PML Diffracting source: executed through a `24 µm` domain and
  `20 µm` aperture; best displayed ROI intensity RMS `4.974%`,
  peak-to-peak `15.184%`, max phase `2.027°`, and `Ez/Ex=6.466%`; rejected
- GPU source-integrity status:
  `BLOCKED_GPU_ONLY_SIX_PML_IDEAL_PLANE_WAVE`
- User-authorized CPU TFSF source gate: six PML, `4×4 µm` lateral domain,
  `2.6 µm` TFSF span, exact central `2×2 µm` ROI
- CPU TFSF PML-24/PML-32 status:
  `VALIDATED_CPU_TFSF_4UM_DOMAIN_2UM_ROI_SOURCE_GATE`
- PML-24 ROI mean-|E|² error / spatial RMS / peak-to-peak:
  `0.0144312% / 0.00000856% / 0.00005593%`
- PML-24 closed-box energy error: `0.00007052%`
- PML-24 native engine / Python run / complete-session wall times:
  `3.270815 / 5.525168 / 10.508481 s`
- PML-32 native engine / Python run / complete-session wall times:
  `4.346625 / 7.462453 / 12.466042 s`
- Validated large-background geometry: FDTD outer x/y `±3.2 µm`, realized
  PML-inner x/y `±2.0 µm`, minimum TFSF-to-PML-inner gap `209.677 nm`
- Design optical endpoints: air `n=1` and actual SiO2 `n=1.38`
- Thermal model: explicit design-SiO2/TaIrTe4/bottom-SiO2/Si domains
- Exposed SiO2/air: Robin `h=10 W/(m2 K)` to `300 K`
- Exposed TaIrTe4 sidewalls: `G_air=1 W/(m2 K)`, not adiabatic
- PTE weighting field: uniform 45-degree direction,
  `grad(psi)=(xhat+yhat)/(4 µm)`
- Flat baseline `P_Q/P_six/closure`:
  `1.3567412718462558e-12 W / 1.3567343935235152e-12 W / 5.06976e-6`
- Mixed rho=0.5 `P_Q/P_six/closure`:
  `1.689091619450848e-12 W / 1.6895947794697648e-12 W / 2.97799e-4`
- GPU adjoint / centered FD (`h=0.01`) gradients:
  `7.316714058728351e-13 / 7.317295351329038e-13 W/rho`
- Direct mixed optical AD–FD relative error: `7.94409e-5`
- CPU/GPU adjoint complex-field NRMSE: `2.19978e-5`
- Local optical-to-thermal Q mapping power/transpose errors:
  `2.39121e-16 / 8.07866e-16`
- Material-support mapping status:
  `VALIDATED_LOCAL_Q_OPTICAL_THERMAL_MAPPING`
- Corrected 4 um / 6 um mapping SHA-256:
  `9971edd6bc61c0028d7fad7a86958099b6bcbe698aa1eedbf6a80a0c903eb290` /
  `73617d249cfa261dd87f1c2b94a38cdb328b6793212b698e31034470430e0ba2`
- Mapped source outside exact TaIrTe4 support:
  `0 W`, `0` nonzero cells
- Fixed-local-Q explicit thermal status:
  `VALIDATED_NAMED_LOCAL_Q_EXPLICIT_THERMAL_ADFD_SCENARIOS`
- Named thermal footprints: `4 × 4 um` and `6 × 6 um`; neither is
  promoted as the unconfirmed fabrication geometry
- Central 2 um TaIrTe4 average DeltaT, 4 um / 6 um:
  `8.27733069135e-8 / 8.06057559735e-8 K`
- TaIrTe4 Tmax, 4 um / 6 um:
  `1.07023617860e-7 / 1.05215987646e-7 K`
- Worst thermal AD-FD / energy / linear-residual errors:
  `1.30271e-4 / 3.51866e-12 / 1.02259e-11`
- Global thermal hotspots: inside TaIrTe4 at approximately
  `(0.05, 0.05, -0.0125) um` in both named scenarios
- Current thermal-source scope: validated local `Omega_Q` only; absorption
  outside the local volume for a truly extended ideal plane wave is omitted
- Remaining physical inputs before combined/full-latent PTE:
  actual finite illumination footprint and actual thermal TaIrTe4 footprint
- Terminal PTE, combined/full-latent PTE, transient, and optimization for
  this large-background plane-wave chain: not executed

### Superseded periodic certificate

- The following section records the immutable 6 µm periodic numerical
  checkpoint only. It does not validate the finite 2 µm problem.

## Inverse-design paper-reduced thermal/PTE AD–FD

- Status: `VALIDATED_PAPER_REDUCED_RHO_DEPENDENT_THERMAL_PTE_ADFD`
- Material label: `n=4 optical proxy + paper SiO2 thermal boundary`
- TaIrTe4 kappa: `diag(14.4, 3.8, 1.0) W/(m K)`
- Fixed substrate Robin G: `7.37e6 W/(m2 K)`
- Design boundary:
  `G(rho_bar)=1+rho_bar*(G_SiO2-1) W/(m2 K)`
- Thermally-grown baseline / evaporated sensitivity:
  `7.37e6 / 7.37e4 W/(m2 K)`
- Thermal-material-only AD–FD errors:
  `1.86887e-8 / 1.17052e-11`
- Combined physical-rho errors at steps `0.0025 / 0.00125`:
  `1.02384% / 0.495604%`
- Combined latent step sweep at `0.01 / 0.005 / 0.0025`:
  `9.96381% / 1.55543% / 6.87508%`
- Selected bracketed latent FD step: `0.005`
- Selected optical / thermal-material / combined directional gradients:
  `3.20854e-19 / 2.72040e-20 / 3.48058e-19`
- Energy balance / linear residual:
  `2.04340e-13 / 8.70484e-12`
- Bulk air/SiO2/Si kappa and SiO2/Si G in this reduced model:
  `omitted`
- Remaining blocker:
  `BLOCKED_PHYSICAL_WEIGHTING_POTENTIAL_OR_FINITE_FLAKE_MASK`
- Terminal current, transient, and PTE optimization: `not executed`

## v261 HEAT material and interface controls

- Branch: `agent/validate-fvm-thermal-physical-model`
- Stacked base: `agent/unblock-heat-material-interface-controls`
- Immutable numerical checkpoint:
  `437ec0644b15a4b9a6919a0151e4aa531fb1e0ab` (PR #4)
- Finite-Q source: PR #3 commit `053260d`
- PR #2 and PR #3 content: unchanged
- Status: `BLOCKED_ANISOTROPIC_K_UNSUPPORTED`
- Independent anisotropic fallback:
  `VALIDATED_DIAGONAL_KAPPA_FVM_CONTROLS`
- Independent internal-interface fallback:
  `VALIDATED_FVM_INTERNAL_INTERFACE_G_CONTROLS`
- Common-physics 3D cross-validation:
  `VALIDATED_3D_ISOTROPIC_HEAT_FVM_CROSS_VALIDATION`
- Finite optical-Q conservative mapping:
  `VALIDATED_FINITE_OPTICAL_Q_FVM_IMPORT`
- Full-device HEAT cases executed: `false`
- Finite optical-Q mapped to FVM control volumes: `true`
- Finite optical-Q used in a thermal solve: `true`
- Multi-material production FVM convergence:
  `VALIDATED_MULTIMATERIAL_FVM_PRODUCTION_CONVERGENCE`
- Physical-model scenarios:
  `VALIDATED_FVM_THERMAL_PHYSICAL_MODEL_SCENARIOS`
- Final experimental prediction promoted: `false`
- Transient/PTE/adjoint/gradient/optimization executed: `false`

### Active blockers

- `BLOCKED_ANISOTROPIC_K_UNSUPPORTED` (native v261 HEAT only)
- `BLOCKED_INTERFACE_G_UNVERIFIED` (native v261 HEAT only)
- `BLOCKED_FABRICATION_GEOMETRY_UNCONFIRMED`

### Key measurements

- Expected finite-Q SHA-256: `7a63f82842751e7623e895701bac4ce92558679ed71bf13a3d404695a150e794`
- Expected finite-Q power: `2.56071371086521e-12 W`
- Reintegrated finite-Q power: `2.56071371086521e-12 W`
- Reintegration relative error: `0`
- Q component-sum relative error: `2.24310e-16`
- `BLOCKED_Q_ARTIFACT_INCOMPATIBLE_WITH_2UM_FOOTPRINT`: release candidate
- Allowed import mismatch: `0.5%`
- DEVICE version: `7.17.4413` from the v261 installation
- DEVICE session startup/save/load/HEAT solve: `passed`
- License-probe temperature range: `[300.0, 300.0499752615] K`
- Requested tensor write/readback before save: `[14.4, 3.8, 1.0] -> [0.0]`
- Requested tensor readback after reload: `[0.0]`
- Fresh vector/row/column/3x3-diagonal encoding probe: all returned scalar
  `0.0`; no tensor encoding round trip passed
- Exhaustive native v261 probe: LSF-native 3x1/1x3/3x3 matrices all returned
  scalar `0.0`; all 11 hidden-property candidates were rejected
- v261 HT database scan: 64 entries, 59 readable scalar conductivity models,
  0 non-scalar conductivity models, 5 unimplemented quaternary-alloy models
- x/y/z effective kappa from solver: `[0.0, 0.0, 0.0] W/(m K)`
- x/y/z heat-flux relative errors: `[100%, 100%, 100%]`
- Isotropic fallback used: `false`
- Independent conservative FVM x/y/z recovered kappa:
  `[14.4000000032, 3.80000000087, 1.00000000130] W/(m K)`
- Independent FVM x/y/z heat-flux relative errors:
  `[2.20e-10, 2.30e-10, 1.30e-9]`
- Independent FVM x/y/z temperature-profile relative errors:
  `[5.01e-11, 3.25e-11, 2.76e-11]`
- Independent FVM status: `VALIDATED_DIAGONAL_KAPPA_FVM_CONTROLS`;
  this is not reported as a v261 HEAT result
- Internal finite-\(G\) candidate: v261 `temperature` BC on the shared
  `material:material` surface with `thermal impedance = 1/G`
- Finite-\(G\) property write/save/reload: `passed` for both requested values
- \(G=7.37e6\) jump: `1.13687e-13 K` versus expected `6.55977 K`
- \(G=7.37e6\) jump/flux/transmission/energy errors:
  `[100%, 86.46%, 57.54%, 28.77%]`
- \(G=1.1e9\) jump: `2.27374e-13 K` versus expected `0.0565884 K`
- \(G=1.1e9\) jump/flux/transmission/energy errors:
  `[100%, 56.18%, 119.13%, 59.57%]`
- Finite-\(G\) candidate control status:
  `FAILED_INTERFACE_G_ANALYTIC_CONTROL`
- Verified internal-\(G\) path status: `BLOCKED_INTERFACE_G_UNVERIFIED`
- Perfect-contact mesh controls (100/50/25 nm): `passed`
- Perfect-contact interface jumps:
  `[2.27374e-13, 1.13687e-13, 2.84217e-13] K`
- Perfect-contact heat-flux errors: all below `1.1e-13`

### Independent FVM internal-interface controls

- Status: `VALIDATED_FVM_INTERNAL_INTERFACE_G_CONTROLS`
- Internal face law:
  \(R''=\Delta z_1/(2k_1)+1/G+\Delta z_2/(2k_2)\)
- Conditions: \(G=7.37\times10^6\), \(G=1.1\times10^9\)
  \(\mathrm{W/(m^2K)}\), and perfect contact
- Meshes for every condition: `100/50/25 nm`
- Total cases: `9`; passed: `9`
- \(G=7.37\times10^6\) analytic/numerical interface jump:
  `3.518029903254 / [3.518029903254, 3.518029903257, 3.518029903248] K`
- \(G=1.1\times10^9\) analytic/numerical interface jump:
  `0.03623188405797 / [0.03623188405828, 0.03623188405885,
  0.03623188405572] K`
- Finite-\(G\) jump relative errors: all below `6.3e-11`
- Analytic series-resistance heat-flux relative errors: all below `5.4e-12`
- Material-1/material-2 flux mismatch: all below `9.4e-12`
- Global energy-balance relative errors: all below `2.3e-11`
- Temperature-profile relative errors: all below `1.1e-12`
- Perfect-contact extrapolated jump: roundoff (`<2.3e-12 K`)
- Perfect-contact raw adjacent-cell difference:
  `0.5 -> 0.25 -> 0.125 K`; finest/coarsest ratio `0.25`
- Solver attribution: independent conservative Cartesian Python/SciPy FVM;
  not a Lumerical HEAT result
- Subsequent 3D cross-validation and finite-Q import gates: `completed`
- Full-device thermal calculation after these prerequisite gates:
  `executed with the independent FVM path`

### 3D isotropic/perfect-contact HEAT-FVM cross-validation

- Status: `VALIDATED_3D_ISOTROPIC_HEAT_FVM_CROSS_VALIDATION`
- Lumerical: v261 DEVICE `7.17.4413`, `772706` tetrahedral elements,
  `128099` nodes
- FVM: `40 x 40 x 30 = 48000` Cartesian cells at `50 nm`
- Geometry: two scalar materials, `k=[10,2] W/(m K)`, perfect contact
- Source: asymmetric grid-aligned synthetic cuboid,
  `Q=1e15 W/m3`, prescribed power `1.92e-4 W`
- Boundary conditions: bottom `300 K`; all other external faces adiabatic
- \(T_{\max}\) difference / maximum FVM temperature rise: `0.226837%`
- Mean-\(T\) difference / maximum FVM temperature rise: `0.0440171%`
- Full 3D field NRMSE / maximum FVM temperature rise: `0.107756%`
- Full 3D field correlation: `0.999983190756`
- Source-power cross-solver difference: `0.400326%`
- Boundary-power cross-solver difference: `0.400034%`
- Lumerical/FVM energy errors: `2.90729e-6 / 1.76324e-11`
- Non-gating pointwise diagnostics: 99th percentile `0.452070%`; maximum
  source-edge point `1.05266%`
- Independent fresh-project rerun reproduced all declared metrics within
  `1e-10`
- Finite optical-Q mapped into thermal control volumes: `true`
- Finite optical-Q used in a thermal solve: `true`
- Subsequent finite-Q conservative import gate: `completed`

### Finite optical-Q conservative FVM import

- Status: `VALIDATED_FINITE_OPTICAL_Q_FVM_IMPORT`
- PR #3 artifact SHA-256:
  `7a63f82842751e7623e895701bac4ce92558679ed71bf13a3d404695a150e794`
- Shape/order: `[80,80,41]`, `x,y,z`
- Incident-intensity normalization: `1 W/m2`
- Mapping: elementwise Q copy; FVM cell widths equal the original
  trapezoidal quadrature weights
- Original/mapped Q-array SHA-256:
  `ff1484537aadfc36d90c2035280da9ad3a2e59895e9ba06a65bea30623e3715d`
- Original nested-trapezoid power:
  `2.56071371086521e-12 W`
- FVM `sum(Q*dV)` power:
  `2.56071371086521e-12 W`
- Mapping relative error: `0`
- Interpolation, clipping, smoothing, gain, rescaling, crop, tiling, and
  outside-flake deletion: all `false`
- `5772` nonzero boundary samples at `z=5.790264e-23 m` are excluded by the
  stored strict boolean mask but lie inside the explicit `1e-15 m`
  roundoff-inclusive physical mask; all were preserved unchanged
- Exact-flake production source: `[76,76,21]` cells with bounds exactly
  `[-1,1] um x [-1,1] um x [-100,0] nm`
- Exact-flake mapping: one-to-one deposition of each original
  `Q*w_x*w_y*w_z` nodal energy parcel into its physical boundary/interior
  cell
- Source-energy/mapped-cell-power SHA-256:
  `dece160abd9965047d2902e6d1bf07fad0146fc306a543a60d79b51a7fd31caf`
- Exact-flake summed power: `2.56071371086521e-12 W`; relative error `0`
- Nonzero source energy deleted: `0 W`
- Empirical gain, global rescaling, and sample averaging in exact-flake
  deposition: all `false`
- Independent import rerun reproduced SHA and power exactly
- Finite optical-Q used in thermal solve: `true`
- Subsequent anisotropic finite-G production and convergence gate:
  `completed`

### Multi-material anisotropic finite-G FVM production

- Status: `VALIDATED_MULTIMATERIAL_FVM_PRODUCTION_CONVERGENCE`
- Attribution: independent conservative Cartesian Python/SciPy FVM;
  not a Lumerical HEAT result
- Numerical-convergence checkpoint: `32 um x 32 um` lateral domain,
  `20 um` Si depth, native optical x/y source grid
- Active solid cells: `1,625,064`
- Material conductivity:
  TaIrTe4 `diag(14.4, 3.8, 1.0)`, SiO2 `1.38`, Si `145 W/(m K)`
- Interfaces:
  `G_bottom=G_top=7.37e6 W/(m2 K)`,
  `G_SiO2/Si=1.1e9 W/(m2 K)`
- Exact source power: `2.56071371086521e-12 W`
- Q mapping relative error in every case: `0`
- Reference maximum unit response:
  `3.12002156771575e-7 K/(W/m2)`
- Reference TaIrTe4 volume-average unit response:
  `2.25508130625815e-7 K/(W/m2)`
- Reference energy-balance relative error: `3.36166e-12`
- Total sensitivity cases: `22`; equation/conservation passes: `22`
- Final `16 -> 32 um` lateral-domain changes
  (`Tmax`, flake average, 3D probe NRMSE):
  `[0.00489969%, 0.00676634%, 0.00517751%]`
- Final `10 -> 20 um` Si-depth changes:
  `[0.0178338%, 0.0246859%, 0.0189037%]`
- Final native -> refined thermal-mesh changes:
  `[0.140694%, 0.0933887%, 0.0666590%]`
- \(G_{\rm bottom}\) sweep:
  `1e6, 3e6, 7.37e6, 1.5e7, 3e7, 1e8, perfect`
- \(G_{\rm top}\) sweep:
  `7.37e4, 7.37e5, 7.37e6, 7.37e7, perfect`
- SiO2/Si `1.1e9` versus perfect contact: completed
- Exposed-surface adiabatic versus `h=10 W/(m2 K)`: completed
- Refined source treatment: native optical x/y cells, piecewise-constant
  `2x` subdivision in z with exact child-power conservation
- No Q clipping, smoothing, gain, global rescaling, periodic tiling, or
  outside-flake deletion was used
- TaIrTe4 `kz=1.0 W/(m K)` remains an estimated physical input; interface-G
  results retain the full sensitivity sweep
- This checkpoint parameter set is not a unique final experimental
  prediction

### FVM thermal physical-model sensitivity

- Status: `VALIDATED_FVM_THERMAL_PHYSICAL_MODEL_SCENARIOS`
- Fabrication status: `BLOCKED_FABRICATION_GEOMETRY_UNCONFIRMED`
- \(G_{\rm top}=7.37e6\) W/(m2 K): named numerical-convergence checkpoint
  scenario
- \(G_{\rm top}=7.37e4\) W/(m2 K): named earlier evaporated-SiO2 estimate
  scenario
- Neither \(G_{\rm top}\) value is promoted as uniquely correct
- \(G_{\rm top}=7.37e4\) versus checkpoint:
  \(T_{\max}\) `+7.48897%`, flake average `-0.0800617%`,
  common flake 3D NRMSE `2.15495%`
- TaIrTe4 \(k_z=[0.5,1.0,2.0]\) W/(m K):
  numerical scenarios, not a confidence interval; \(k_x=14.4\),
  \(k_y=3.8\) unchanged
- \(k_z=0.5\): \(T_{\max}\) `+12.3111%`; \(k_z=2.0\):
  \(T_{\max}\) `-6.29652%`
- Far-x/y fixed versus adiabatic with fixed bottom:
  \(T_{\max}\) change `+0.0475768%`
- Exposed convection `h=[0,5,10,20] W/(m2 K)`: completed
- Lateral/bottom fractions are numerical truncation-boundary fluxes, not
  physical heat-path fractions
- Geometry A: suspended/overhanging disk outside the flake
- Geometry B: 100 nm SiO2 support annulus connects the disk overhang to the
  surrounding bottom oxide
- Geometry B versus A: \(T_{\max}\) `-39.7356%`, flake average `-37.0430%`,
  common flake 3D NRMSE `27.5386%`
- Geometry-B native-to-refined numerical changes:
  `[0.789170%, 0.743380%, 0.522514%]` for
  `[Tmax, flake average, common flake 3D NRMSE]`
- Physical support-geometry variation is much larger than its numerical mesh
  error
- Published promoted metadata:
  `provisional_until_sensitivity_passes=false`,
  `next_required_gate=null`
- Raw per-case JSON metadata remains unchanged for provenance
- PR #3 commit is not in PR #4 ancestry; clean reproduction requires an
  external artifact with SHA-256
  `7a63f82842751e7623e895701bac4ce92558679ed71bf13a3d404695a150e794`
- Missing or mismatched PR #3 artifacts fail closed before import or solve

### Final control artifacts

- Execution:
  `validation/photothermal_stage1/29_validate_heat_material_interface_controls.py`
- Summary:
  `validation/photothermal_stage1/30_summarize_heat_material_interface_controls.py`
- Native anisotropy probe and validated FVM controls:
  `validation/photothermal_stage1/31_resolve_anisotropic_kappa.py`
- Conservative diagonal-tensor solver:
  `validation/photothermal_stage1/anisotropic_heat_fvm.py`
- Independent FVM interface execution:
  `validation/photothermal_stage1/33_validate_fvm_internal_interface_controls.py`
- Independent FVM interface report:
  `reports/fvm_internal_interface_controls/FVM_INTERNAL_INTERFACE_G_CONTROL_REPORT.md`
- Independent FVM interface summary/cases/raw manifest:
  `reports/fvm_internal_interface_controls/`
- 3D cross-validation execution:
  `validation/photothermal_stage1/34_validate_3d_isotropic_heat_fvm_crosscheck.py`
- 3D cross-validation report:
  `reports/fvm_3d_isotropic_cross_validation/HEAT_FVM_3D_ISOTROPIC_CROSS_VALIDATION_REPORT.md`
- 3D cross-validation summary/cases/raw manifest:
  `reports/fvm_3d_isotropic_cross_validation/`
- Finite-Q import execution:
  `validation/photothermal_stage1/35_validate_finite_q_fvm_import.py`
- Finite-Q import report:
  `reports/fvm_finite_q_import/FINITE_OPTICAL_Q_FVM_IMPORT_REPORT.md`
- Finite-Q import summary/cases/raw manifest:
  `reports/fvm_finite_q_import/`
- Multi-material production execution:
  `validation/photothermal_stage1/36_run_fvm_multimaterial_thermal.py`
- Domain/depth/mesh/interface/boundary sensitivity:
  `validation/photothermal_stage1/37_run_fvm_production_sensitivity.py`
- Production report generation:
  `validation/photothermal_stage1/38_summarize_fvm_multimaterial_thermal.py`
- Multi-material production report/summary/cases/convergence/raw manifest:
  `reports/fvm_multimaterial_thermal/`
- Physical-model scenario execution:
  `validation/photothermal_stage1/39_validate_fvm_thermal_physical_model.py`
- Clean-checkout fail-closed reproduction:
  `validation/photothermal_stage1/40_reproduce_fvm_thermal_physical_model.py`
- Physical-model report generation:
  `validation/photothermal_stage1/41_summarize_fvm_thermal_physical_model.py`
- Physical-model report/summary/cases/raw manifest:
  `reports/fvm_thermal_physical_model/`
- Anisotropic-\(\kappa\) report:
  `reports/heat_material_interface_controls/HEAT_ANISOTROPIC_K_SOLVER_REPORT.md`
- Internal-\(G\) report:
  `reports/heat_material_interface_controls/HEAT_INTERNAL_INTERFACE_G_SOLVER_REPORT.md`
- Machine-readable summary/cases/raw manifest:
  `reports/heat_material_interface_controls/`

Native v261 HEAT still cannot represent the requested conductivity tensor.
The validated FVM path now resolves the anisotropic equation and finite
internal-G law independently, and its common 3D scalar-isotropic/perfect-
contact solution agrees with v261 HEAT. The finite optical-Q mapping now
preserves the PR #3 source exactly. The independent anisotropic, finite-G,
multi-material FVM solve and its domain, substrate-depth, mesh, interface-G,
and exposed-boundary sensitivity are now complete. The reported temperature
is a unit response, not a finite-power laser temperature. Physical-model
sensitivity shows that disk-support geometry and uncertain material/interface
inputs dominate the remaining interpretation; no single scenario is called a
final experimental prediction. No transient, PTE, adjoint, gradient, or
optimization is claimed at this checkpoint. No isotropic fallback or
modification of the finite optical-Q artifact was used.

## Mechanical/MAPDL route probe

- Official capability: orthotropic/full-anisotropic thermal conductivity and
  finite thermal contact conductance are supported
- Generated material path: `MP,KXX/KYY/KZZ`
- Generated interface path: `TARGE170/CONTA174`, pure thermal
  `KEYOPT(1)=2`, bonded `KEYOPT(12)=5`, and `TCC=G` at real constant 14
- Controls generated: x/y/z anisotropic kappa, `G=7.37e6`, `G=1.1e9`,
  and perfect-contact meshes at 100/50/25 nm
- Input-deck static audit:
  `PASSED_MECHANICAL_INPUT_DECK_STATIC_AUDIT`
- MAPDL executable:
  `BLOCKED_MECHANICAL_EXECUTABLE_UNAVAILABLE`
- Mechanical license feature:
  `BLOCKED_MECHANICAL_LICENSE_UNAVAILABLE`
- License server: reachable, but only Lumerical/optislang features are
  advertised; no `ansys`, `mech_1`, `mech_2`, `struct`, or `preppost`
- Actual Mechanical solver executed: `false`
- Mechanical solver validation claimed: `false`
- Execution:
  `validation/photothermal_stage1/32_validate_mechanical_thermal_controls.py`
- Reports:
  `reports/mechanical_thermal_controls/`

The Mechanical route is physically and API-capability compatible, but it
cannot be solver-validated on this host until Mechanical/MAPDL is installed
and an applicable Mechanical license feature is added.

## Finite in-flake SiO2 proxy optical Q

- Branch: `agent/validate-inflake-proxy-optical-q`
- Base: PR #3 head `053260da6fd0caec28ce155221bd18f683a0e5e7`
- Status: `VALIDATED_FINITE_INFLAKE_PROXY_OPTICAL_Q`
- PR #2–#5: unchanged
- PR #3 radius-1.5-µm artifact: not reused or cropped

Fresh v261 GPU FDTD was run for a centered radius-0.8-µm, 600-nm-high SiO2
disk completely inside the 2 µm × 2 µm × 100 nm TaIrTe4 footprint. Outside
the disk is air, with no support annulus, overhang support, or oxide pillar.
The finite Gaussian source uses a 2 µm waist, 6.8 µm aperture, 3–6 µm source
band, 4 µm analysis point, and measured central incident intensity of 1 W/m2.

The promoted x-polarized result uses a 16 µm lateral domain, 24 PML layers,
and 5 nm TaIrTe4 dz:

- `P_Q=2.0361088604691824e-12 W`
- `P_six=2.040668004695463e-12 W`
- six-face closure `0.223414304%`
- `Qx/Q=0.993324070`, `Qy/Q=0.006675930`, `Qz/Q=0`
- raw NPZ SHA-256
  `2ecdb8a8a2a01f85635914357ce05aab834576a66069cdc024a5dca49b0c71c3`

Final convergence changes are:

- domain 12→16 µm: P_Q 0.0240581%, P_six 0.0232486%, spatial L2 0.025513%;
- PML 16→24: P_Q 0.000270435%, P_six 0.00134641%, spatial L2 0.000594892%;
- flake dz 5→2.5 nm: P_Q 0.0769457%, P_six 0.0503751%, spatial L2 0.608514%.

Source-off, empty-stack x/y/45-degree, finite-flat x/y/45-degree, proxy,
six-face closure, domain, PML, mesh, finite-value, geometry, and P_Q
reintegration gates pass. Raw NPZ/FSP files are not committed. Thermal, PTE,
adjoint, gradient, and optimization were not run.
# Extended full-latent AD–FD — 10 directions — 2026-07-29

- Status: `VALIDATED_EXTENDED_FULL_LATENT_PERTURBATION_ADFD`.
- The original five-direction certificate is unchanged. Five fresh
  `h=0.005` centered-FD directions were added: uniform, x-antisymmetric,
  y-antisymmetric, diagonal-quadrupole, and radial-ring.
- Total directions per named thermal scenario: `10`.
- 4 µm: slope `1.00015826`, R2 `0.999999953`, NRMSE `0.026509%`,
  angle `0.012183 deg`, worst individual error `0.304123%`.
- 6 µm: slope `1.00013367`, R2 `0.999999943`, NRMSE `0.025968%`,
  angle `0.012754 deg`, worst individual error `0.236677%`.
- Worst new optical closure: `0.012878%`; worst Q remap error:
  `3.56e-16`; thermal energy balance: `2.84e-12`; linear residual:
  `1.02e-11`.
- No clipping, empirical normalization, gradient rescaling, or optimization.
- Separate figures:
  `inverse_design_pte_adfd/figures/21_full_latent_adfd_4um_10directions.png`
  and
  `inverse_design_pte_adfd/figures/22_full_latent_adfd_6um_10directions.png`.

# Final finite 81x81 full-latent PTE AD–FD — 2026-07-28

- Status: `VALIDATED_FULL_LATENT_COMBINED_PTE_ADFD_WITH_USER_ACCEPTED_FD_NOISE`.
- End-to-end chain: finite nonperiodic latent → 500 nm conic filter → beta=8
  projection → component Yee Maxwell Q → conservative thermal remap →
  explicit anisotropic/material/interface thermal solve → uniform-45° PTE.
- At `h=0.005`, five directions pass in both named 4 and 6 µm thermal
  scenarios. Global normalized error is `0.014397%`, gradient angle is
  `0.002225 deg`, and the largest individual directional error is
  `0.172688%`.
- The earlier physical-rho near-null strict plateau failure remains preserved.
  Its plateau gate was waived by the user; no empirical normalization or
  gradient rescaling was used.
- Coupled gray-law sensitivity is complete. The choice is materially
  important and remains explicit rather than being promoted as a measured
  material law.
- Optimization was not run.
- Report:
  `inverse_design_pte_adfd/FINAL_FULL_LATENT_PTE_ADFD_REPORT.md`.
