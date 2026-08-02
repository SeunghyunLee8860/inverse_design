# Device-A Fig.3 sanity check - staged re-verification protocol

Motivation: the incremental campaign accumulated defects that were each
caught only by after-the-fact cross-checks (thermal frame
misregistration, origin-referenced "central intensity" naming, gate
relaxations applied ad hoc, a sub-diffraction waist request, and a
comparator misaligned with the paper's map-extremum definition).  This
protocol rebuilds every conclusion from frozen inputs through
pre-registered numeric gates, executed strictly in order; a stage may
not start before the previous stage's gates all pass and its table is
committed.  One consolidated report (V5) supersedes all earlier
incremental result claims, including the preliminary extremum ratio
0.86, which is VOID (it was computed from misregistered thermal runs).

Execution notes: no new GPU FDTD is required; the thermal stage is the
repo's own scipy FVM and is executed as 12 parallel CPU processes
(128-core host, ~2 GB/case), wall ~15-30 min.

## V0 - input freeze (no gates, manifest only)

* The 14 span-40 scan optical artifacts
  `scan40_w6p83_palik_{empty,finite}_{a,b}[_{s}]_gpu{4,3}_20260801`
  and the 4 span-50 scenario artifacts (auxiliary cross-check), each
  pinned by the SHA-256 of `case_result.json` and of
  `finite_q_on_artifact.npz`.
* Geometry contract JSON SHA-256; runner code state = production repo
  branch `sio2-lossy-scenario` (thermal frame fix included).
* Claims to re-derive are exactly the tables of V1-V5; no number from
  earlier chat/report text may be quoted without re-derivation.

## V1 - optical stage (frozen artifacts, read-only)

| gate | criterion |
|---|---|
| G1.1 status | all 14 case_result present, status COMPLETED |
| G1.2 closure | per-case relative AND incident-referenced closure recorded; absolute (incident-referenced) < 1% every case; acceptance rule (strict 0.5% / rel<2% / abs<1%) labelled per case |
| G1.3 shutoff | final auto-shutoff <= 1e-5 every case (no waiver used) |
| G1.4 source invariance | `measured_source_power_native_W` spread across the 6 positions < 0.01% per polarization |
| G1.5 beam realization | empty-reference profile: total vs ideal-Gaussian excess < 2%; power beyond r=2.5 w0 < 1%; realized 1/e2 radii reported (x/y) |
| G1.6 normalization identity | P_inc == P_src x scale (exact); C_thermal == 285 uW / P_inc (exact); stored-Q path == sourcepower path, relative difference < 1e-9 |
| G1.7 cross-span physics | span-50 vs span-40 s=3 physical absorbed power agree < 1e-4 relative |

## V2 - optics-to-thermal mapping

Primary boundary treatment is the **material-overlap remap**
(`--q-remap material-overlap`): each optical cell's absorbed power
p_m = Q_m V_m is split over thermal flake-support cells i in
proportion to the geometric overlap |Omega_m ^ Omega_i ^ F| /
|Omega_m ^ F| (power conserved analytically; interface-cell power is
divided by the actual absorbing-material volume rather than moved to
one nearest cell).  F is the thermal grid's flake support; the
conformal-effective-epsilon caveat (coverage is a volume-fraction
approximation of the solver's conformal average) is stated in V5.
The legacy nearest-support projection is retained as the sensitivity
cross-check.

| gate | criterion |
|---|---|
| G2.1 frame | thermal runner frame span == optical artifact span (read fail-closed from case_result); recorded per case |
| G2.2 registration | diagnostic mapped-power-outside-flake-before-support < 5% every case (frame-consistent baseline is ~3%); with material-overlap the final outside-support power is 0 by construction |
| G2.3 conservation | mapping relative power error < 1e-9 (analytic conservation); zero-overlap residual fraction < 0.5% and recorded |
| G2.4 metal routing | metal-excluded power recorded; isolated vs perfect degeneracy < 1% at one position (spot check) |
| G2.5 method sensitivity | per-case current difference between material-overlap and nearest-support remaps reported in V5 (pilot case first; full 12x2 if the pilot difference exceeds 1%) |

## V3 - thermal solve

| gate | criterion |
|---|---|
| G3.1 | energy-balance relative error < 1% every case |
| G3.2 | linear-solve relative residual < 1e-8 every case |
| G3.3 | identical thermal grid across all 12 cases |

## V4 - PTE / weighting stage

| gate | criterion |
|---|---|
| G4.1 | stored Laplace psi == re-solved Laplace psi, max abs diff < 1e-12 |
| G4.2 | sigma-weighted solver identity: sigma=(1,1) reproduces Laplace < 1e-10 |
| G4.3 | Laplace-weighted current recomputed offline == runner-stored current, < 1e-9 relative, every case |
| G4.4 | sheet/volume integral equivalence < 1e-12 (runner-recorded) |
| G4.5 | dual weighting: currents under Laplace AND sigma-weighted psi for ALL 12 cases; extremum position re-located independently per weighting and polarization |

## V5 - comparator and consolidated report

* I(s) tables and profiles for both weightings; per-polarization,
  per-weighting edge-lobe extrema; extremum ratio and pointwise ratios
  vs the digitized paper value 0.8366 +/- 0.0085.
* Laplace psi is the primary comparator (paper Eq. S7); the
  sigma-weighted result is reported alongside as model sensitivity.
* Mandatory caveat list: simulated realized beam (7.3/7.0 um 1/e2)
  is smaller than the experiment's NA-0.4 diffraction bound
  (w0 >= 8.75 um at 11 um); SiO2 self-heating channel absent (Palik
  SiO2 absorbs optically but is not a thermal source); electrode
  comparator (paper S4.B, Ia>Ib at contacts) outside the scan's
  aperture-clearance reach; eps_c = eps_b closure; single empty
  reference per polarization justified by G1.4/G1.5; full gate-
  relaxation table reproduced verbatim.
* The V5 report supersedes: the preliminary 0.86 claim, all
  misregistered scan thermal numbers, and the fixed-position ratios
  (1.618/1.20) as *comparator* statements (they remain valid as
  fixed-position diagnostics of the earlier model variants).

## Execution order

1. V0 manifest -> commit.
2. V1 script + table -> commit (gates pass required).
3. Restart the 12 thermal cases in parallel (fixed-frame runner) ->
   V2+V3 tables -> commit.
4. V4 dual-weighting script -> commit.
5. V5 consolidated report + figures -> commit; mark earlier reports
   superseded.
