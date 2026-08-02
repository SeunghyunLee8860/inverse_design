# Device A LWIR |I_a|/|I_b|: the discrepancy is an optical-model difference (2026-08-02)

## Summary

The long-standing mismatch — our |I_a|/|I_b| ≈ 1.19 at the off-axis edge versus the
0.8366 digitized from Blevins et al. Fig. 3J — is **not a bug in this pipeline**.
Feeding the *paper's own optical model* into our unchanged thermal / weighting-potential /
Shockley-Ramo code reproduces the paper-side answer (**0.683**). The entire gap is
created in the optics: a full-wave calculation resolves an E∥a absorption hotspot in
the first ~1 µm inside the flake edge, which the paper's model cannot express by
construction.

## What the paper actually models (AFM Methods 2.4; SI Eq. S1-S2; thesis Eq. 4.3-4.4, §5.4.4)

> "we combined the **transfer-matrix method** to solve for the **total optical absorption**
> into TaIrTe4 for E∥a vs. E∥b … We then solved the **2D heat equation** at the 45° edge,
> treating the laser as a **Gaussian heat source**, the crystal edge as an insulating boundary"

Polarization enters *only* as a scalar (the TMM absorption). The in-plane heat-source
shape is the same Gaussian for both polarizations. Fig. 3G plots "max ∇T_x at the edge";
Fig. 3J compares the **measured** current ratio against that **∇T** prediction — it is not
a photocurrent calculation.

## Two-model comparison (identical geometry, thermal model, ψ, and Ramo integral)

| d (µm) | I_a PM | I_b PM | **r PM** | I_a FW | I_b FW | **r FW** |
|---:|---:|---:|---:|---:|---:|---:|
| −2 | 11.23 | 16.44 | 0.6829 | 18.73 | 15.41 | 1.2148 |
| −1 | 13.02 | 19.06 | 0.6829 | 21.12 | 17.59 | 1.2006 |
|  0 | 14.21 | 20.81 | 0.6828 | 22.60 | 19.00 | 1.1897 |
| +1 | 14.12 | 20.68 | 0.6828 | 22.29 | 18.79 | 1.1866 |
| +2 | 12.60 | 18.45 | 0.6829 | 20.10 | 16.72 | 1.2023 |
| +3 |  9.82 | 14.38 | 0.6829 | 16.30 | 13.18 | 1.2375 |
| +5 |  2.67 |  3.90 | 0.6843 |  6.72 |  3.76 | 1.7874 |

PM = paper-replication optics (TMM total × Gaussian × Beer-Lambert, this work).
FW = full-wave FDTD optics. Currents in nA at 285 µW incident.

**The paper-model ratio is position-independent to 4 digits** (spread < 0.002) and sits on
the TMM absorption ratio 0.671 — exactly as the construction forces. The full-wave ratio
varies with beam position because the edge hotspot is sampled differently.

Note this also bears on the paper: its own Fig. 3I shows the E∥a and E∥b profiles with
*different shapes*, which a position-independent ratio cannot produce.

## The physical mechanism (smooth 45° edge, no staircase, no electrodes)

Absorbed power vs. distance inside the edge:

| n from edge (µm) | 0.25 | 0.50 | 0.75 | 1.5 | 2.5 | total |
|---:|---:|---:|---:|---:|---:|---:|
| Q_a / Q_b | 1.18 | **1.28** | 1.22 | 0.73 | 0.57 | **0.801** |

E∥a absorption is concentrated in a ~1 µm layer at the edge; the decay length matches
λ/|n_a| = 0.76 µm. This is a metallic-termination field enhancement, dominated by the
in-plane E_a component (Q_z is only 0.4–0.9% there, so it is *not* an ε_c artifact).

## Waist sensitivity (paper's own smooth 45° edge, w₀ 8.75 → 11.58 µm, +32%)

| metric (a/b) | w₀ = 8.75 | w₀ = 11.58 | change |
|---|---:|---:|---:|
| max ∇T along a (**paper Fig. 3G comparator**) | 1.5435 | **1.7629** | +14.2% |
| max ∇T edge-normal | 1.8770 | 2.4011 | +27.9% |
| p99 ∇T edge-normal | 1.9012 | 2.4358 | +28.1% |
| absorbed power | 0.8012 | 0.7799 | −2.7% |
| area-average ΔT | 0.8010 | 0.7803 | −2.6% |

A larger beam makes the effect **stronger**, not weaker: ∇T is dominated by the sharpest
feature, and a wider beam flattens the bulk profile while the 1 µm edge feature is
unchanged. This eliminates the "beam-size ambiguity explains the discrepancy" branch —
the paper quotes only a spot size range (λ/2NA = 9–16 µm over 7–13 µm), and under either
reading (w₀ = λ/(π·NA) = 8.75 µm, or w₀ = 11.7 µm from FWHM = λ/2NA) the mechanism survives.

## Hypotheses eliminated with data

| hypothesis | verdict |
|---|---|
| SiO₂ self-heating channel missing | absorbs 0.35% (a) / 0.49% (b) of incident — cannot move the ratio |
| ε_c = ε_b closure drives the hotspot | Q_z is 0.38–0.95% at the hotspot; it is 70–88% Q_a (in-plane) |
| digitized staircase edge geometry | the **smooth** 45° control shows the same effect |
| Au/Ti plasmonic contamination (optics-only, no metal heat sink) | excluding a 10 µm halo around the contacts moves r 1.190 → 1.113 (**7%**) — real, not causal |
| mesh non-convergence | previously certified converged at 25 nm |

## Parameters verified against the papers (all correct)

ε axis mapping (lab x = b dielectric, lab y = a metallic; matches Fig. 3A axes a↑ b→),
ε values (solver fit error 5e-6…2e-5), λ = 11 µm, SiO₂ = 285 nm (Fig. 3B), flake = 130 nm,
285 µW, κ = (14.4, 3.8, 1.0) mapped to (a = lab y, b = lab x, c = z), S = (−6, +27) µV/K,
σ = (4.91e5, 1.1e5) S/m, G = 7.37e6 / 1 W m⁻²K⁻¹, ψ from the isotropic Laplace equation
(SI Eq. S7), J_loc = −σS∇T into the Ramo integral (SI Eq. S6).

Current sign: all 14 full-wave currents share one sign under our fixed ψ polarity
(top contact ψ = 1). The paper's negative edge current corresponds to the opposite
electrode/amplifier convention; the convention-invariant check (a and b same sign at the
same edge) passes.

## Discrepancies internal to the paper (absolute-magnitude comparison is unreliable)

* Fig. 3H colorbar is in **pA** (−100…+200) while Fig. 3I, the profile taken along the
  dashed line in 3H, is in **nA** (to ≈ −140) — a 1000× inconsistency.
* Fig. 3D shows Abs(E∥b) ≈ 70%, but TMM with the stated ε and stack gives ≈ 26%.
  Fig. 3D's own absorption ratio (≈0.28) also disagrees with Fig. 3G's ∇T ratio (0.6–0.8),
  although in their model ∇T ∝ absorbed power.
* SiO₂ thickness: 300 nm (Methods) vs 285 nm (Fig. 3B).
* Time response: 32 ± 1.6 / 31 ± 5.0 µs (Fig. 2K,L) vs 30 ± 2.5 / 26 ± 3.3 µs (Fig. S5).
* Experimental systematic flagged by the author (thesis §4.6.3): polarization was switched
  by **rotating the sample**, and QCL beams are elliptical — the exact combination the
  thesis warns produces spurious 180°-period trends in spatially sensitive PTE devices.

## Provenance

* Full-wave Device-A scan and paper-replication scan: branch `sio2-lossy-scenario`
  (`edgetrue_*_20260802`, `papermodel_*_w8.75`).
* Straight-45 controls and waist sensitivity: branch `agent/validate-inverse-design-pte-adfd`
  (its runner has the `waist-sensitivity` execution contract, which `sio2-lossy-scenario`
  does not). Both waists were run on that same branch, so that comparison is internally
  consistent.
* The paper-model optics generator (`make_paper_model_optics.py`) writes a drop-in optical
  case dir, so the thermal/ψ/Ramo code path is byte-identical between the two tracks.
  It masks Q to the flake polygon (without the mask the power-conserving remap injects the
  off-flake beam tail into the flake: Tmax 11 K instead of 0.37 K) and renormalizes the
  discrete Beer-Lambert depth profile (the flake spans only ~14 z-cells and 1/β differs 4×
  between polarizations, biasing the ratio by +9.5% for a if left raw).

## Open / not done

* The paper's Fig. 3I profile is a **vertical** line cut (along a) through the off-axis
  edge; our scan is along the edge normal (36.4° from x). A registered vertical-cut scan
  was not run here (a parallel Codex session was working on exactly that).
* A full-wave Device-A scan at w₀ = 11.58 µm is not possible in the frozen 60 µm domain:
  the ≥0.999 aperture-capture gate demands a 50 µm source span, which no longer clears the
  PML for the off-centre scan positions. The waist test was therefore done on the
  straight-45 geometry, where the beam sits at the domain centre.
