# Meeting guide — paper-derived inverse-T with TaIrTe4 substitution

## What was actually solved

- 2024 Supplementary MIR inverse-T scenario at 4.75 um.
- Period 1500 nm x 1000 nm; normal-incidence periodic plane wave.
- Lumerical x = TaIrTe4 b, y = TaIrTe4 a, z = c with epsilon_c=epsilon_b closure.
- 100-nm TaIrTe4 / 35-nm Al2O3 / opaque Au mirror / optional 33-nm Au T.
- x/y Periodic and z PML; conformal variant 1; 10-nm x/y and 5-nm z local mesh.
- Four independent GPU forwards: T_Eb, T_Ea, bare_Eb, bare_Ea.

The T outline is digitized from Supplementary Fig. 14 axes because author CAD
vertices are not published. This is a paper-derived TaIrTe4 substitution
scenario, not a graphene-experiment reproduction.

## How to read each case folder

1. `01_structure_and_source`: top view and polarization.
2. `02--04_Qx/Qy/Qz`: native staggered-Yee component absorption. These are
   separate physical component grids.
3. `05_Qtotal`: Qx/Qy/Qz are conservatively moved to the common monitor grid
   before summation. It is not a same-index native sum.
4. `06_power_and_material_breakdown`: pabs/flux/native/common power and
   geometric material-support partition.
5. `07_setup_xy_xz_yz`: source, monitors, layers, periodic boundaries and PML.
6. `08_top_monitor_total_field_components`: total field at z=450 nm. It
   contains incident plus reflected/scattered fields and is not called the
   pure incident field.
7. `09_Qtotal_xy_xz_yz_sections`: volumetric heat-source sections.
8. `10_Q_component_depth_profiles`: where each component is absorbed in z.
9. `11_geometric_material_Q_maps`: TaIrTe4/T-envelope/mirror masks. Conformal
   interface residual is retained and never deleted or reassigned silently.

## Main numerical result

- E||b: adding the T changes total P_Q by +10.266% and geometric TaIrTe4 Q by +6.085%.
- E||a: adding the T changes total P_Q by -5.415% and geometric TaIrTe4 Q by -8.342%.
- bare total Eb/Ea = 0.96831; with T total Eb/Ea = 1.12885.
- with-T geometric TaIrTe4 Eb/Ea = 1.09449.

Thus the T creates polarization-selective active-layer absorption. This does
not yet prove a temperature-gradient or PTE-current improvement: no thermal,
electrical, adjoint, or optimization calculation is included here.

## Likely questions and precise answers

**Was only Qx plotted for E||b?**  No in this package. Qx/Qy/Qz are separate,
and Qtotal is a conservative common-grid sum. An earlier T-vs-bare diagnostic
showed only the incident-dominant component in its lower panel.

**Why is Qx dominant for E||b and Qy dominant for E||a?**  The normal incident
field is aligned with x=b or y=a, and the TaIrTe4 permittivity tensor is
diagonal in that coordinate system. The patterned T creates smaller cross and
out-of-plane components, which are retained.

**Is all absorption inside TaIrTe4?**  No. The report separates geometric
TaIrTe4, the T-envelope/interface, Au mirror and unresolved conformal-interface
power. The primary total certificate is pabs versus flux closure.

**Can Qx/Qy/Qz be added directly by array index?**  No. Their longitudinal
coordinates are staggered. The published Qtotal uses conservative deposition;
the measured power error is approximately machine precision.

**Is the 2022 Z result included?**  No Maxwell result yet. The paper supplies
L/W/P/D but not a unique arm offset/junction CAD, so it remains fail-closed.

**Can these results be called PTE enhancement?**  Not yet. They establish an
optical heat-source effect. Explicit thermal and electrical solves are the
next separate gates.

## Suggested slide order

1. `00_four_case_scalar_summary.png`: state the conclusion and the strict
   optical-only scope.
2. `T_Ea/07_setup_xy_xz_yz.png`: establish the complete E||a simulation
   contract before showing a result.
3. `T_Ea/00_case_overview.png`: show that all three loss components were
   retained and that Qtotal is conservative.
4. `comparisons/01_Ea_T_vs_bare_Qtotal.png`: explain that the T suppresses,
   rather than enhances, E||a absorption at this wavelength.
5. `T_Ea/09_Qtotal_xy_xz_yz_sections.png` and
   `T_Ea/10_Q_component_depth_profiles.png`: show the volumetric/depth
   evidence behind the scalar comparison.
6. `comparisons/03_T_Eb_vs_Ea_Qtotal.png`: close with polarization selectivity.

Do not call the top-monitor field a pure incident beam, do not call geometric
material masks an exact conformal material decomposition, and do not claim a
PTE-current enhancement from these optical figures.
