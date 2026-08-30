# Finite-187T optical-to-PTE handoff audit

Status: `BLOCKED_FINITE_PTE_GEOMETRY_UNDEFINED`

The validated optical model has a finite 11 x 17 Au inverse-T array, but its TaIrTe4 layer and lower stack extend laterally through the PML. A terminal-current calculation instead needs a finite conducting flake and two explicit contacts. Those dimensions are not defined by the current paper-architecture contract.

The existing Q was reintegrated without modification. Audit rectangles show how much of that source would be deleted by an after-the-fact crop:

| Audit rectangle | Q inside | Q outside |
|---|---:|---:|
| finite_T_array_footprint (16.5 x 17.0 um) | 88.416% | 11.584% |
| legacy_top_bottom_24x20 (24.0 x 20.0 um) | 94.632% | 5.368% |
| legacy_left_right_20x24 (20.0 x 24.0 um) | 94.667% | 5.333% |
| diagnostic_30x30 (30.0 x 30.0 um) | 99.028% | 0.972% |

These are diagnostics, not promoted device geometries. In particular, cropping to 24 x 20 um would delete about 5.37% of the existing optical power and would also miss finite-flake/contact scattering. No crop, deletion, gain, smoothing, or rescaling was performed.

The next physically valid step is to freeze the finite TaIrTe4 footprint, electrode footprints/polarity, and physical thermal stack; rerun Maxwell with those objects; then map the unmodified volumetric Q conservatively into the explicit thermal/electrical solve.
