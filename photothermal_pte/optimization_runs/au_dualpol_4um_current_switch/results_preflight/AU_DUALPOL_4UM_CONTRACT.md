# 4 um Au dual-polarization PTE inverse-design contract

Status: **AUDITED_AU_DUALPOL_4UM_PREFLIGHT_NOT_YET_SOLVED**

The fixed TaIrTe4 flake is 16 x 16 x 0.1 um and the centered floating Au
design window is 8 x 8 x 0.05 um. The design has 80 x
80 physical cells at 100 nm pitch. The optical excitation is
a centered, normally incident scalar Gaussian at 4 um with w0=4 um. At the
flake/source-aperture boundary the requested infinite-Gaussian intensity is
0.03355% of its peak.

Lumerical x is crystal b and y is crystal a. The low/high electrical terminal
boundary conditions are imposed on the fixed flake at x_min/x_max. Positive
current is left-to-right. Therefore the requested switch is

- E||a: I_a < 0 (right-to-left),
- E||b: I_b > 0 (left-to-right).

Production uses an epigraph objective: maximize t subject to -I_a >= t and
I_b >= t. This prevents one polarization from becoming large while the other
remains weak.

The patterned Au is electrically floating, not an optical model of the
measurement electrodes. It must be included in optical absorption, thermal
spreading/Au-Ta contact, and electrical shunting/weighting-field response.

The 500 nm solid and void requirements are enforced through differentiable
constraints during continuation and a separate exact thresholded morphology
audit. Final promotion requires zero exact bad cells. This preflight does not
claim that the initial uniform rho=0.5 design is manufacturable.

Au/Ta thermal and electrical contact values are explicitly named numerical
scenarios because direct TaIrTe4/Au values have not been experimentally fixed.
They require sensitivity analysis before an experimental prediction claim.

No Maxwell, thermal, electrical, adjoint, or optimization solve is claimed by
this checkpoint.
