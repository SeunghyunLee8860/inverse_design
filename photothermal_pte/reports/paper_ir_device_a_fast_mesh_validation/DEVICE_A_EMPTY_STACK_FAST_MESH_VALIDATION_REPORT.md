# Device-A empty-stack fast-mesh validation

Status: `VALIDATED_DEVICE_A_EMPTY_STACK_FAST_MESH`

This is an empty Palik SiO2/Si propagation and source-normalization check.
It does **not** validate finite Device-A absorption, thermal temperature, PTE
current, or optimization.

The raw case JSON files were not changed.  For an intentionally offset beam,
the old opposite-face *ratio* is ill-conditioned when both lateral powers are
nearly zero.  The separate audit uses the maximum absolute lateral face flux
relative to incident power, with a `1e-4` gate.

## Comparison

- incident-power relative difference: `0.023728%`
- central-intensity relative difference: `0.356057%`
- normalized target-plane intensity NRMSE: `0.049718%`
- normalized target-plane intensity correlation: `0.999999878594`
- centroid displacement: `0.012716 nm`
- second-moment waist relative difference x/y: `0.056932%` / `0.031584%`

## Gates

- reference_corrected_empty_stack_acceptance: `True`
- candidate_corrected_empty_stack_acceptance: `True`
- incident_power_relative_difference_lt_0p5_percent: `True`
- central_intensity_relative_difference_lt_0p5_percent: `True`
- normalized_spatial_intensity_NRMSE_lt_0p5_percent: `True`
- both_second_moment_waists_relative_difference_lt_0p5_percent: `True`
