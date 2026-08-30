# Production design-window selection

Status: `VALIDATED_PRODUCTION_DESIGN_WINDOW_SELECTION`

The window is selected from the absolute L1 mass of the already validated
combined physical-density gradient. No Maxwell, thermal, adjoint, FD, or
optimization solve was run in this checkpoint.

| original reviewed candidate | area (µm²) | retained | passes 90% |
|---|---:|---:|---:|
| a_positive_strip_12x6 | 72.00 | 26.649% | False |
| a_negative_strip_12x6 | 72.00 | 20.241% | False |
| b_positive_strip_6x12 | 72.00 | 25.111% | False |
| b_negative_strip_6x12 | 72.00 | 26.879% | False |
| centered_control_10x10 | 100.00 | 42.008% | False |

Every original 12×6 µm strip and the centered 10×10 µm control fails by a
large margin. They were not silently promoted.

The promoted centered window is
`x,y=[-9.3,9.3] µm`, or 18.6×18.6 µm. It retains
`90.887297%` of the full-canvas
absolute combined gradient. The immediately smaller 18.4×18.4 µm control
retains only
`89.465223%`
and fails the 90% gate. At 50 nm production spacing the promoted nodal design
has shape `373×373`.

The selected area is `86.49%` of the
20×20 µm coarse canvas. This modest reduction is a physical consequence of
the broad 8.5 µm-waist illumination; a much smaller window would discard most
of the available sensitivity.
