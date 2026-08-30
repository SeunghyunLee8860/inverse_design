# 10 um FDTDX substrate and binary-Au endpoint control

Status: **VALIDATED_FDTDX_DIAGNOSTIC_SUBSTRATE_BINARY_ENDPOINT_CLOSURE**

## Outcome

The component-specific Yee dual-volume integration error at the Si/SiO2
transition was identified and corrected.  `Ex` and `Ey` occupy z-edge dual
volumes, while `Ez` occupies the cell z width.  Applying one cell-centered
volume to all three components over-counted the 285-nm SiO2 optical support
when a 15-nm oxide cell touched a coarse Si cell.

The promoted diagnostic uses a matched z grid at the Si/SiO2 interface,
32 optical periods, a four-period late window, and direct material-loss versus
deep-box time-domain Poynting balance.  There is no background subtraction,
clipping, smoothing, gain, or result rescaling.

## Optical contract

- wavelength: 10 um; scalar Gaussian; requested `w0=8.5 um`; `E || b`
- domain: 48 x 48 x 16 um; six PML; 9,870,336 Yee cells
- TaIrTe4: 20 x 20 x 0.1 um; `epsilon_x=epsilon_b`, `epsilon_y=epsilon_a`,
  `epsilon_z=epsilon_b`
- SiO2: 285 nm, Kitamura value `epsilon=[7.3490019303043495, 1.9899687286880576]`
- Si: diagnostic lossless `n=3.4215`; installed-Lumerical Palik readback is
  still blocked and this checkpoint is therefore not a production material
  certificate
- Au: 50 nm, `epsilon=[-4642.2300000000005, 1674.64]`

## Endpoint results

| case | P_Q (W) | Au (W) | TaIrTe4 (W) | SiO2 (W) | primary closure | late-window change |
|---|---:|---:|---:|---:|---:|---:|
| empty | 1.317944033e-13 | 0.000000000e+00 | 0.000000000e+00 | 1.317944077e-13 | 0.4742% | 0.0067% |
| au0 | 4.165061181e-13 | 0.000000000e+00 | 3.377255172e-13 | 7.878059009e-14 | 0.1101% | 0.0160% |
| au1 | 1.964830392e-13 | 1.180901106e-14 | 1.456746476e-13 | 3.899937489e-14 | 0.1678% | 0.0131% |


Worst primary closure is `0.4742%`; worst late-window change
is `0.0160%`.  Both pass the 0.5% gates.  Near/deep phasor
box differences remain in the CSV as detector-convergence diagnostics rather
than being hidden or substituted for the primary time-domain balance.

The 1D planar matched-grid control independently gives time-domain closure
`0.2777%`.

## Interpretation and remaining blocker

Adding Au changes the complete electromagnetic field: it introduces Au loss
but also reduces TaIrTe4 and SiO2 loss in this binary endpoint.  Therefore Au
power must not be appended to a fixed TaIrTe4 Q map after the Maxwell solve.

This checkpoint validates the FDTDX loss/flux bookkeeping for the stated
diagnostic substrate.  It does **not** yet validate the substrate-bearing Au
density gradient, the thermal/electrical coupled PTE gradient, or an inverse
design.  Palik Si readback from the installed Lumerical material database also
remains fail-closed.
