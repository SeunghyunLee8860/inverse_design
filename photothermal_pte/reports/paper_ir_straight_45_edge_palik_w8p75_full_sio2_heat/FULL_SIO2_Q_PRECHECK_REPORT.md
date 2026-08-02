# Full-SiO2 Maxwell heat-source precheck

Status: `BLOCKED_FULL_SIO2_Q_SIX_FACE_CLOSURE`

The requested material-resolved path is implemented.  The GPU Maxwell
artifact now preserves separate, disjoint volumetric arrays for the complete
130-nm TaIrTe4 layer and complete 285-nm SiO2 layer.  The expanded thermal FVM
maps each array independently into cells of the same material.  It does not
project SiO2 absorption into TaIrTe4 and it applies no clipping, smoothing,
gain, global rescaling, or tiling.

## Thermal-material contract

- TaIrTe4: `k_lab=(3.8,14.4,1.0) W/(m K)` for lab `x=b, y=a, z=c`.
- SiO2: `k=1.38 W/(m K)`.
- Si: `k=145 W/(m K)`.
- Air: `k=0.026 W/(m K)`.

The SiO2 and Si values are named approximately-300-K bulk-reference scenario
assumptions in the expanded FVM.  They are not supplied by the TaIrTe4 paper,
not measured for this fabricated oxide/Si wafer, and do not close thin-film,
process, doping, or temperature sensitivity.

## GPU precheck

| oxide dz | P_Q full (W) | P_Q TaIrTe4 (W) | P_Q SiO2 (W) | P_Q Ta+SiO2 (W) | closure | auto-shutoff |
|---:|---:|---:|---:|---:|---:|---:|
| 10 nm | 1.261187935992e-11 | 1.174108848212e-11 | 5.168273162834e-13 | 1.225791579840e-11 | 1.249170% | 9.81673e-6 |
| 5 nm | 1.261110455146e-11 | 1.174043658404e-11 | 5.235329082206e-13 | 1.226396949226e-11 | 1.249223% | 9.68584e-6 |

The 10-to-5-nm changes are:

- full control-volume power: `0.006144%`;
- TaIrTe4 power: `0.005553%`;
- SiO2 power: `1.280835%`;
- combined TaIrTe4+SiO2 power: `0.049362%`;
- normalized depth-integrated SiO2-Q NRMSE: `0.324782%`;
- depth-integrated SiO2-Q spatial correlation: `0.9999953552`.

Lumerical's analysis-group `Pabs_total`, rescaled only by the measured incident
reference, is `1.261111530072e-11 W`.  It agrees with the independently
reconstructed `P_Q=1.261110455146e-11 W`; therefore the blocker is not the
Python component integration.  The remaining difference is between volume
absorption and the subtraction of the large top/bottom fluxes.

## Existing-artifact flux audit

No new solve was used for this audit.  Common-grid trapezoid, common-grid
bounded, native-Yee trapezoid, native-Yee bounded, and Lumerical `Pabs_total`
all give closure in the narrow range `1.249138%--1.249223%`.  Native bounded
and Lumerical `Pabs_total` differ by only `8.29e-9` relative.

The six-face balance is cancellation sensitive:

- `sum(abs(P_face))/abs(P_six) = 7.885904`;
- z-min outward power is `+4.396774e-11 W`;
- z-max outward power is `-5.673933e-11 W`;
- all four lateral faces contribute only `7.444e-5*abs(P_six)` in absolute
  aggregate.

The audit also found that the Pabs/Q bounds and realized six-face bounds are
offset by `50 nm` in x/y.  This is too small to explain the closure because
the lateral flux is negligible, but it invalidates the previous phrase
“matched-volume.”  Future runs now fail closed on that readback mismatch.

## Fail-closed decision

The required closure gate is `<0.5%`; both meshes give about `1.249%`, and the
realized-volume match gate is also false.  Refining the oxide from 10 to 5 nm
does not improve closure, so no 2.5-nm run is justified.  The `E||b` full-oxide
run and physical thermal solves were not started.  The failed artifacts remain
useful diagnostics, but are not production heat sources.  Running a clearly
labelled diagnostic thermal sensitivity with them would require an explicit
decision to bypass the optical gates.
