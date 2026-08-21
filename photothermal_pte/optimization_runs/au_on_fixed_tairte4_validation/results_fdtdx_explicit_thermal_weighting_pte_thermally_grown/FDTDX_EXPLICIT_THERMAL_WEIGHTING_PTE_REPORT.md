# FDTDX explicit thermal and Au-aware weighting/PTE forward

Status: **VALIDATED_FDTDX_EXPLICIT_THERMAL_AU_AWARE_WEIGHTING_PTE_FORWARD**

Scenario: **thermally_grown** TaIrTe4/SiO2 contact with
`G=7.370000e+06 W/(m2 K)`.

The validated spatial Au+TaIrTe4+SiO2 Maxwell source is conservatively placed
in the explicit 3-D Au/TaIrTe4/SiO2/Si FVM. The literal source power is
`2.477953932988e-13 W`; no experimental-power scaling is applied.
The GPU solve gives `Tmax=5.577354020816e-10 K`, residual
`8.261e-10`, and energy-balance error
`0.000000%`.

The thickness-averaged TaIrTe4 temperature is then passed to the already
validated two-layer electrical operator. The Au topology changes lateral Au
conductance, finite Au/TaIrTe4 contact, and therefore the weighting potential.
Au thermopower is zero in this control. The resulting literal-normalization
PTE current is `2.684875438916e-18 A`; electrical residual is
`7.803e-12`.

`G_Au/TaIrTe4=1.724138e+07 W/(m2 K)` is an Au/MoS2 analogue and the
electrical contact is a numerical scenario. Neither is promoted as measured
TaIrTe4 data. The gray Au/air layer uses an area-fraction thermal/electrical
relaxation and is not claimed to be a fabricated effective medium.

This validates a forward chain only. It is not yet a combined Maxwell+
thermal+electrical gradient certificate and does not authorize optimization.
