# 4 µm dual-polarization multiphysics rho=0.5 forward

Status: **HISTORICAL_O3_TE1_FORWARD_REVALIDATION_REQUIRED**

This result used historical O3 optical Q with TE1 downstream operators. It is
not a certificate for the corrected shared-linear production path.

The same source-only incident-power calibration is applied to both polarizations.
This historical result predates the corrected Shockley-Ramo sign and is stale;
its signed currents must not be interpreted or relabeled. In the current code,
positive current is internal conventional current along solver +x, from x_min
to x_max.
The switching target is Ia>0 and Ib<0; the uniform design already has the requested signs, but is not optimized.

| polarization | P_Q (W) | Tmax (K) | current (nA) | runtime (s) |
|---|---:|---:|---:|---:|
| Ea | 2.47959910e-05 | 1.92326179e-01 | 0.31329048 | 3.98 |
| Eb | 5.96851455e-05 | 4.67888778e-01 | -0.18766851 | 1.74 |
