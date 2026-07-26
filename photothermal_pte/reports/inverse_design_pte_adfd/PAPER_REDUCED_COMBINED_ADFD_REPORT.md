# Paper-reduced rho-dependent thermal/PTE AD–FD

**Status: `VALIDATED_PAPER_REDUCED_RHO_DEPENDENT_THERMAL_PTE_ADFD`**

The immutable fixed-K 2.04% checkpoint remains a numerical control. This new
certificate adds the omitted rho-dependent thermal boundary terms while
keeping the optical endpoint explicitly labeled **n=4 optical proxy + paper
SiO2 thermal boundary**.

## Reduced thermal contract

- TaIrTe4 kappa: diag(14.4, 3.8, 1.0) W/(m K).
- Bath: 300 K; the stable solve uses theta=T-300 K.
- Fixed substrate Robin G: 7.37e6 W/(m2 K).
- Design face: `G(rho_bar)=1+rho_bar*(G_SiO2-1)` W/(m2 K).
- Thermally-grown baseline: G_SiO2=7.37e6 W/(m2 K).
- Evaporated sensitivity: G_SiO2=7.37e4 W/(m2 K).
- No bulk air/SiO2/Si kappa or SiO2/Si G is introduced.

Each face adds `g=A*G` to `K_T` and `g*T_bath` to `b_T`. The added adjoint
term is `lambda_i*A_i*(G_SiO2-G_air)*(T_bath-T_i)`.

## AD–FD results

| Space | step | AD | FD | relative error | 5% gate |
|---|---:|---:|---:|---:|---|
| physical_rho | 0.00125 | 1.590246e-20 | 1.598166e-20 | 0.495604% | True |
| physical_rho | 0.0025 | 1.590246e-20 | 1.606696e-20 | 1.023838% | True |
| latent | 0.0025 | 3.480580e-19 | 3.737539e-19 | 6.875084% | False |
| latent | 0.005 | 3.480580e-19 | 3.426442e-19 | 1.555431% | True |
| latent | 0.01 | 3.480580e-19 | 3.133782e-19 | 9.963809% | False |


The latent sweep is U-shaped: h=0.01 is dominated by nonlinearity, h=0.0025
is below the observed v261 FD numerical floor, and the bracketed stable step
h=0.005 gives 1.555431% error. Failed side-control rows
are retained rather than discarded.

At selected h=0.005, the optical-Q directional term is
3.208540e-19; the newly implemented
thermal-material term is
2.720400e-20; their combined
directional derivative is
3.480580e-19.

Energy balance is 2.043405e-13
and the linear residual is
8.704844e-12.

## Remaining scope limit

`BLOCKED_PHYSICAL_WEIGHTING_POTENTIAL_OR_FINITE_FLAKE_MASK` remains. The
reported objective is still the finite-local-mask A m surrogate, not terminal
current in A. No PTE optimization, transient solve, or final device prediction
is claimed.
