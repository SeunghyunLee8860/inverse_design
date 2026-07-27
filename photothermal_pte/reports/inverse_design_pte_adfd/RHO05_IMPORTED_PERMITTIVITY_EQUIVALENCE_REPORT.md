# Uniform rho=0.5 imported-permittivity equivalence

Status: `VALIDATED_RHO05_IMPORTED_PERMITTIVITY_EQUIVALENCE`

This is the missing gray-state representation check. It compares the
previously certified scalar-index rho=0.5 CPU-TFSF solve with a new,
otherwise matched v261 solve using 81×81×13 imported `n` samples. There is
no bitwise-equality requirement and no phase fitting, empirical
normalization, gradient rescaling, thermal solve, or optimization.

The imported chain is
`epsilon=1+rho*(1.38^2-1)`, `n=sqrt(epsilon)`, with rho=0.5. The object
bounds are x,y=[-1,1] µm and z=[0,600] nm. The scalar and imported records
have an exact match for the selected geometry, source, mesh, PML,
simulation-time, and incident-normalization contract: `True`.

| metric | relative difference |
| --- | ---: |
| P_Q | 0.000000000e+00 |
| P_six | 0.000000000e+00 |
| complex field NRMSE | 0.000000000e+00 |
| spatial Q NRMSE | 0.000000000e+00 |
| index NRMSE | 4.152262160e-17 |
| major Q-component power | 0.000000000e+00 |

Worst metric: `4.152262160e-17`; required: `<5.000000000e-03`.
Maximum imported-object bounds error: `0.000000000e+00 m`; required:
`<2.000000000e-18 m`.

Raw FSP and NPZ artifacts remain outside Git. Their paths, sizes, and
SHA-256 values are recorded in the manifest.
