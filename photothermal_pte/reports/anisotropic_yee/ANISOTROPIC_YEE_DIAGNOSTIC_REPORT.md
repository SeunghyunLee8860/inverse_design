# Anisotropic native-Yee Pabs audit

## Scope

- Optical FDTD only; HEAT was not run.
- No Pabs/flux gain and no geometric clipping were applied.
- Native fields and component-specific index arrays were obtained with `getdata(...,1)`.

## pabs_adv script

- Full script: `LOCAL_OUTPUT/20260724T030000Z_anisotropic_yee_diagnostics/script_audit/pabs_adv_analysis_script.lsf`
- The script does not use scalar epsilon.
- It reads `index_x^2`, `index_y^2`, and `index_z^2` separately and evaluates:
  - `Qx = 0.5*eps0*omega*abs(Ex)^2*imag(eps_x)`
  - `Qy = 0.5*eps0*omega*abs(Ey)^2*imag(eps_y)`
  - `Qz = 0.5*eps0*omega*abs(Ez)^2*imag(eps_z)`
- Each Qi is evaluated at its native Yee position before scalar interpolation.
- The sampled anisotropic material is handled through the solver-produced component arrays `index_x/y/z`; pabs_adv does not reinterpret the input material table.

Lumerical records frequency-domain fields with the Fourier transform `integral exp(+i*omega*t) E(t) dt`, corresponding to inverse time dependence `exp(-i*omega*t)`. For passive materials in this convention Im(epsilon)>0 and the positive-sign formula above gives positive loss.

## Native Yee component integrals

| geometry | A_Qx | A_Qy | A_Qz | total | local flux | mismatch |
|---|---:|---:|---:|---:|---:|---:|
| disk | 0.453957262961 | 0.054062184637 | 2.468e-12 | 0.508019447600 | 0.453709357422 | 11.970238% |
| square | 0.363579457678 | 0.108725895083 | 2.484e-12 | 0.472305352764 | 0.363376055266 | 29.977016% |

The conservative common-grid mapping preserves every native component integral to floating-point precision. Native totals reproduce the Lumerical pabs_adv totals to about 1e-10 absolute.

## Disk excess channel

- DeltaA_Qx: `0.197329303121`
- DeltaA_Qy: `0.054062184637`
- DeltaA_Qz: `2.468e-12`
- measured excess: `0.054307055515`
- integral(DeltaQ-DeltaQx): `0.054062184640`
- correlation[(DeltaQ-DeltaQx), DeltaQy]: `0.999999999999997`

## False Ez hypothesis

- scalar eps_x applied to Ez: `0.135422489872` (2.494x the actual disk mismatch)
- scalar eps_y applied to Ez: `0.024739690259` (0.456x the actual disk mismatch)
- residual vs hypothetical false-Ez spatial correlation: `0.207935`
- residual vs Delta|Ez|^2 spatial correlation: `0.182851`

The scalar-epsilon false-Ez hypothesis is therefore not numerically consistent with the measured excess. The observed extra channel is Qy.

## Periodic correction and analysis padding

- The outer periodic-correction switch is enabled.
- X-periodic and Y-periodic branches are enabled; Z is disabled.
- The monitor spans the full 6 um x 6 um periodic unit cell, as required by the script.
- In z, the analysis region is approximately -150 to +50 nm around TaIrTe4 (-100 to 0 nm), providing at least one nonabsorbing mesh cell on both sides.
- No lateral nonabsorbing padding exists because TaIrTe4 fills the periodic unit cell. This is the periodic-unit-cell case handled by the enabled x/y correction, not a finite isolated absorber.

## Six-face and isotropic controls

| case | pabs_adv | top-bottom | six-face | pabs-six | mismatch |
|---|---:|---:|---:|---:|---:|
| disk anisotropic | 0.508019447504 | 0.453709357422 | 0.453709357422 | 0.054310090082 | 11.970238% |
| square anisotropic | 0.472305352679 | 0.363376055266 | 0.363376055266 | 0.108929297413 | 29.977016% |
| disk isotropic eps_a | 0.451881767402 | 0.451586714618 | 0.451586714618 | 0.000295052785 | 0.065337% |
| square isotropic eps_a | 0.398697968480 | 0.398469786982 | 0.398469786982 | 0.000228181498 | 0.057264% |

The patterned isotropic controls close within 1%, so the large mismatch is anisotropy-specific rather than a generic finite-edge interpolation effect.
