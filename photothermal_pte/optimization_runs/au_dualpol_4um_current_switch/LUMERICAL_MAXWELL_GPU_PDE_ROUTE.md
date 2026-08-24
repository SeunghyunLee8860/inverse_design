# Lumerical density-topology Maxwell + custom GPU-PDE route

Status: `SOLVER_FREE_NK_LAW_IMPLEMENTED_BLOCKED_PENDING_B200_GATES`

## Selected architecture

- Maxwell forward/adjoint and native-Yee absorption: Ansys Lumerical FDTD on
  NVIDIA B200;
- steady heat equation: repository custom CUDA finite-volume solver;
- weighting potential, PTE current, and electrical adjoint: repository custom
  CUDA solver.

No Lumerical HEAT or CHARGE license is assumed. FDTDX/JAX is not an allowed
production Maxwell substitute.

## The design variable

This route uses density topology, not shape/level-set optimization:

```text
latent rho
  -> 500-nm spatial filter
  -> tanh projection with beta continuation
  -> projected topology occupancy rho_bar in [0,1]
  -> documented optical, thermal, and electrical constitutive maps
  -> final thresholded 0/1 mask
  -> independent ordinary dispersive-Au Lumerical reevaluation
```

`rho_bar` is not an electron or hole density and is not claimed to be a
fabricated gray Au alloy. It is the differentiable relaxation of the binary
topology problem. The identical `rho_bar` array, shape, and SHA-256 must reach
all three constitutive maps, but the maps need not use the same exponent or
formula because permittivity, thermal conductivity, and electrical
conductivity are different physical quantities.

## Optical material law: no rho cubed

At the 4-um optimization frequency, use the nonlinear metal/dielectric
interpolation

```text
n(rho_bar) = n_bg + rho_bar (n_Au - n_bg)
k(rho_bar) = k_bg + rho_bar (k_Au - k_bg)
epsilon(rho_bar) = [n(rho_bar) + i k(rho_bar)]^2
```

with the passive `n+i k` convention. The frozen Ordal endpoint is
`n_Au+i k_Au = 2.2+28.9i`, hence
`epsilon_Au = -830.37+127.16i` at 4 um. The implementation and analytic
complex derivative are in `au_density_relaxation.py`.

This is the physically motivated nonlinear interpolation proposed for
metallic topology optimization by Christiansen et al.
([DOI 10.1016/j.cma.2018.08.034](https://doi.org/10.1016/j.cma.2018.08.034))
and used in the plasmonic FDTD inverse-design framework of Zeng et al.
([DOI 10.1021/acsphotonics.1c00260](https://doi.org/10.1021/acsphotonics.1c00260)).
It avoids assigning an unsupported cubic law to Au oscillator strength.

`rho**3` is not used. Binarization is produced by the filter/projection
continuation and final discrete audit, not by pretending that a physical Au
property scales cubically.

## Lumerical carrier and its limitation

The first Lumerical-compatible implementation candidate is a complex
`importnk2` layer generated from the equation above. The repository already
has a validated precedent for this implementation pattern in
`legacy_v261_optical_support`: a nonuniform complex density was mapped through
the actual Lumerical component-Yee mesh, a sparse material Jacobian was built,
and a full latent/filter/projection AD-FD gate passed for the earlier TaIrTe4
optimization.

That precedent proves the software pattern, not the Au physics. Au adds a
large negative real permittivity and a possible intermediate-density
zero-crossing resonance. Therefore the 4-um Au carrier remains blocked until
the Au-specific gates below pass.

`importnk2` supplies a spatial complex index for the single-frequency
relaxation; it is not being called an exact broadband Au material. A final
binary candidate must use an ordinary sampled-data dispersive Au material.
The imported endpoint must first agree with that ordinary material for the
actual source spectrum. If the source bandwidth is too broad for this
single-frequency relaxation, production remains blocked until a
GPU-supported causal spatial-dispersion carrier is demonstrated. A custom
Flexible Material Plugin is not a B200 solution because Lumerical GPU does
not support that plugin framework.

The endpoint/final control builder `lumerical_4um_exact_au.py` is retained for
this distinction. It samples Ordal Au, anisotropic TaIrTe4, and Kitamura SiO2
over a 3.2--4.8 um guard band around the 3.6--4.4 um source pulse, hashes the
complete physical 0/1 geometry, and maps it to non-overlapping ordinary-Au
prisms. Its sampled inputs still require actual Lumerical MCM fit readback.
`lumerical_4um_mesh_contract.py` defines the sequential source/time/z/x-y/PML
and domain-clearance controls for the exact endpoint/final cases. These files
do not replace the density carrier or its uniform-rho resonance/AD-FD gates.

The earlier `np density` proposal remains rejected. It is a semiconductor
carrier-density attribute, not topology occupancy, and it is unnecessary for
this route.

## Maxwell derivative

Do not use the bundled LumOpt metal gradient without an independent gate. The
installed legacy and LumOpt2 topology implementations discard information
needed for lossy negative-real Au in parts of their material derivative path.

Use the repository's explicit discrete construction instead:

1. update `rho_bar -> n+i k -> importnk2`;
2. read the realized component-Yee permittivity on the frozen Lumerical mesh;
3. build `J_c = d epsilon_Yee,c / d rho_bar` by colored centered material-map
   finite differences without one Maxwell solve per pixel;
4. verify every JVP/VJP transpose identity;
5. contract `J_c^T` with the Lumerical forward/adjoint field product;
6. add direct-loss, thermal-material, and electrical-material derivatives;
7. pull the result through projection and filter transposes;
8. compare the complete latent directional derivative with independently
   rebuilt central finite differences for both polarizations and several
   steps/directions.

No empirical gradient scaling is allowed.

## Required Au gates on the B200

1. Empty layer, uniform `rho_bar=0`, uniform `rho_bar=1`, and ordinary
   sampled-data Au controls must pass material readback, time stationarity,
   native-Yee Q, and six-face flux closure.
2. Imported `rho_bar=1` and ordinary dispersive Au must agree over the actual
   source/monitor bandwidth, not only at one tabulated wavelength.
3. Uniform `rho_bar` from 0 to 1 must be swept to detect artificial field/Q
   peaks and optimizer-favored gray resonances. Passivity of the algebraic
   material law alone is insufficient.
4. The nonuniform density-to-component-Yee map must pass multi-direction
   centered FD and transpose tests on the exact frozen mesh.
5. Optical and complete Maxwell/thermal/electrical latent AD-FD must pass for
   `Ea` and `Eb` before LD_MMA is enabled.
6. Full x/y/z/PML mesh convergence, source recalibration, and time/Q closure
   must pass on the same route.
7. The final 500-nm solid/void mask must be independently rebuilt with
   ordinary dispersive Au and reevaluated for `Ia>0`, `Ib<0`.

The current host is not a B200, so this checkout can implement and test the
solver-free constitutive law but cannot issue any of these Maxwell
certificates.

## Thermal and electrical maps

The historical O3/TE1 defect was that optical used `rho**3` while thermal and
electrical used `rho`. The correction is not to force every property to share
one arbitrary exponent. The correction is:

- share exactly one `rho_bar` and its hash;
- give each physical coefficient an explicit endpoint-correct law;
- differentiate every law through the same `rho_bar`;
- pass fixed-Q and combined AD-FD;
- verify the final exact-binary endpoint independently.

The present shared-linear thermal/electrical maps remain provisional until
their mixture/bound and void-floor sensitivity studies are complete. They are
not promoted merely because the optical rho-cubed law was removed.
