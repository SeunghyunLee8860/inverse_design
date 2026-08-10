# Run012 validation checkpoint

The selected geometry has a finite 24 x 24 x 0.1 um TaIrTe4 support, fixed
top/bottom 24 x 2 um TaIrTe4 contact strips, and a 24 x 20 um intervening
design region.  There is no fixed left/right frame, symmetry, or material-use
constraint.  Lumerical coordinates remain x=b and y=a.

- Geometry/runsetup: passed; design mesh dx,dy <= 100 nm and flake dz <= 10 nm.
- Uniform rho=0.5 E||a forward: closure `3.84247e-5`, auto-shutoff `9.35837e-8`.
- Component Yee Jacobian: worst mapping FD `2.48730e-9`; worst transpose error `2.20855e-15`.
- Electrical weighting AD-FD: finest-step error `7.04698e-7`.
- Fixed-Q thermal AD-FD: worst finest-step error `1.02720e-5`.
- Combined physical-rho AD-FD: error `1.38248e-5`.
- Thermal residual and energy balance: `9.92019e-11` and `1.05829e-12`.
- Pilot evaluation time: 175.6 s initial, 160.2 s first candidate.
- Pilot current: `1.18256593e-23 A` -> `1.18474896e-18 A`.
- Connectivity guard: `Gmin=0.00491 S`; first candidate `G=0.0139536 S`.

The conductance guard prevents a numerically regularized but physically
disconnected sheet.  It does not constrain TaIrTe4 area or volume.

Every solver evaluation now receives a separate PNG, including rejected
candidates and beta reprojections.  Black denotes rho=1 TaIrTe4 and white
denotes rho=0 void.
