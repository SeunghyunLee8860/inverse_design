# Matched scalar versus thin-lens vectorial Gaussian plan

This is a plan, not a completed vector-source result.

Both source models will use the same 48 µm lateral FDTD domain, six PML
boundaries, 24 PML layers, 11 µm analysis wavelength, 7–13 µm source band,
32 µm injection aperture, focus at the centre of the 130 nm TaIrTe4 film,
and paper-consistent `epsilon_c=epsilon_b` material.

The thin-lens source will not be created by merely toggling
`use scalar approximation`.  An empty 285 nm SiO2/Si stack is used first to
match the *realized* incident field:

1. set the thin-lens focus to the same physical z coordinate;
2. choose NA/pupil fill so the flake-plane 1/e² intensity radius is 6.5 µm;
3. set the source amplitude so measured incident power matches the scalar
   source, without multiplying or globally rescaling the resulting Q;
4. require incident-power difference <0.5%, waist-radius difference <1%,
   focus-position difference <0.1 µm, normalized flake-plane intensity
   NRMSE <1%, and aperture-edge/central intensity <5%;
5. only then run one identical straight-edge finite-flake case.

The finite-edge comparison records Ex/Ey/Ez, Qx/Qy/Qz, total P_Q, six-face
closure, native Yee coordinates, and the edge-normal areal-Q profile.  Failure
of the incident-field matching stage prevents a material/edge comparison.
No post-hoc Q gain, clipping, smoothing, or gradient rescaling is allowed.
