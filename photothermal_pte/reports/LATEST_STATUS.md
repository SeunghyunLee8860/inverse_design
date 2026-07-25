# Latest photothermal validation status

## Finite 2 um TaIrTe4 optical Q

- Branch: `agent/validate-finite-2um-optical-q`
- Baseline: `be2cbc2c9c77bbcc0265ce2c293affdbb08105de`
- Status: `CONVERGENCE_PASSED_FINAL_REPORT_PACKAGING`
- HEAT Draft PR #2: unchanged and still blocked
- Periodic production optical path: unchanged
- New finite Q artifact validated: `false` (final report packaging pending)

The actual v261 GPU solver rejected TFSF as unsupported. The validated source
is therefore a finite Gaussian beam, never called a plane wave: 3–6 µm
broadband, 4 µm evaluation, 2 µm waist focused at the flake center, and
measured empty-stack E/H intensity normalization. Source-off, x/y/45-degree
empty-stack, flat x/y/45-degree, and fixed-design x controls all pass.

The final fixed-design x result uses a 16 µm lateral domain, 24 PML layers,
5 nm TaIrTe4 dz, and unit central incident intensity. It has
`P_Q=2.56071371e-12 W`, `P_six=2.56486066e-12 W`, 0.161683% six-face closure,
`sigma_abs=2.56071371e-12 m2`, and `sigma_abs/A_geo=0.64017843`.

Final successive convergence changes are:

- domain 12→16 µm: P_Q 0.01996%, spatial L2 0.02121%;
- PML 16→24 layers: P_Q 0.000210%, spatial L2 0.000639%;
- mesh 5→2.5 nm: P_Q 0.12778%, P_six 0.10913%, spatial L2 0.28295%;
- waist 1.75→2 µm: P_Q 0.61986%, spatial L2 1.87002%.

The final raw NPZ is intentionally not committed. Its server path, size,
SHA-256, generating command, commit, and reproduction instructions are
recorded in the finite optical raw-artifact manifest. HEAT, adjoint,
gradients, optimization, and PTE were not run.
