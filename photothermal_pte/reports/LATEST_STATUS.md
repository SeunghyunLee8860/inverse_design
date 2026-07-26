# Latest photothermal validation status

## Finite in-flake SiO2 proxy optical Q

- Branch: `agent/validate-inflake-proxy-optical-q`
- Base: PR #3 head `053260da6fd0caec28ce155221bd18f683a0e5e7`
- Status: `VALIDATED_FINITE_INFLAKE_PROXY_OPTICAL_Q`
- PR #2–#5: unchanged
- PR #3 radius-1.5-µm artifact: not reused or cropped

Fresh v261 GPU FDTD was run for a centered radius-0.8-µm, 600-nm-high SiO2
disk completely inside the 2 µm × 2 µm × 100 nm TaIrTe4 footprint. Outside
the disk is air, with no support annulus, overhang support, or oxide pillar.
The finite Gaussian source uses a 2 µm waist, 6.8 µm aperture, 3–6 µm source
band, 4 µm analysis point, and measured central incident intensity of 1 W/m2.

The promoted x-polarized result uses a 16 µm lateral domain, 24 PML layers,
and 5 nm TaIrTe4 dz:

- `P_Q=2.0361088604691824e-12 W`
- `P_six=2.040668004695463e-12 W`
- six-face closure `0.223414304%`
- `Qx/Q=0.993324070`, `Qy/Q=0.006675930`, `Qz/Q=0`
- raw NPZ SHA-256
  `2ecdb8a8a2a01f85635914357ce05aab834576a66069cdc024a5dca49b0c71c3`

Final convergence changes are:

- domain 12→16 µm: P_Q 0.0240581%, P_six 0.0232486%, spatial L2 0.025513%;
- PML 16→24: P_Q 0.000270435%, P_six 0.00134641%, spatial L2 0.000594892%;
- flake dz 5→2.5 nm: P_Q 0.0769457%, P_six 0.0503751%, spatial L2 0.608514%.

Source-off, empty-stack x/y/45-degree, finite-flat x/y/45-degree, proxy,
six-face closure, domain, PML, mesh, finite-value, geometry, and P_Q
reintegration gates pass. Raw NPZ/FSP files are not committed. Thermal, PTE,
adjoint, gradient, and optimization were not run.
