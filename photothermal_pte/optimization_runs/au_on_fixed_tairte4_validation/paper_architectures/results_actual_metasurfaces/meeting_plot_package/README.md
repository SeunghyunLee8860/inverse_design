# Meeting plot package — 2024 inverse-T / TaIrTe4 optical controls

Each subfolder is one independently solved GPU case. `Qx`, `Qy`, and `Qz` are
native component-grid depth integrals. `Qtotal` is **not** a same-index sum:
each component is conservatively deposited from its staggered coordinate onto
the common monitor grid first. The JSON records the resulting power error.

- `T_Eb`: inverse-T present, E parallel TaIrTe4 b (Lumerical x)
- `T_Ea`: inverse-T present, E parallel TaIrTe4 a (Lumerical y)
- `bare_Eb`: matched stack without top T, E parallel b
- `bare_Ea`: matched stack without top T, E parallel a

Files `00--06` provide the compact certificate. Files `07--11` add the full
xy/xz/yz setup, top-monitor total fields, volumetric Q sections,
component-resolved depth profiles, and geometric material/support maps.
`comparisons/` contains matched T-versus-bare and E||a-versus-E||b figures.
Use `MEETING_GUIDE.md` for the slide order, exact wording, limitations, and
anticipated questions. `all_case_metrics.csv` is the four-case numerical table.

The package is a single-wavelength optical-forward result. It is not a thermal,
PTE, adjoint, or optimized-metal result.
