# Meeting plot package — 2024 inverse-T / TaIrTe4 optical controls

Each subfolder is one independently solved GPU case. `Qx`, `Qy`, and `Qz` are
native component-grid depth integrals. `Qtotal` is **not** a same-index sum:
each component is conservatively deposited from its staggered coordinate onto
the common monitor grid first. The JSON records the resulting power error.

- `T_Eb`: inverse-T present, E parallel TaIrTe4 b (Lumerical x)
- `T_Ea`: inverse-T present, E parallel TaIrTe4 a (Lumerical y)
- `bare_Eb`: matched stack without top T, E parallel b
- `bare_Ea`: matched stack without top T, E parallel a

The package is a single-wavelength optical-forward result. It is not a thermal,
PTE, adjoint, or optimized-metal result.
