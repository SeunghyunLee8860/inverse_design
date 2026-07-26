# Mechanical internal-interface-G solver report

**Status: `BLOCKED_MECHANICAL_EXECUTABLE_UNAVAILABLE`.**

The generated two-slab controls use separate coincident meshes joined
by TARGE170/CONTA174. `KEYOPT(1)=2` selects pure thermal contact,
`KEYOPT(12)=5` keeps the interface bonded, and real constant 14
sets `TCC=G` in W/(m2 K). Cases are generated for `7.37e6` and
`1.1e9 W/(m2 K)`.

- Canonical input-deck static audit: `PASSED_MECHANICAL_INPUT_DECK_STATIC_AUDIT`
- Actual Mechanical solver executed: `False`

Perfect-contact controls use a shared-node material interface at
100, 50, and 25 nm axial mesh spacing; the expected temperature jump
is exactly zero and the analytic heat flux is `4.0e7 W/m2`.

The solver-side acceptance criteria are `<1%` for transmitted heat
flux, `Delta T=q''/G`, and global energy balance. These criteria have
not been evaluated by Mechanical on this server because the executable
and license are absent.

The same execution command runs the finite-G and perfect-contact
controls together with the anisotropic controls.

Official documentation:
https://ansyshelp.ansys.com/public/Views/Secured/corp/v261/en/pdf/ANSYS_Mechanical_APDL_Contact_Technology_Guide.pdf
https://ansyshelp.ansys.com/public/Views/Secured/corp/v251/en/ans_elem/Hlp_E_CONTA174.html
