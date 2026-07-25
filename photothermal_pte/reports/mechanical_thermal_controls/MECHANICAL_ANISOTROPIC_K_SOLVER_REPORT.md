# Mechanical anisotropic-kappa solver report

**Status: `BLOCKED_MECHANICAL_EXECUTABLE_UNAVAILABLE`.**

Mechanical/MAPDL officially supports orthotropic conductivity through
`MP,KXX`, `MP,KYY`, and `MP,KZZ`; the generated controls assign
`diag(14.4, 3.8, 1.0) W/(m K)` to SOLID70 bricks and independently
drive heat in x, y, and z.

- Executable probe: `BLOCKED_MECHANICAL_EXECUTABLE_UNAVAILABLE`
- License probe: `BLOCKED_MECHANICAL_LICENSE_UNAVAILABLE`
- Actual Mechanical solver executed: `False`
- Canonical input-deck static audit: `PASSED_MECHANICAL_INPUT_DECK_STATIC_AUDIT`
- Isotropic average used: `false`

The input decks include database `SAVE`, `/CLEAR`, `RESUME`, material
readback, boundary reaction heat flow, nodal temperature output, and
energy-balance quantities. On this server they were not executed
because neither a MAPDL executable nor a Mechanical license feature is
available. Offline analytic values are references only.

To execute after installing MAPDL and adding an `ansys`, `mech_1`,
`mech_2`, or `struct` solver license:

```bash
python photothermal_pte/validation/photothermal_stage1/32_validate_mechanical_thermal_controls.py \
  --executable /path/to/ansys261
```

Official documentation:
https://ansyshelp.ansys.com/public/Views/Secured/corp/v261/en/ans_mat/thermalmat.html
https://ansyshelp.ansys.com/public/Views/Secured/corp/v252/en/ans_cmd/Hlp_C_MP.html
