# FDTDX physical-device blocker audit

## Outcome

The implemented rectangular Shockley--Ramo operator has the correct internal
sign algebra, but it is not a model of a confirmed target device.  The audit
therefore validates the blocker and keeps every production mesh/optimizer
gate closed.

This work used one CPU diagnostic solve.  It did not use a GPU, run Maxwell,
run a thermal solve, launch or modify Lumerical, or start an optimization.

## Certificate

- code commit: `5d8cf7f12a01b64ae99b89547ca00c56f0ec8f84`
- certificate:
  `/home/seunghyun200/fdtdx_results/fdtdx_physical_device_blocker_audit_5d8cf7f1/FDTDX_PHYSICAL_DEVICE_BLOCKER_AUDIT.json`
- certificate SHA-256:
  `2a3ba7ce4428cb2a5b6db2a470d55f26510ca7530b95a1931b03d856fcec8890`
- generator SHA-256:
  `268cee94c98b5491553d3d0f7654b3c3e6208a96edc04013389dd45c0a827975`
- status: `VALIDATED_BLOCKED_FDTDX_PHYSICAL_DEVICE_AUDIT`
- failed integrity checks: none
- full regression suite: `467 passed, 7 subtests passed in 26.57 s`
- complete command wall time: `0.54 s`

`VALIDATED_BLOCKED` means that the code/provenance checks and the negative
decision are valid.  It does not certify the present geometry as a device.

## What is currently assumed

`physical_device_contract.json` records all of the following as code
assumptions, not measurements:

- `16 um x 16 um x 100 nm` rectangular TaIrTe4;
- solver `x=b`, `y=a`, with no in-plane rotation;
- ideal terminals covering the complete `x_min` and `x_max` flake edges;
- `8 um x 8 um x 50 nm` centered, floating patterned Au in direct
  thermal/electrical contact with TaIrTe4;
- `285 nm` SiO2 on Si;
- centered normal-incidence `4 um` Gaussian, `w0=4 um`, `285 uW`;
- no optical electrode/pad geometry;
- prototype thermal outer boundaries and electrical void floors.

All ten required confirmations in that JSON remain exactly `false`.

## What the CPU sign audit proves

The audit removes patterned Au electrically and solves the ideal `160 x 160`
TaIrTe4 rectangle with full x-edge terminals.  The solve took `0.279 s` inside
the audit.

- the computed weighting potential matches the analytic x ramp to
  `1.4352e-11` maximum absolute error;
- the free-system residual is `9.8223e-12`;
- terminal current balance error is `8.8249e-12`;
- a prescribed `+x` temperature gradient gives
  `-4.752000000000095e-7 A`, while the closed discrete formula gives
  `-4.7519999999999997e-7 A`;
- a pure y gradient gives only `-1.696e-22 A` because the rectangular
  full-edge weighting field has no y component;
- swapping the 0/1 terminals changes the current to
  `+4.7520000000000955e-7 A`;
- the current-density map integrates back to the scalar objective.

This proves that the present discrete sign, thickness factor, terminal swap,
and current-map identity are internally consistent.  It also demonstrates why
the target current sign cannot be separated from the real terminal geometry:
changing the weighting field changes the collected projection.

## Unsupported target-device physics

The audit records these active blockers:

1. electrode polygons are unsupported; only complete x edges are terminals;
2. arbitrary crystal rotation and the resulting off-diagonal optical,
   thermal, electrical-conductivity, and Seebeck tensors are unsupported;
3. the electrical weighting problem is a 2-D thin-sheet reduction, not 3-D;
4. patterned Au changes the electrical weighting network but contributes no
   thermoelectric source, which is an implicit `S_Au=0` assumption;
5. real electrode/pad metal is absent from the optical model;
6. patterned-Au electrical role and Au-TaIrTe4 contact are unconfirmed;
7. electrical pitch/contact/void sensitivity is not converged on actual
   geometry;
8. every target-device confirmation is missing;
9. the current local papers folder lacks the cited AFM supporting-information
   PDF.

The main AFM paper in `/home/seunghyun200/papers` is present with SHA-256
`7a573dd775483e5c5af3ac95e07554027bad5cd45fb3d72074eddf157ad930ff`.
The repository also contains an older Device-A reconstruction contract, but
its embedded files point to `/home/seunghyun/...`, are unavailable now, and
its main-paper bytes differ from the current local file.  That historical
reconstruction is evidence about a different paper device, not authority for
the user's target 4-um inverse-design device.

## Required target-device inputs

Before thermal/electrical production mesh convergence, provide and freeze:

1. flake outline/CAD and TaIrTe4 thickness;
2. a-axis angle in device coordinates;
3. measurement-electrode and pad polygons, plus the signed output contact;
4. whether patterned Au touches TaIrTe4 electrically and whether it is
   floating or terminal-connected;
5. SiO2 thickness and relevant Si stack;
6. 4-um beam power, waist definition, center, incidence, and polarization
   convention;
7. accepted or measured Au-TaIrTe4 thermal/electrical contact ranges.

If the user's intended target is deliberately the ideal rectangle rather than
an experimental device, that must be stated explicitly.  It would make the
geometry a defined mathematical design target, but it would not turn the
result into a prediction for an unspecified fabricated flake.

## Next numerical order after the contract is supplied

1. implement the actual polygonal flake/electrodes and rotated tensors;
2. include every optically relevant electrode/pad in the independently owned
   Maxwell model;
3. converge electrical pitch, Au-contact bounds, patterned-Au role, and the
   exact-zero void limit on that geometry;
4. rerun thermal mesh/interface convergence using the actual support and
   independently validated Q;
5. complete coupled AD-FD, then reevaluate only exact-binary ordinary-Au
   candidates.

Until then, the existing same-sign Ea/Eb result is a prototype diagnostic and
must not be used to reject or validate the physical polarization switch.
