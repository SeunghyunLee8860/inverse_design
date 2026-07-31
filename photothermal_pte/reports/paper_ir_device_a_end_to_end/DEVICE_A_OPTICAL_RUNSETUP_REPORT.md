# Device A optical runsetup audit

Status: `DEVICE_A_OPTICAL_RUNSETUP_AUDITED_NOT_SOLVED`

This checkpoint created and saved the v261 geometry, invoked `runsetup`, and
read the realized contract. It did **not** time-step Maxwell and is not an
optical, thermal, or current result.

## Frozen contract

- all six boundaries: PML; no periodic/Bloch boundary
- scalar Gaussian, 11 um, explicit-assumption physical waist 12 um
- source span / lateral domain: 50 / 60 um
- local mesh: 50 nm nested in a 100 nm outer material region
- TaIrTe4 and metal-region dz: 5 nm
- TaIrTe4 axes: code x=b, y=a, z=c=b
- Device A: Figure-digitized approximation, not unpublished CAD
- electrodes: digitized 5 nm Ti / 50 nm Au top and bottom polygons

The digitized structure and the beam were translated together by
`[7.231221193496584, -0.5972383224697246]` um. This changes only the
coordinate origin and preserves every beam/device/contact relative position.
The resulting nominal lateral PML clearances are x=2.547784 um and
y=5.000000 um.

## Material readback at 11 um

- Au: built-in `Au (Gold) - CRC`, n =
  7.63606715 +
  78.7750414i
- Ti: built-in `Ti (Titanium) - CRC`, n =
  4.59346378 +
  21.6788929i

Both CRC sampled-data ranges include 11 um. TaIrTe4 fitted epsilon-z equals
epsilon-x exactly under the documented epsilon-c=epsilon-b 3D closure.

## Gate

Empty and finite runsetup checks both passed. A GPU-only Maxwell solve is the
next authorized step. No CPU fallback, Q modification, thermal solve, PTE,
adjoint, AD-FD, or optimization occurred in this checkpoint.
