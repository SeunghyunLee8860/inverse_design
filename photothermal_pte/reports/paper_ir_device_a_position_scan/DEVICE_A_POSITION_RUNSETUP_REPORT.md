# Device-A position-scan runsetup audit

Status: `DEVICE_A_POSITION_RUNSETUP_AUDITED_NOT_SOLVED`

All four `s0±1 um`, `E||a/b` v261 sessions opened on the host, saved, ran
`runsetup`, and passed every pre-run check. No Maxwell time stepping occurred.

The Device-A flake/electrode polygons, six PML boundaries, monitors, material,
and every local-mesh override bound are exactly identical across the four
contracts: `True`. Only source center and polarization change. The
50-nm local mesh remains fixed at the pre-registered `s0` center.

The first sandboxed session attempt failed before opening Lumerical because it
could not join the host ANSYSLI sharing context. The host runsetup sessions then
opened normally; no existing Lumerical process or `.ansys` state was deleted.
