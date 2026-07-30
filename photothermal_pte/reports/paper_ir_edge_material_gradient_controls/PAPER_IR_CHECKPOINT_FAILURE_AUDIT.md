# Existing paper-IR checkpoint and GPU-failure audit

**Status: `AUDITED_EXISTING_PAPER_IR_CHECKPOINTS_UNRESOLVED_ENGINE_TERMINATION_AND_EDGE_METRIC`**

## Repository and provenance

- Branch: `agent/validate-inverse-design-pte-adfd`
- Immutable audit basis: `651797fefbcaa254737bcec3cac854979ae2bfef`
- Report-generation HEAD: `98f58323961f515d071ec4ee9c44422690174d26` (descends from the audit basis:
  `True`)
- Dirty/untracked at audit start: `[]`
- All six 200/100/50 nm paper-reduced cases use the same geometry, Robin
  boundary, source, and remap contract within each polarization.  Only the
  intended lateral core step and resulting grid shape differ.
- Exact paths, byte sizes, and SHA-256 values are in the audit JSON.

## Thermal comparator decision

The full per-case table is `paper_reduced_thermal_audit_cases.csv`.
For the analytic a-polarization source, the 100-to-50 nm fitted-x strip mean
changes by `0.609595%`,
but its fitted normal strip mean changes by
`2.991614%`.
For legacy Maxwell-Q b polarization, the fitted-x strip mean changes by
`12.851007%`.
Raw maxima remain diagnostic only.  Therefore the local edge-gradient gate is
not promoted even where a single fitted-x aggregate is below 1%.

## GPU termination evidence

Retry 4 acquired the solve license, completed meshing, initialized GPU 2, and
started 39,362 time steps.  The log stops at
`3.3357%`.
The kernel recorded an `fdtd-solutions-app` remote-messenger segfault at the
same wall time, but the external `fdtd-engine` exit code was not captured.
The formal classification is therefore `UNRESOLVED_ENGINE_TERMINATION`, with
a confirmed secondary `FDTD_SOLUTIONS_REMOTE_MESSENGER_SEGFAULT`.

There is no contemporaneous OOM, GPU Xid/reset, timeout, or license failure in
retry 4.  The engine estimated `15.169 GiB` GPU
memory against 49,140 MiB capacity, and the host reported
`903.354 GiB` available.  This makes memory
exhaustion unlikely but does not prove the fdtd-engine's internal cause.

## Why the run became large

The logged grid is `1461 x 1461 x 161` =
`343657881` gridpoints.  There is no x/y mesh override; only
the flake-region z mesh is fixed to 10 nm.  The production straight-edge
TaIrTe4 half-plane spans the 48-um lateral domain, so accuracy-5 automatic
meshing resolves its high complex index laterally across most of the domain.
The nominal average lateral interval is
`32.877 nm`.

The incomplete HDF5 contains monitor sampling coordinates, not the native
FDTD mesh.  They are reported separately in
`partial_h5_monitor_coordinate_audit.csv`; native x/y/z min/median/max steps
remain unavailable because the API failed before the post-run solver-mesh
readback.

No new solver calculation was performed by this audit.
