# v261 FDTD license and API probe

Status: `PASSED_V261_FDTD_LICENSE_API_PROBE`

## Result

- Installation: `/home/seunghyun/lumerical_r12/opt/lumerical/v261`
- Application version read through the live session: `8.35.4522`
- FDTD session startup: passed
- Lumerical-script variable round trip: passed
- Project save: passed
- Project reload: passed
- Solver execution: not performed by this license/API-only checkpoint
- Optimization: not performed by this checkpoint
- Complete session wall time: `5.471355520188808 s`

The saved project contains only a `1 µm × 1 µm × 1 µm` empty FDTD
region. It is a session/save/load probe, not a Maxwell physics result.

## Diagnosis of the initial failed attempts

The first direct attempts did not establish a usable session. The decisive
license-client message was:

`Could not create a license socket. Operation not permitted`

The v261 license logs independently showed successful
`lum_fdtd_gui` and `lum_fdtd_solve` checkouts, including a solve checkout of
`9/50` seats by another already-running job. Therefore the initial failure
was not evidence of unavailable entitlement or exhausted seats. It was caused
by the restricted execution sandbox blocking the localhost Ansys license
proxy socket. Running the same probe outside that sandbox opened the session.

An earlier implementation of the probe also attempted to set an inactive
default object's `name` property. The final probe explicitly clears the CAD
tree, adds a fresh FDTD region, and uses the documented `version` script
command. This API setup error is not classified as a license failure.

## Concurrent process observation

At probe time, a separate pre-existing
`run_constrained_inverse_design.py` process was using GPU FDTD. This
checkpoint did not start, alter, or terminate it. Its presence must be
accounted for before timing or launching the approved matched CPU-TFSF case,
because concurrent work would contaminate runtime measurements even though
the license pool itself is available.

## Artifact policy

The raw `.fsp` and raw JSON remain outside Git. Their absolute paths, byte
sizes, SHA-256 values, and exact generation command are recorded in
`V261_FDTD_LICENSE_API_PROBE_RAW_ARTIFACT_MANIFEST.json`.
