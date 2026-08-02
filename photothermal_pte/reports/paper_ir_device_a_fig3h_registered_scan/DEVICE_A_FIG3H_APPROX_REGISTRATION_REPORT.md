# Device-A Figure 3H approximate registration plan

Status: `READY_FOR_GPU_PHASE1_WHEN_RESOURCE_AVAILABLE`

This is a named approximation, not raw SPCM stage metrology. No FDTD,
thermal, PTE, adjoint, AD-FD, or optimization solve was run here.

## Registration assumption

- image right is `+b`; image up is `+a`; no rotation or shear is introduced;
- the visible 10-um bar gives `11.2` pixels/um;
- the Figure-3H map-panel centre is aligned to the Figure-2 digitized flake
  bounding-box centre;
- Figure-3I distance zero is placed at that common centre;
- the nominal Figure-3I peak coordinate is `+3 um` along `+a`.

The resulting nominal beam centre is
`(b,a)=(-16.562500,3.000000) um`.  Its signed
distance from the nearest digitized flake boundary is
`3.081837 um` (positive means outside the flake).  This differs
qualitatively from the old, unregistered point placed 3 um *inside* a
non-boundary chord.

## Source/PML consequence

The registered source plus the full digitized electrode envelope does not
fit the old 60-um domain with the unchanged 50-um source span and the
runner's 0.5-um minimum PML-clearance gate.  That case has x clearance
`-0.891747 um`.
The phase-1 plan therefore preserves the 50-um source and uses a 64-um
lateral domain, giving x clearance
`1.108253 um`.

## Comparison contract

The experimental quantity is obtained from a scan profile.  The final
comparison must use separate maxima of `|Ia|` and `|Ib|` over the same
registered scan, not one arbitrary source point.  Phase 1 runs only the
nominal `+3 um` a/b pair.  The sparse `-1, +1, +3, +5, +7 um` scan is
authorized only after both optical gates pass.
