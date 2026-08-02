# Device-A Figure 3H registered runsetup

Status: `VALIDATED_DEVICE_A_FIG3H_REGISTERED_RUNSETUP_READY_FOR_GPU_PHASE1`

The registered Figure-3H approximation opened a fresh v261 session, saved a
project, ran `runsetup`, and read back the actual native mesh. Maxwell time
stepping was not executed.

- registered digitized beam centre: `[-16.5625, 3.0] um`;
- signed distance to the nearest digitized flake boundary:
  `3.081837 um` (outside);
- lateral domain/source span: `64/50 um`;
- minimum source/PML clearance:
  `1.108253 um`;
- native mesh: `629 x 691 x 88`;
- estimated native Yee cells: `37698840`;
- realized minimum dx/dy/dz:
  `50.000000 / 50.000000 / 9.802632 nm`;
- empirical estimate: `5.899 GiB`,
  `274.9 s` per optical case.

Every pre-run check passed. The registration remains an explicit affine
figure-reading assumption, not raw experimental stage metrology. Raw FSP and
case JSON remain outside Git and are SHA-pinned in the manifest.
