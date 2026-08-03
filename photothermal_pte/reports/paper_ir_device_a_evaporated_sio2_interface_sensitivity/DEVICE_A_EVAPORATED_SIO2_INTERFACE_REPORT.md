# Device-A evaporated-SiO2 interface-G sensitivity

Status: `VALIDATED_DEVICE_A_EVAPORATED_SIO2_INTERFACE_G_SENSITIVITY`

Only the internal TaIrTe4/SiO2 face conductance changed from
`7.37e6` to `7.37e4 W/(m2 K)`. This is a named evaporated-interface
sensitivity, not a replacement paper baseline and not a fabrication
prediction. Maxwell Q, bulk materials, all other interfaces/boundaries, and
the electrical weighting field are unchanged. No Q rescaling was used.

| d (um) | pol | grown Tavg (K) | evaporated Tavg (K) | Tavg ratio | grown I (nA) | evaporated I (nA) |
|---:|:---:|---:|---:|---:|---:|---:|
| 1 | a | 0.0093852 | 0.301365 | 32.110692 | 13.722020 | 214.348352 |
| 1 | b | 0.0107563 | 0.34255 | 31.846371 | 11.155379 | 209.874872 |
| 3 | a | 0.00775779 | 0.249303 | 32.135873 | 13.699612 | 219.161461 |
| 3 | b | 0.00873465 | 0.278446 | 31.878335 | 11.462797 | 215.562908 |
| 5 | a | 0.00599413 | 0.192936 | 32.187451 | 12.291305 | 197.040216 |
| 5 | b | 0.00660889 | 0.211043 | 31.933242 | 10.382508 | 192.365940 |


| d (um) | grown production Ib/Ia | evaporated production Ib/Ia | evaporated strict Ib/Ia | evaporated face Ib/Ia | production ratio change |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.812955 | 0.979130 | 0.983117 | 0.979571 | 20.441% |
| 3 | 0.836724 | 0.983580 | 0.986876 | 0.983872 | 17.551% |
| 5 | 0.844703 | 0.976278 | 0.979363 | 0.976493 | 15.576% |


All numerical gates are recorded in the summary JSON. Production current is
the unchanged full-volume implementation. Strict four-neighbour and symmetric
internal-face values are diagnostics that test sensitivity to the earlier
boundary-gradient concern.

The lower G raises TaIrTe4 average temperature by `31.85--32.19x` and
production current by `15.62--18.81x`. It moves `Ib/Ia` from `0.813--0.845`
to `0.976--0.984`, but all three current discretizations remain below one.
Therefore this interface scenario strongly reduces, but does not reverse, the
present simulated polarization trend and does not reproduce the paper's
approximate `Ib/Ia~1.17`.

Absolute current is not certified because the digitized Device-A geometry's
two-terminal resistance remains far from the measured resistance. The
evaporated value is a literature-based numerical scenario; actual fabrication
must determine whether the full bottom contact is thermally grown, evaporated,
or spatially mixed. The temperature-change image uses independent per-panel
color scales and is diagnostic only; scalar JSON metrics must be used for
cross-panel comparison.

No FDTD, new Q, adjoint, AD-FD, or optimization was run. Raw NPZ inputs remain
external and SHA-pinned; derived 3D temperatures are reproducible and are not
committed.
