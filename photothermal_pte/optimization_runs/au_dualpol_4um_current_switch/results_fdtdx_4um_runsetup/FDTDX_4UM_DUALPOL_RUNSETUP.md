# FDTDX 4 um dual-polarization runsetup audit

Status: **AUDITED_FDTDX_4UM_DUALPOL_RUNSETUP_NOT_YET_SOLVED**

This checkpoint placed the exact Ea and Eb models but did not run Maxwell,
thermal, electrical, adjoint, or optimization solves.  Both polarizations use
the same 186 x 186 x 40 realized grid;
only the source vector changes (Ea=y, Eb=x).  The finite 16 um TaIrTe4 flake
has a 1.000 um air gap to each lateral PML and the centered Au
design is 8 x 8 x 0.05 um.

The source is a normally incident scalar Gaussian with w0=4 um and a 16 um
square support.  Its requested intensity at the aperture boundary is
0.03355% of the peak.

The carrier-frequency material closure is fitted after float32 coefficient
rounding. The largest realized complex-susceptibility relative error is
4.548e-07;
the seed and adjusted damping plus realized c1/c2/c3 are recorded in JSON.

The production optical-gradient implementation is the checkpoint-free harmonic
two-solve method. Its historical O3 result is not a validation of the current
shared-linear Au law. This audit does not promote the 4 um combined PTE
gradient; full-mesh convergence and a new hash-linked AD-FD certificate still
have to pass.
