# FDTDX dispersive reversible-adjoint report

Date: 2026-08-25 (Asia/Seoul)

Status: `PASS_SMALL_SCENE_ADE_CPML_PHASOR_VJP`; production feasibility is not
yet established.

This work replaces neither the certified forward controls nor the final
Lumerical CV0/finer-mesh reevaluation. It investigates an alternative to the
checkpointed FDTDX optical reverse path, which was closed after its optimistic
two-polarization optical projection exceeded 87 minutes per iteration.

## Why the pinned reversible path could not simply be enabled

The pinned FDTDX reversible wrapper omits the dispersive polarization state and
returns only permittivity/permeability cotangents. The parity model encodes the
Au density law in three Lorentz-pole `c3` arrays, so removing the library's
dispersion guard would silently lose the design gradient.

For the target contract (`conductivity=None`, diagonal epsilon, `c4=None`), the
implemented inverse uses the exact discrete relations

```text
P_next = c1 P_curr + c2 P_prev + c3 E_prev
E_next = E_prev + inv_eps drive + inv_eps sum(P_curr - P_next)
E_prev = E_next - inv_eps drive - inv_eps sum(P_curr - P_next)
P_prev = (P_next - c1 P_curr - c3 E_prev) / c2
```

Padded non-dispersive entries are handled separately from the physical,
nonzero `c2` recurrence. One-step and multistep tests use the actual pinned
FDTDX forward update rather than a parallel toy forward equation.

## CPML state audit

The existing FDTDX reversible recorder stores both E- and H-side PML interface
data at every time step. On the exact `186 x 186 x 286`, 256,163-step problem,
the six faces contain 281,976 spatial interface cells including the library's
corner duplication. The payload is 6,767,424 bytes per step and exceeds 1.5
TiB for one polarization, so that recorder is not a production option.

The pinned CPML memory recurrence has the form

```text
psi_next = b psi_prev + a derivative
```

All six placed PML objects have finite, nonzero `b`, allowing algebraic
reconstruction of `psi_prev`. The implementation reproduces the exact raw
directional derivatives used by pinned `curl_E`/`curl_H`; it does not substitute
a continuum PML formula.

## Proven small-scene stages

The tests use a placed 12 x 12 x 12 FDTDX scene with a real point source, a
dispersive Lorentz slab, and two-cell CPML on every face.

1. Actual E/H/ADE state passes one-step and eight-step reverse round trips.
2. Actual CPML psi state passes the same round trips on all six faces.
3. A 24-step, no-recorder custom VJP matches direct unrolled FDTDX for final
   field value and the complete `c3` gradient.
4. A three-component `PhasorDetector`, enabled only after step 8, is added to
   the same dispersive CPML scene. Its phasor-power objective and complete `c3`
   gradient match direct unrolled FDTDX with `rtol=1e-3` and a nonzero gradient.

The detector optimization is valid because the production `PhasorDetector`
update is an additive affine recurrence:

```text
phasor_next = phasor_prev +/- scale * window(time) * phase(time) * field
```

Its prior-state Jacobian is identity and its field Jacobian is independent of
the accumulated phasor value. The prototype therefore accepts exact
`PhasorDetector` objects only and fails closed on inherited/nonlinear detector
types. This covers the retained production `au_late` and `tairte4_late`
detectors after gradient-detector pruning; it does not claim support for the
validation flux detectors.

Relevant commits are:

- `ae61ccbb`: reversible dispersive source/algebra contract;
- `fd48c0d8`: actual pinned FDTDX ADE reverse step and one-step VJP;
- `0f146ba8`: multistep dispersive custom VJP without PML;
- `d7575aeb`: actual six-face CPML inverse and recorder audit;
- `d991e02c`: combined ADE+CPML custom VJP;
- the commit containing this report: late PhasorDetector custom VJP.

After the phasor addition, the complete target-folder CPU suite passes
`224 passed`.

## What is not validated

The current prototype reverses all 24 steps without resets. Both Lorentz
damping and CPML damping become amplification during algebraic reconstruction;
roundoff will therefore accumulate over the production 256,163-step horizon.
The following remain open:

- sliced reverse execution with exact primal reset states;
- sparse storage of ADE polarization only on Au/TaIrTe4 support;
- measured reconstruction/gradient error versus slice length;
- bounded exact-grid latent-density AD-FD for both Ea and Eb;
- exact-grid reversible runtime and peak-memory measurements;
- material-resolved Q, thermal, electrical, signed current, and optimizer
  gradients.

No full 40-period gradient, 16-forward certificate, PDE/current evaluation, or
optimizer is authorized until those gates pass. No Lumerical, HEAT, or CHARGE
call was made during this work.

## Next action

Implement sliced exact resets and prove them first on a longer small scene.
The cotangent must remain continuous across slice boundaries while the reverse
primal E/H/CPML/ADE state is replaced with the corresponding saved forward
state. Only after direct-gradient parity is stable across multiple slice
lengths should a short exact-grid GPU timing probe be launched on a freshly
verified-idle GPU UUID.
