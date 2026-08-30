# Lumerical exact-binary Au nanostructure endpoints on fixed TaIrTe4

Status: `VALIDATED_LUMERICAL_AU_TAIRTE4_BINARY_ENDPOINTS`

## What was validated

Au is the **designable optical nanoantenna/nanocube material**, not an
electrode.  Two v261 GPU FDTD cases use the same 10 um scalar Gaussian,
`w0=8.5 um`, 48 x 48 um lateral domain, six PML boundaries, fixed anisotropic
TaIrTe4, component-specific native-Yee absorption, and exact scalar material
endpoints.  The only material difference is absence/presence of a 10 x 10 x
0.05 um Au block in direct face contact with the fixed flake.

| endpoint | P_Q (W) | P_six (W) | closure | auto-shutoff | P_Q/source | GPU |
|---|---:|---:|---:|---:|---:|---:|
| Au absent | 6.163222225290e-14 | 6.162976383701e-14 | 0.003989% | 9.634e-08 | 44.767564% | 4 |
| exact Au | 3.021175778533e-14 | 3.021402721358e-14 | 0.007511% | 9.264e-08 | 21.944800% | 4 |

The source powers match to `0.000e+00` relative.  Adding this exact
Au block changes total absorbed power by
`-50.980580%`.  This is
a raw optical consequence of reflection, field redistribution, Au loss, and
changed TaIrTe4 loss; it is not a fitted or equal-power-normalized result.

## Material and numerical gates

- TaIrTe4 axes: Lumerical `x=b`, `y=a`, `z=c=b` repository closure.
- exact Au at 10 um: `epsilon=-4642.23+1674.64i`.
- maximum component material readback error: `3.169e-05`.
- all raw `Qx/Qy/Qz` cells are finite and nonnegative.
- source, mesh, source power, GPU log, auto-shutoff, material readback, raw
  artifact hashes, and six-face closure pass fail-closed gates.
- no Q clipping, smoothing, gain, global rescaling, or material-power
  reassignment is used.

The component-grid geometric masks account for TaIrTe4 and Au interiors.  A
conformal/interface residual of `3.317211%`
of `P_Q` remains in the Au-present case.  It is reported as a residual rather
than being deleted or forcibly assigned to either material.

## What this resolves—and what it does not

This closes exact-binary Lumerical endpoint stability, fitted material
readback, native-Yee Q extraction, GPU execution, and energy closure for Au
on fixed TaIrTe4.  Independently, the fixed-grid causal FDTDX/JAX route has
status `VALIDATED_AU_ON_FIXED_TAIRTE4_OPTICAL_ADFD_CONTROL` with finest-step strong directional error
`0.014831%`.

The two solvers do **not** use the same compact geometry or absolute source
normalization, so this report does not claim cross-solver equality of raw
power.  It also does not rehabilitate the failed Lumerical moving/conformal
metal boundary derivative or gray `importnk2` route.  The promoted
differentiable optical path remains the causal fixed-grid dispersive route;
thermal, electrical/PTE, and production optimization require their own next
gates.

## Reproduction

```bash
/home/eidl/miniconda3/envs/EIDL-Lumapi/bin/python \
  photothermal_pte/optimization_runs/au_on_fixed_tairte4_validation/43_run_lumerical_au_on_tairte4_binary_endpoint.py --au-endpoint 0 \
  --gpu-device 'GPU 4' --output-dir /external/raw/au0

/home/eidl/miniconda3/envs/EIDL-Lumapi/bin/python \
  photothermal_pte/optimization_runs/au_on_fixed_tairte4_validation/43_run_lumerical_au_on_tairte4_binary_endpoint.py --au-endpoint 1 \
  --gpu-device 'GPU 4' --output-dir /external/raw/au1

/home/eidl/miniconda3/envs/EIDL-Lumapi/bin/python \
  photothermal_pte/optimization_runs/au_on_fixed_tairte4_validation/44_summarize_lumerical_au_on_tairte4_binary_endpoints.py \
  --au0 /external/raw/au0/case_result.json \
  --au1 /external/raw/au1/case_result.json
```
