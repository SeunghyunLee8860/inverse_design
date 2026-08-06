# Run 002 Gaussian-source GPU report

Status: `VALIDATED_GAUSSIAN10_W8P5_SOURCE_ONLY`

This is a homogeneous-air **source-only** certificate. It contains no
TaIrTe4, SiO2, optical absorption Q, thermal solve, PTE current, adjoint, or
optimization result.

## Fixed contract and readback

- analysis wavelength: 10 µm;
- scalar Gaussian, waist-size-and-position definition;
- target-plane requested waist: 8.5 µm;
- one-step calibrated source-object waist: 8.36043075475035 µm;
- source span: 40×40 µm²;
- FDTD lateral span: 48×48 µm²;
- z bounds: -8 to +8 µm; source/focus z: +5/0 µm;
- all six boundaries PML, 24 layers; periodic/Bloch disabled;
- conformal variant 1, mesh accuracy 3;
- GPU FDTD only; no CPU fallback.

## Measured source gate

| metric | result | gate |
|---|---:|---:|
| fitted waist x | 8.491592 µm | within 0.5% of 8.5 µm |
| fitted waist y | 8.521228 µm | within 0.5% of 8.5 µm |
| effective fitted waist | 8.506397 µm | diagnostic |
| Gaussian-fit NRMSE | 0.090394% | <0.5% |
| ellipticity | 0.348394% | <0.5% |
| incident-power closure | 0.150817% | <0.5% |
| target boundary max/peak | 1.757049e-05 | <1e-3 |
| source-square captured fraction | 0.999992265 | >=0.999 |
| auto shutoff | 5.385550e-10 | <=1e-5 |
| solver wall time | 3.458 s | diagnostic |
| GPU memory | 0.054 GiB | diagnostic |

All acceptance booleans are true. The uncalibrated 8.5 µm source-object run,
which realized 8.64190 µm and failed only the waist gate, remains preserved as
a diagnostic raw artifact; it was not relabeled as passing.

## Consequence

The source contract is now authorized for the next **single forward material
smoke test**. It does not authorize optimization. The remaining gates are the
complex 10 µm SiO2/TaIrTe4 material readback and closure, material-resolved Q
remap, coarse physical-gradient design-window selection, Gaussian combined
AD–FD smoke test, and production-scale CUDA thermal parity.
