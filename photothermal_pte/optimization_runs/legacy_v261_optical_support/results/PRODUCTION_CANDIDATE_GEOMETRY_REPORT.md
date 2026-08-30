# Run 002 production-candidate geometry audit

Status: `VALIDATED_RUN002_PRODUCTION_CANDIDATE_RUNSETUP`

This is a Lumerical runsetup/readback certificate.  No Maxwell, thermal,
adjoint, or optimization solve was run.

## Frozen candidate

- FDTD: 48×48 µm lateral, z=-8..8 µm, six PML, 24 layers.
- Source: calibrated scalar Gaussian, λ=10 µm, target-plane waist 8.5 µm,
  source aperture 40×40 µm, source z=5 µm, focus z=0.
- Long optical TaIrTe4 background: 60×60×0.1 µm, extending beyond the
  transverse PML bounds; no artificial optical flake edge.
- Bottom stack: 285 nm Kitamura-SiO2 on Palik-Si.
- Coarse design canvas: 20×20×1 µm, 201×201×21 imported nodes,
  100 nm lateral and 50 nm vertical node spacing.
- Matched Q/six-face box: x,y=-20..20 µm and z=-1.25..1.25 µm.

The exact layer order is Si → bottom SiO2 → TaIrTe4 → imported design → air.
All adjacent z interfaces were read back as contiguous.

## Material values at 10 µm

- TaIrTe4 epsilon_x=epsilon_b:
  `[13.778527727027518, 23.688463190816908]`
- TaIrTe4 epsilon_y=epsilon_a:
  `[-39.87819057091918, 187.50005695474056]`
- TaIrTe4 epsilon_z=epsilon_b closure:
  `[13.778527727027518, 23.688463190816908]`
- SiO2 n+ik: `[2.7352020978104523, 0.36376996242453913]`
- Si n+ik: `[3.4214999999999995, 6.759999999999991e-05]`

`epsilon_c=epsilon_b` is an explicit paper-consistent 3D closure, not an
independent c-axis measurement.

## Realized mesh

- mesh lines: `[445, 445, 104]`
- grid-point product: `20,594,600`
- minimum dx/dy/dz:
  `59.701` /
  `59.701` /
  `10.000` nm
- maximum dx/dy/dz:
  `198.000` /
  `198.000` /
  `583.333` nm

The 100 nm value is the design-node spacing, not the minimum Yee step.  v261
realized about 59.7 nm minimum lateral spacing in this runsetup.

## Corrected diagnostic

The first runsetup requested an asymmetric z control volume, while the pabs
internal monitors remained centered.  It is preserved as a failed diagnostic.
The promoted v2 uses an exactly matched symmetric control volume.  No field
solve was performed with the mismatched geometry.
