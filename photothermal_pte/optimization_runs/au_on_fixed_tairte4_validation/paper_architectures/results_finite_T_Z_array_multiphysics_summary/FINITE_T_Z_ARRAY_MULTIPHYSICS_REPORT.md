# Finite T11x15 / Z1x3 Maxwell–thermal–electrical report

Status: **VALIDATED_FINITE_T11X15_Z1X3_MAXWELL_THERMAL_ELECTRICAL_AU_EFFECT_FORWARD**

The requested finite arrays are T=11x15 and Z=1x3. Both use a finite 20x20 um TaIrTe4 flake, finite Gaussian illumination, six optical PML boundaries, and the same explicit 32x32 um thermal/electrical contract. The optical stage is nonperiodic.

## Primary results at 285 uW incident

| case | absorbed (uW) | Tmax (K) | Ta avg (K) | I top-bottom (nA) | I left-right (nA) | closure |
|---|---:|---:|---:|---:|---:|---:|
| T11x15_Ea_Au_on | 51.79359 | 0.646573 | 0.143024 | 0.00866823 | -3.25212e-09 | 0.00296% |
| T11x15_Eb_Au_on | 59.26788 | 0.748648 | 0.163002 | 0.0456538 | 6.47337e-08 | 0.00063% |
| Z1x3_Ea_Au_on | 49.13490 | 0.447395 | 0.057004 | 0.511701 | 0.916864 | 0.00286% |
| Z1x3_Eb_Au_on | 153.21692 | 1.523934 | 0.175763 | 2.54696 | 3.33785 | 0.05682% |

## Interpretation

- The T 11x15 array is nearly mirror-symmetric in x, so left-right signed current cancels to a near-null value. Its top-bottom current is 0.00867 nA for Ea and 0.04565 nA for Eb.
- The Z 1x3 array breaks both terminal symmetries. It produces 0.512/0.917 nA (top-bottom/left-right) for Ea and 2.547/3.338 nA for Eb.
- Top-Au absorbed power is only a small fraction of total absorption. The exact decomposition shows that floating-Au electrical redistribution and Au-induced redistribution of absorption in the non-Au materials can dominate direct Au heating.
- These statements apply to the modeled thermal/electrical contact scenarios. They do not certify unknown experimental Au/TaIrTe4 contacts.

## Au effect decomposition

| case/orientation | electrical | direct Au heat | thermal shunt | optical redistribution | total array-bare |
|---|---:|---:|---:|---:|---:|
| T11x15_Ea_Au_on / top_bottom | 0.00400535 | 0.000830639 | -0.0060315 | 0.00986373 | 0.00866822 |
| T11x15_Ea_Au_on / left_right | 1.43213e-09 | 2.31621e-08 | -1.09668e-10 | 2.47305e-08 | 4.9215e-08 |
| T11x15_Eb_Au_on / top_bottom | 0.0640578 | -0.0206051 | 0.00249911 | -0.000297944 | 0.0456538 |
| T11x15_Eb_Au_on / left_right | -1.11118e-08 | 1.5451e-08 | -8.27052e-10 | 1.02963e-08 | 1.38085e-08 |
| Z1x3_Ea_Au_on / top_bottom | 0.53117 | 0.000906398 | -0.00593543 | -0.0144394 | 0.511701 |
| Z1x3_Ea_Au_on / left_right | 1.04047 | 0.0040452 | 0.0157455 | -0.143395 | 0.916864 |
| Z1x3_Eb_Au_on / top_bottom | 2.56829 | 0.0037509 | -0.0196789 | -0.00539588 | 2.54696 |
| Z1x3_Eb_Au_on / left_right | 4.4548 | 0.0076853 | 0.0839891 | -1.20863 | 3.33785 |

Raw FSP/NPZ files are not committed. Their paths, sizes, and SHA-256 hashes are recorded in the manifest.
