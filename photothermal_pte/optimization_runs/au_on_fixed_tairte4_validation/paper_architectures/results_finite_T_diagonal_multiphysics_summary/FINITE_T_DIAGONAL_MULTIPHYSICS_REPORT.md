# Finite T11x15 axial/diagonal polarization Maxwell-to-current report

Status: `VALIDATED_FINITE_T11X15_FOUR_LINEAR_POLARIZATION_MAXWELL_THERMAL_ELECTRICAL`

The electrodes and weighting solves are unchanged between polarizations. Only the coherent linear source angle changes.

| polarization | absorbed power at 285 uW (uW) | Tmax (K) | top-bottom I (nA) | left-right I (nA) | closure |
|---|---:|---:|---:|---:|---:|
| E||a | 51.7935885 | 0.646573329 | 0.00866823171 | -3.25211765e-09 | 0.002964% |
| E||b | 59.26788 | 0.748648127 | 0.0456538347 | 6.47337111e-08 | 0.000634% |
| +45 deg | 55.5271745 | 0.692928042 | 0.0273616536 | -0.118196822 | 0.010039% |
| -45 deg | 55.5271757 | 0.692928046 | 0.027361661 | 0.118196765 | 0.010037% |

The +/-45 cases are independent coherent Maxwell solves, not arithmetic averages of Ea/Eb Q.

## Diagonal-polarization mirror audit

The two diagonal sources have equal total absorption to numerical precision, but their spatial fields are mirrored across Lumerical x=b.
The left-right current integrand acquires an additional sign reversal, whereas the top-bottom integrand does not.

- full-3D Q direct relative L2: `0.543450412`
- full-3D Q after x reflection: `5.70752094e-07`
- TaIrTe4 Q after x reflection: `1.97014574e-07`
- 3D temperature after x reflection: `4.39035272e-08`
- left-right integrand after x reflection and sign flip: `1.04129334e-07`
- top-bottom integrand after x reflection: `7.63745929e-08`

Raw NPZ/FSP remain outside Git; no clipping, smoothing, gain, global rescaling, or tiling was used.
