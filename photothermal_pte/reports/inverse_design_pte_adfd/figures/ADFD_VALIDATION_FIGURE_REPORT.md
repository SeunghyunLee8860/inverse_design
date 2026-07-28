# AD–FD validation figure suite

- Immutable combined status: `FAILED_CORRECTED_COMBINED_PHYSICAL_RHO_PTE_ADFD`
- Near-null extension status: `FAILED_SCALE_ADAPTIVE_NEAR_NULL_COMBINED_ADFD`
- Figures are derived from SHA-pinned raw JSON/NPZ artifacts.
- No normalization or gradient rescaling is applied to AD or FD values.

## 01 combined adfd parity

Five-direction corrected Maxwell–thermal–PTE AD versus centered FD.

![01_combined_adfd_parity](01_combined_adfd_parity.png)

## 02 combined relative error vs step

Relative directional error versus FD step; near-null h=0.02 is included when supplied.

![02_combined_relative_error_vs_step](02_combined_relative_error_vs_step.png)

## 03 combined fd over ad vs step

Signed FD/AD deviation, which exposes bias separately from magnitude.

![03_combined_fd_over_ad_vs_step](03_combined_fd_over_ad_vs_step.png)

## 04 combined error heatmap

Direction-by-step matrix for the immutable five-direction sweep.

![04_combined_error_heatmap](04_combined_error_heatmap.png)

## 05 combined direction maps

The exact 81×81 physical-density perturbation fields.

![05_combined_direction_maps](05_combined_direction_maps.png)

## 06 combined gradient maps

Baseline density and optical, thermal-material, and total gradients.

![06_combined_gradient_maps](06_combined_gradient_maps.png)

## 07 combined gradient norm decomposition

L2 norms of optical and thermal contributions.

![07_combined_gradient_norm_decomposition](07_combined_gradient_norm_decomposition.png)

## 08 thermal only adfd parity

Independent fixed-local-Q thermal-material adjoint certificate.

![08_thermal_only_adfd_parity](08_thermal_only_adfd_parity.png)

## 09 thermal only relative error vs step

Thermal-only centered-FD step convergence.

![09_thermal_only_relative_error_vs_step](09_thermal_only_relative_error_vs_step.png)

## 10 optical dz downstream convergence

Q, temperature, PTE, and gradient dependence on optical flake dz.

![10_optical_dz_downstream_convergence](10_optical_dz_downstream_convergence.png)

## 11 supporting gate margins

Measured-to-limit ratios for closure, mapping, residual, and Jacobian gates.

![11_supporting_gate_margins](11_supporting_gate_margins.png)

## 12 adfd validation dashboard

Compact status dashboard with parity, near-null behavior, and gradients.

![12_adfd_validation_dashboard](12_adfd_validation_dashboard.png)

## 13 near null scale adaptive plateau

Dedicated 0.02→0.01→0.005 near-null plateau test.

![13_near_null_scale_adaptive_plateau](13_near_null_scale_adaptive_plateau.png)

## 14 filter projection contract

Finite nonperiodic filter support and tanh projection law.

![14_filter_projection_contract](14_filter_projection_contract.png)

## 15 optical thermal gradient scatter

Pixelwise relation between optical-Q and thermal-material sensitivity.

![15_optical_thermal_gradient_scatter](15_optical_thermal_gradient_scatter.png)

## 16 gradient central linecuts

Central x/y line cuts through each gradient contribution.

![16_gradient_central_linecuts](16_gradient_central_linecuts.png)

## 17 directional derivative dynamic range

Derivative dynamic range and h=0.005 error annotations.

![17_directional_derivative_dynamic_range](17_directional_derivative_dynamic_range.png)
