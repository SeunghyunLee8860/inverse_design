# Corrected Z M2 paired optical→thermal diagnostic

Status: `DIAGNOSTIC_Z2022_M2_PAIRED_THERMAL_BLOCKED_LOCAL_Q_OSCILLATION`

Both `E||a` and `E||b` use the same corrected figure-period geometry,
incident intensity, conservative remap, material tensors, periodic lateral
boundaries, bottom bath, and top adiabatic boundary.  The plot contains the
depth-integrated signed heat source, TaIrTe4 thickness-averaged temperature,
`dT/db`, `dT/da`, and in-plane gradient magnitude for **both** polarizations.

This is not a promoted physical thermal certificate.  The native Lumerical
volumetric-loss monitor failed the matched-volume closure gate, while the
independent Poynting-divergence construction closes by conservation but retains
signed metal-interface oscillations.  No clipping, smoothing, gain, or
rescaling was used.  The maps are therefore a fail-closed diagnostic only.

The periodic unit cell has no terminal pair, so weighting potential and PTE
current are not defined in this result.

## Case metrics

```json
{
  "Ea": {
    "P_Q_W_at_1_W_m2_incident": 2.0558720739823003e-12,
    "Q_mapping_error_relative": 1.463135398705982e-16,
    "remapped_negative_to_positive_power": 0.03673130441217047,
    "Tmin_K_per_W_m2": 1.77999276980944e-11,
    "Tmax_K_per_W_m2": 1.7416203253597315e-07,
    "TaIrTe4_Tmax_K_per_W_m2": 8.227866785938859e-08,
    "max_abs_dT_db_K_m_per_W_m2": 0.06241534563944208,
    "max_abs_dT_da_K_m_per_W_m2": 0.03661911219042425,
    "residual_relative": 9.841197675200147e-11,
    "energy_balance_relative": 1.5035089600359492e-12,
    "solve_wall_time_s": 2.2766839042305946
  },
  "Eb": {
    "P_Q_W_at_1_W_m2_incident": 7.31620197921141e-12,
    "Q_mapping_error_relative": 2.3024114978139765e-15,
    "remapped_negative_to_positive_power": 0.21752502227110254,
    "Tmin_K_per_W_m2": -1.5827803226846717e-06,
    "Tmax_K_per_W_m2": 2.77297004867284e-06,
    "TaIrTe4_Tmax_K_per_W_m2": 4.934334109670144e-07,
    "max_abs_dT_db_K_m_per_W_m2": 0.8689456568918321,
    "max_abs_dT_da_K_m_per_W_m2": 0.9880464856241118,
    "residual_relative": 8.67837224174244e-11,
    "energy_balance_relative": 2.8989669713886946e-12,
    "solve_wall_time_s": 1.7599617429077625
  }
}
```
