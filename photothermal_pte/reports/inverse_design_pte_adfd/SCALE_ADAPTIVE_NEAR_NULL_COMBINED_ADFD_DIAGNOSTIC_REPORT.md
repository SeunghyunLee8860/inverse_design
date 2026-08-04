# Scale-adaptive near-null combined AD–FD diagnostic

- Status: `FAILED_SCALE_ADAPTIVE_NEAR_NULL_COMBINED_ADFD`
- Passed: `false`
- Empirical normalization / gradient rescaling / clipping: `false`
- Original sequence: `0.01 -> 0.005 -> 0.0025`
- Near-null sequence: `0.02 -> 0.01 -> 0.005`

## Cases

- 4um central_localized: selected error `0.144072%`, plateau `0.170038%` (fail)
- 4um fixed_seed_random: selected error `0.0265038%`, plateau `0.137484%` (fail)
- 6um central_localized: selected error `0.194467%`, plateau `0.144701%` (fail)
- 6um fixed_seed_random: selected error `0.00232609%`, plateau `0.0745501%` (pass)

The original five-direction raw result remains unchanged. A rejected orphan-recovery run, when listed in the manifest, is retained only as a provenance diagnostic and is not used to promote a certificate.
