# Native-Yee thermal-source adjoint pullback

Status: **VALIDATED_NATIVE_YEE_THERMAL_SOURCE_ADJOINT_PULLBACK**

The explicit thermal source adjoint is pulled back through both conservative
power maps. Every Ex/Ey/Ez component uses its own physical Yee coordinates,
dual widths, and overlap operator; no same-index component pairing is used.

The worst two-stage transpose dot-test error is `6.634e-15` and
the base native-weighted source contraction agrees with the explicit thermal
grid contraction to `5.484e-16` relative. The weights
have units `A/W` and are not normalized or rescaled.

This is a mapping certificate only. It does not yet validate the reverse
Maxwell solve or authorize Au optimization.
