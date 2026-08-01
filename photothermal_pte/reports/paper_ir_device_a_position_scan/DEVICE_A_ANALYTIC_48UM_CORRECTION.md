# Device-A analytic 48-um diagnostic correction

Status: `DIAGNOSTIC_WRONG_DOMAIN_NOT_SAME_AS_IMMUTABLE_S0`

The first analytic three-position artifact used a 48-um lateral thermal
domain. The immutable Device-A `s0` thermal artifacts use a 60-um lateral
domain (`x,y = [-30,30] um`). Therefore the earlier files without a `60um`
suffix are preserved for provenance but are not eligible for the promoted
Maxwell--analytic comparison.

This mismatch was exposed fail-closed when the full optical-Q support could
not be conservatively embedded in the 48-um grid. No source cropping,
deletion, gain, or rescaling was used to bypass the failure.
