# Rejected: using `np density` as an Au topology variable

Status: `REJECTED_NOT_AN_AU_GEOMETRY_REPRESENTATION`

## Correction

Lumerical's `np density` grid attribute represents semiconductor electron and
hole densities used by an index-perturbation material. It is not a gold
occupancy field, an Au/void boundary, or a general topology-optimization
variable.

An earlier checkpoint fitted its Drude parameters so that the `n` endpoint
matched one frozen complex Au permittivity at 4 um. That calculation was only
a numerical single-frequency material surrogate. It did not turn carrier
density into physical Au, did not create an exact Au/void interface, and did
not establish the required broadband dispersive material response. It must
not be used for forward solves, gradients, mesh certificates, or final
inverse-design claims in this project.

The probe, implementation facade, and tests for that route have been removed.
Repository history retains the checkpoint for auditability, but it is not an
authorized production option.

## Consequence for Lumerical and R1.3

The earlier statement that Lumerical 2026 R1.3 was required for this project
because it adds GPU support for `np density` was wrong. Since `np density` is
rejected, that feature is irrelevant to the Au inverse-design route. A
different Lumerical version may still be needed after an actual B200
compatibility test, but this repository currently has no evidence requiring
R1.3 for the exact-Au problem.

No Lumerical HEAT or CHARGE license is involved. Maxwell remains Lumerical
FDTD; heat and weighting-potential equations remain the repository's custom
CUDA solvers.

## Authorized topology variable

Rejecting semiconductor `np density` does not reject density topology. The
authorized topology field is the repository's own dimensionless projected
occupancy:

```text
latent rho
  -> 500-nm filter and tanh projection
  -> 81x81 projected nodal occupancy rho_bar in [0,1]
  -> nonlinear n-k optical relaxation in Lumerical
  -> exact four-node cell map into custom thermal/electrical constitutive laws
  -> final exact 0/1 mask and ordinary dispersive-Au reevaluation
```

`rho_bar` is a numerical relaxation and must never be relabeled as electron or
hole density. The selected optical map is the published Christiansen
`n-k`-then-square interpolation in `au_density_relaxation.py`, not a diluted
Drude plasma and not `rho**3`. The unresolved gate is its Au-specific B200
4-um endpoint parity, quantified source-band error, resonance sweep,
component-Yee Jacobian, and
combined AD-FD validation. See `LUMERICAL_MAXWELL_GPU_PDE_ROUTE.md`.
