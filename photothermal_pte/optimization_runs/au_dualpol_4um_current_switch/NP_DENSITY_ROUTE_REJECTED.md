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

## Authorized Au representation

Every physical evaluation must use:

```text
continuous shape/level-set optimizer parameters
  -> geometry and 500-nm DFM mapping
  -> exact binary Au mask m in {0,1}
  -> ordinary exact dispersive-Au geometry in Lumerical FDTD
  -> the identical binary geometry in custom thermal/electrical solvers
```

Continuous parameters may move an exact material boundary. They may not be
interpreted as a gray Au fraction, diluted Drude plasma, electron/hole
density, or different optical/thermal/electrical powers of one density.

The unresolved research problem is therefore the derivative or search method
for exact-binary geometry. The next gate is a 4-um exact-Au shape/level-set
AD-FD experiment in Lumerical. If a trustworthy shape derivative cannot be
certified, use an exact-geometry derivative-free or stochastic estimator
(for example central finite differences for a compact parameterization or
SPSA with independent central-FD validation). In all cases, both perturbed
evaluations contain ordinary binary dispersive Au.
