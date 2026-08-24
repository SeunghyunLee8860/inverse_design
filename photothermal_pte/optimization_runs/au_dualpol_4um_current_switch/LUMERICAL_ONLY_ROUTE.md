# Lumerical-only Au inverse-design route

Status: `BLOCKED_UNTIL_B200_AND_DEVICE_CONTRACT_ARE_AVAILABLE`

## Decision

The new production route uses Ansys Lumerical v261 for all three field
solvers:

- optical: 3-D Lumerical FDTD on an NVIDIA B200;
- thermal: Lumerical Multiphysics HEAT;
- electrical/weighting: Lumerical Multiphysics CHARGE.

HEAT and CHARGE are CPU solvers in this Lumerical release.  The B200
requirement applies to FDTD time stepping; Lumerical meshing, scripts, HEAT,
and CHARGE do not become GPU solvers by running on a B200 host.

FDTDX and the old JAX Maxwell gradient are excluded from this route.  The old
files remain in Git only as historical diagnostics and cannot certify or
launch a new production result.

## Why the bundled topology adjoint is not used for Au

This is a material-model limitation, not a claim that Lumerical cannot solve
Maxwell fields around gold.  Exact scalar dispersive Au has already produced
stable v261 GPU forward solutions.  The failure is the *generic differentiable
topology representation*:

1. Legacy `lumopt.TopologyOptimization2D` accepts scalar `eps_min/eps_max`,
   forms a real `dF_dEps`, and is documented with dielectric examples.
2. Installed `lumopt2.Topology` types its material indices as real floats.
3. Installed `lumopt2` takes `real(index**2)`, takes the real part of the
   sparse epsilon difference, clips negative real epsilon, and fits a lossless
   Cauchy model under an `n >> k ~ 0` assumption.
4. At 4 um the frozen Au endpoint is
   `epsilon=-830.37+127.16i`; deleting either the negative real part or loss is
   not a small approximation.
5. Earlier same-step tests found that the moving/conformal Au volume-epsilon
   contraction missed independently re-solved Maxwell finite differences.

`21_audit_lumerical_only_preflight.py` checks the installed v261 source for
these assumptions.  Patching the Python wrapper to keep complex numbers would
not by itself create the missing discrete dispersive/conformal material
Jacobian, so such a patch is not accepted as validation.

## Physical design variable

The only field allowed into FDTD, HEAT, and CHARGE is one exact binary array

```text
m[i,j] in {0,1};  0 = void, 1 = 50 nm Au.
```

The shape, dtype, and bytes have one SHA-256.  Every solver artifact must name
that same hash.  Gray optical `rho**3` and thermal/electrical `rho` laws are
therefore removed from the new physical evaluation path rather than made
artificially equal.  Continuous or stochastic optimizer state may exist, but
it must be converted to a 500-nm-DFM-valid binary mask before *any* physics
solve.

The Lumerical-only optimizer will consequently use exact-binary candidate
evaluations (a gradient-free/stochastic search with paired candidates), not a
false Au adjoint.  Each candidate requires both `Ea` and `Eb` Maxwell solves,
followed by HEAT and CHARGE evaluations of the same geometry.  The signed
objective remains

```text
maximize min(+I(E||a), -I(E||b)).
```

## Fail-closed B200 launch

All Maxwell entry points must be launched through:

```bash
LUMERICAL_B200_GPU_INDEX=<physical-index> \
  ./run_lumerical_b200.sh <python-script> [arguments]
```

The launcher checks the requested physical device with `nvidia-smi`, requires
its reported name to contain `B200`, records the physical index, and refuses
to start otherwise.  The completed FDTD artifact must additionally prove GPU
time stepping from the Lumerical engine log; the preflight alone is not a run
certificate.

The current host `aigpu1123` exposes eight RTX 6000 Ada GPUs and no B200.
Accordingly, this session may build and test layouts but must not generate a
claimed B200 Maxwell result here.

## Gates before optimization

1. Confirm the experimental flake outline/thickness, contact polygons,
   crystal angle, Au electrical role, stack, beam, and interface parameters.
2. Build and audit exact-binary Lumerical layouts for empty, full-Au, and
   nonuniform masks.
3. On B200, close source calibration, time stationarity, native-Yee absorbed
   power, and six-face energy balance for both polarizations.
4. Converge the complete x/y/z/PML mesh, including Au, TaIrTe4, SiO2, Si,
   surrounding air, and every geometry edge.
5. Transfer one mask and optical heat map into Lumerical HEAT; converge the
   unstructured mesh and energy balance.
6. Transfer the same mask into Lumerical CHARGE; solve the device-specific
   weighting problem and close terminal-current balance.
7. Validate the exact-binary search estimator with repeated seeds and direct
   candidate reevaluation before a long optimization.

