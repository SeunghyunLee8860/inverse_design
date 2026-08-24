# Lumerical Maxwell + custom GPU-PDE Au inverse-design route

Status: `BLOCKED_PENDING_LUMERICAL_AU_AD_FD_AND_B200`

## Corrected solver architecture

The production architecture is the same split already used by the earlier
TaIrTe4-flake topology optimization:

- optical forward/adjoint fields and native-Yee absorbed power: Ansys
  Lumerical FDTD v261, with production runs required on an NVIDIA B200;
- steady thermal equation: the repository's custom CUDA finite-volume sparse
  solver;
- weighting potential, Shockley-Ramo PTE current, and electrical adjoint: the
  repository's custom CUDA finite-element sparse solver.

No Lumerical HEAT or CHARGE license is assumed. The phrase "use Lumerical" in
this route means that Lumerical replaces FDTDX for Maxwell; it does not replace
the validated custom thermal/electrical PDE implementations.

The reference implementation of this split is
`tairte4_flake_topology/evaluate_objective_gradient.py` on branch
`agent/pte-electrode-boundary-adjoint`: it calls Lumerical for one forward and
one adjoint Maxwell solve, maps native-Yee Q into the thermal mesh, uses
`PersistentCudaCSR` for thermal and electrical systems, and sums the optical,
thermal-material, and electrical-material gradient terms.

## Continuous topology variable versus exact binary Au

Inverse design still needs a differentiable continuous relaxation. The
correct state sequence is

```text
latent x
  -> one spatial density filter
  -> one tanh projection with beta continuation
  -> one shared physical Au fraction f_Au in [0,1]
  -> optical, thermal, and electrical material maps
  -> exact 0/1, 500-nm-DFM final mask
  -> exact dispersive-Au endpoint reevaluation
```

Therefore gray values are not categorically forbidden during optimization.
What is forbidden is silently giving different physical design states to the
three solvers. Each evaluation must record the shape and SHA-256 of the same
projected `f_Au` array used by all material maps. The old O3/TE1 path violated
that requirement because Maxwell effectively received `rho**3` while the
thermal/electrical models received `rho`.

The exact-binary condition applies to endpoint controls and final candidate
promotion. A topology optimizer that binarizes before every field solve would
remove the useful derivative almost everywhere and reduce the problem to an
impractical 6400-bit discrete search.

## What remains unresolved in Lumerical

Lumerical can forward-simulate exact dispersive Au. The unresolved issue is
how the continuous `f_Au` changes the lossy, negative-real dispersive Au
response while retaining a correct derivative.

The installed generic LumOpt topology paths are not accepted without a new
same-step AD-FD certificate:

1. legacy `lumopt.TopologyOptimization2D` is based on scalar endpoint
   permittivities and takes a real `dF/dEps`;
2. installed `lumopt2.Topology` types its material index as a real float;
3. installed `lumopt2` takes `real(index**2)`, takes the real part of the
   sparse epsilon difference, clips negative real epsilon, and fits a
   lossless Cauchy model under an `n >> k ~ 0` assumption;
4. these operations are incompatible with treating the 4-um Au endpoint
   (`epsilon=-830.37+127.16i` in the frozen dataset) as a small dielectric
   perturbation;
5. earlier repository controls found that fixed-geometry material
   differentiation could agree, while moving/conformal Au and some imported
   density carriers did not agree with independently re-solved finite
   differences.

This does **not** prove that Lumerical Au inverse design is impossible. It
means the next gate is an Au-specific, fixed-grid differentiable material
carrier using Lumerical forward/adjoint fields, with exact void and dispersive
Au endpoints. Its gradient may enter optimization only after central FD over
multiple step sizes agrees for the same objective and the same discretization.
No FDTDX result may substitute for that gate.

## Shared material-map rule

`f_Au` must be identical across all physics, but the material properties need
not be numerically identical functions because they have different units and
constitutive meanings. Each law must be explicit and differentiated through
the same `f_Au`:

```text
Maxwell:     dispersive constitutive carrier M_opt(f_Au)
thermal:     k(f_Au), interface conductance G(f_Au), heat capacity if transient
electrical:  sigma(f_Au), and sigma*S(f_Au)
```

Using `f_Au**3` only in Maxwell and `f_Au` in the other two is prohibited.
Using one `f_Au` with documented, endpoint-correct constitutive interpolation
laws is allowed, but the full chain derivative must pass combined AD-FD. The
currently committed shared-linear law is only a provisional consistency
baseline; it is not yet a physical or mesh-converged certificate.

## B200 launch and current-host limitation

Maxwell entry points are launched through:

```bash
LUMERICAL_B200_GPU_INDEX=<physical-index> \
  ./run_lumerical_b200.sh <python-script> [arguments]
```

The launcher verifies that the requested physical GPU is reported as a B200.
The current host exposes RTX 6000 Ada GPUs, so this session can audit code and
prepare layouts but cannot honestly produce a B200 execution certificate.
Thermal/electrical GPU tests on another device are development tests only.

## Gates before optimization

1. Confirm the experimental flake outline/thickness, contact polygons,
   crystal axes, stack, beam, Au electrical role, and interface parameters.
2. Reproduce the earlier Lumerical-forward + custom-CUDA-PDE data path for the
   fixed device before changing the Au design representation.
3. Establish a Lumerical fixed-grid continuous Au carrier with exact endpoints
   and pass same-step optical AD-FD over several step sizes.
4. Pass exact empty/full/nonuniform binary endpoint controls in Lumerical.
5. Connect one filtered/projected `f_Au` to all material maps and pass thermal,
   electrical, and combined latent-variable AD-FD.
6. On B200, converge time stationarity, native-Yee Q, six-face energy balance,
   and the complete x/y/z/PML mesh for both polarizations.
7. Run LD_MMA continuation, finalize a 500-nm-DFM exact binary mask, and
   independently reevaluate both polarizations with exact dispersive Au.
