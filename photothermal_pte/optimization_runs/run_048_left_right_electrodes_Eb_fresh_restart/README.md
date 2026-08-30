# Run048: left/right-electrode Eb fresh optimization

Run048 is queued behind Run047 and uses the identical left/right-electrode
geometry, material, optical, thermal, electrical, mapping, LD_MMA, beta
continuation, and 500 nm DFM contract.  The only physical change is the source
polarization:

- Run047: `E || a` (Lumerical `y`, source polarization angle 90 degrees)
- Run048: `E || b` (Lumerical `x`, source polarization angle 0 degrees)

The run starts from exact uniform physical density `rho=0.5`, `beta=1`, with a
new NLopt LD_MMA state.  It does not warm-start from Run047.

Before optimization, the pipeline runs a fail-closed uniform-`rho=0.5` combined
physical-density AD-FD certificate for `E || b`.  The component-Yee Jacobian is
reused because the density-to-material layout and geometry are unchanged; the
Maxwell forward/adjoint/FD solves are newly executed with `E || b`.

`queue_after_run047.py` waits for a successful Run047 `FINAL_RESULT.json`, then
starts this pipeline through `runres` with nine reserved licenses on GPU 5.

