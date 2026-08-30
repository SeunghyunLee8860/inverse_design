# Binary Au representation control

Status: `FAILED_DENSITY_ROUTE_UNIFORM_AU_IMPORTNK2_DIVERGENCE_FALLBACK_SHARP_INTERFACE`

This is an isolated material-representation control, not a fixed-TaIrTe4
device prediction.

## Stable scalar-Au reference

The 20 x 20 x 0.05 um exact scalar `(n,k)` Au film completed on GPU with:

- `P_Q = 1.529552448066e-15 W`
- `P_six = 1.531206319772e-15 W`
- six-face closure = `0.108011%`
- all component epsilon medians equal the requested Ordal endpoint
- no Q clipping, smoothing, gain, or rescaling

The earlier 10 x 10 um control had approximately 0.9% relative closure because
Au absorption was a small difference of large incident/reflected fluxes.
Changing `dz=5 -> 2.5 nm` left that closure essentially unchanged. Increasing
the finite control area raised the absorption signal and closed to 0.5%.

## Density/imported endpoint failure

With the same 20 x 20 x 0.05 um geometry and mesh, uniform `rho=1`
`importnk2` Au diverged after 1,919 FDTD iterations. The scalar material stayed
stable. CPU fallback was prohibited and no gray-density or AD-FD run followed.

Because the density representation fails already at the binary Au endpoint,
the approved option 2 is rejected for production. The next route is option 1:
binary scalar-Au geometry with a sharp-interface level-set/shape derivative.
The failed FSP/log/JSON are preserved outside Git; their hashes are recorded.
