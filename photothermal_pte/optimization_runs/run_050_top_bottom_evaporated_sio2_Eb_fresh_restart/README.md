# Run050 — top/bottom electrodes, evaporated-SiO2 interface, E||b

Fresh uniform-density optimization using the same explicit 3D optical,
thermal, and electrical geometry as Run049. The only physical source change is
the incident polarization from `E||a` to `E||b`.

- Lumerical axes: `x=b`, `y=a`, `z=c`
- terminals: bottom `psi=0`, top `psi=1`
- TaIrTe4/SiO2 interface: evaporated scenario, `G=7.37e4 W/(m2 K)`
- bulk SiO2/Si geometry remains explicit and unchanged
- initial latent density: uniform `0.5`
- optimizer: NLopt LD_MMA with beta continuation and 500 nm solid/void DFM
- exact cleanup is accepted only if its fresh binary objective changes by no
  more than 1%; a failed cleanup is retained and continuation proceeds

The pipeline first requires a new combined physical-density AD-FD certificate
for `E||b` under the evaporated-interface contract. It is launched through
`runres` so nine Lumerical licenses remain reserved for the complete job.
