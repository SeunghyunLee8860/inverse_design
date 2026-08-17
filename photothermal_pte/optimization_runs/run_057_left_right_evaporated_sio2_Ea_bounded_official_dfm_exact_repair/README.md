# Run057 — left/right electrodes, evaporated SiO2, E || a

Fresh optimization from uniform latent density 0.5 using the bounded official
Ansys-DFM plus exact-repair production driver.

- Lumerical axes: `x=b`, `y=a`, `z=c`
- electrical terminals: left/right (`contact_axis=x`)
- illumination: `E || a` (Lumerical y polarization)
- TaIrTe4/SiO2 interface scenario: evaporated, `G=7.37e4 W/(m^2 K)`
- minimum solid and void feature: 500 nm
- optimizer: NLopt LD_MMA
- continuation: beta 1, 2, 4, 8, 16, 32, 64, 128
- final exact-binary repair and fresh full-physics validation
- raw FSP/NPZ artifacts remain outside Git under `/data/seunghyun`

Launch on GPU 0:

```bash
RUN057_GPU=0 /home/eidl/miniconda3/envs/EIDL-Lumapi/bin/python -u run.py
```
