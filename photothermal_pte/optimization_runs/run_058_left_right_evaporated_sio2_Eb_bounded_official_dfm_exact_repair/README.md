# Run058 — left/right electrodes, evaporated SiO2, E || b

Polarization-paired counterpart to Run057. It starts from the same uniform
latent density 0.5 and changes only the source polarization from `E || a` to
`E || b`.

- Lumerical axes: `x=b`, `y=a`, `z=c`
- electrical terminals: left/right (`contact_axis=x`)
- illumination: `E || b` (Lumerical x polarization)
- TaIrTe4/SiO2 interface scenario: evaporated, `G=7.37e4 W/(m^2 K)`
- minimum solid and void feature: 500 nm
- optimizer: NLopt LD_MMA
- continuation: beta 1, 2, 4, 8, 16, 32, 64, 128
- final exact-binary repair and fresh full-physics validation
- raw FSP/NPZ artifacts remain outside Git under `/data/seunghyun`

Launch on GPU 0 in a persistent tmux session:

```bash
tmux new-session -d -s run058 \
  "cd /home/seunghyun/tairte4/worktrees/pte_true_mma && \
   exec env RUN058_GPU=0 /home/eidl/miniconda3/envs/EIDL-Lumapi/bin/python -u \
   photothermal_pte/optimization_runs/run_058_left_right_evaporated_sio2_Eb_bounded_official_dfm_exact_repair/run.py"
```
