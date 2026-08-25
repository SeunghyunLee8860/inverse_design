# Active Lumerical production run

Snapshot time: 2026-08-25 17:32 UTC. The external manifest below is the
authoritative live state; this committed file is only the launch handoff.

- Branch: `agent/optimize-au-dualpol-4um-pte`
- Immutable run commit: `790a5ade69307ed1cf7ac5a9cbf3f9011d3321dc`
- Detached run worktree:
  `/home/seunghyun/tairte4/worktrees/au_lumerical_continuation_790a5ade`
- tmux session: `au4um_lum_prod_790a5ade`
- GPU: physical GPU 5, RTX 6000 Ada,
  `GPU-aa047452-9c73-d10f-675f-8af800915acf`
- Reserved license project: nine `lum_fdtd_solve` tasks held by `runres`
- External output root:
  `/home/seunghyun/tairte4_raw_artifacts/au_dualpol_4um_lumerical_production/continuation_790a5ade6930`
- Live manifest:
  `.../continuation_790a5ade6930/production_manifest.json`
- Resume checkpoint:
  `.../continuation_790a5ade6930/continuation_checkpoint.npz`

At this snapshot the job was in the first beta-1, exact-uniform-rho=0.5 Ea
forward. No current or FOM had been produced yet. The GPU engine log proved
the requested GPU UUID and reported normal simulation progress. There was no
optimization error.

Monitor without changing the run:

```bash
tmux capture-pane -pt au4um_lum_prod_790a5ade -S -120
jq '{status,latest,stage_count:(.stages|length),error}' \
  /home/seunghyun/tairte4_raw_artifacts/au_dualpol_4um_lumerical_production/continuation_790a5ade6930/production_manifest.json
```

Do not run another copy against the same output root. If the process exits,
rerun the committed launcher from the same detached worktree; the driver will
accept only its same-commit checkpoint. Do not delete the `runres` parent
while a child is alive because the parent owns and releases the reservation.
