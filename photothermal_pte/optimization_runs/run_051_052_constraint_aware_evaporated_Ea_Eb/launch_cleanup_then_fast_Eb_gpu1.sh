#!/usr/bin/env bash
set -euo pipefail

export PATH="/home/eidl/miniconda3/envs/EIDL-Lumapi/bin:${PATH}"
export LD_LIBRARY_PATH="/home/eidl/miniconda3/envs/EIDL-Lumapi/lib"
export RUN051_CLEANUP_GPU=1

cd /home/seunghyun/tairte4/worktrees/pte_true_mma
exec /home/dhkim/bin/runres \
  --reserve-count 9 \
  --reserve-wait 1800 \
  --reserve-tag run051_cleanup_fast_run052_Eb_gpu1 \
  photothermal_pte/optimization_runs/run_051_052_constraint_aware_evaporated_Ea_Eb/run_cleanup_then_fast_Eb.py \
  -th 8 -GPU 1 \
  > /data/seunghyun/tairte4/artifacts/tairte4_contact_anchored/run051_cleanup_fast_run052_Eb_gpu1.log \
  2>&1
