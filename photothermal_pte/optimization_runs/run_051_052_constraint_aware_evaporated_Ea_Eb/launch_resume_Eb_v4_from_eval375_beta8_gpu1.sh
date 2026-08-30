#!/usr/bin/env bash
set -euo pipefail

export PATH="/home/eidl/miniconda3/envs/EIDL-Lumapi/bin:${PATH}"
export LD_LIBRARY_PATH="/home/eidl/miniconda3/envs/EIDL-Lumapi/lib"

cd /home/seunghyun/tairte4/worktrees/pte_true_mma

exec /home/dhkim/bin/runres \
  --reserve-count 9 \
  --reserve-wait 1800 \
  --reserve-tag run052_v7_beta8_resume_eval375_gpu1 \
  photothermal_pte/optimization_runs/run_051_052_constraint_aware_evaporated_Ea_Eb/run_resume_Eb_v4_from_eval375_beta8.py \
  -th 8 -GPU 1
