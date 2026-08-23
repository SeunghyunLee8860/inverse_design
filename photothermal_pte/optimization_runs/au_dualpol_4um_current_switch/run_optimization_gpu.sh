#!/usr/bin/env bash
set -euo pipefail

repository="/home/seunghyun/tairte4/worktrees/au_dualpol_4um"
gpu="${AU_DUALPOL_GPU:-0}"

cd "$repository"
export PYTHONPATH="$repository"
export CUDA_VISIBLE_DEVICES="$gpu"
export THERMAL_CUDA_DEVICE=0
export MPLCONFIGDIR="/tmp/seunghyun_au_dualpol_4um_matplotlib"
exec photothermal_pte/optimization_runs/au_dualpol_4um_current_switch/run_combined_gpu_python.sh -u \
  photothermal_pte/optimization_runs/au_dualpol_4um_current_switch/10_optimize_4um_dualpol_au_ld_mma.py
