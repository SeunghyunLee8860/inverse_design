#!/usr/bin/env bash
set -euo pipefail

repository="/home/seunghyun/tairte4/worktrees/au_dualpol_4um"
python_bin="/home/seunghyun/.venvs/fdtdx-thermal-py312/bin/python"
gpu="${AU_DUALPOL_GPU:-0}"

cd "$repository"
export PYTHONPATH="$repository"
export CUDA_VISIBLE_DEVICES="$gpu"
export THERMAL_CUDA_DEVICE=0
export MPLCONFIGDIR="/tmp/seunghyun_au_dualpol_4um_matplotlib"
exec "$python_bin" -u \
  photothermal_pte/optimization_runs/au_dualpol_4um_current_switch/10_optimize_4um_dualpol_au_ld_mma.py
