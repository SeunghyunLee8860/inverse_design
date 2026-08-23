#!/usr/bin/env bash
set -euo pipefail
cd /home/seunghyun/tairte4/worktrees/au_dualpol_4um
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export THERMAL_CUDA_DEVICE=0
exec photothermal_pte/optimization_runs/au_dualpol_4um_current_switch/run_combined_gpu_python.sh -u \
  photothermal_pte/optimization_runs/au_dualpol_4um_current_switch/15_validate_4um_z_mesh_convergence.py "$@"
