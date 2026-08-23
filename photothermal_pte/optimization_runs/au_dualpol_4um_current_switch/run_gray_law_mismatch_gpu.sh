#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repository="$(git -C "$script_dir" rev-parse --show-toplevel)"
cd "$repository"
export PYTHONPATH="$repository${PYTHONPATH:+:$PYTHONPATH}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export THERMAL_CUDA_DEVICE=0
exec photothermal_pte/optimization_runs/au_dualpol_4um_current_switch/run_combined_gpu_python.sh -u \
  photothermal_pte/optimization_runs/au_dualpol_4um_current_switch/14_diagnose_gray_law_mismatch.py
