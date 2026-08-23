#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repository="$(git -C "$script_dir" rev-parse --show-toplevel)"
gpu="${AU_DUALPOL_GPU:-0}"
if [[ $# -gt 0 && "$1" != --* ]]; then
  gpu="$1"
  shift
fi
cd "$repository"
export PYTHONPATH="$repository${PYTHONPATH:+:$PYTHONPATH}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_VISIBLE_DEVICES="$gpu"
exec photothermal_pte/optimization_runs/au_dualpol_4um_current_switch/run_combined_gpu_python.sh -u \
  photothermal_pte/optimization_runs/au_dualpol_4um_current_switch/20_validate_4um_stable_time_contract.py "$@"
