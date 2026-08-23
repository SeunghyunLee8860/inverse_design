#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repository="$(git -C "$script_dir" rev-parse --show-toplevel)"
gpu="${AU_DUALPOL_GPU:-${1:-0}}"
cd "$repository"
export PYTHONPATH="$repository"
export CUDA_VISIBLE_DEVICES="$gpu"
export THERMAL_CUDA_DEVICE=0
export MPLCONFIGDIR="/tmp/seunghyun_au_dualpol_exact_matplotlib"
exec photothermal_pte/optimization_runs/au_dualpol_4um_current_switch/run_combined_gpu_python.sh -u \
  photothermal_pte/optimization_runs/au_dualpol_4um_current_switch/12_optimize_exact_binary_au_topology.py
