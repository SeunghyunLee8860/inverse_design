#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repository="$(git -C "$script_dir" rev-parse --show-toplevel)"
python_bin="${FDTDX_FRESH_PYTHON:-/home/seunghyun200/.venvs/fdtdx-fresh-py312/bin/python}"
source_dir="${FDTDX_SOURCE_DIR:-/home/seunghyun200/dependencies/fdtdx-f26f84b70a8cceec9b889553955a868624736bf1}"
gpu_index="${FDTDX_FRESH_GPU_INDEX:?set FDTDX_FRESH_GPU_INDEX to one idle physical NVIDIA GPU index}"
output_dir="${FDTDX_FRESH_OUTPUT_DIR:?set FDTDX_FRESH_OUTPUT_DIR to a new absolute raw-result directory}"

if [[ ! -x "$python_bin" ]]; then
  echo "missing executable Python: $python_bin" >&2
  exit 2
fi
if [[ ! "$gpu_index" =~ ^[0-9]+$ ]]; then
  echo "FDTDX_FRESH_GPU_INDEX must be one non-negative physical GPU index" >&2
  exit 2
fi
if [[ "$output_dir" != /* || ! -d "$output_dir" || ! -w "$output_dir" ]]; then
  echo "FDTDX_FRESH_OUTPUT_DIR must be an existing writable absolute directory" >&2
  exit 2
fi
shopt -s nullglob dotglob
output_entries=("$output_dir"/*)
shopt -u nullglob dotglob
if (( ${#output_entries[@]} != 0 )); then
  echo "FDTDX_FRESH_OUTPUT_DIR must be empty to prevent result overwrite" >&2
  exit 2
fi
if [[ $# -eq 0 ]]; then
  echo "pass a Python script or -m module after this wrapper" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="$gpu_index"
export FDTDX_FRESH_OUTPUT_DIR="$output_dir"
export FDTDX_SOURCE_DIR="$source_dir"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$source_dir/src:$repository${PYTHONPATH:+:$PYTHONPATH}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

"$python_bin" -m \
  photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_runtime_preflight \
  --source "$source_dir" --gpu-index "$gpu_index" --require-ready

exec "$python_bin" "$@"
