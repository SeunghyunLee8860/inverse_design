#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 GPU_INDEX Ea|Eb /absolute/empty/output_directory" >&2
  exit 2
fi

gpu_index="$1"
polarization="$2"
output_directory="$3"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repository="$(git -C "$script_dir" rev-parse --show-toplevel)"
python_bin="${FDTDX_INCREMENT_PYTHON:-/home/seunghyun200/.venvs/fdtdx-fresh-py312/bin/python}"
source_dir="${FDTDX_INCREMENT_SOURCE_DIR:-/home/seunghyun200/dependencies/fdtdx-increment-state-05d8e9ba}"

if [[ ! "$gpu_index" =~ ^[0-9]+$ ]]; then
  echo "GPU_INDEX must be a non-negative physical NVIDIA GPU index" >&2
  exit 2
fi
if [[ "$polarization" != "Ea" && "$polarization" != "Eb" ]]; then
  echo "polarization must be Ea or Eb" >&2
  exit 2
fi
if [[ ! -x "$python_bin" ]]; then
  echo "missing executable Python: $python_bin" >&2
  exit 2
fi
if [[ ! -d "$source_dir" ]]; then
  echo "missing patched FDTDX source: $source_dir" >&2
  exit 2
fi
if [[ "$output_directory" != /* || ! -d "$output_directory" || ! -w "$output_directory" ]]; then
  echo "output directory must be existing, writable, and absolute" >&2
  exit 2
fi
shopt -s nullglob dotglob
output_entries=("$output_directory"/*)
shopt -u nullglob dotglob
if (( ${#output_entries[@]} != 0 )); then
  echo "output directory must be empty" >&2
  exit 2
fi

gpu_row="$(nvidia-smi --query-gpu=index,uuid,name --format=csv,noheader,nounits | awk -F ', ' -v gpu_id="$gpu_index" '$1 == gpu_id {print}')"
if [[ -z "$gpu_row" ]]; then
  echo "physical GPU index $gpu_index does not exist" >&2
  exit 2
fi
busy_rows="$(nvidia-smi -i "$gpu_index" --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader,nounits 2>/dev/null || true)"
if [[ -n "$busy_rows" ]]; then
  echo "refusing busy GPU $gpu_index; existing compute processes:" >&2
  echo "$busy_rows" >&2
  exit 3
fi

echo "using verified-idle GPU: $gpu_row" >&2
nvidia-smi -i "$gpu_index" --query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu --format=csv,noheader >&2

cd "$repository"
export CUDA_VISIBLE_DEVICES="$gpu_index"
export FDTDX_SOURCE_DIR="$source_dir"
export JAX_PLATFORMS=cuda
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$source_dir/src:$repository${PYTHONPATH:+:$PYTHONPATH}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

exec "$python_bin" -u \
  photothermal_pte/optimization_runs/au_dualpol_4um_current_switch/fdtdx_increment_state_exact_binary_control.py \
  --output-directory "$output_directory" \
  --source "$source_dir" \
  --polarization "$polarization"
