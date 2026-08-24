#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repository="$(git -C "$script_dir" rev-parse --show-toplevel)"
python_bin="${AU_LUMERICAL_PYTHON:-/home/eidl/miniconda3/envs/EIDL-Lumapi/bin/python}"
gpu_index="${LUMERICAL_B200_GPU_INDEX:?set LUMERICAL_B200_GPU_INDEX to the physical NVIDIA B200 index}"

if [[ ! -x "$python_bin" ]]; then
  echo "missing executable Python: $python_bin" >&2
  exit 2
fi

export PYTHONPATH="/home/seunghyun/lumerical_r12/opt/lumerical/v261/api/python:$repository${PYTHONPATH:+:$PYTHONPATH}"
export VC_LUMERICAL_ROOT="/home/seunghyun/lumerical_r12/opt/lumerical/v261"
export LUMERICAL_ROOT="$VC_LUMERICAL_ROOT"

"$python_bin" \
  "$script_dir/21_audit_lumerical_maxwell_preflight.py" \
  --gpu-index "$gpu_index" --require-ready

export CUDA_VISIBLE_DEVICES="$gpu_index"
export LUMERICAL_SESSION_GPU_DEVICE="GPU $gpu_index"
exec "$python_bin" "$@"
