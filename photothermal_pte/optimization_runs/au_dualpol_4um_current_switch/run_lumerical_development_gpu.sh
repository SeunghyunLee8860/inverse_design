#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repository="$(git -C "$script_dir" rev-parse --show-toplevel)"
python_bin="${AU_LUMERICAL_PYTHON:-/home/eidl/miniconda3/envs/EIDL-Lumapi/bin/python}"
# The system /opt tree is 2026 R1.0 build 4413.  Its lumapi/FieldRegion
# importdataset path is incompatible with this adjoint pipeline.  Keep the
# Python API, CAD, and fdtd-engine on the audited R1.2 build 4522 tree.
lumerical_root="${AU_LUMERICAL_ROOT:-/home/seunghyun/lumerical_r12/opt/lumerical/v261}"
gpu_index="${LUMERICAL_GPU_INDEX:?set LUMERICAL_GPU_INDEX to one physical NVIDIA GPU index}"

if [[ ! -x "$python_bin" ]]; then
  echo "missing executable Python: $python_bin" >&2
  exit 2
fi
if [[ ! -f "$lumerical_root/api/python/lumapi.py" ]]; then
  echo "missing Lumerical lumapi.py below: $lumerical_root" >&2
  exit 2
fi

export AU_LUMERICAL_ACCELERATOR_POLICY="development"
export AU_LUMERICAL_ROOT="$lumerical_root"
export PYTHONPATH="$lumerical_root/api/python:$repository${PYTHONPATH:+:$PYTHONPATH}"
export VC_LUMERICAL_ROOT="$lumerical_root"
export LUMERICAL_ROOT="$VC_LUMERICAL_ROOT"

"$python_bin" \
  "$script_dir/21_audit_lumerical_maxwell_preflight.py" \
  --gpu-index "$gpu_index" --accelerator-policy development --require-ready

export CUDA_VISIBLE_DEVICES="$gpu_index"
export LUMERICAL_SESSION_GPU_DEVICE="GPU $gpu_index"
exec "$python_bin" "$@"
