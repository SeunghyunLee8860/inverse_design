#!/usr/bin/env bash
set -euo pipefail

# Isolated Python 3.12 environment: EIDL PyTorch CUDA + pinned FDTDX/JAX.
# The venv-specific cuDNN must precede the inherited EIDL CUDA libraries so
# JAX and PyTorch can coexist in the same process.
export LD_LIBRARY_PATH="/home/seunghyun/.venvs/fdtdx-thermal-py312/lib/python3.12/site-packages/nvidia/cudnn/lib:/home/seunghyun/.venvs/fdtdx-thermal-py312/lib/python3.12/site-packages/nvidia/nvshmem/lib:/home/eidl/miniconda3/envs/EIDL-Lumapi/lib/python3.12/site-packages/nvidia/cublas/lib:/home/eidl/miniconda3/envs/EIDL-Lumapi/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/home/eidl/miniconda3/envs/EIDL-Lumapi/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib:/home/eidl/miniconda3/envs/EIDL-Lumapi/lib/python3.12/site-packages/nvidia/cuda_cupti/lib:/home/eidl/miniconda3/envs/EIDL-Lumapi/lib/python3.12/site-packages/nvidia/cufft/lib:/home/eidl/miniconda3/envs/EIDL-Lumapi/lib/python3.12/site-packages/nvidia/curand/lib:/home/eidl/miniconda3/envs/EIDL-Lumapi/lib/python3.12/site-packages/nvidia/cusolver/lib:/home/eidl/miniconda3/envs/EIDL-Lumapi/lib/python3.12/site-packages/nvidia/cusparse/lib:/home/eidl/miniconda3/envs/EIDL-Lumapi/lib/python3.12/site-packages/nvidia/nvjitlink/lib:/home/eidl/miniconda3/envs/EIDL-Lumapi/lib/python3.12/site-packages/nvidia/nccl/lib:/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH="/home/seunghyun/.local/fdtdx_main_src/src:${PYTHONPATH:-.}"

exec /home/seunghyun/.venvs/fdtdx-thermal-py312/bin/python "$@"
