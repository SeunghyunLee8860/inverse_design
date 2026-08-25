#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repository="$(git -C "$script_dir" rev-parse --show-toplevel)"
runres_bin="${AU_RUNRES_BIN:-/home/dhkim/bin/runres}"
run_bin="${MSOPT_RUN_CMD:-/home/eidl/miniconda3/envs/EIDL-Lumapi/bin/run}"
gpu_index="${LUMERICAL_GPU_INDEX:?set LUMERICAL_GPU_INDEX to the physical GPU index}"
threads="${FDTD_THREADS:-8}"
raw_root="${AU_LUMERICAL_OPT_OUTPUT_ROOT:?set AU_LUMERICAL_OPT_OUTPUT_ROOT to a new external directory}"
lifecycle_root="${AU_LUMERICAL_RUNRES_LOG_ROOT:-/home/seunghyun/tairte4/lumerical_runres}"

ea_source_default="/home/seunghyun/tairte4_raw_artifacts/au_dualpol_4um_lumerical_development/r12_gpu5_source_only_Ea_z2p5_bulk50_cv0_MCM6/source_only_Ea_fine_z2p5_bulk50_xy100_cv0_pml8_span20_z6_t1ps.json"
eb_source_default="/home/seunghyun/tairte4_raw_artifacts/au_dualpol_4um_lumerical_development/r12_gpu5_source_only_Eb_z2p5_bulk50_cv0_MCM6/source_only_Eb_fine_z2p5_bulk50_xy100_cv0_pml8_span20_z6_t1ps.json"

export AU_LUMERICAL_EA_SOURCE_CALIBRATION="${AU_LUMERICAL_EA_SOURCE_CALIBRATION:-$ea_source_default}"
export AU_LUMERICAL_EB_SOURCE_CALIBRATION="${AU_LUMERICAL_EB_SOURCE_CALIBRATION:-$eb_source_default}"
export AU_LUMERICAL_ACCELERATOR_POLICY="${AU_LUMERICAL_ACCELERATOR_POLICY:-development}"
export AU_LUMERICAL_ROOT="${AU_LUMERICAL_ROOT:-/home/seunghyun/lumerical_r12/opt/lumerical/v261}"
export AU_LUMERICAL_PYTHON="${AU_LUMERICAL_PYTHON:-/home/eidl/miniconda3/envs/EIDL-Lumapi/bin/python}"
export LUMERICAL_ROOT="$AU_LUMERICAL_ROOT"
export VC_LUMERICAL_ROOT="$AU_LUMERICAL_ROOT"
export LUMERICAL_PYTHONPATH="$AU_LUMERICAL_ROOT/api/python"
export LUMERICAL_BIN_DIR="$AU_LUMERICAL_ROOT/bin"
export PATH="$LUMERICAL_BIN_DIR:${PATH}"
export MSOPT_RUN_CMD="$run_bin"
export PYTHONUNBUFFERED=1

if [[ "$gpu_index" != "5" ]]; then
  echo "The default source calibrations are UUID-bound to physical GPU 5." >&2
  echo "Provide matching Ea/Eb source calibrations before selecting another GPU." >&2
  exit 2
fi
if [[ -e "$raw_root" ]]; then
  echo "Refusing an existing smoke output root: $raw_root" >&2
  exit 2
fi

cd "$repository"
exec "$runres_bin" \
  --reserve-count 9 \
  --reserve-wait "${AU_LUMERICAL_RESERVE_WAIT_S:-21600}" \
  --reserve-tag au4um_lumerical_dualpol_smoke \
  "$script_dir/40_optimize_lumerical_4um_dualpol_smoke.py" \
  -th "$threads" -GPU "$gpu_index" \
  --busy-gpu-util-threshold 0 \
  --tag au4um_lumerical_dualpol_smoke \
  --outdir "$lifecycle_root"
