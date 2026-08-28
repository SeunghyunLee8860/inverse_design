#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repository="$(git -C "$script_dir" rev-parse --show-toplevel)"
output_root="${1:?usage: $0 NEW_UNIFORM_OUTPUT_ROOT B200_CALIBRATION_ROOT}"
calibration_root="${2:?usage: $0 NEW_UNIFORM_OUTPUT_ROOT B200_CALIBRATION_ROOT}"
gpu_index="${LUMERICAL_B200_GPU_INDEX:?set the physical B200 GPU index}"
python_bin="${AU_LUMERICAL_PYTHON:?set the absolute Lumerical Python path}"
lumerical_root="${AU_LUMERICAL_ROOT:?set the absolute Lumerical v261 root}"
license_mode="direct_checkout"
threads="${FDTD_THREADS:-8}"
coarse_label="fine_z2p5_bulk50_xy100_cv0_pml8_span20_z6_t1ps"
fine_label="fine_z2p5_bulk50_xy50_cv0_pml8_span20_z6_t1ps"


if [[ -e "$output_root" ]]; then
  echo "refusing existing production output root: $output_root" >&2
  exit 2
fi
for required in \
  "$python_bin" \
  "$lumerical_root/api/python/lumapi.py"; do
  if [[ ! -e "$required" ]]; then
    echo "missing B200 continuation prerequisite: $required" >&2
    exit 2
  fi
done

export AU_LUMERICAL_EA_SOURCE_CALIBRATION="$calibration_root/xy100/Ea/source_only_Ea_${coarse_label}.json"
export AU_LUMERICAL_EB_SOURCE_CALIBRATION="$calibration_root/xy100/Eb/source_only_Eb_${coarse_label}.json"
export AU_LUMERICAL_EA_FINAL_XY50_SOURCE_CALIBRATION="$calibration_root/xy50/Ea/source_only_Ea_${fine_label}.json"
export AU_LUMERICAL_EB_FINAL_XY50_SOURCE_CALIBRATION="$calibration_root/xy50/Eb/source_only_Eb_${fine_label}.json"
for calibration in \
  "$AU_LUMERICAL_EA_SOURCE_CALIBRATION" \
  "$AU_LUMERICAL_EB_SOURCE_CALIBRATION" \
  "$AU_LUMERICAL_EA_FINAL_XY50_SOURCE_CALIBRATION" \
  "$AU_LUMERICAL_EB_FINAL_XY50_SOURCE_CALIBRATION"; do
  if [[ ! -f "$calibration" ]]; then
    echo "missing B200 source calibration: $calibration" >&2
    exit 2
  fi
done

export PATH="$(dirname -- "$python_bin"):${PATH}"
export PYTHONPATH="${EIDL_LUMAPI_ROOT:-/home/eidl/EIDL-Lumapi}:$lumerical_root/api/python:$repository${PYTHONPATH:+:$PYTHONPATH}"
export XDG_CONFIG_HOME="${AU_LUMERICAL_XDG_CONFIG_HOME:-$output_root/.xdg_config}"
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-offscreen}"
export OMP_NUM_THREADS="$threads"
export MKL_NUM_THREADS="$threads"
export OPENBLAS_NUM_THREADS="$threads"
export NUMEXPR_NUM_THREADS="$threads"
export VC_LUMERICAL_ROOT="$lumerical_root"
export LUMERICAL_ROOT="$lumerical_root"
export AU_LUMERICAL_ROOT="$lumerical_root"
export AU_LUMERICAL_PYTHON="$python_bin"
export AU_LUMERICAL_ACCELERATOR_POLICY="b200"
export AU_LUMERICAL_OPT_BETA="1"
export AU_LUMERICAL_OPT_OUTPUT_ROOT="$output_root"
unset AU_LUMERICAL_RESTART_CHECKPOINT
unset AU_LUMERICAL_RESTART_MANIFEST
export LUMERICAL_GPU_INDEX="$gpu_index"
export LUMERICAL_B200_GPU_INDEX="$gpu_index"
export CUDA_VISIBLE_DEVICES="$gpu_index"
export LUMERICAL_SESSION_GPU_DEVICE="GPU $gpu_index"
export AU_LUMERICAL_LICENSE_MODE="$license_mode"
: "${ANSYSLMD_LICENSE_FILE:?set the B200 direct-checkout endpoint; do not source the runres reservation environment}"
export FDTD_THREADS="$threads"
export AU_LUMERICAL_LICENSE_AUDIT_WAIT_S="${AU_LUMERICAL_LICENSE_AUDIT_WAIT_S:-1800}"
export AU_LUMERICAL_LICENSE_AUDIT_POLL_S="${AU_LUMERICAL_LICENSE_AUDIT_POLL_S:-5}"

# Solver-free: validates the B200 UUID/build source records and writes a fresh
# beta-1 checkpoint whose production code requires exact uniform latent rho=0.5.
"$script_dir/run_lumerical_b200.sh" \
  "$script_dir/41_optimize_lumerical_4um_dualpol_continuation.py" \
  --preflight-only

cd "$repository"
exec "$python_bin" -u \
  "$script_dir/41_optimize_lumerical_4um_dualpol_continuation.py"
