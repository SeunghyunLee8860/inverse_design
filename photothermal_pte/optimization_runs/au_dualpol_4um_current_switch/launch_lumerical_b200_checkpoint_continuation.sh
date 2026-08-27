#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repository="$(git -C "$script_dir" rev-parse --show-toplevel)"
output_root="${1:?usage: $0 NEW_OUTPUT_ROOT B200_CALIBRATION_ROOT}"
calibration_root="${2:?usage: $0 NEW_OUTPUT_ROOT B200_CALIBRATION_ROOT}"
gpu_index="${LUMERICAL_B200_GPU_INDEX:?set the physical B200 GPU index}"
python_bin="${AU_LUMERICAL_PYTHON:?set the absolute Lumerical Python path}"
lumerical_root="${AU_LUMERICAL_ROOT:?set the absolute Lumerical v261 root}"
license_mode="${AU_LUMERICAL_LICENSE_MODE:-reservation_audit}"
runres_bin="${AU_RUNRES_BIN:-}"
reserve_module="${LUM_RESERVE_MODULE_DIR:-}"
threads="${FDTD_THREADS:-8}"
bundle="$script_dir/b200_migration"
coarse_label="fine_z2p5_bulk50_xy100_cv0_pml8_span20_z6_t1ps"
fine_label="fine_z2p5_bulk50_xy50_cv0_pml8_span20_z6_t1ps"

if [[ "$license_mode" != "reservation_audit" && "$license_mode" != "direct_checkout" ]]; then
  echo "invalid AU_LUMERICAL_LICENSE_MODE: $license_mode" >&2
  exit 2
fi
if [[ "$license_mode" == "reservation_audit" ]]; then
  : "${runres_bin:?set the absolute runres path}"
  : "${reserve_module:?set the directory containing lum_reserve.py}"
fi

if [[ -e "$output_root" ]]; then
  echo "refusing existing production output root: $output_root" >&2
  exit 2
fi
for required in \
  "$python_bin" \
  "$lumerical_root/api/python/lumapi.py" \
  "$bundle/continuation_checkpoint.npz" \
  "$bundle/terminal_stage_state.npz" \
  "$bundle/restart_manifest.json" \
  "$bundle/bundle_manifest.json"; do
  if [[ ! -e "$required" ]]; then
    echo "missing B200 continuation prerequisite: $required" >&2
    exit 2
  fi
done
if [[ "$license_mode" == "reservation_audit" ]]; then
  for required in "$runres_bin" "$reserve_module/lum_reserve.py"; do
    if [[ ! -e "$required" ]]; then
      echo "missing B200 reservation prerequisite: $required" >&2
      exit 2
    fi
  done
fi

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
export PYTHONPATH="$lumerical_root/api/python:$repository${PYTHONPATH:+:$PYTHONPATH}"
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
export AU_LUMERICAL_RESTART_CHECKPOINT="$bundle/continuation_checkpoint.npz"
export AU_LUMERICAL_RESTART_MANIFEST="$bundle/restart_manifest.json"
export LUMERICAL_GPU_INDEX="$gpu_index"
export LUMERICAL_B200_GPU_INDEX="$gpu_index"
export CUDA_VISIBLE_DEVICES="$gpu_index"
export LUMERICAL_SESSION_GPU_DEVICE="GPU $gpu_index"
export AU_LUMERICAL_LICENSE_MODE="$license_mode"
if [[ -n "$reserve_module" ]]; then
  export LUM_RESERVE_MODULE_DIR="$reserve_module"
fi
export FDTD_THREADS="$threads"
export AU_LUMERICAL_LICENSE_AUDIT_WAIT_S="${AU_LUMERICAL_LICENSE_AUDIT_WAIT_S:-1800}"
export AU_LUMERICAL_LICENSE_AUDIT_POLL_S="${AU_LUMERICAL_LICENSE_AUDIT_POLL_S:-5}"

# Solver-free, but validates the new B200 UUID/build source records and the
# portable checkpoint/state hashes before any reserved Maxwell job starts.
"$script_dir/run_lumerical_b200.sh" \
  "$script_dir/41_optimize_lumerical_4um_dualpol_continuation.py" \
  --preflight-only

cd "$repository"
if [[ "$license_mode" == "direct_checkout" ]]; then
  exec "$python_bin" -u \
    "$script_dir/41_optimize_lumerical_4um_dualpol_continuation.py"
fi
exec "$runres_bin" \
  --reserve-count 9 \
  --reserve-wait "${AU_LUMERICAL_RESERVE_WAIT_S:-1800}" \
  --reserve-tag au4um_lumerical_b200_checkpoint_continuation \
  "$script_dir/41_optimize_lumerical_4um_dualpol_continuation.py" \
  -th "$threads" -GPU "$gpu_index"
