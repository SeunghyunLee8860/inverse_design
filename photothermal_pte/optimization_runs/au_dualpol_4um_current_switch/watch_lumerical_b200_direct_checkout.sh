#!/usr/bin/env bash
set -uo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repository="$(git -C "$script_dir" rev-parse --show-toplevel)"
output_root="${1:?usage: $0 NEW_OUTPUT_ROOT B200_CALIBRATION_ROOT [RESTART_CHECKPOINT RESTART_MANIFEST]}"
calibration_root="${2:?usage: $0 NEW_OUTPUT_ROOT B200_CALIBRATION_ROOT [RESTART_CHECKPOINT RESTART_MANIFEST]}"
restart_checkpoint="${3:-}"
restart_manifest="${4:-}"
if [[ -n "$restart_checkpoint" || -n "$restart_manifest" ]]; then
  if [[ -z "$restart_checkpoint" || -z "$restart_manifest" ]]; then
    echo "restart checkpoint and manifest must be provided together" >&2
    exit 2
  fi
  export AU_LUMERICAL_RESTART_CHECKPOINT="$restart_checkpoint"
  export AU_LUMERICAL_RESTART_MANIFEST="$restart_manifest"
else
  unset AU_LUMERICAL_RESTART_CHECKPOINT
  unset AU_LUMERICAL_RESTART_MANIFEST
fi
gpu_index="${LUMERICAL_B200_GPU_INDEX:?set the physical B200 GPU index}"
uniform_launcher="$script_dir/launch_lumerical_b200_uniform_rho0p5.sh"
resume_launcher="$script_dir/run_lumerical_b200.sh"
optimizer_script="$script_dir/41_optimize_lumerical_4um_dualpol_continuation.py"
manifest="$output_root/production_manifest.json"
event_log="${output_root}.watchdog_events.log"
console_log="${output_root}.watchdog_console.log"
lock_file="${output_root}.watchdog.lock"
coarse_label="fine_z2p5_bulk50_xy100_cv0_pml8_span20_z6_t1ps"
fine_label="fine_z2p5_bulk50_xy50_cv0_pml8_span20_z6_t1ps"

exec 9>"$lock_file"
if ! flock -n 9; then
  exit 0
fi

export ANSYSLMD_LICENSE_FILE="${ANSYSLMD_LICENSE_FILE:-1055@166.104.112.74}"
export AU_LUMERICAL_PYTHON="${AU_LUMERICAL_PYTHON:-/home/seunghyun200/.venvs/thermal-cu130-py312/bin/python}"
export AU_LUMERICAL_ROOT="${AU_LUMERICAL_ROOT:-/home/eidl/lumerical/r12/v261}"
export EIDL_LUMAPI_ROOT="${EIDL_LUMAPI_ROOT:-/home/eidl/EIDL-Lumapi}"
export AU_LUMERICAL_OPT_OUTPUT_ROOT="$output_root"
export AU_LUMERICAL_EA_SOURCE_CALIBRATION="$calibration_root/xy100/Ea/source_only_Ea_${coarse_label}.json"
export AU_LUMERICAL_EB_SOURCE_CALIBRATION="$calibration_root/xy100/Eb/source_only_Eb_${coarse_label}.json"
export AU_LUMERICAL_EA_FINAL_XY50_SOURCE_CALIBRATION="$calibration_root/xy50/Ea/source_only_Ea_${fine_label}.json"
export AU_LUMERICAL_EB_FINAL_XY50_SOURCE_CALIBRATION="$calibration_root/xy50/Eb/source_only_Eb_${fine_label}.json"
export AU_LUMERICAL_ACCELERATOR_POLICY="b200"
export AU_LUMERICAL_OPT_BETA="1"
export AU_LUMERICAL_LICENSE_MODE="direct_checkout"
export AU_LUMERICAL_LICENSE_AUDIT_WAIT_S="${AU_LUMERICAL_LICENSE_AUDIT_WAIT_S:-1800}"
export AU_LUMERICAL_LICENSE_AUDIT_POLL_S="${AU_LUMERICAL_LICENSE_AUDIT_POLL_S:-5}"
export LUMERICAL_GPU_INDEX="$gpu_index"
export CUDA_VISIBLE_DEVICES="$gpu_index"
export LUMERICAL_SESSION_GPU_DEVICE="GPU $gpu_index"
export FDTD_THREADS="${FDTD_THREADS:-8}"
export OMP_NUM_THREADS="$FDTD_THREADS"
export MKL_NUM_THREADS="$FDTD_THREADS"
export OPENBLAS_NUM_THREADS="$FDTD_THREADS"
export NUMEXPR_NUM_THREADS="$FDTD_THREADS"
export XDG_CONFIG_HOME="$output_root/.xdg_config"
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-offscreen}"
timestamp() {
  date --iso-8601=seconds
}

final_certified() {
  jq -e '(.passed == true) and (.final != null)' "$manifest" >/dev/null 2>&1
}

transient_license_failure() {
  local evidence
  evidence="$(tail -n 200 "$console_log" 2>/dev/null)"
  case "$evidence" in
    *"could not match resource name provided or the resource may not be active"* | \
    *"FlexNet Licensing error:-4,132"* | \
    *"Licensed number of users already reached"* | \
    *"Insufficient FlexNet Publisher"*)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

run_once() {
  if [[ ! -e "$output_root" && -z "$restart_checkpoint" ]]; then
    "$uniform_launcher" "$output_root" "$calibration_root"
  else
    "$resume_launcher" "$optimizer_script"
  fi
}

printf '%s watchdog_started policy=direct_checkout_license_retry gpu=%s commit=%s\n' \
  "$(timestamp)" "$gpu_index" "$(git -C "$repository" rev-parse HEAD)" >>"$event_log"

while true; do
  printf '%s optimizer_start output=%s\n' "$(timestamp)" "$output_root" >>"$event_log"
  : >"$console_log"
  if run_once >>"$console_log" 2>&1; then
    rc=0
  else
    rc=$?
  fi
  printf '%s optimizer_exit rc=%s\n' "$(timestamp)" "$rc" >>"$event_log"

  if final_certified; then
    printf '%s watchdog_exit final_certified=true\n' "$(timestamp)" >>"$event_log"
    exit 0
  fi
  if transient_license_failure; then
    printf '%s transient_license_failure retry_after_s=60\n' "$(timestamp)" >>"$event_log"
    sleep 60
    continue
  fi

  status="$(jq -r '.status // "NO_MANIFEST"' "$manifest" 2>/dev/null)"
  printf '%s watchdog_stop non_license_failure rc=%s status=%s\n' \
    "$(timestamp)" "$rc" "$status" >>"$event_log"
  if [[ "$rc" -eq 0 ]]; then
    exit 2
  fi
  exit "$rc"
done
