#!/usr/bin/env bash
set -uo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
watchdog="$script_dir/watch_lumerical_b200_direct_checkout.sh"
main_session="${1:?usage: $0 MAIN_TMUX_SESSION OUTPUT_ROOT CALIBRATION_ROOT GPU_INDEX [RESTART_CHECKPOINT RESTART_MANIFEST]}"
output_root="${2:?usage: $0 MAIN_TMUX_SESSION OUTPUT_ROOT CALIBRATION_ROOT GPU_INDEX [RESTART_CHECKPOINT RESTART_MANIFEST]}"
calibration_root="${3:?usage: $0 MAIN_TMUX_SESSION OUTPUT_ROOT CALIBRATION_ROOT GPU_INDEX [RESTART_CHECKPOINT RESTART_MANIFEST]}"
gpu_index="${4:?usage: $0 MAIN_TMUX_SESSION OUTPUT_ROOT CALIBRATION_ROOT GPU_INDEX [RESTART_CHECKPOINT RESTART_MANIFEST]}"
restart_checkpoint="${5:-}"
restart_manifest="${6:-}"
manifest="$output_root/production_manifest.json"
event_log="${output_root}.supervisor_events.log"

timestamp() {
  date --iso-8601=seconds
}

manifest_status() {
  jq -r '.status // "NO_MANIFEST"' "$manifest" 2>/dev/null
}

final_certified() {
  jq -e '(.passed == true) and (.final != null)' "$manifest" >/dev/null 2>&1
}

launch_main_session() {
  tmux new-session -d -s "$main_session" \
    env \
    ANSYSLMD_LICENSE_FILE=1055@166.104.112.74 \
    AU_LUMERICAL_PYTHON=/home/seunghyun200/.venvs/thermal-cu130-py312/bin/python \
    AU_LUMERICAL_ROOT=/home/eidl/lumerical/r12/v261 \
    EIDL_LUMAPI_ROOT=/home/eidl/EIDL-Lumapi \
    LUMERICAL_B200_GPU_INDEX="$gpu_index" \
    FDTD_THREADS=8 \
    "$watchdog" \
    "$output_root" \
    "$calibration_root" \
    "$restart_checkpoint" \
    "$restart_manifest"
}

printf '%s supervisor_started main_session=%s gpu=%s\n' \
  "$(timestamp)" "$main_session" "$gpu_index" >>"$event_log"

while true; do
  if final_certified; then
    printf '%s supervisor_exit final_certified=true\n' "$(timestamp)" >>"$event_log"
    exit 0
  fi

  if tmux has-session -t "$main_session" 2>/dev/null; then
    sleep 30
    continue
  fi

  status="$(manifest_status)"
  case "$status" in
    RUNNING_*)
      printf '%s main_session_missing status=%s action=restart\n' \
        "$(timestamp)" "$status" >>"$event_log"
      if launch_main_session; then
        printf '%s main_session_restarted status=%s\n' \
          "$(timestamp)" "$status" >>"$event_log"
      else
        printf '%s main_session_restart_failed status=%s retry_after_s=30\n' \
          "$(timestamp)" "$status" >>"$event_log"
      fi
      ;;
    *)
      printf '%s supervisor_stop non_running_status=%s\n' \
        "$(timestamp)" "$status" >>"$event_log"
      exit 2
      ;;
  esac
  sleep 30
done
