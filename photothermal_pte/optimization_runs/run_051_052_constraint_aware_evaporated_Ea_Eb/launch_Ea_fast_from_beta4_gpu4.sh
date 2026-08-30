#!/usr/bin/env bash
set -euo pipefail

export PATH="/home/eidl/miniconda3/envs/EIDL-Lumapi/bin:${PATH}"
export LD_LIBRARY_PATH="/home/eidl/miniconda3/envs/EIDL-Lumapi/lib"
export RUN053_EA_GPU=4

cd /home/seunghyun/tairte4/worktrees/pte_true_mma

log=/data/seunghyun/tairte4/artifacts/tairte4_contact_anchored/run053_Ea_fast_from_beta4_gpu4.log
attempt_log=/tmp/run053_Ea_fast_from_beta4_gpu4_reservation.$$.log

while true; do
  printf '\n[%s] requesting one 9-license reservation\n' "$(date -u +%FT%TZ)" >> "${log}"
  if /home/dhkim/bin/runres \
    --reserve-count 9 \
    --reserve-wait 1800 \
    --reserve-tag run053_Ea_fast_from_beta4_gpu4 \
    photothermal_pte/optimization_runs/run_051_052_constraint_aware_evaporated_Ea_Eb/run_Ea_fast_from_beta4.py \
    -th 8 -GPU 4 \
    > "${attempt_log}" 2>&1; then
    tee -a "${log}" < "${attempt_log}" >/dev/null
    rm -f "${attempt_log}"
    exit 0
  else
    status=$?
  fi
  tee -a "${log}" < "${attempt_log}" >/dev/null
  if grep -q 'job was NOT started' "${attempt_log}"; then
    printf '[%s] no license was acquired; retrying in 30 s\n' "$(date -u +%FT%TZ)" >> "${log}"
    sleep 30
    continue
  fi
  rm -f "${attempt_log}"
  exit "${status}"
done
