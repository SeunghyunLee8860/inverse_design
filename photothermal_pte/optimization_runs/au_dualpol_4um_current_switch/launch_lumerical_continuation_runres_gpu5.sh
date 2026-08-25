#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repository="$(git -C "$script_dir" rev-parse --show-toplevel)"
commit="$(git -C "$repository" rev-parse --short=12 HEAD)"

export PATH="/home/eidl/miniconda3/envs/EIDL-Lumapi/bin:${PATH}"
export LD_LIBRARY_PATH="/home/eidl/miniconda3/envs/EIDL-Lumapi/lib"
export LUM_RESERVE_MODULE_DIR="${LUM_RESERVE_MODULE_DIR:-/home/seunghyun/tairte4/worktrees/pte_true_mma/tools/lumerical_runres}"
export AU_LUMERICAL_ACCELERATOR_POLICY="development"
export AU_LUMERICAL_OPT_BETA="1"
export AU_LUMERICAL_LICENSE_AUDIT_WAIT_S="${AU_LUMERICAL_LICENSE_AUDIT_WAIT_S:-1800}"
export AU_LUMERICAL_LICENSE_AUDIT_POLL_S="${AU_LUMERICAL_LICENSE_AUDIT_POLL_S:-5}"
export AU_LUMERICAL_OPT_OUTPUT_ROOT="${AU_LUMERICAL_OPT_OUTPUT_ROOT:-/home/seunghyun/tairte4_raw_artifacts/au_dualpol_4um_lumerical_production/continuation_${commit}}"
export AU_LUMERICAL_EA_SOURCE_CALIBRATION="${AU_LUMERICAL_EA_SOURCE_CALIBRATION:-/home/seunghyun/tairte4_raw_artifacts/au_dualpol_4um_lumerical_development/r12_gpu5_source_only_Ea_z2p5_bulk50_cv0_MCM6/source_only_Ea_fine_z2p5_bulk50_xy100_cv0_pml8_span20_z6_t1ps.json}"
export AU_LUMERICAL_EB_SOURCE_CALIBRATION="${AU_LUMERICAL_EB_SOURCE_CALIBRATION:-/home/seunghyun/tairte4_raw_artifacts/au_dualpol_4um_lumerical_development/r12_gpu5_source_only_Eb_z2p5_bulk50_cv0_MCM6/source_only_Eb_fine_z2p5_bulk50_xy100_cv0_pml8_span20_z6_t1ps.json}"
: "${AU_LUMERICAL_EA_FINAL_XY50_SOURCE_CALIBRATION:?set the passed Ea 50-nm source-only JSON before production launch}"
: "${AU_LUMERICAL_EB_FINAL_XY50_SOURCE_CALIBRATION:?set the passed Eb 50-nm source-only JSON before production launch}"
export AU_LUMERICAL_EA_FINAL_XY50_SOURCE_CALIBRATION
export AU_LUMERICAL_EB_FINAL_XY50_SOURCE_CALIBRATION
export FDTD_THREADS="${FDTD_THREADS:-8}"

if [[ ! -f "$LUM_RESERVE_MODULE_DIR/lum_reserve.py" ]]; then
  echo "missing runres reservation module: $LUM_RESERVE_MODULE_DIR/lum_reserve.py" >&2
  exit 2
fi

cd "$repository"
exec /home/dhkim/bin/runres \
  --reserve-count 9 \
  --reserve-wait 1800 \
  --reserve-tag au4um_lumerical_beta_continuation_gpu5 \
  photothermal_pte/optimization_runs/au_dualpol_4um_current_switch/41_optimize_lumerical_4um_dualpol_continuation.py \
  -th "$FDTD_THREADS" -GPU 5
