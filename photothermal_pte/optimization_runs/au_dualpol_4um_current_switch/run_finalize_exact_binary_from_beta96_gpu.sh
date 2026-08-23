#!/usr/bin/env bash
set -euo pipefail

# Promote no further continuous stage.  Apply both exact 500 nm binary repair
# orders to the last performance-retaining beta=96 checkpoint, audit them, and
# rerun the full two-polarization Maxwell -> thermal -> electrical chain.
raw_root="${AU_DUALPOL_RAW_ROOT:-/home/seunghyun/tairte4/raw/au_dualpol_4um_current_switch}"
export AU_DUALPOL_RESUME_STAGE_NPZ="$raw_root/optimization_ld_mma/stage_11_beta_96.npz"
export AU_DUALPOL_RESUME_EVALUATION="183"
export AU_DUALPOL_BETAS=""
export AU_DUALPOL_FINALIZE_ONLY="1"

exec "$(dirname "$0")/run_optimization_gpu.sh" "${1:-0}"
