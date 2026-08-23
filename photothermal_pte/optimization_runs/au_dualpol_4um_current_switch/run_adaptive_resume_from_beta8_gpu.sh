#!/usr/bin/env bash
set -euo pipefail

# Resume only from the immutable, completed beta=8 raw checkpoint.  The
# production output/history remains in the same published directory, but the
# failed beta=16 jump is truncated at evaluation 66 before new evaluations are
# written.  No CPU Maxwell or thermal fallback is permitted by the wrapped
# GPU runner.
export AU_DUALPOL_RESUME_STAGE_NPZ="/home/seunghyun/tairte4/raw/au_dualpol_4um_current_switch/optimization_ld_mma/stage_03_beta_8.npz"
export AU_DUALPOL_RESUME_EVALUATION="66"
export AU_DUALPOL_BETAS="12,16,24,32,48,64,96,128"

exec "$(dirname "$0")/run_optimization_gpu.sh" "${1:-0}"
