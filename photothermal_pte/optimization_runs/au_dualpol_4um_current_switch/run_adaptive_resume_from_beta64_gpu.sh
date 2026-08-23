#!/usr/bin/env bash
set -euo pipefail

# The direct beta=64 -> 96 transition was feasible but lost 59% of the
# epigraph objective.  Resume the last promoted beta=64 checkpoint and insert
# factor-1.25/1.2 continuation stages.  The Python driver also refuses any
# stage whose returned objective is below 90% of its predecessor.
raw_root="${AU_DUALPOL_RAW_ROOT:-/home/seunghyun/tairte4/raw/au_dualpol_4um_current_switch}"
export AU_DUALPOL_RESUME_STAGE_NPZ="$raw_root/optimization_ld_mma/stage_09_beta_64.npz"
export AU_DUALPOL_RESUME_EVALUATION="153"
export AU_DUALPOL_BETAS="80,96,112,128"

exec "$(dirname "$0")/run_optimization_gpu.sh" "${1:-0}"
