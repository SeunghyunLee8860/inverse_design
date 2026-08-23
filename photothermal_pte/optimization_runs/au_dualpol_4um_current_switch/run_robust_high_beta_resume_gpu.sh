#!/usr/bin/env bash
set -euo pipefail
export AU_ROBUST_RESUME_HIGH_BETA=1
exec "$(dirname "$0")/run_robust_projection_gpu.sh" "${1:-0}"
