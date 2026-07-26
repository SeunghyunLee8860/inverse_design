#!/bin/bash
# Phase B: simulation-time convergence under the exact production environment.
#   GPU="GPU 5" BASELINE=/path/to/phaseA_run ./run_phaseB.sh [outdir]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-/home/eidl/miniconda3/envs/EIDL-Lumapi/bin/python}"
command -v "$PY" >/dev/null 2>&1 || { echo "[fatal] no python: $PY" >&2; exit 3; }
GPU="${GPU:-GPU 5}"
[ -n "${BASELINE:-}" ] || { echo "[fatal] set BASELINE=<passed phaseA outdir>" >&2; exit 3; }
OUT="${1:-$HERE/runs/phaseB_$(date -u +%Y%m%dT%H%M%SZ)}"
cd "$HERE"
case "$OUT" in /*) : ;; *) OUT="$HERE/$OUT" ;; esac

. "$HERE/env_production.sh"

echo "[provenance] $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "  gpu=$GPU  out=$OUT  baseline=$BASELINE"
echo "  r12_root=$R12  PYTHONPATH=${PYTHONPATH:-<unset>}"
echo "  source_sha=$(git -C "$HERE" rev-parse HEAD 2>/dev/null || echo unknown)"

exec "$PY" "$HERE/phaseB_simtime_convergence.py" "$OUT" \
    --baseline-dir "$BASELINE" --times "${TIMES:-3e-12,2e-12}" \
    --beta "${BETA:-4.0}"
