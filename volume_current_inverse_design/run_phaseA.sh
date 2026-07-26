#!/bin/bash
# Phase A driver under the exact production environment.
#   GPU="GPU 5" ./run_phaseA.sh [outdir]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-/home/eidl/miniconda3/envs/EIDL-Lumapi/bin/python}"
command -v "$PY" >/dev/null 2>&1 || { echo "[fatal] no python: $PY" >&2; exit 3; }
GPU="${GPU:-GPU 5}"
OUT="${1:-$HERE/runs/phaseA_$(date -u +%Y%m%dT%H%M%SZ)}"
cd "$HERE"
case "$OUT" in /*) : ;; *) OUT="$HERE/$OUT" ;; esac

. "$HERE/env_production.sh"

echo "[provenance] $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "  gpu=$GPU  out=$OUT"
echo "  r12_root=$R12  PYTHONPATH=${PYTHONPATH:-<unset>}"
echo "  source_sha=$(git -C "$HERE" rev-parse HEAD 2>/dev/null || echo unknown)"

exec "$PY" "$HERE/phaseA_split_vs_combined.py" "$OUT" --beta "${BETA:-4.0}"
