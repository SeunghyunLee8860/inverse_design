#!/bin/bash
# Single-evaluator smoke under the EXACT production environment.
#
#   GPU="GPU 0" ./run_smoke.sh [output-dir]
#
# Sources env_production.sh, the same file run_inverse_design.sh sources, so
# there is no hand-copied environment that can drift from production.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-/home/eidl/miniconda3/envs/EIDL-Lumapi/bin/python}"
GPU="${GPU:-GPU 0}"
OUT="${1:-$HERE/runs/smoke_$(date -u +%Y%m%dT%H%M%SZ)}"
cd "$HERE"
case "$OUT" in /*) : ;; *) OUT="$HERE/$OUT" ;; esac

. "$HERE/env_production.sh"

echo "[provenance] $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "  gpu=$GPU  out=$OUT"
echo "  r12_root=$R12"
echo "  lumapi=$R12/api/python/lumapi.py"
echo "  engine=$R12/bin/fdtd-engine"
echo "  PYTHONPATH=${PYTHONPATH:-<unset>}"

exec "$PY" "$HERE/smoke_single_evaluator.py" "$OUT" --pol "${POL:-x}"
