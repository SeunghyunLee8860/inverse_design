#!/bin/bash
# Phase D: final full-chain AD/FD certification under the SELECTED runtime
# configuration (combined adjoint + converged sim time + validated fast bulk).
#
#   GPU="GPU 5" SIM_TIME=2e-12 BULK=fast_bulk ./run_phaseD.sh [logdir]
#
# Runs the SAME certification tests that produced the 4 ps/split/auto baseline
# (safe beta=4: 1.80%, safe beta=32: 2.33%, exact beta=8: 1.20%):
#   tests/test_full_chain_filter_project_adfd.py
# Production approval requires all three AD/FD relative errors < 5%.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-/home/eidl/miniconda3/envs/EIDL-Lumapi/bin/python}"
command -v "$PY" >/dev/null 2>&1 || { echo "[fatal] no python: $PY" >&2; exit 3; }
GPU="${GPU:-GPU 5}"
SIM_TIME="${SIM_TIME:-4e-12}"
BULK="${BULK:-auto}"
ADJ="${ADJ:-combined}"
OUT="${1:-$HERE/runs/phaseD_$(date -u +%Y%m%dT%H%M%SZ)}"
cd "$HERE"
case "$OUT" in /*) : ;; *) OUT="$HERE/$OUT" ;; esac
mkdir -p "$OUT"

. "$HERE/env_production.sh"
export VC_SIM_TIME_S="$SIM_TIME"
export BULK_MESH_MODE="$BULK"
export VC_ADJOINT_COMPONENT_MODE="$ADJ"

echo "[provenance] $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "  gpu=$GPU  sim_time=$SIM_TIME  bulk=$BULK  adjoint=$ADJ"
echo "  r12_root=$R12  PYTHONPATH=${PYTHONPATH:-<unset>}"
echo "  source_sha=$(git -C "$HERE" rev-parse HEAD 2>/dev/null || echo unknown)"

# -s so the [adfd] rel-error lines land in the log; failures are the verdict.
exec "$PY" -m pytest tests/test_full_chain_filter_project_adfd.py -q -s \
    2>&1 | tee "$OUT/phaseD_pytest.log"
