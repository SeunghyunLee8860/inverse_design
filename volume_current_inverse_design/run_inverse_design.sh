#!/bin/bash
# TaIrTe4 volume-current coherent-FOM inverse design, 500 nm length scale.
# Portable launcher (derives its own root); runs BOTH design variants:
#
#   ./run_inverse_design.sh connected     # no air border (connected film)
#   ./run_inverse_design.sh isolated      # 500 nm air moat -> isolated islands
#
# Env overrides: GPU="GPU 3", PY=/path/to/python, OUT=runs/foo, RESUME=1,
#                FDTD_THREADS=16, VC_LUMERICAL_ROOT=/path/to/lumerical/v261
#
# Adaptive beta continuation (fixed ladder, adaptive per-stage eval count):
#   BETA_SCHEDULE=2,4,8,16,32,64   MAXEVAL=12   MIN_EVALS_PER_STAGE=3
#   VC_CONV_WINDOW=3  VC_OBJ_REL_TOL=5e-3
#   VC_LATENT_RMS_TOL=1e-3  VC_LATENT_MAX_TOL=1e-2
# A stage advances early only when the objective plateaus while constraint-
# feasible with a quiet latent; an infeasible plateau or an all-infeasible
# maxeval stage ABORTS the run (exit 7).  See inverse_design/adaptive_stage.py.
#
# FAILURE PROPAGATION (review P0-3): optimizer or DRC/FDTD failure aborts with a
# nonzero exit; a stale final_design.npz from a previous attempt is deleted
# before optimizing; the launcher exits 0 ONLY when this run wrote SUCCESS.json.
# Success = exact binary design that PASSES the independent geometry DRC and has
# an exact-binary FDTD FOM -- NOT "beta reached N".
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# A detached/setsid launch does not inherit an interactive PATH, so bare
# "python" is often absent there.  Fall back to the lab conda interpreter and
# fail closed rather than dying at the first solve.
PY="${PY:-python}"
command -v "$PY" >/dev/null 2>&1 || PY=/home/eidl/miniconda3/envs/EIDL-Lumapi/bin/python
command -v "$PY" >/dev/null 2>&1 || { echo "[fatal] no usable python; set PY=" >&2; exit 3; }
MODE="${1:-connected}"
GPU="${GPU:-GPU 1}"
OUT="${OUT:-runs/${MODE}_500nm}"
cd "$HERE"
# P0-1: make OUT absolute so the runner (resolves --output vs CWD) and this
# script agree on where final_design.npz / SUCCESS.json live.
case "$OUT" in /*) : ;; *) OUT="$HERE/$OUT" ;; esac

# Single source of truth for the production environment; the smoke test sources
# the same file so it cannot drift from what production actually runs under.
. "$HERE/env_production.sh"

GAPARG=""
if [ "$MODE" = "isolated" ]; then
  export PERIODIC_ISOLATION_GAP_UM=0.5
  GAPARG="--min-gap-um 0.5"
  echo "[run] ISOLATED islands: 500 nm air moat between periodic cells"
elif [ "$MODE" = "connected" ]; then
  unset PERIODIC_ISOLATION_GAP_UM || true
  echo "[run] CONNECTED film: no forced air border"
else
  echo "usage: $0 {connected|isolated}"; exit 2
fi

echo "[preflight] $(date)"
$PY - <<'PYEOF'
import importlib
for m in ("numpy", "scipy", "autograd", "nlopt"):
    importlib.import_module(m)
print("  python deps OK (numpy scipy autograd nlopt)")
PYEOF

# Preflight the API/engine pair in a THROWAWAY process: if the wrong lumapi
# would be loaded, fail here rather than minutes into the first adjoint.
$PY - <<'PYEOF'
import json, sys
sys.path.insert(0, "."); sys.path.insert(0, "bundle")
import eqc_lib as lib
lib.bootstrap_env()
import lumapi
lib.assert_approved_lumapi(lumapi)
print("  lumapi preflight OK: " + json.dumps(lib.import_provenance(), default=str))
PYEOF

echo "[provenance] $(date)"
echo "  mode=$MODE  gpu=$GPU  out=$OUT"
echo "  r12_root=$R12"
echo "  lumapi=$R12/api/python/lumapi.py"
echo "  engine=$R12/bin/fdtd-engine"
echo "  PYTHONPATH=${PYTHONPATH:-<unset>}"
echo "  source_sha=$(cd "$HERE" && git rev-parse HEAD 2>/dev/null || echo 'not-a-git-checkout')"

echo "[optimize] $(date)  mode=$MODE gpu=$GPU out=$OUT"
rm -f "$OUT/final_design.npz" 2>/dev/null || true   # never finalise a prior attempt's design
# BETA_SCHEDULE / MAXEVAL exist so the pre-production smoke can run the REAL
# entrypoint with a short schedule instead of a separate near-copy of it.
$PY inverse_design/run_constrained_inverse_design.py --output "$OUT" \
    --mfs-um 0.5 --mgs-um 0.5 --beta-schedule "${BETA_SCHEDULE:-2,4,8,16,32,64}" \
    --maxeval-per-stage "${MAXEVAL:-12}" --rho-step 0.001 --gpu "$GPU" ${RESUME:+--resume}

if [ ! -f "$OUT/final_design.npz" ]; then
  echo "[finalize] no final_design.npz -> optimizer produced nothing; ABORT"; exit 4
fi
echo "[finalize] exact binary -> independent DRC -> exact FDTD"
frc=0
$PY inverse_design/final_projection.py "$OUT/final_design.npz" \
    --mfs-um 0.5 --mgs-um 0.5 --gpu "$GPU" $GAPARG \
    --output "$OUT/final_projection" || frc=$?
# P0-3: a zero finalizer rc with NO SUCCESS.json must still be a launcher failure.
if [ "$frc" -ne 0 ]; then
  echo "[finalize] FAILED: final_projection rc=$frc"; exit "$frc"
fi
if [ ! -f "$OUT/final_projection/SUCCESS.json" ]; then
  echo "[finalize] FAILED: no SUCCESS.json"; exit 5
fi
echo "[done] SUCCESS $(date) -> $OUT/final_projection/SUCCESS.json"
