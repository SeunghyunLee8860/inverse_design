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
# Requires: Lumerical v261 + license + an idle GPU, and `pip install -r
# requirements.txt` (numpy scipy autograd nlopt).  Success = exact binary design
# that PASSES the independent geometry DRC and has an exact-binary FDTD FOM;
# it is NOT "beta reached N".
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-python}"
MODE="${1:-connected}"          # connected | isolated
GPU="${GPU:-GPU 1}"
OUT="${OUT:-runs/${MODE}_500nm_$(date +%Y%m%d 2>/dev/null || echo run)}"
cd "$HERE"

export CL_GPU_DEVICE="$GPU" LUMERICAL_SESSION_GPU_DEVICE="$GPU"
export FDTD_THREADS="${FDTD_THREADS:-16}"
export PERIOD_UM=6.0 TARGET_WL_UM=4.0
export SOURCE_WL_START_UM=3.0 SOURCE_WL_STOP_UM=6.0        # broadband source
export MATERIAL_FIT_START_UM=2.7 MATERIAL_FIT_STOP_UM=13.2 # material fit range
export BULK_MESH_MODE=auto MESH_ACCURACY=5
export VC_AUTO_SHUTOFF_MIN=1e-8 VC_SIM_TIME_S=4e-12
export MFS_UM=0.5 MGS_UM=0.5 ID_SEED=7 ID_SEED_AMP=0.10
export MSOPT_MAPPING=periodic_constrained

GAPARG=""
if [ "$MODE" = "isolated" ]; then
  export PERIODIC_ISOLATION_GAP_UM=0.5      # total tile-to-tile air gap (um)
  GAPARG="--min-gap-um 0.5"
  echo "[run] ISOLATED islands: 500 nm air moat between periodic cells"
elif [ "$MODE" = "connected" ]; then
  unset PERIODIC_ISOLATION_GAP_UM
  echo "[run] CONNECTED film: no forced air border"
else
  echo "usage: $0 {connected|isolated}"; exit 2
fi

echo "[preflight] $(date)"
$PY - <<'PYEOF' || { echo "[preflight] FAILED (install requirements / check Lumerical)"; exit 3; }
import importlib
for m in ("numpy", "scipy", "autograd", "nlopt"):
    importlib.import_module(m)
print("  python deps OK (numpy scipy autograd nlopt)")
PYEOF

echo "[optimize] $(date)  mode=$MODE gpu=$GPU out=$OUT"
$PY inverse_design/run_constrained_inverse_design.py --output "$OUT" \
    --mfs-um 0.5 --mgs-um 0.5 --beta-schedule "2,4,8,16,32,64" \
    --maxeval-per-stage 12 --rho-step 0.001 --gpu "$GPU" \
    ${RESUME:+--resume}

echo "[finalize] exact binary -> independent DRC -> exact FDTD"
if [ -f "$OUT/final_design.npz" ]; then
  $PY inverse_design/final_projection.py "$OUT/final_design.npz" \
      --mfs-um 0.5 --mgs-um 0.5 --gpu "$GPU" $GAPARG \
      --output "$OUT/final_projection"
  echo "[finalize] exit $? (0=SUCCESS.json written, 2=DRC failed -> no SUCCESS)"
fi
echo "[done] $(date)"
