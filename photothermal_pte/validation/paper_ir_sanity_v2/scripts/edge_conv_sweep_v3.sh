#!/bin/bash
# Edge optical-Q mesh-convergence sweep v3.
# 4 ps + acceptance check that tolerates ONLY the auto-shutoff-floor gate
# (same precedent as the accepted w2 planar/edge diagnostics).
# usage: edge_conv_sweep_v3.sh <polarization a|b> <gpu-index> <initial-delay-s>
set -u
POL=$1
GPU=$2
DELAY=${3:-0}
STAMP=20260801
PY=/home/eidl/miniconda3/envs/EIDL-Lumapi/bin/python
REPO=/home/seunghyun/tairte4/pte_inverse_design_adfd
OUTROOT=/data/seunghyun/tairte4/artifacts/paper_ir_lumerical_sanity
cd "$REPO" || exit 1
sleep "$DELAY"

acceptable () {
  $PY - "$1" <<'PYEOF'
import json, sys
from pathlib import Path
path = Path(sys.argv[1]) / "case_result.json"
if not path.is_file():
    sys.exit(1)
d = json.loads(path.read_text())
rr = d.get("run_result") or {}
acc = rr.get("acceptance") or {}
if not acc:
    sys.exit(1)
failed = [k for k, v in acc.items() if not v]
shutoff_only = set(failed) <= {"auto_shutoff_reached_requested_threshold"}
completed = (rr.get("auto_shutoff") or {}).get(
    "simulation_completed_successfully", False
)
sys.exit(0 if (shutoff_only and completed) else 1)
PYEOF
}

for MESH in 50 25 12.5; do
  TAGMESH=${MESH/./p}
  OUT=$OUTROOT/w2edge_conv_${POL}_xy${TAGMESH}_dz5_t4_gpu${GPU}_${STAMP}
  OK=0
  for ATTEMPT in 1 2 3; do
    if acceptable "$OUT"; then
      echo "ACCEPTED $OUT"
      OK=1
      break
    fi
    if [ -d "$OUT" ]; then
      echo "REMOVE unacceptable $OUT"
      rm -rf "$OUT"
    fi
    echo "=== START pol=$POL mesh=${MESH}nm gpu=$GPU attempt=$ATTEMPT $(date) ==="
    $PY photothermal_pte/validation/paper_ir_sanity/run_lumerical_device_a_ir_q.py \
      --output-dir "$OUT" \
      --case finite-flake --geometry straight-45-edge --polarization "$POL" \
      --domain-um 12 --source-span-um 6 --waist-um 2 \
      --pml-layers 24 --flake-dz-nm 5 \
      --local-xy-mesh-nm "$MESH" \
      --simulation-time-ps 4 --auto-shutoff-min 1e-5 \
      --execution-contract edge-isolation-smoke \
      --epsilon-c-model paper-b-closure \
      --gpu-device "GPU $GPU" --threads 3
    STATUS=$?
    echo "=== END pol=$POL mesh=${MESH}nm attempt=$ATTEMPT status=$STATUS $(date) ==="
    if acceptable "$OUT"; then
      echo "ACCEPTED $OUT"
      OK=1
      break
    fi
    echo "RETRY in 180 s"
    sleep 180
  done
  if [ $OK -ne 1 ]; then
    echo "GIVE-UP pol=$POL mesh=${MESH}nm after 3 attempts; continuing"
  fi
done
echo "=== SWEEP COMPLETE pol=$POL $(date) ==="
