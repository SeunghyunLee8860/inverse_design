#!/bin/bash
# Device-A named-scenario sweep: w0=6.83 um + Palik lossy SiO2.
# Runs empty-stack reference then finite-flake for one polarization.
# usage: scenario_sweep.sh <polarization a|b> <gpu-index> <initial-delay-s>
set -u
POL=$1
GPU=$2
DELAY=${3:-0}
STAMP=20260801
PY=/home/eidl/miniconda3/envs/EIDL-Lumapi/bin/python
REPO=/home/seunghyun/tairte4/pte_inverse_design_adfd
OUTROOT=/data/seunghyun/tairte4/artifacts/paper_ir_device_a_end_to_end
GEOJSON=photothermal_pte/reports/paper_ir_device_a_end_to_end/device_a_geometry_digitization.json
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
if d.get("validated", False):
    sys.exit(0)
rr = d.get("run_result") or {}
acc = rr.get("acceptance") or {}
failed = [k for k, v in acc.items() if not v]
shutoff_only = bool(acc) and set(failed) <= {
    "auto_shutoff_reached_requested_threshold"
}
completed = (rr.get("auto_shutoff") or {}).get(
    "simulation_completed_successfully", False
)
sys.exit(0 if (shutoff_only and completed) else 1)
PYEOF
}

run_case () {
  local CASE=$1
  local OUT=$2
  shift 2
  local OK=0
  for ATTEMPT in 1 2 3; do
    if acceptable "$OUT"; then
      echo "ACCEPTED $OUT"
      return 0
    fi
    if [ -d "$OUT" ]; then
      echo "REMOVE unacceptable $OUT"
      rm -rf "$OUT"
    fi
    echo "=== START case=$CASE pol=$POL gpu=$GPU attempt=$ATTEMPT $(date) ==="
    $PY photothermal_pte/validation/paper_ir_sanity/run_lumerical_device_a_ir_q.py \
      --output-dir "$OUT" \
      --case "$CASE" --polarization "$POL" --geometry device-a-polygon \
      --device-a-geometry-json "$GEOJSON" --include-electrodes \
      --domain-um 60 --source-span-um 50 --waist-um 12 \
      --scenario-waist-um 6.83 --sio2-model palik-lossy \
      --pml-layers 24 --flake-dz-nm 5 \
      --local-xy-mesh-nm 50 --refinement-half-span-um 15 \
      --simulation-time-ps 4 --auto-shutoff-min 1e-5 \
      --execution-contract production \
      --epsilon-c-model paper-b-closure \
      --gpu-device "GPU $GPU" --threads 3 "$@"
    STATUS=$?
    echo "=== END case=$CASE pol=$POL attempt=$ATTEMPT status=$STATUS $(date) ==="
    if acceptable "$OUT"; then
      echo "ACCEPTED $OUT"
      return 0
    fi
    echo "RETRY in 180 s"
    sleep 180
  done
  echo "GIVE-UP case=$CASE pol=$POL after 3 attempts"
  return 1
}

EMPTY=$OUTROOT/scenario_w6p83_palik_empty_${POL}_gpu${GPU}_${STAMP}
FINITE=$OUTROOT/scenario_w6p83_palik_finite_${POL}_gpu${GPU}_${STAMP}
run_case empty-stack "$EMPTY" || exit 1
run_case finite-flake "$FINITE" --incident-reference "$EMPTY/case_result.json" || exit 1
echo "=== SCENARIO SWEEP COMPLETE pol=$POL $(date) ==="
