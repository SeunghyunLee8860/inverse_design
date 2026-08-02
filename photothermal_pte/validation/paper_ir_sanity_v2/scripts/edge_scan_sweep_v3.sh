#!/bin/bash
# Paper-faithful Fig.3I edge line scan under the corrected model
# (Palik lossy SiO2 + w0=6.83um), span 40um for clearance.
# usage: edge_scan_sweep.sh <polarization a|b> <gpu-index> <initial-delay-s>
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
allowed = {
    "auto_shutoff_reached_requested_threshold",
    "six_face_closure_lt_0p5_percent",
}
ok = bool(acc) and set(failed) <= allowed
if ok and "six_face_closure_lt_0p5_percent" in failed:
    closure = rr.get("six_face_relative_closure")
    p_q = rr.get("P_Q_W")
    incident = (rr.get("normalization") or {}).get(
        "incident_power_W_at_1_W_m2"
    )
    ok = (
        None not in (closure, p_q, incident)
        and incident
        and abs(closure) * abs(p_q) / abs(incident) < 0.01
    )
completed = (rr.get("auto_shutoff") or {}).get(
    "simulation_completed_successfully", False
)
sys.exit(0 if (ok and completed) else 1)
PYEOF
}

run_case () {
  local CASE=$1
  local OUT=$2
  shift 2
  for ATTEMPT in 1 2 3; do
    if acceptable "$OUT"; then
      echo "ACCEPTED $OUT"
      return 0
    fi
    if [ -d "$OUT" ]; then
      echo "REMOVE unacceptable $OUT"
      rm -rf "$OUT"
    fi
    echo "=== START case=$CASE pol=$POL out=$(basename $OUT) attempt=$ATTEMPT $(date) ==="
    $PY photothermal_pte/validation/paper_ir_sanity/run_lumerical_device_a_ir_q.py \
      --output-dir "$OUT" \
      --case "$CASE" --polarization "$POL" --geometry device-a-polygon \
      --device-a-geometry-json "$GEOJSON" --include-electrodes \
      --domain-um 60 --source-span-um 40 --waist-um 12 \
      --scenario-waist-um 6.83 --sio2-model palik-lossy \
      --pml-layers 24 --flake-dz-nm 5 \
      --local-xy-mesh-nm 50 --refinement-half-span-um 15 \
      --simulation-time-ps 4 --auto-shutoff-min 1e-5 \
      --execution-contract production \
      --epsilon-c-model paper-b-closure \
      --gpu-device "GPU $GPU" --threads 3 "$@"
    echo "=== END case=$CASE pol=$POL attempt=$ATTEMPT status=$? $(date) ==="
    if acceptable "$OUT"; then
      echo "ACCEPTED $OUT"
      return 0
    fi
    echo "RETRY in 180 s"
    sleep 180
  done
  echo "GIVE-UP case=$CASE pol=$POL $(basename $OUT)"
  return 1
}

EMPTY=$OUTROOT/scan40_w6p83_palik_empty_${POL}_gpu${GPU}_${STAMP}
run_case empty-stack "$EMPTY" || exit 1

# signed s from edge (um, + = inward); offset = (s-3) * n_hat,
# n_hat = (0.80533, -0.59281) from the digitized off-axis edge.
run_scan_point () {
  local LABEL=$1 DX=$2 DY=$3
  local OUT=$OUTROOT/scan40_w6p83_palik_finite_${POL}_${LABEL}_gpu${GPU}_${STAMP}
  run_case finite-flake "$OUT" \
    --incident-reference "$EMPTY/case_result.json" \
    --beam-offset-x-um "$DX" --beam-offset-y-um "$DY"
}

run_scan_point sm1p5 -3.6240 2.6676
run_scan_point s0    -2.4160 1.7784
run_scan_point s1    -1.6107 1.1856
run_scan_point s2    -0.8053 0.5928
run_scan_point s3     0.0000 0.0000
run_scan_point s5     1.6107 -1.1856
echo "=== EDGE SCAN SWEEP COMPLETE pol=$POL $(date) ==="
