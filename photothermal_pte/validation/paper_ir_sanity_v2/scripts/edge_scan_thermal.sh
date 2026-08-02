#!/bin/bash
# Thermal/PTE stage for the Fig.3I edge line scan (isolated bound only;
# isolated and perfect were degenerate at s=3 under this scenario).
set -u
PY=/home/eidl/miniconda3/envs/EIDL-Lumapi/bin/python
REPO=/home/seunghyun/tairte4/pte_inverse_design_adfd
OUTROOT=/data/seunghyun/tairte4/artifacts/paper_ir_device_a_end_to_end
GEOJSON=photothermal_pte/reports/paper_ir_device_a_end_to_end/device_a_geometry_digitization.json
STAMP=20260801
declare -A GPUOF=( [a]=4 [b]=3 )
cd "$REPO" || exit 1
for POL in a b; do
  for LABEL in sm1p5 s0 s1 s2 s3 s5; do
    OPT=$OUTROOT/scan40_w6p83_palik_finite_${POL}_${LABEL}_gpu${GPUOF[$POL]}_${STAMP}
    OUT=$OUTROOT/scan40_thermal_${POL}_${LABEL}_${STAMP}
    if [ -f "$OUT/summary.json" ]; then
      echo "SKIP existing $OUT"
      continue
    fi
    if [ ! -f "$OPT/case_result.json" ]; then
      echo "MISSING-OPTICS $OPT — skipping"
      continue
    fi
    rm -rf "$OUT"
    echo "=== THERMAL START pol=$POL pos=$LABEL $(date) ==="
    $PY photothermal_pte/validation/paper_ir_sanity/run_device_a_explicit_thermal_pte.py \
      --optical-case-dir "$OPT" \
      --output-dir "$OUT" \
      --thermal-domain-um 60 --si-depth-um 20 \
      --core-step-nm 100 --flake-dz-nm 10 \
      --geometry device-a-polygon \
      --geometry-contract-json "$GEOJSON" \
      --thermal-model expanded \
      --metal-thermalization isolated-lower-bound
    echo "=== THERMAL END pol=$POL pos=$LABEL status=$? $(date) ==="
  done
done
echo "=== SCAN THERMAL COMPLETE $(date) ==="
