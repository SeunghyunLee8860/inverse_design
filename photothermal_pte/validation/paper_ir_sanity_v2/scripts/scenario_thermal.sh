#!/bin/bash
# Thermal/PTE stage for the w0=6.83um + Palik-SiO2 named scenario.
# Runs (a,b) x (isolated, perfect) sequentially on CPU.
set -u
PY=/home/eidl/miniconda3/envs/EIDL-Lumapi/bin/python
REPO=/home/seunghyun/tairte4/pte_inverse_design_adfd
OUTROOT=/data/seunghyun/tairte4/artifacts/paper_ir_device_a_end_to_end
GEOJSON=photothermal_pte/reports/paper_ir_device_a_end_to_end/device_a_geometry_digitization.json
STAMP=20260801
cd "$REPO" || exit 1
declare -A GPUOF=( [a]=4 [b]=0 )
for POL in a b; do
  OPT=$OUTROOT/scenario_w6p83_palik_finite_${POL}_gpu${GPUOF[$POL]}_${STAMP}
  for MODE in isolated-lower-bound perfect-to-flake-upper-bound; do
    SHORT=${MODE%%-*}
    OUT=$OUTROOT/scenario_thermal_${POL}_${SHORT}_${STAMP}
    if [ -f "$OUT/summary.json" ]; then
      echo "SKIP existing $OUT"
      continue
    fi
    rm -rf "$OUT"
    echo "=== THERMAL START pol=$POL mode=$MODE $(date) ==="
    $PY photothermal_pte/validation/paper_ir_sanity/run_device_a_explicit_thermal_pte.py \
      --optical-case-dir "$OPT" \
      --output-dir "$OUT" \
      --thermal-domain-um 60 --si-depth-um 20 \
      --core-step-nm 100 --flake-dz-nm 10 \
      --geometry device-a-polygon \
      --geometry-contract-json "$GEOJSON" \
      --thermal-model expanded \
      --metal-thermalization "$MODE"
    echo "=== THERMAL END pol=$POL mode=$MODE status=$? $(date) ==="
  done
done
echo "=== THERMAL SWEEP COMPLETE $(date) ==="